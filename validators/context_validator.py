"""
validators/context_validator.py
-------------------------------
Context-first DPR validation. Replaces the brittle string/number matcher with a
retrieve-then-understand approach:

  1. RULE = a REQUIREMENT (from the rulebook). FACT = a CLAIM (from the DPR).
  2. For each rule, gather candidate DPR facts — both syntactically (shared whole
     word) and semantically (local mxbai embeddings, top-k) — each WITH its
     surrounding context.
  3. One bounded LLM call (Qwen-14B) reads requirement + candidates-in-context and:
       - picks the value that actually corresponds to the rule's intent
         (design speed ≠ average/sectional speed; a length ≠ a gradient),
       - applies dimensional sanity (slope ≠ length, % ≠ metres),
       - judges compliance, comparing only like-with-like.

No knowledge graph. Embeddings stay local (Ollama mxbai-embed-large).
"""

import json
import uuid
from collections import Counter
from datetime import datetime

import numpy as np
from loguru import logger

from config.settings import NodeLabel, RelType, Severity, OUTPUT_DIR, get_applicable_sectors
from utils.neo4j_client import run_read
from utils.llm_router import generate, generate_json, get_model_for_task, TaskType
from config.backend_settings import LMSTUDIO_SEED, VALIDATION_VOTES
from validators.quantity import NUMERIC_OPS, numeric_verdict
# reuse the (fixed, local, timed) embedding call and the whole-word tokenizer
from validators.validation_engine import _embed_texts_local, _attr_tokens, _describe_operator

SEVERITY_WEIGHTS = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1, Severity.INFO: 1}
TOP_K_SEMANTIC   = 6      # semantic candidates per rule
MAX_CANDIDATES   = 10     # cap sent to the LLM
# Below this vote-agreement the judgment was not stable across samples — surface it as
# "review" rather than presenting a jittery verdict as firm. (confidence = fraction of votes
# that agreed on the chosen candidate; deterministic/clear rows score 1.0.)
REVIEW_CONFIDENCE = 0.6


# ─── Load requirements (rules) and claims (facts) ──────────────────────────────

def _load_rules(sector: str) -> list[dict]:
    applicable = get_applicable_sectors(sector)
    return run_read(
        f"""
        MATCH (r:{NodeLabel.RULE})-[:{RelType.BELONGS_TO}]->(s:{NodeLabel.SECTOR})
        WHERE s.name IN $applicable
        RETURN r.rule_id AS rid, r.attribute AS attr, r.operator AS op,
               r.threshold AS thr, r.unit AS unit, r.condition AS cond,
               r.rule_text AS text, r.severity AS sev, r.standard_name AS std,
               r.clause AS clause
        """,
        {"applicable": applicable},
    )


def _load_facts(doc_id: str) -> list[dict]:
    return run_read(
        f"""
        MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $id}})-[:{RelType.HAS_FACT}]->(f:{NodeLabel.FACT})
        WHERE f.fact_type <> 'table_row'
        RETURN f.fact_id AS fid, f.attribute AS attr, f.subject AS subj,
               f.value AS val, f.unit AS unit, f.context AS ctx, f.nature AS nature,
               f.source_page AS page, f.printed_page AS ppage
        LIMIT 4000
        """,
        {"id": doc_id},
    )


# ─── Candidate selection: syntactic ∪ semantic ─────────────────────────────────

def _candidates(rule: dict, facts: list[dict], fact_embs, rule_emb) -> list[dict]:
    rtokens = _attr_tokens(rule.get("attr", ""))
    picked: dict[int, dict] = {}

    # syntactic: facts sharing a whole word with the rule's parameter
    for i, f in enumerate(facts):
        if _attr_tokens(f.get("attr", "")) & rtokens:
            picked[i] = f

    # semantic: nearest facts by embedding (catches synonyms/wording)
    if fact_embs is not None and fact_embs.size and rule_emb is not None:
        sims = fact_embs @ rule_emb
        for i in np.argsort(-sims)[:TOP_K_SEMANTIC]:
            if float(sims[i]) >= 0.55:
                picked.setdefault(int(i), facts[int(i)])

    return list(picked.values())[:MAX_CANDIDATES]


# ─── LLM judgment (context-aware) ──────────────────────────────────────────────

_SYSTEM = (
    "You are an infrastructure-DPR compliance auditor. You decide whether a value PROPOSED in a "
    "project document satisfies a STANDARD REQUIREMENT. You reason about MEANING, not text, and "
    "you run the SAME procedure for every parameter — no parameter is special-cased.\n\n"
    "Model BOTH the requirement and each candidate as a QUANTITY with four facets, resolved from "
    "context (not from labels or raw digits):\n"
    "  1. REFERENT — the physical thing measured, identified by meaning. Differently-worded "
    "labels can be the SAME quantity; the same word can be DIFFERENT quantities. Judge by what "
    "is referred to.\n"
    "  2. DIMENSION — the kind of quantity (length, slope/ratio, speed, mass, count, currency, "
    "…).\n"
    "  3. MAGNITUDE — the size in a canonical unit, AFTER normalising the notation: units and "
    "prefixes, ratios/percentages to a decimal, and inverse or non-linear encodings to their "
    "true magnitude. Never compare raw printed numbers.\n"
    "  4. FRAME — the operator's direction (which way 'satisfies' runs) and any CONDITION under "
    "which the requirement applies.\n\n"
    "Then judge by commensurability, and DEFER rather than force a verdict when you cannot judge "
    "soundly:\n"
    "  • Compare only commensurable quantities. Same dimension → compare magnitudes. Different "
    "but physically related (linked by a known relationship) → convert, then compare. Otherwise "
    "do not compare.\n"
    "  • Pick the value the DPR actually ADOPTS ('proposed') over one it merely quotes "
    "('standard_reference').\n"
    "  • Verdicts: 'Not Applicable' if the requirement's condition does not apply to THIS "
    "project; 'Not Found' if no candidate refers to the required quantity; 'Needs Human' if the "
    "quantities are incommensurable OR the requirement itself is ill-formed / internally "
    "inconsistent / its unit does not fit its referent (a likely extraction error) — do not "
    "guess a Compliant/Non-Compliant in that case; only 'Compliant'/'Non-Compliant' when a sound "
    "like-with-like comparison is actually possible."
)


def _project_profile(facts: list[dict]) -> str:
    """One general, doc-agnostic summary of THIS project's defining attributes, derived
    once from the DPR's own facts. Fed into every requirement judgment so condition
    applicability and word-sense disambiguation are grounded in the actual project rather
    than guessed per rule. No hardcoded parameters — the model decides what is defining."""
    # Prefer the DPR's own ('proposed') short factual claims; cap to keep the call light.
    sample, seen = [], set()
    for f in facts:
        if (f.get("nature") or "proposed") != "proposed":
            continue
        key = (f.get("attr", ""), str(f.get("val", "")))
        if key in seen:
            continue
        seen.add(key)
        sample.append(f"- {f.get('attr','')}: {f.get('val','')} {f.get('unit','')}".strip())
        if len(sample) >= 70:
            break
    if not sample:
        return "(no project facts available)"
    prompt = (
        "From these facts extracted from an infrastructure DPR, write a SHORT profile (4-6 "
        "lines) of the project's DEFINING attributes — only what's needed to decide which "
        "standards/conditions apply. Cover, when present: project type (e.g. new line / "
        "doubling / 3rd-4th line / gauge conversion), number of tracks, electrification, "
        "gauge, route/section, speed class/category, traction, and any other defining trait. "
        "State only what the facts support; if something isn't stated, omit it. Plain text.\n\n"
        + "\n".join(sample)
    )
    try:
        txt = generate(prompt, model=get_model_for_task(TaskType.VALIDATION), temperature=0.0)
        return (txt or "").strip()[:800] or "(project profile unavailable)"
    except Exception as e:  # never let profiling crash the whole validation
        logger.warning(f"[context-validation] project profile failed: {e}")
        return "(project profile unavailable)"


def _load_or_make_profile(doc_id: str, facts: list[dict]) -> str:
    """Return the cached project profile for this doc, computing+saving it once. Pinning the
    profile makes the validation context identical across runs (delete the file to refresh)."""
    cache = OUTPUT_DIR / f".profile_{doc_id}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8")).get("profile") or _project_profile(facts)
        except Exception:
            pass
    prof = _project_profile(facts)
    try:
        cache.write_text(json.dumps({"profile": prof}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[context-validation] could not cache profile: {e}")
    return prof


def _judge(rule: dict, cands: list[dict], profile: str = "") -> dict:
    if not cands:
        return {"verdict": "Not Found", "reason": "No DPR value found for this requirement.",
                "confidence": 1.0, "candidate_index": -1}

    req = _describe_operator(rule.get("op", ""), str(rule.get("thr", "")), str(rule.get("unit", "")))
    lines = []
    for i, c in enumerate(cands):
        lines.append(json.dumps({
            "i": i,
            "value": f"{c.get('val','')} {c.get('unit','')}".strip(),
            "attribute": c.get("attr", ""),
            "nature": c.get("nature") or "proposed",   # 'proposed' = DPR's own; 'standard_reference' = quoted code
            "context": (c.get("ctx") or "")[:180],
            "page": c.get("ppage") or c.get("page"),
        }, ensure_ascii=False))

    prompt = (
        f"PROJECT CONTEXT (what kind of project this DPR is — use it for applicability and meaning):\n"
        f"{profile or '(not available)'}\n\n"
        f"REQUIREMENT (from standard '{rule.get('std','')}' {rule.get('clause','') or ''}):\n"
        f"  Parameter : {rule.get('attr','')}\n"
        f"  Rule text : {rule.get('text','')}\n"
        f"  Requires  : {req}\n"
        f"  Condition : {rule.get('cond','') or '(none)'}\n\n"
        "CANDIDATE VALUES proposed in the DPR (one per line, with context):\n"
        + "\n".join(lines) + "\n\n"
        "Your job is the SEMANTIC judgment only — a separate deterministic step does the unit "
        "normalisation and the ≤/≥/= arithmetic, so do NOT compute compliance for numeric "
        "requirements yourself. Resolve the four facets (referent, dimension, magnitude, frame) "
        "and answer:\n"
        "  • applicable: does the requirement's Condition apply to THIS project (per PROJECT "
        "CONTEXT)? false → it will be marked Not Applicable.\n"
        "  • well_formed: is the requirement internally consistent and does its unit/dimension fit "
        "its referent? false (a likely extraction error) → it will be marked Needs Human.\n"
        "  • candidate_index: the ONE candidate whose REFERENT is the required quantity (match by "
        "meaning, not words — a synonym counts, the same word in another sense does not). Prefer "
        "nature 'proposed' over 'standard_reference', BUT if the only candidate(s) referring to "
        "the quantity are 'standard_reference', still pick the best one — use -1 (Not Found) ONLY "
        "when the quantity is genuinely absent from the DPR, not merely when it appears as a "
        "quoted/standard value.\n"
        "  • verdict: your overall judgment, used ONLY for non-numeric requirements; for numeric "
        "ones the deterministic step decides.\n"
        'Return ONLY JSON: {"applicable": <bool>, "well_formed": <bool>, "candidate_index": <int>, '
        '"verdict": "Compliant"|"Non-Compliant"|"Not Applicable"|"Not Found"|"Needs Human", '
        '"reason": "<one sentence>"}'
    )

    # Self-consistency: vote over a fixed seed-set so the run is REPRODUCIBLE yet robust to the
    # thinking model's residual nondeterminism. The deterministic numeric verdict (below) then
    # overrides for numeric operators, so flip-flop only ever touches the semantic fields.
    base = LMSTUDIO_SEED if LMSTUDIO_SEED is not None else 0
    samples = []
    for i in range(max(1, VALIDATION_VOTES)):
        r = generate_json(prompt, system=_SYSTEM, model=get_model_for_task(TaskType.VALIDATION),
                          temperature=0.0, seed=base + i)
        if isinstance(r, dict):
            samples.append(r)
    if not samples:
        return {"verdict": "Needs Human", "reason": "Adjudication failed to parse.",
                "confidence": 0.0, "candidate_index": -1}

    def _mode(key, default):
        return Counter(s.get(key, default) for s in samples).most_common(1)[0][0]

    applicable  = _mode("applicable", True)
    well_formed = _mode("well_formed", True)
    llm_verdict = _mode("verdict", "Needs Human")

    # Candidate pick: plain majority INCLUDING -1, so a genuinely-absent quantity (most samples
    # say -1) correctly wins Not Found instead of being force-matched to a spurious candidate by a
    # single noisy sample. The prompt's sole-reference rule is what keeps real-but-quoted values
    # (e.g. a standard_reference parameter) from being dropped, so no recall bias is needed here.
    ci = _mode("candidate_index", -1)
    if not (isinstance(ci, int) and 0 <= ci < len(cands)):
        ci = -1
    reason_llm = next((s.get("reason", "") for s in samples if s.get("candidate_index") == ci),
                      samples[0].get("reason", ""))
    confidence = sum(1 for s in samples if s.get("candidate_index") == ci) / len(samples)

    # ── Deterministic decision (code is authoritative for the numeric core) ──
    if not applicable:
        return {"verdict": "Not Applicable", "candidate_index": -1, "confidence": confidence,
                "reason": reason_llm or "Requirement condition does not apply to this project."}
    if not well_formed:
        return {"verdict": "Needs Human", "candidate_index": -1, "confidence": confidence,
                "reason": reason_llm or "Requirement appears ill-formed (likely extraction error)."}
    if not (isinstance(ci, int) and 0 <= ci < len(cands)):
        return {"verdict": "Not Found", "candidate_index": -1, "confidence": confidence,
                "reason": reason_llm or "No candidate denotes the required quantity."}

    op = rule.get("op", "")
    chosen = cands[ci]
    if op in NUMERIC_OPS:
        v_text = f"{chosen.get('val','')} {chosen.get('unit','')}".strip()
        t_text = f"{rule.get('thr','')} {rule.get('unit','')}".strip()
        verdict, reason = numeric_verdict(op, v_text, t_text)
    else:                                  # text/date rules (must_be, requires, before…) → LLM
        verdict, reason = llm_verdict, reason_llm
    return {"verdict": verdict, "reason": reason, "confidence": confidence, "candidate_index": ci}


# ─── Public entry point ────────────────────────────────────────────────────────

def run_context_validation(doc_id: str, sector: str) -> dict:
    rules = _load_rules(sector)
    facts = _load_facts(doc_id)
    logger.info(f"Context validation: {len(rules)} requirements vs {len(facts)} DPR facts")

    # Embed facts once and each rule (local mxbai). Empty/zero arrays => semantic off.
    fact_texts = [f"{f.get('attr','')} {f.get('subj','')} {(f.get('ctx') or '')[:120]}".strip() for f in facts]
    fact_embs  = _embed_texts_local(fact_texts) if facts else None
    rule_texts = [f"{r.get('attr','')} {r.get('text','')}".strip() for r in rules]
    rule_embs  = _embed_texts_local(rule_texts) if rules else None

    # One project profile, derived from the DPR's own facts, shared by every judgment.
    # CACHED per doc: the profile is an LLM call and varies run-to-run; pinning it removes a
    # major source of cross-run verdict drift (every run then judges against the same context).
    profile = _load_or_make_profile(doc_id, facts)
    logger.info(f"Project profile:\n{profile}")

    results = []
    for ri, rule in enumerate(rules):
        r_emb = rule_embs[ri] if (rule_embs is not None and rule_embs.size) else None
        cands = _candidates(rule, facts, fact_embs, r_emb)
        j = _judge(rule, cands, profile)

        ci = j.get("candidate_index", -1)
        chosen = cands[ci] if isinstance(ci, int) and 0 <= ci < len(cands) else None
        sev = rule.get("sev") or Severity.HIGH
        results.append({
            "check_area":   rule.get("attr", ""),
            "requirement":  _describe_operator(rule.get("op", ""), str(rule.get("thr", "")), str(rule.get("unit", ""))),
            "rule_text":    rule.get("text", ""),
            "condition":    rule.get("cond", ""),
            "dpr_value":    (f"{chosen.get('val','')} {chosen.get('unit','')}".strip() if chosen else "Not found in DPR"),
            "verdict":      j.get("verdict", "Needs Human"),
            "reason":       j.get("reason", ""),
            "confidence":   float(j.get("confidence", 0.0) or 0.0),
            "severity":     sev,
            "standard":     f"{rule.get('std','')} {rule.get('clause','') or ''}".strip(),
            "source_page":  (chosen.get("ppage") or chosen.get("page") or 0) if chosen else 0,
        })

    return _build_report(doc_id, sector, results)


def _build_report(doc_id: str, sector: str, results: list[dict]) -> dict:
    # Flag rows whose verdict was not stable across votes — these need a human look even when
    # they happen to land on a pass/fail this run. Deterministic numeric rows score 1.0 here.
    for r in results:
        r["review"] = float(r.get("confidence", 1.0)) < REVIEW_CONFIDENCE

    judged = [r for r in results if r["verdict"] in ("Compliant", "Non-Compliant")]
    comp_w = sum(SEVERITY_WEIGHTS.get(r["severity"], 2) for r in judged if r["verdict"] == "Compliant")
    tot_w  = sum(SEVERITY_WEIGHTS.get(r["severity"], 2) for r in judged)
    score  = round(comp_w / tot_w * 100, 1) if tot_w else 0.0
    verdict = "NO_CHECKS" if not judged else next(
        (lbl for thr, lbl in [(90, "GOOD"), (75, "SATISFACTORY"), (50, "NEEDS IMPROVEMENT"), (0, "POOR")] if score >= thr), "POOR")

    def _count(v): return sum(1 for r in results if r["verdict"] == v)
    report = {
        "doc_id": doc_id, "sector": sector, "generated_at": datetime.now().isoformat(),
        "method": "context-aware LLM validation (local embeddings + Qwen judgment)",
        "score": {
            "weighted_score": score, "verdict": verdict,
            "compliant": _count("Compliant"), "non_compliant": _count("Non-Compliant"),
            "not_applicable": _count("Not Applicable"),
            "not_found": _count("Not Found"), "needs_human": _count("Needs Human"),
            "low_confidence": sum(1 for r in results if r["review"]),
            "total_requirements": len(results),
        },
        "results": sorted(results, key=lambda r: {"Non-Compliant": 0, "Needs Human": 1,
                                                  "Not Found": 2, "Not Applicable": 3,
                                                  "Compliant": 4}.get(r["verdict"], 5)),
    }
    out = OUTPUT_DIR / f"context_validation_{doc_id}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Context validation report saved: {out}")
    return report
