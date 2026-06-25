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
from datetime import datetime

import numpy as np
from loguru import logger

from config.settings import NodeLabel, RelType, Severity, OUTPUT_DIR, get_applicable_sectors
from utils.neo4j_client import run_read
from utils.llm_router import generate, generate_json, get_model_for_task, TaskType
# reuse the (fixed, local, timed) embedding call and the whole-word tokenizer
from validators.validation_engine import _embed_texts_local, _attr_tokens, _describe_operator

SEVERITY_WEIGHTS = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1, Severity.INFO: 1}
TOP_K_SEMANTIC   = 6      # semantic candidates per rule
MAX_CANDIDATES   = 10     # cap sent to the LLM


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
    "You are a senior Indian Railways DPR auditor. You compare a STANDARD REQUIREMENT "
    "(from a rulebook) against values PROPOSED in a DPR. Apply the SAME reasoning to EVERY "
    "parameter — these are general rules, never special-cased per parameter:\n"
    "  • MEANING over words. The same word can mean different things — 'speed' → design vs "
    "average vs sectional; 'gradient' → a slope vs its length; 'width' → formation vs "
    "carriageway vs track-centre; 'load' → axle vs structure. Use the context to pick the "
    "sense the requirement is actually about.\n"
    "  • DIMENSION & UNITS. State the dimension/unit of BOTH the requirement and the value, "
    "convert to common units when compatible (m↔km, t↔kg), and NEVER compare incompatible "
    "quantities — length ≠ slope, % ≠ metres, count ≠ ratio, a raw length ≠ a ratio, mass ≠ "
    "length. If they cannot be made comparable, the verdict is 'Needs Human'.\n"
    "  • APPLICABILITY. A requirement may carry a CONDITION (e.g. only for doubling / 3rd-4th "
    "line / electrified / a gauge / a bridge type). Check it against the PROJECT CONTEXT. If "
    "the condition does not apply to THIS project, the verdict is 'Not Applicable' — do not "
    "score it.\n"
    "  • Reason identically whatever the parameter is (speed, gradient, width, radius, load, "
    "count, ratio, cost, spacing, depth, …)."
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
        "Work through these steps IN ORDER, the same way for any parameter:\n"
        "1. APPLICABILITY: if the requirement's Condition does not apply to THIS project (per the "
        "PROJECT CONTEXT), stop and return verdict 'Not Applicable' (candidate_index -1).\n"
        "2. PICK: choose the ONE candidate index that genuinely matches this requirement's "
        "parameter and intent — use the context to get the right SENSE (design vs average speed, "
        "slope vs length, etc.). Prefer nature 'proposed' over 'standard_reference'. If none truly "
        "corresponds, return 'Not Found' (index -1).\n"
        "3. DIMENSION: state the requirement's and the chosen value's dimension/unit; convert to "
        "common units if compatible. If they are NOT comparable (e.g. a raw length vs a ratio, % vs "
        "metres), return 'Needs Human'.\n"
        "4. COMPLY: only now judge compliance, comparing like-with-like.\n"
        'Return ONLY JSON: {"candidate_index": <int>, "verdict": '
        '"Compliant"|"Non-Compliant"|"Not Applicable"|"Not Found"|"Needs Human", '
        '"reason": "<one sentence naming the dimension you compared>", "confidence": <0.0-1.0>}'
    )

    res = generate_json(prompt, system=_SYSTEM,
                        model=get_model_for_task(TaskType.VALIDATION), temperature=0.0)
    if not isinstance(res, dict):
        return {"verdict": "Needs Human", "reason": "Adjudication failed to parse.",
                "confidence": 0.0, "candidate_index": -1}
    return res


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
    profile = _project_profile(facts)
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
