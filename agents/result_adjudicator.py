"""
agents/result_adjudicator.py
----------------------------
STEP 6 (final): LLM "adjudicator" agent that reviews the automated outputs and
produces the final, judgment-applied result.

Why this exists:
    The deterministic validation engine is intentionally literal — it can still be
    too strict (treating a minimum/range as an exact value, mishandling unusual
    units, or punting compound thresholds to "Needs Review"). This agent re-reads
    each Non-Compliant and Needs-Review check and re-judges it the way a senior
    DPR auditor would: a value meeting a minimum is compliant, a value inside a
    stated range is compliant, a "ruling/maximum" limit means flatter/smaller is
    fine, and a rule whose condition doesn't apply is Not Applicable.

It then:
    1. Recomputes a final weighted score from the adjudicated verdicts
       (Compliant / Non-Compliant judged; Not-Applicable / Needs-Human excluded).
    2. Writes an LLM executive summary that folds in the consistency + anomaly
       engine findings.
    3. Saves output/final_report_<doc_id>.json and prints a console summary.

Inputs (produced by earlier steps, no Neo4j required):
    output/validation_report_<doc_id>.json   (run_validation.py)
    output/engine_results_<doc_id>.json      (run_engines.py)   [optional]

Run:  python run_adjudicator.py --doc-id <id>
"""

import json
from pathlib import Path
from datetime import datetime

from loguru import logger

from config.settings import OUTPUT_DIR, Severity
from utils.llm_router import generate_json, generate, get_model_for_task, TaskType

# Weights mirror validators.validation_engine.SEVERITY_WEIGHTS
_SEVERITY_WEIGHTS = {
    Severity.CRITICAL: 4,
    Severity.HIGH:     3,
    Severity.MEDIUM:   2,
    Severity.LOW:      1,
    Severity.INFO:     1,
}

_BATCH_SIZE = 12   # checks per LLM adjudication call

# Determinism + conservatism: run at temperature 0 and only let the LLM OVERRIDE
# the automated verdict when it is at least this confident. Below the threshold we
# keep a safe fallback (non-compliances stay non-compliant; reviews stay open), so
# the final score is stable run-to-run instead of drifting on low-confidence flips.
_TEMPERATURE       = 0.0
_MIN_CONFIDENCE    = 0.80

# Verdicts the adjudicator may assign. Only the first two count toward the score.
_JUDGED   = {"Compliant", "Non-Compliant"}
_EXCLUDED = {"Not Applicable", "Needs Human"}


# ─── IO helpers ────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not read {path.name}: {e}")
        return None


# ─── Adjudication ──────────────────────────────────────────────────────────────

_ADJUDICATOR_SYSTEM = (
    "You are a senior Indian Railways DPR auditor giving the FINAL verdict on the "
    "output of an automated compliance checker. The automated checker is known to be "
    "OVER-STRICT: it sometimes treats a minimum or a range as if it were an exact "
    "required value, mishandles units, and fails checks whose threshold it could not "
    "parse. Re-judge each check using sound engineering judgment."
)

_RULES_GUIDANCE = (
    "Apply these principles:\n"
    "- A value that MEETS or EXCEEDS a stated minimum is Compliant.\n"
    "- A value at or below a stated maximum / 'ruling' limit is Compliant "
    "(for gradient, a flatter/smaller slope is fine).\n"
    "- A value within a stated range is Compliant.\n"
    "- If the rule's condition does not apply to this value (wrong line config, "
    "wrong element, unit mismatch that makes comparison meaningless), return 'Not Applicable'.\n"
    "- Mark 'Non-Compliant' ONLY when the value genuinely violates the requirement.\n"
    "- If you truly cannot decide from the data given, return 'Needs Human'.\n"
)


def _adjudicate_batch(batch: list[dict]) -> dict[int, dict]:
    """Send one batch of checks to the LLM. Returns {id: {verdict, confidence, rationale}}."""
    lines = []
    for row in batch:
        lines.append(json.dumps({
            "id":             row["_id"],
            "check_area":     row.get("check_area", ""),
            "dpr_value":      row.get("dpr_value", ""),
            "rule_requires":  row.get("rule_expected", ""),
            "severity":       row.get("severity", ""),
            "auto_verdict":   row.get("classification", ""),
            "auto_reason":    row.get("reason", "")[:200],
        }, ensure_ascii=False))

    prompt = (
        f"{_RULES_GUIDANCE}\n"
        "Here are automated compliance checks to re-judge. For EACH one, decide the "
        "final verdict.\n\n"
        "CHECKS (one JSON object per line):\n" + "\n".join(lines) + "\n\n"
        "Return ONLY a JSON array, one object per check, each with:\n"
        '{"id": <id>, "verdict": "Compliant"|"Non-Compliant"|"Not Applicable"|"Needs Human", '
        '"confidence": <0.0-1.0>, "rationale": "<one short sentence>"}'
    )

    result = generate_json(prompt, system=_ADJUDICATOR_SYSTEM,
                           model=get_model_for_task(TaskType.VALIDATION),
                           temperature=_TEMPERATURE)
    out: dict[int, dict] = {}
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "id" in item:
                try:
                    out[int(item["id"])] = item
                except (ValueError, TypeError):
                    continue
    return out


def _adjudicate_rows(rows: list[dict]) -> list[dict]:
    """
    Re-judge the given report rows. Returns the same rows annotated with
    'final_verdict', 'confidence', 'rationale'. Rows the LLM didn't return keep
    their original classification (fail-safe).
    """
    verdicts: dict[int, dict] = {}
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i:i + _BATCH_SIZE]
        try:
            verdicts.update(_adjudicate_batch(batch))
        except Exception as e:
            logger.warning(f"Adjudication batch {i // _BATCH_SIZE} failed: {e}")

    for row in rows:
        v = verdicts.get(row["_id"])
        auto = row["classification"]
        if not v:
            row["final_verdict"] = auto
            row["confidence"]    = 0.0
            row["rationale"]     = "Adjudicator did not return a verdict; kept automated result."
            continue

        verdict = v.get("verdict", auto)
        conf    = float(v.get("confidence", 0.0) or 0.0)
        rationale = v.get("rationale", "")

        # Only OVERRIDE the automated verdict when confident. On a low-confidence
        # disagreement, fall back conservatively so the result is stable:
        #   auto Non-Compliant  -> stays Non-Compliant (don't clear a failure on a hunch)
        #   auto Needs Review   -> Needs Human (stays open for a person)
        if verdict != auto and conf < _MIN_CONFIDENCE:
            row["final_verdict"] = auto if auto == "Non-Compliant" else "Needs Human"
            row["confidence"]    = conf
            row["rationale"]     = (
                f"Low adjudicator confidence ({conf:.2f}) to change verdict; "
                f"kept conservative result. (LLM suggested: {verdict} — {rationale})"
            )
        else:
            row["final_verdict"] = verdict
            row["confidence"]    = conf
            row["rationale"]     = rationale
    return rows


# ─── Scoring on adjudicated verdicts ──────────────────────────────────────────

def _weight_for(row: dict) -> int:
    return row.get("weight") or _SEVERITY_WEIGHTS.get(row.get("severity", ""), 2)


def _final_score(all_rows: list[dict]) -> dict:
    """Recompute weighted compliance using final_verdict. Excludes NA / Needs-Human."""
    judged = [r for r in all_rows if r.get("final_verdict") in _JUDGED]
    total_w = sum(_weight_for(r) for r in judged)
    comp_w  = sum(_weight_for(r) for r in judged if r["final_verdict"] == "Compliant")
    score   = round(comp_w / total_w * 100, 1) if total_w else 0.0

    verdict = "NO_CHECKS_RUN" if not judged else "POOR"
    for threshold, label in [(90, "GOOD"), (75, "SATISFACTORY"), (50, "NEEDS IMPROVEMENT"), (0, "POOR")]:
        if judged and score >= threshold:
            verdict = label
            break

    return {
        "weighted_score":      score,
        "verdict":             verdict,
        "compliant":           sum(1 for r in all_rows if r.get("final_verdict") == "Compliant"),
        "non_compliant":       sum(1 for r in all_rows if r.get("final_verdict") == "Non-Compliant"),
        "not_applicable":      sum(1 for r in all_rows if r.get("final_verdict") == "Not Applicable"),
        "needs_human":         sum(1 for r in all_rows if r.get("final_verdict") == "Needs Human"),
        "total_checks":        len(all_rows),
    }


# ─── Executive summary ─────────────────────────────────────────────────────────

def _executive_summary(doc_id: str, sector: str, score: dict,
                       final_nc: list[dict], engine: dict | None) -> str:
    """One LLM call: a crisp executive summary folding in engine findings."""
    nc_lines = [
        f"- [{r.get('severity')}] {r.get('check_area')}: DPR={r.get('dpr_value')} "
        f"vs requires {r.get('rule_expected')} ({r.get('rationale','')})"
        for r in final_nc[:25]
    ]

    cons = (engine or {}).get("consistency", {})
    anom = (engine or {}).get("anomaly", {})
    cons_issues = [i.get("description", "") for i in cons.get("issues", [])[:8]]
    anom_flags  = [f.get("description", "") for f in anom.get("flags", [])[:8]]

    prompt = (
        f"Write a concise executive summary (6-10 sentences) of a {sector} DPR appraisal.\n\n"
        f"Final weighted compliance score: {score['weighted_score']}% "
        f"(verdict: {score['verdict']}). "
        f"{score['compliant']} compliant, {score['non_compliant']} non-compliant, "
        f"{score['not_applicable']} not applicable, {score['needs_human']} need human review.\n\n"
        f"Key confirmed non-compliances:\n" + ("\n".join(nc_lines) or "  (none)") + "\n\n"
        f"Consistency issues:\n" + ("\n".join(f"- {c}" for c in cons_issues) or "  (none)") + "\n\n"
        f"Anomaly flags:\n" + ("\n".join(f"- {a}" for a in anom_flags) or "  (none)") + "\n\n"
        "Summarise the overall readiness of the DPR, the most material risks, and the "
        "top corrective actions. Be specific and factual; do not invent numbers."
    )
    try:
        return generate(prompt, model=get_model_for_task(TaskType.VALIDATION), temperature=_TEMPERATURE).strip()
    except Exception as e:
        logger.warning(f"Executive summary generation failed: {e}")
        return "(executive summary unavailable)"


# ─── Public entry point ────────────────────────────────────────────────────────

def run_adjudication(doc_id: str) -> dict:
    """Adjudicate the validation report for doc_id and write output/final_report_<doc_id>.json."""
    report = _load_json(OUTPUT_DIR / f"validation_report_{doc_id}.json")
    if not report:
        raise FileNotFoundError(
            f"validation_report_{doc_id}.json not found. Run run_validation.py first."
        )
    engine = _load_json(OUTPUT_DIR / f"engine_results_{doc_id}.json")

    sector  = report.get("sector", "")
    results = report.get("results", [])
    for i, r in enumerate(results):
        r["_id"] = i

    # Only re-judge what the automated pass flagged; keep Compliants as-is.
    to_review = [r for r in results if r.get("classification") in ("Non-Compliant", "Needs Review")]
    logger.info(f"Adjudicating {len(to_review)} flagged checks (of {len(results)} total)...")
    _adjudicate_rows(to_review)

    # Compliant rows carry forward unchanged.
    for r in results:
        if "final_verdict" not in r:
            r["final_verdict"] = r["classification"]
            r["confidence"]    = 1.0
            r["rationale"]     = "Automated check passed; not re-judged."

    score    = _final_score(results)
    final_nc = sorted(
        [r for r in results if r.get("final_verdict") == "Non-Compliant"],
        key=lambda r: _SEVERITY_WEIGHTS.get(r.get("severity", ""), 0), reverse=True,
    )
    summary = _executive_summary(doc_id, sector, score, final_nc, engine)

    final_report = {
        "doc_id":            doc_id,
        "sector":            sector,
        "generated_at":      datetime.now().isoformat(),
        "automated_score":   report.get("score", {}).get("weighted_score"),
        "final_score":       score,
        "executive_summary": summary,
        "results":           [
            {
                "final_verdict":  r.get("final_verdict"),
                "auto_verdict":   r.get("classification"),
                "confidence":     r.get("confidence"),
                "check_area":     r.get("check_area"),
                "dpr_value":      r.get("dpr_value"),
                "rule_expected":  r.get("rule_expected"),
                "severity":       r.get("severity"),
                "standard":       r.get("standard"),
                "source_page":    r.get("source_page"),
                "rationale":      r.get("rationale"),
            }
            for r in results
        ],
    }

    out_path = OUTPUT_DIR / f"final_report_{doc_id}.json"
    out_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Final adjudicated report saved: {out_path}")
    return final_report
