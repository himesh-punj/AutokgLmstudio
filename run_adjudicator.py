#!/usr/bin/env python3
"""
run_adjudicator.py
------------------
STEP 6 (final): LLM adjudicator over the validation + engine outputs.

Re-judges the over-strict automated checks with engineering judgment, recomputes
the final compliance score, and writes an executive summary.

    python run_adjudicator.py --doc-id <id>
    python run_adjudicator.py            # reads doc-id from output/.extraction_state.json

Reads:  output/validation_report_<id>.json, output/engine_results_<id>.json
Writes: output/final_report_<id>.json
No Neo4j required — operates on the JSON artifacts from earlier steps.
"""

import sys
import json
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Severity
from agents.result_adjudicator import run_adjudication

console = Console()


def _resolve_doc_id(arg_id: str | None) -> str | None:
    if arg_id:
        return arg_id
    state = Path("output/.extraction_state.json")
    if state.exists():
        try:
            return json.loads(state.read_text(encoding="utf-8")).get("dpr", {}).get("doc_id")
        except Exception:
            return None
    return None


def _verdict_color(v: str) -> str:
    return {
        "GOOD": "bold green", "SATISFACTORY": "green",
        "NEEDS IMPROVEMENT": "yellow", "POOR": "red",
    }.get(v, "white")


def main():
    ap = argparse.ArgumentParser(description="Final LLM adjudication of DPR validation results")
    ap.add_argument("--doc-id", type=str, help="Document ID (defaults to extraction state)")
    args = ap.parse_args()

    doc_id = _resolve_doc_id(args.doc_id)
    if not doc_id:
        console.print("[red]No --doc-id given and none found in output/.extraction_state.json[/red]")
        sys.exit(1)

    console.rule(f"[bold]Adjudicating doc {doc_id}[/bold]")
    console.print("[dim]Re-judging flagged checks with the LLM (this runs several calls)...[/dim]")

    try:
        report = run_adjudication(doc_id)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    s = report["final_score"]
    vc = _verdict_color(s["verdict"])

    console.print(Panel(
        f"[{vc}]{s['verdict']}[/{vc}]   Final Weighted Score: [{vc}]{s['weighted_score']:.1f}%[/{vc}]\n"
        f"[dim]Automated score was: {report.get('automated_score')}%[/dim]",
        title=f"Final Adjudicated Result — {report.get('sector','')} DPR",
        subtitle=(
            f"[dim]Compliant: {s['compliant']}  |  Non-Compliant: {s['non_compliant']}  |  "
            f"Not Applicable: {s['not_applicable']}  |  Needs Human: {s['needs_human']}  |  "
            f"Total: {s['total_checks']}[/dim]"
        ),
        border_style=vc.replace("bold ", ""),
    ))

    console.print(Panel(report["executive_summary"],
                        title="[bold]Executive Summary[/bold]", border_style="cyan"))

    # Show checks the adjudicator OVERTURNED (auto said fail/review, agent cleared or vice versa)
    overturned = [
        r for r in report["results"]
        if r["final_verdict"] != r["auto_verdict"]
        and r["auto_verdict"] in ("Non-Compliant", "Needs Review")
    ]
    if overturned:
        t = Table(title=f"Adjudicator changed {len(overturned)} verdicts", border_style="magenta", show_lines=False)
        t.add_column("Check Area", min_width=20)
        t.add_column("DPR Value", min_width=12)
        t.add_column("Auto", min_width=14)
        t.add_column("Final", min_width=14)
        t.add_column("Why", min_width=40)
        for r in overturned[:25]:
            fv = r["final_verdict"]
            fc = {"Compliant": "green", "Non-Compliant": "red"}.get(fv, "yellow")
            t.add_row(
                str(r.get("check_area", ""))[:28],
                str(r.get("dpr_value", ""))[:14],
                f"[red]{r['auto_verdict']}[/red]",
                f"[{fc}]{fv}[/{fc}]",
                str(r.get("rationale", ""))[:60],
            )
        console.print(t)
        if len(overturned) > 25:
            console.print(f"[dim]  ... and {len(overturned) - 25} more changes.[/dim]")

    # Confirmed non-compliances (final)
    final_nc = [r for r in report["results"] if r["final_verdict"] == "Non-Compliant"]
    if final_nc:
        console.print(f"\n[bold red]Confirmed Non-Compliant ({len(final_nc)}):[/bold red]")
        sev_w = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        for r in sorted(final_nc, key=lambda x: sev_w.get(x.get("severity", ""), 4))[:15]:
            console.print(
                f"  [red][{r.get('severity')}][/red] {r.get('check_area')} — "
                f"DPR {r.get('dpr_value')} vs {r.get('rule_expected')} "
                f"[dim](p.{r.get('source_page')})[/dim]"
            )

    console.print(f"\n[green]Final report:[/green] output/final_report_{doc_id}.json")


if __name__ == "__main__":
    main()
