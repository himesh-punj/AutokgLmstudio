#!/usr/bin/env python3
"""
run_context_validation.py
-------------------------
Context-first validation (no KG). For each rulebook REQUIREMENT it retrieves the
relevant DPR claims (syntactic + local-embedding semantic) and has Qwen judge them
in context — disambiguating things like design vs average speed and refusing to
compare a length against a gradient.

    python run_context_validation.py --doc-id <id> [--sector "Rail Infrastructure"]
"""

import sys
import json
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from utils.neo4j_client import init_schema, run_read
from config.settings import NodeLabel
from validators.context_validator import run_context_validation

console = Console()


def _resolve(doc_id, sector):
    if not doc_id:
        st = Path("output/.extraction_state.json")
        if st.exists():
            dpr = json.loads(st.read_text(encoding="utf-8")).get("dpr", {})
            doc_id = dpr.get("doc_id")
            sector = sector or dpr.get("sector")
    if doc_id and not sector:
        rows = run_read(f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id:$id}}) RETURN d.sector AS s", {"id": doc_id})
        sector = rows[0]["s"] if rows else "Rail Infrastructure"
    return doc_id, sector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", type=str)
    ap.add_argument("--sector", type=str)
    args = ap.parse_args()

    init_schema()
    doc_id, sector = _resolve(args.doc_id, args.sector)
    if not doc_id:
        console.print("[red]No --doc-id and none in state file.[/red]"); sys.exit(1)

    console.rule(f"[bold]Context-aware validation — {sector}[/bold]")
    console.print(f"[dim]doc-id={doc_id} — retrieving DPR context per requirement and judging with Qwen...[/dim]")
    report = run_context_validation(doc_id, sector)

    s = report["score"]
    vc = {"GOOD": "bold green", "SATISFACTORY": "green", "NEEDS IMPROVEMENT": "yellow",
          "POOR": "red", "NO_CHECKS": "dim"}.get(s["verdict"], "white")
    console.print(Panel(
        f"[{vc}]{s['verdict']}[/{vc}]   Weighted Compliance: [{vc}]{s['weighted_score']:.1f}%[/{vc}]",
        subtitle=(f"[dim]Compliant {s['compliant']} | Non-Compliant {s['non_compliant']} | "
                  f"N/A {s.get('not_applicable', 0)} | Not Found {s['not_found']} | "
                  f"Needs Human {s['needs_human']} | Requirements {s['total_requirements']}[/dim]"),
        border_style=vc.replace("bold ", ""),
    ))

    t = Table(border_style="cyan", show_lines=False)
    for col in ("Verdict", "Requirement", "DPR Value", "Expected", "Pg", "Why"):
        t.add_column(col)
    vcol = {"Compliant": "green", "Non-Compliant": "red", "Not Applicable": "dim",
            "Not Found": "yellow", "Needs Human": "magenta"}
    for r in report["results"]:
        t.add_row(
            f"[{vcol.get(r['verdict'],'white')}]{r['verdict']}[/{vcol.get(r['verdict'],'white')}]",
            str(r["check_area"])[:26], str(r["dpr_value"])[:16], str(r["requirement"])[:22],
            str(r["source_page"]), str(r["reason"])[:48],
        )
    console.print(t)
    console.print(f"\n[green]Report:[/green] output/context_validation_{doc_id}.json")


if __name__ == "__main__":
    main()
