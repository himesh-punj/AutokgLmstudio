#!/usr/bin/env python3
"""
backfill_printed_pages.py
-------------------------
Populate f.printed_page on already-extracted Fact nodes WITHOUT a full re-extraction.

Re-opens the source PDF, detects the printed page number on each physical page
(footer/header), then maps it onto facts by their stored source_page (1-indexed
physical page). Use after pulling the printed-page feature into an existing doc.

    python backfill_printed_pages.py --doc-id <id> --pdf "data/dpr/<file>.pdf"
    python backfill_printed_pages.py --doc-id <id>          # auto-find single PDF
    python backfill_printed_pages.py --doc-id <id> --dry-run  # just show the mapping
"""

import sys
import argparse
from pathlib import Path

import pdfplumber
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DPR_INPUT_DIR
from extractors.document_loader import infer_page_number_offset
from utils.neo4j_client import run_read, run_write

console = Console()


def build_map(pdf_path: Path) -> dict[int, int]:
    """Return {physical_page_1indexed: printed_page} using a document-wide offset."""
    with pdfplumber.open(pdf_path) as pdf:
        texts = [(p.extract_text() or "") for p in pdf.pages]
    offset = infer_page_number_offset(texts)
    if offset is None:
        return {}
    console.print(f"   Detected page-number offset: printed = physical [bold]{offset:+d}[/bold]")
    return {i + 1: i + 1 + offset for i in range(len(texts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--pdf", type=Path, help="Source PDF (auto-detected if omitted)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pdf_path = args.pdf
    if not pdf_path:
        pdfs = sorted(DPR_INPUT_DIR.glob("*.pdf"))
        if len(pdfs) != 1:
            console.print(f"[red]Specify --pdf (found {len(pdfs)} PDFs in {DPR_INPUT_DIR})[/red]")
            sys.exit(1)
        pdf_path = pdfs[0]
    if not pdf_path.exists():
        console.print(f"[red]PDF not found: {pdf_path}[/red]")
        sys.exit(1)

    console.print(f"Reading printed page numbers from [cyan]{pdf_path.name}[/cyan] ...")
    mapping = build_map(pdf_path)
    console.print(f"Detected printed numbers on [green]{len(mapping)}[/green] pages.")

    # Show a sample of the physical->printed offset
    sample = sorted(mapping.items())[:10]
    for phys, printed in sample:
        console.print(f"   physical p.{phys}  ->  printed p.{printed}")

    if args.dry_run:
        console.print("[yellow]Dry run — no Neo4j writes.[/yellow]")
        return

    # Facts store source_page = physical (1-indexed). Set printed_page from the map.
    facts = run_read(
        "MATCH (d:Document {doc_id:$id})-[:HAS_FACT]->(f:Fact) "
        "RETURN f.fact_id AS fid, f.source_page AS sp",
        {"id": args.doc_id},
    )
    updated = 0
    for f in facts:
        printed = mapping.get(int(f["sp"])) if f["sp"] is not None else None
        if printed is not None:
            run_write(
                "MATCH (f:Fact {fact_id:$fid}) SET f.printed_page = $pp",
                {"fid": f["fid"], "pp": printed},
            )
            updated += 1
    console.print(f"[green]Updated printed_page on {updated} of {len(facts)} facts.[/green]")
    console.print("Re-run: [bold]python run_validation.py --doc-id "
                  f"{args.doc_id}[/bold] to see printed pages in the report.")


if __name__ == "__main__":
    main()
