"""
extractors/table_extractor.py
------------------------------
Deterministic 3-stage table extraction cascade.

Stage 1: pdfplumber  (fast, no external deps)
Stage 2: camelot     (better for bordered/lattice tables)
Stage 3: Vision LLM  (Ollama llama3.2-vision — guaranteed fallback)

Each stage produces a list[dict] (rows as dicts with column headers as keys).
A stage is accepted if the fill rate (non-null cells / total cells) >= TABLE_MIN_FILL.
"""

import threading
import tempfile
from pathlib import Path
from typing import Optional

import pdfplumber
import camelot
import fitz  # PyMuPDF — fast, accurate table finder used between pdfplumber and camelot
import pandas as pd
from loguru import logger

from config.settings import TABLE_MIN_FILL, PAGE_DPI
from utils.llm_router import vision_extract_table_json


# ─── Reusable per-thread PDF handle ───────────────────────────────────────────
# The cascade is called once per page across a thread pool. Opening the PDF on
# every call re-parses the whole document (hundreds of times for a big DPR) and
# causes I/O contention between workers. Instead we keep ONE open pdfplumber
# handle per (thread, path) and reuse it for every page that thread touches.
# Call close_cached_pdfs() once after extraction to release the handles.

_pdf_lock: threading.Lock = threading.Lock()
_open_pdfs:  dict = {}   # (thread_id, path_str) -> pdfplumber.PDF
_open_fitz:  dict = {}   # (thread_id, path_str) -> fitz.Document


def _get_pdf(pdf_path: Path):
    key = (threading.get_ident(), str(pdf_path))
    with _pdf_lock:
        pdf = _open_pdfs.get(key)
        if pdf is None:
            pdf = pdfplumber.open(pdf_path)
            _open_pdfs[key] = pdf
        return pdf


def _get_fitz(pdf_path: Path):
    key = (threading.get_ident(), str(pdf_path))
    with _pdf_lock:
        doc = _open_fitz.get(key)
        if doc is None:
            doc = fitz.open(str(pdf_path))
            _open_fitz[key] = doc
        return doc


def close_cached_pdfs():
    """Close all cached PDF handles (call once after extraction finishes)."""
    with _pdf_lock:
        for pdf in _open_pdfs.values():
            try:
                pdf.close()
            except Exception:
                pass
        for doc in _open_fitz.values():
            try:
                doc.close()
            except Exception:
                pass
        _open_pdfs.clear()
        _open_fitz.clear()


# ─── Acceptance check ─────────────────────────────────────────────────────────

def _fill_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    total = sum(len(r) for r in rows)
    filled = sum(1 for r in rows for v in r.values() if v is not None and str(v).strip() != "")
    return filled / total if total > 0 else 0.0


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename duplicate and nan column headers.
    - nan headers (from merged cells) → col_1, col_2, etc.
    - Duplicate headers → Value, Value_2, Value_3 etc.
    Common in engineering tables with merged or repeated header cells.
    """
    import math
    seen: dict[str, int] = {}
    new_cols = []
    col_counter = 0
    for col in df.columns:
        col_counter += 1
        # Handle nan/None/float nan from merged cells
        if col is None or (isinstance(col, float) and math.isnan(col)):
            col_str = f"col_{col_counter}"
        else:
            col_str = str(col).strip()
            # pdfplumber sometimes returns "nan" as string
            if col_str.lower() in ("nan", "none", ""):
                col_str = f"col_{col_counter}"

        if col_str in seen:
            seen[col_str] += 1
            new_cols.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 1
            new_cols.append(col_str)
    df.columns = new_cols
    return df


def _df_to_rows(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to list-of-dicts, using first row as header if needed."""
    if df is None or df.empty:
        return []
    # If headers are 0,1,2,... (unnamed), promote first row to header
    if all(isinstance(c, int) for c in df.columns):
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    # Deduplicate column names before conversion (avoids pandas UserWarning + data loss)
    df = _dedup_columns(df)
    # Replace empty strings with None
    df = df.replace(r"^\s*$", None, regex=True)
    return df.to_dict(orient="records")


# ─── Stage 1: pdfplumber ──────────────────────────────────────────────────────

def _try_pdfplumber(pdf_path: Path, page_num: int) -> list[dict]:
    """page_num is 0-indexed. Reuses a per-thread open PDF handle."""
    try:
        pdf = _get_pdf(pdf_path)
        if page_num >= len(pdf.pages):
            return []
        page = pdf.pages[page_num]

        # Try explicit table settings first (lattice-like)
        tables = page.extract_tables(table_settings={
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
        })
        if not tables:
            # Fallback to text-based detection
            tables = page.extract_tables()

        all_rows = []
        for tbl in tables:
            if not tbl:
                continue
            df = pd.DataFrame(tbl[1:], columns=tbl[0]) if tbl else pd.DataFrame()
            rows = _df_to_rows(df)
            all_rows.extend(rows)

        # Free this page's parsed-object cache so reusing the handle across the
        # whole document doesn't accumulate memory.
        try:
            page.flush_cache()
        except Exception:
            pass
        return all_rows
    except Exception as e:
        logger.debug(f"pdfplumber table extraction failed p{page_num}: {e}")
        return []


# ─── Stage 2: PyMuPDF find_tables ─────────────────────────────────────────────

def _try_pymupdf(pdf_path: Path, page_num: int) -> list[dict]:
    """
    Fast, accurate table detection via PyMuPDF. Sits between pdfplumber and the
    slow camelot/vision stages — catches many tables (esp. ruled/bordered ones)
    in-process without spawning ghostscript. Reuses a per-thread fitz handle.
    page_num is 0-indexed.
    """
    try:
        doc = _get_fitz(pdf_path)
        if page_num >= doc.page_count:
            return []
        page = doc[page_num]
        finder = page.find_tables()
        all_rows = []
        for tbl in getattr(finder, "tables", []) or []:
            try:
                df = tbl.to_pandas()
            except Exception:
                continue
            all_rows.extend(_df_to_rows(df))
        return all_rows
    except Exception as e:
        logger.debug(f"pymupdf table extraction failed p{page_num}: {e}")
        return []


# ─── Stage 3: camelot ────────────────────────────────────────────────────────

def _try_camelot(pdf_path: Path, page_num: int) -> list[dict]:
    """page_num is 0-indexed; camelot uses 1-indexed pages."""
    try:
        page_1indexed = page_num + 1
        # Try lattice first (bordered tables)
        tables = camelot.read_pdf(
            str(pdf_path), pages=str(page_1indexed),
            flavor="lattice", suppress_stdout=True
        )
        if tables.n == 0:
            # Try stream (borderless / text-aligned)
            tables = camelot.read_pdf(
                str(pdf_path), pages=str(page_1indexed),
                flavor="stream", suppress_stdout=True,
                edge_tol=50, row_tol=10
            )

        all_rows = []
        for tbl in tables:
            rows = _df_to_rows(tbl.df)
            all_rows.extend(rows)
        return all_rows
    except Exception as e:
        logger.debug(f"camelot table extraction failed p{page_num}: {e}")
        return []


# ─── Stage 3: Vision LLM fallback ────────────────────────────────────────────

def _rasterise_page(pdf_path: Path, page_num: int, dpi: int = PAGE_DPI) -> Optional[Path]:
    """
    Rasterise a single PDF page using the already-open pdfplumber handle
    (page.to_image), avoiding a pdftoppm subprocess and a fresh PDF open.
    Returns the path to a temporary PNG, or None on failure. page_num is 0-indexed.
    """
    try:
        # Render via PyMuPDF's pixmap (robust + thread-safe per handle). pdfplumber's
        # page.to_image proved fragile under concurrency ("PDFium: Data format error").
        doc = _get_fitz(pdf_path)
        if page_num >= doc.page_count:
            return None
        pix = doc[page_num].get_pixmap(dpi=dpi)
        out = Path(tempfile.mkdtemp()) / f"page-{page_num + 1}.png"
        pix.save(str(out))
        return out
    except Exception as e:
        logger.warning(f"Rasterise page {page_num + 1} failed: {e}")
        return None


def _try_vision_llm(pdf_path: Path, page_num: int, context: str = "") -> list[dict]:
    """Rasterise page and send to vision model. Final guaranteed fallback."""
    logger.info(f"  → Vision LLM fallback for table on page {page_num + 1}")
    img_path = _rasterise_page(pdf_path, page_num)
    if img_path is None:
        logger.error(f"Could not rasterise page {page_num + 1}")
        return []

    rows = vision_extract_table_json(img_path, context=context)

    # Cleanup temp file
    try:
        img_path.unlink()
        img_path.parent.rmdir()
    except Exception:
        pass

    return rows or []


# ─── Public API: extract_tables_from_page ────────────────────────────────────

def extract_tables_from_page(
    pdf_path: Path,
    page_num: int,
    context: str = "",
    min_fill: float = TABLE_MIN_FILL,
    use_vision: bool = True,
) -> list[dict]:
    """
    Extract all tables from a single PDF page using the 3-stage cascade.

    Args:
        pdf_path:  Path to the PDF file.
        page_num:  0-indexed page number.
        context:   Short description of what the page is about (helps vision LLM).
        min_fill:  Minimum fill rate to accept a stage's output (0–1).

    Returns:
        List of row dicts. Empty list if no tables found.
    """
    log_prefix = f"Page {page_num + 1}:"

    # Stage 1 — pdfplumber
    logger.debug(f"{log_prefix} Trying pdfplumber...")
    rows = _try_pdfplumber(pdf_path, page_num)
    fill = _fill_rate(rows)
    if rows and fill >= min_fill:
        logger.debug(f"{log_prefix} pdfplumber accepted (fill={fill:.0%}, rows={len(rows)})")
        return rows
    best = rows  # best result so far across stages (by fill rate, then row count)
    logger.debug(f"{log_prefix} pdfplumber fill={fill:.0%} < threshold, trying PyMuPDF...")

    def _better(a, b):
        return a if (_fill_rate(a), len(a)) >= (_fill_rate(b), len(b)) else b

    # Stage 2 — PyMuPDF (fast, in-process)
    rows = _try_pymupdf(pdf_path, page_num)
    fill = _fill_rate(rows)
    if rows and fill >= min_fill:
        logger.debug(f"{log_prefix} PyMuPDF accepted (fill={fill:.0%}, rows={len(rows)})")
        return rows
    best = _better(best, rows)
    logger.debug(f"{log_prefix} PyMuPDF fill={fill:.0%} < threshold, trying camelot...")

    # Stage 3 — camelot
    rows = _try_camelot(pdf_path, page_num)
    fill = _fill_rate(rows)
    if rows and fill >= min_fill:
        logger.debug(f"{log_prefix} camelot accepted (fill={fill:.0%}, rows={len(rows)})")
        return rows
    best = _better(best, rows)

    # Stage 4 — Vision LLM (slow accuracy backstop). Skipped only in fast mode.
    if not use_vision:
        logger.debug(f"{log_prefix} vision skipped (fast mode); returning best so far ({len(best)} rows)")
        return best
    logger.debug(f"{log_prefix} camelot fill={fill:.0%} < threshold, falling back to vision LLM...")

    rows = _try_vision_llm(pdf_path, page_num, context=context)
    fill = _fill_rate(rows)
    logger.info(f"{log_prefix} vision LLM result: {len(rows)} rows, fill={fill:.0%}")
    return _better(best, rows)


def extract_all_tables(
    pdf_path: Path,
    page_nums: list[int],
    context_map: dict[int, str] = None,
) -> dict[int, list[dict]]:
    """
    Extract tables from multiple pages. Returns {page_num: [rows]}.
    context_map: optional {page_num: context_string} for vision fallback.
    """
    results = {}
    for pn in page_nums:
        ctx = (context_map or {}).get(pn, "")
        results[pn] = extract_tables_from_page(pdf_path, pn, context=ctx)
    return results