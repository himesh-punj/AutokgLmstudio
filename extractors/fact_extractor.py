"""
extractors/fact_extractor.py
-----------------------------
Extracts structured engineering facts from DPR text using Ollama LLM.

Each "fact" is a structured record:
    {
        "fact_id": <uuid>,
        "fact_type": "parameter" | "measurement" | "material" | "cost" | "schedule" | "assumption",
        "subject": "pier foundation",
        "attribute": "bearing capacity",
        "value": "250",
        "unit": "kN/m²",
        "context": "surrounding sentence",
        "source_page": 14,
        "confidence": 0.9,
        "sector": "bridges",
        "doc_id": "...",
    }

Facts are chunked by page and extracted with a sector-aware prompt.
Results are written directly to Neo4j.
"""

import uuid
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from config.settings import CHUNK_SIZE, PAGE_EXTRACT_CHUNK_CHARS, SECTOR_KEYS, NodeLabel, RelType
from utils.llm_router import generate_json
from utils.neo4j_client import run_write

# ─── Sector-specific extraction hints ─────────────────────────────────────────
# These are injected into the prompt to guide the LLM on what to look for.
# Fully config-driven — no sector logic hardcoded in extraction code.

SECTOR_EXTRACTION_HINTS: dict[str, str] = {
    "rail": (
        "Focus on: track geometry (gauge, gradient, curvature), axle loads, "
        "speed limits, ballast depth, rail section, sleeper spacing, signal block lengths, "
        "station platforms, earthwork quantities, formation width."
    ),
    "bridges": (
        "Focus on: span lengths, number of spans, deck width, carriageway width, "
        "foundation type, soil bearing capacity (SBC), pile dimensions, "
        "design flood level (HFL/DFL), scour depth, live loads (IRC class), "
        "material grades (concrete, steel), seismic zone."
    ),
    "tunnels": (
        "Focus on: tunnel length, diameter/cross-section dimensions, overburden depth, "
        "rock mass rating (RMR/Q-value), support system (shotcrete thickness, bolt spacing), "
        "lining thickness, portal dimensions, ventilation requirements, groundwater inflow."
    ),
    "metro": (
        "Focus on: corridor length, number of stations, station depth/height, "
        "headway, design speed, rolling stock capacity, viaduct span, "
        "depressed/elevated/underground ratio, ridership projections, fare."
    ),
    "mobility": (
        "Focus on: freight volume, commodity type, route length, modal split, "
        "vehicle counts, logistics park area, warehouse capacity, connection to NH/rail."
    ),
    "highways": (
        "Focus on: carriageway width, number of lanes, pavement composition "
        "(DBM/BC/GSB thickness), subgrade CBR, design traffic (MSA), "
        "formation width, ROW, gradient, curve radius, cross-drainage structures."
    ),
    "ports": (
        "Focus on: berth length, draft depth, cargo capacity, quay wall type, "
        "fender system, dredging depth, equipment (cranes, reach stackers), "
        "hinterland connectivity, navigational channel dimensions."
    ),
    "airports": (
        "Focus on: runway length and width, PCN/ACN, pavement type, "
        "taxiway dimensions, apron area, terminal capacity (MPPA), "
        "approach category (CAT I/II/III), wind rose, RESA dimensions."
    ),
}

# ─── Extraction prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior infrastructure engineer extracting structured facts 
from a Detailed Project Report (DPR). You extract ONLY facts explicitly stated in the text — 
never infer or assume values. Each fact must have a numeric or clearly measurable value."""

def _build_extraction_prompt(text: str, sector: str, page_num: int, doc_id: str) -> str:
    sector_key = SECTOR_KEYS.get(sector, "")
    hints = SECTOR_EXTRACTION_HINTS.get(sector_key, "Focus on engineering parameters, quantities, and costs.")

    return f"""Extract ALL engineering facts from this {sector} DPR text (page {page_num + 1}).

{hints}

Also extract:
- Project costs (e.g. "total cost is Rs. 63,246 crore" → subject: project, attribute: total cost, value: 63246, unit: crore Rs.)
- Quantities in narrative sentences (e.g. "237 bore holes of 30m depth" → two facts)
- Operational parameters (speed, headway, capacity, frequency)
- Any measurement mentioned even in passing

TEXT:
\"\"\"
{text}
\"\"\"

Return a JSON array. Each object MUST have:
- "fact_type": one of ["parameter", "measurement", "material", "cost", "schedule", "assumption", "design_value"]
- "subject": engineering element described (e.g. "pile foundation", "corridor 3", "station")
- "attribute": the SPECIFIC property, KEEPING its qualifier — never drop the adjective that
  changes the meaning. Use "design speed", "average speed", "sectional speed", "ruling gradient",
  "gradient length", "maximum gradient", "minimum radius" — NOT a bare "speed"/"gradient"/"radius".
- "value": the value as a string — numeric preferred (e.g. "63246", "M30", "80")
- "unit": the unit EXACTLY as written (e.g. "crore Rs.", "km", "kmph", "m", "%", "1 in 200") —
  empty string if none. The unit/form carries dimension; keep "1 in 200" as a ratio, "%" as percent.
- "context": the full sentence/phrase containing this fact (≤ 200 chars) — keep enough words to
  tell WHAT the value describes and whether it is the DPR's own proposal or a quoted standard.
- "nature": "proposed" if this is a value the DPR adopts/uses, or "standard_reference" if the DPR
  is merely quoting a code/guideline requirement.
- "confidence": 0.85–1.0 for clearly stated facts, 0.7–0.85 for implied facts

Extract aggressively — include facts stated in prose, tables, and lists.
Extract costs, distances, counts, speeds, capacities, depths, heights.
If no engineering facts at all, return [].
Do not repeat identical facts."""


# ─── Neo4j writer ─────────────────────────────────────────────────────────────

def _write_facts_to_neo4j(facts: list[dict], doc_id: str, sector: str, page_num: int):
    """Upsert extracted facts as Fact nodes connected to the Document node."""
    for fact in facts:
        fact_id = str(uuid.uuid4())
        run_write(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
            MERGE (f:{NodeLabel.FACT} {{fact_id: $fact_id}})
            SET f.fact_type  = $fact_type,
                f.subject    = $subject,
                f.attribute  = $attribute,
                f.value      = $value,
                f.unit       = $unit,
                f.context    = $context,
                f.confidence = $confidence,
                f.source_page = $page_num,
                f.sector     = $sector,
                f.doc_id     = $doc_id
            MERGE (d)-[:{RelType.HAS_FACT}]->(f)
            WITH f
            MATCH (s:{NodeLabel.SECTOR} {{name: $sector_name}})
            MERGE (f)-[:{RelType.BELONGS_TO}]->(s)
            """,
            {
                "doc_id":    doc_id,
                "fact_id":   fact_id,
                "fact_type": fact.get("fact_type", "parameter"),
                "subject":   fact.get("subject", ""),
                "attribute": fact.get("attribute", ""),
                "value":     str(fact.get("value", "")),
                "unit":      fact.get("unit", ""),
                "context":   fact.get("context", "")[:400],
                "confidence": float(fact.get("confidence", 0.5)),
                "page_num":  page_num,
                "sector":    sector,
                "sector_name": sector,
            }
        )


# ─── Table facts ─────────────────────────────────────────────────────────────

def _write_table_facts_to_neo4j(table_rows: list[dict], doc_id: str, sector: str, page_num: int):
    """
    Convert extracted table rows into Fact nodes.
    Each row becomes a separate fact with the row data stored as JSON string.
    """
    import json
    for i, row in enumerate(table_rows):
        fact_id = str(uuid.uuid4())
        run_write(
            f"""
            MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})
            MERGE (f:{NodeLabel.FACT} {{fact_id: $fact_id}})
            SET f.fact_type   = 'table_row',
                f.subject     = 'table',
                f.attribute   = 'row_data',
                f.value       = $row_json,
                f.unit        = '',
                f.context     = 'Extracted from table',
                f.confidence  = 0.85,
                f.source_page = $page_num,
                f.row_index   = $row_index,
                f.sector      = $sector,
                f.doc_id      = $doc_id
            MERGE (d)-[:{RelType.HAS_FACT}]->(f)
            """,
            {
                "doc_id":    doc_id,
                "fact_id":   fact_id,
                "row_json":  json.dumps(row),
                "page_num":  page_num,
                "row_index": i,
                "sector":    sector,
            }
        )


# ─── Fast-model routing (optional, off by default) ────────────────────────────

def _use_fast_routing() -> bool:
    try:
        from config.backend_settings import USE_FAST_PAGE_ROUTING
        return bool(USE_FAST_PAGE_ROUTING)
    except Exception:
        return False


def _is_context_or_400(e: Exception) -> bool:
    """
    True for a context-size overflow OR any HTTP 400. The 400's HTTPError string is
    just "400 Client Error" (no body), so string-matching alone misses it — check the
    response status code too. These are deterministic, so we recover by truncating.
    """
    s = str(e).lower()
    if any(k in s for k in ("context", "exceed", "too large", "too long")):
        return True
    resp = getattr(e, "response", None)
    return resp is not None and getattr(resp, "status_code", None) == 400


def _chunk_text(text: str, max_chars: int, overlap: int = 200) -> list[str]:
    """
    Split page text into <= max_chars chunks, preferring to break on a line
    boundary, with a small overlap so a fact isn't cut across chunks. Returns
    [text] unchanged when it already fits in one chunk (single-shot).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            nl = text.rfind("\n", start + max_chars - 400, end)
            if nl > start:
                end = nl
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _is_low_density(text: str) -> bool:
    """Short page with few engineering values → safe for a smaller model."""
    import re as _re
    if len(text.strip()) > 600:
        return False
    hits = len(_re.findall(
        r'\d+\.?\d*\s*(?:m|km|mm|%|kmph|kg|t|rs|crore|lakh|ha|cum|mpa|kn)', text, _re.I))
    return hits < 3


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_facts_from_page(
    text: str,
    doc_id: str,
    sector: str,
    page_num: int,
    write_to_db: bool = True,
) -> list[dict]:
    """
    Extract engineering facts from a single page of text.
    Optionally writes results to Neo4j.
    Returns list of fact dicts.
    """
    if not text or len(text.strip()) < 30:
        return []

    def _call(txt: str):
        """One bounded LLM call on a chunk of page text."""
        p = _build_extraction_prompt(txt, sector, page_num, doc_id)
        # Optional speed lever: route low-density chunks to a small Ollama model.
        if _use_fast_routing() and _is_low_density(txt):
            from utils.ollama_client import generate_json as _fast_json
            from config.backend_settings import FAST_PAGE_MODEL
            return _fast_json(p, system=_SYSTEM_PROMPT, model=FAST_PAGE_MODEL)
        return generate_json(p, system=_SYSTEM_PROMPT)

    def _gen(txt: str) -> list:
        """Call once; on a context-overflow / 400, retry that chunk halved. Returns a list."""
        try:
            r = _call(txt)
        except Exception as e:
            if _is_context_or_400(e):
                try:
                    r = _call(txt[: max(600, len(txt) // 2)])
                except Exception:
                    return []
            else:
                raise
        return r if isinstance(r, list) else []

    # Multi-shot: split the page into small chunks so each LLM call stays light
    # (bounded VRAM/compute — avoids the spill-to-RAM freeze and GPU TDR). Small
    # pages are a single shot; dense pages are processed in several. Facts are
    # de-duplicated across the (overlapping) chunks.
    chunks = _chunk_text(text, PAGE_EXTRACT_CHUNK_CHARS, overlap=200)
    facts, seen = [], set()
    for ch in chunks:
        for f in _gen(ch):
            subj = str(f.get("subject") or "").strip()
            val  = str(f.get("value") or "").strip()
            if not (subj and val):
                continue
            key = (subj.lower(), str(f.get("attribute") or "").strip().lower(), val.lower())
            if key in seen:
                continue
            seen.add(key)
            facts.append(f)

    logger.debug(f"Page {page_num + 1}: extracted {len(facts)} facts from {len(chunks)} shot(s)")

    if write_to_db and facts:
        _write_facts_to_neo4j(facts, doc_id, sector, page_num)

    return facts


def write_table_facts(
    table_rows: list[dict],
    doc_id: str,
    sector: str,
    page_num: int,
):
    """Write table rows extracted by table_extractor as Fact nodes."""
    if table_rows:
        _write_table_facts_to_neo4j(table_rows, doc_id, sector, page_num)
        logger.debug(f"Page {page_num + 1}: wrote {len(table_rows)} table row facts")