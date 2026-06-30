"""
utils/glm_ocr_client.py
-----------------------
Drop-in alternative to the VISION half of utils.ollama_client, backed by a
GLM-OCR model served through Ollama. GLM-OCR models are tuned for reading text,
tables and structured layout out of document images — a better fit for DPR
table / scanned-page extraction than a general vision model.

Public API mirrors the vision functions in utils.ollama_client exactly so it
can be swapped in via utils.llm_router with no changes to the extractors:
    vision_extract(image_path, prompt, model)        -> str
    vision_extract_table_json(image_path, context)   -> list[dict] | None
    vision_describe_image(image_path, sector)        -> str

Still uses the `ollama` python client (the model is served by Ollama). Set the
exact model tag in config/backend_settings.GLM_OCR_MODEL.
"""

import json
import re
import base64
from pathlib import Path
from typing import Optional

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from loguru import logger


def _is_retryable(exc: Exception) -> bool:
    """Don't retry 4xx (e.g. a 400 'context size exceeded') — it's deterministic."""
    sc = getattr(exc, "status_code", None)
    if sc is not None and 400 <= sc < 500:
        return False
    return True

from config.backend_settings import (
    GLM_OCR_MODEL, GLM_OCR_NUM_CTX, GLM_OCR_TEMPERATURE,
)
from config.settings import OLLAMA_MAX_RETRIES


# ─── Low-level vision call ─────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(OLLAMA_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def vision_extract(image_path: Path, prompt: str, model: str = None) -> str:
    """Send an image + prompt to the GLM-OCR model. Returns raw text."""
    m = model or GLM_OCR_MODEL

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = ollama.chat(
        model=m,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
        options={"temperature": GLM_OCR_TEMPERATURE, "num_ctx": GLM_OCR_NUM_CTX},
    )
    return response["message"]["content"].strip()


# ─── Table extraction ──────────────────────────────────────────────────────────

def _rows_from_raw(raw: str) -> list[dict] | None:
    """Best-effort recovery of a JSON array of row objects from a messy OCR string.
    Multi-strategy (mirrors the text clients) so a minor format slip no longer drops
    the whole table: strip reasoning/fences → direct parse → array slice → json_repair."""
    if not raw:
        return None
    # drop any <think>…</think> and markdown fences
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

    def _aslist(obj):
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)] or None
        if isinstance(obj, dict):
            # sometimes wrapped as {"rows": [...]} or a single row object
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return [obj]
        return None

    # 1) direct
    try:
        r = _aslist(json.loads(raw))
        if r:
            return r
    except json.JSONDecodeError:
        pass
    # 2) first [...] slice
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            r = _aslist(json.loads(m.group(0)))
            if r:
                return r
        except json.JSONDecodeError:
            pass
    # 3) json_repair (truncated / trailing-comma / unquoted)
    try:
        import json_repair
        r = _aslist(json_repair.loads(raw))
        if r:
            return r
    except Exception:
        pass
    return None


def vision_extract_table_json(image_path: Path, context: str = "") -> list[dict] | None:
    """
    OCR a table from an image and return it as a list of row dicts.
    Used as the final fallback when pdfplumber and camelot both fail. Never raises —
    a hard vision failure or unparseable output returns None so one bad page is skipped,
    not retried noisily or propagated.
    """
    prompt = (
        f"{'Context: ' + context + chr(10) if context else ''}"
        "This image contains an engineering table from a Detailed Project Report (DPR). "
        "Read ALL rows and columns exactly as printed. "
        "Return ONLY a JSON array where each element is an object representing one row. "
        "Use the column headers as keys. Preserve numeric values exactly as shown. "
        "If a cell is empty, use null. "
        "Do not add any explanation. Start directly with [."
    )
    try:
        raw = vision_extract(image_path, prompt)
    except Exception as e:
        logger.warning(f"[GLM-OCR] vision call failed for {Path(image_path).name}: {e}")
        return None

    rows = _rows_from_raw(raw)
    if rows is None:
        logger.warning(f"[GLM-OCR] table JSON unparseable for {Path(image_path).name} "
                       f"(snippet: {(raw or '')[:120]!r}) — page skipped")
    return rows


# ─── Image description ─────────────────────────────────────────────────────────

def vision_describe_image(image_path: Path, sector: str = "") -> str:
    """Get a textual description of an engineering diagram/image for fact extraction."""
    prompt = (
        f"This is an engineering diagram from a {'[' + sector + '] ' if sector else ''}DPR. "
        "Read and describe all engineering information, measurements, labels, and annotations "
        "visible in the image. Be specific about numbers, units, materials, and structural elements."
    )
    return vision_extract(image_path, prompt)


# ─── Health check ──────────────────────────────────────────────────────────────

def ping() -> tuple[bool, bool, list[str]]:
    """
    Check Ollama is up and whether GLM_OCR_MODEL is present.
    Returns (ollama_ok, model_present, [available_tags]).
    Used by check_backends.py.
    """
    try:
        listed = ollama.list()
        tags = [m.get("model") or m.get("name") for m in listed.get("models", [])]
        present = any(t == GLM_OCR_MODEL or (t or "").startswith(GLM_OCR_MODEL) for t in tags)
        return True, present, [t for t in tags if t]
    except Exception as e:
        logger.error(f"[GLM-OCR] Ollama not reachable: {e}")
        return False, False, []
