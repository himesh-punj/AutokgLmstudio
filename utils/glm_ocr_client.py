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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from config.backend_settings import (
    GLM_OCR_MODEL, GLM_OCR_NUM_CTX, GLM_OCR_TEMPERATURE,
)
from config.settings import OLLAMA_MAX_RETRIES


# ─── Low-level vision call ─────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(OLLAMA_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception_type(Exception),
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

def vision_extract_table_json(image_path: Path, context: str = "") -> list[dict] | None:
    """
    OCR a table from an image and return it as a list of row dicts.
    Used as the final fallback when pdfplumber and camelot both fail.
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
    raw = vision_extract(image_path, prompt)

    # Strip fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("[GLM-OCR] table extraction: JSON parse failed")
            return None
    return None


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
