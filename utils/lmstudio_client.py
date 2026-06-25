"""
utils/lmstudio_client.py
------------------------
Drop-in alternative to utils.ollama_client for TEXT generation, backed by
LM Studio's OpenAI-compatible server (http://localhost:1234/v1 by default).

Why LM Studio:
  - Just-in-time (JIT) model loading: request a model by id and LM Studio loads
    it on demand, then unloads on idle — no manual `ollama run`, no 14B model
    pinned in VRAM for the whole pipeline.
  - Cleaner model management / swapping during development.

Public API mirrors utils.ollama_client exactly so it can be swapped in via
utils.llm_router with no changes to the extractors / engines:
    generate(prompt, system, model, temperature) -> str
    generate_json(prompt, system, model)         -> dict | list | None
    classify_sector(text_sample, sectors)        -> (sector, confidence)
    get_model_for_task(task) -> str ;  TaskType

Talks plain HTTP (requests) to /v1/chat/completions — no extra heavy deps.
"""

import json
import re
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from loguru import logger


def _is_retryable(exc: Exception) -> bool:
    """Retry transient errors (timeouts, connection drops, 5xx) but NOT 4xx —
    a 400 (e.g. context-size exceeded) is deterministic, so retrying just wastes time."""
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return True

from config.backend_settings import (
    LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY,
    LMSTUDIO_TEXT_MODEL, LMSTUDIO_FAST_MODEL,
    LMSTUDIO_TIMEOUT, LMSTUDIO_MAX_RETRIES, LMSTUDIO_MAX_TOKENS,
    USE_THINKING_FOR_VALIDATION, LMSTUDIO_VALIDATION_MODEL,
    LMSTUDIO_VALIDATION_MAX_TOKENS,
)

# ─── Model router (mirrors ollama_client.TaskType) ─────────────────────────────
# Heavy structured extraction → text model.  Simple/short prompts → fast model.
# Validation reasoning → optional separate "thinking" model (see backend_settings).

class TaskType:
    EXTRACTION   = "extraction"    # fact / triple extraction  → heavy model
    CONCEPT      = "concept"       # concept induction         → fast model
    CLASSIFY     = "classify"      # sector classification     → fast model
    CONSISTENCY  = "consistency"   # consistency / anomaly LLM → heavy model
    VALIDATION   = "validation"    # validation reasoning      → thinking model (if enabled)


def get_model_for_task(task: str) -> str:
    """Route a task to the appropriate LM Studio model id."""
    if task == TaskType.VALIDATION and USE_THINKING_FOR_VALIDATION:
        return LMSTUDIO_VALIDATION_MODEL
    fast_tasks = {TaskType.CONCEPT, TaskType.CLASSIFY}
    return LMSTUDIO_FAST_MODEL if task in fast_tasks else LMSTUDIO_TEXT_MODEL


def _max_tokens_for_model(model: Optional[str]) -> int:
    """The reasoning validation model needs a larger budget — thinking burns
    output tokens before it emits the JSON answer."""
    if USE_THINKING_FOR_VALIDATION and model == LMSTUDIO_VALIDATION_MODEL:
        return LMSTUDIO_VALIDATION_MAX_TOKENS
    return LMSTUDIO_MAX_TOKENS


# ─── Low-level chat call ───────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
    }


@retry(
    stop=stop_after_attempt(LMSTUDIO_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _chat(messages: list[dict], model: str, temperature: float = 0.1,
          max_tokens: int = LMSTUDIO_MAX_TOKENS) -> str:
    """POST /v1/chat/completions and return the assistant message content."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(
        f"{LMSTUDIO_BASE_URL}/chat/completions",
        headers=_headers(),
        json=payload,
        timeout=LMSTUDIO_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ─── Text generation ───────────────────────────────────────────────────────────

def generate(prompt: str, system: str = "", model: str = None, temperature: float = 0.1,
             max_tokens: int = None) -> str:
    """Generate a text response from LM Studio. Low temperature for structured extraction.
    max_tokens defaults per-model (the reasoning validation model gets a larger budget)."""
    m = model or LMSTUDIO_TEXT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _chat(messages, model=m, temperature=temperature,
                 max_tokens=max_tokens or _max_tokens_for_model(m))


def generate_json(prompt: str, system: str = "", model: str = None,
                  temperature: float = 0.05, max_tokens: int = None) -> dict | list | None:
    """
    Generate and parse JSON from LM Studio.
    Handles: markdown fences, multiple objects (wraps into array),
    truncated responses, and extra text after valid JSON.
    (Same multi-strategy parser as utils.ollama_client.generate_json.)
    Pass temperature=0.0 for reproducible structured output.
    """
    json_system = (system + "\n\n" if system else "") + (
        "You MUST respond with valid JSON only. "
        "No markdown, no explanation, no preamble. "
        "Start your response directly with { or [ as appropriate."
    )
    raw = generate(prompt, system=json_system, model=model, temperature=temperature,
                   max_tokens=max_tokens)
    return _extract_json(raw)


# ─── Robust JSON extraction (mirrors ollama_client) ────────────────────────────

def _strip_think(raw: str) -> str:
    """Remove a reasoning model's <think>…</think> block(s). The JSON answer
    follows the reasoning, so we keep only what comes after. Handles a closed
    block, and the truncated case where the closing tag never arrives (then the
    whole thing was thinking → nothing usable)."""
    if "<think>" not in raw and "</think>" not in raw:
        return raw
    if "</think>" in raw:
        # keep everything after the LAST closing tag (the actual answer)
        raw = raw.rsplit("</think>", 1)[1]
    else:
        # opened but never closed → reasoning got truncated, no answer emitted
        raw = raw.split("<think>", 1)[0]
    return raw.strip()


def _extract_json(raw: str) -> dict | list | None:
    """Best-effort recovery of a JSON value from a (possibly messy) LLM string."""
    # Strip a reasoning model's <think>…</think> first, then markdown fences
    raw = _strip_think(raw)
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Try 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try 2: extract first complete array [ ... ]
    match = re.search(r"(\[.*?\])", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try 3: multiple JSON objects {...}{...} → collect and wrap into array
    objects = []
    pos = 0
    while pos < len(raw):
        start = -1
        for i in range(pos, len(raw)):
            if raw[i] in ("{", "["):
                start = i
                break
        if start == -1:
            break
        depth = 0
        open_char  = raw[start]
        close_char = "}" if open_char == "{" else "]"
        for i in range(start, len(raw)):
            if raw[i] == open_char:
                depth += 1
            elif raw[i] == close_char:
                depth -= 1
                if depth == 0:
                    candidate = raw[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            objects.append(parsed)
                        elif isinstance(parsed, list):
                            objects.extend(parsed)
                        pos = i + 1
                    except json.JSONDecodeError:
                        pos = start + 1
                    break
        else:
            break

    if objects:
        return objects

    # Try 4: truncated JSON — json_repair if available
    try:
        import json_repair
        result = json_repair.loads(raw)
        if result:
            return result
    except (ImportError, Exception):
        pass

    logger.warning(f"[LM Studio] JSON parse failed after all attempts. Raw snippet: {raw[:200]}")
    return None


# ─── Sector classification ─────────────────────────────────────────────────────

def classify_sector(text_sample: str, sectors: list[str]) -> tuple[str, float]:
    """
    Zero-shot sector classification. Returns (sector_name, confidence 0-1).
    Uses the fast model — classification is a simple task.
    """
    sectors_formatted = "\n".join(f"- {s}" for s in sectors)
    prompt = (
        f"You are classifying an infrastructure project document.\n\n"
        f"Available sectors:\n{sectors_formatted}\n\n"
        f"Document excerpt:\n\"\"\"\n{text_sample[:3000]}\n\"\"\"\n\n"
        "Which sector does this document belong to? "
        "Respond with a JSON object: "
        '{"sector": "<exact sector name from the list>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}'
    )
    result = generate_json(prompt, model=get_model_for_task(TaskType.CLASSIFY))
    if result and isinstance(result, dict) and "sector" in result:
        return result["sector"], float(result.get("confidence", 0.5))
    logger.warning("[LM Studio] Sector classification failed, defaulting to first sector")
    return sectors[0], 0.1


# ─── Health check ──────────────────────────────────────────────────────────────

def ping() -> tuple[bool, list[str]]:
    """
    Check the LM Studio server is up. Returns (ok, [available_model_ids]).
    Used by check_backends.py.
    """
    try:
        resp = requests.get(f"{LMSTUDIO_BASE_URL}/models", headers=_headers(), timeout=10)
        resp.raise_for_status()
        ids = [m.get("id", "?") for m in resp.json().get("data", [])]
        return True, ids
    except Exception as e:
        logger.error(f"[LM Studio] not reachable at {LMSTUDIO_BASE_URL}: {e}")
        return False, []
