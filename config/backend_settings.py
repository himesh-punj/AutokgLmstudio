"""
config/backend_settings.py
--------------------------
Optional alternative LLM backends. Keeps the original Ollama path 100% untouched
— nothing here is read unless you import utils.llm_router (or one of the new
clients) somewhere. The legacy utils.ollama_client path is unaffected.

Two independent switches:

  USE_LMSTUDIO_FOR_TEXT   → route text generation (facts, rules, validation,
                            sector classification) to LM Studio's
                            OpenAI-compatible server instead of Ollama.
                            LM Studio handles just-in-time (JIT) model
                            load/unload, so you don't have to keep a 14B model
                            pinned in VRAM the whole run.

  USE_GLM_OCR_FOR_VISION  → route image / table extraction to a GLM-OCR model
                            served by Ollama instead of llama3.2-vision.

Set a switch to False to fall back to the original Ollama behaviour for that
half of the pipeline. The two switches are fully independent — you can run text
on LM Studio while keeping vision on Ollama, or vice versa.
"""

# ─── Master switches ──────────────────────────────────────────────────────────

USE_LMSTUDIO_FOR_TEXT  = True   # text generation  → LM Studio (else Ollama)
USE_GLM_OCR_FOR_VISION = True   # vision / OCR      → GLM-OCR on Ollama (else llama3.2-vision)


# ─── LM Studio (OpenAI-compatible server) ──────────────────────────────────────
# In LM Studio:  Developer tab → "Start Server"  (default port 1234)
# Enable "Just-In-Time Model Loading" so a request auto-loads the model by id.

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY  = "lm-studio"   # any non-empty string — LM Studio ignores the value

# Model identifiers MUST match the model keys LM Studio exposes.
# Check them with:  GET http://localhost:1234/v1/models   (or `python check_backends.py`)
# Recommended download: lmstudio-community/Qwen2.5-14B-Instruct-GGUF, Q4_K_M quant
# (NOT the Coder variant, NOT the 1M-context variant — 8192 ctx is all this pipeline uses).
LMSTUDIO_TEXT_MODEL = "qwen2.5-14b-instruct"    # heavy: fact / rule / validation
# Fast model: defaults to the SAME 14B model to avoid a second download and avoid
# GPU model-swapping (sector classification runs only once per document, so the
# 14B cost is negligible). Point this at a downloaded 8B if you prefer.
LMSTUDIO_FAST_MODEL = "qwen2.5-14b-instruct"    # fast:  sector classify / concept induction

LMSTUDIO_TIMEOUT     = 180   # seconds — first call includes JIT model load, keep generous
LMSTUDIO_MAX_RETRIES = 3
LMSTUDIO_MAX_TOKENS  = 4096  # max generated tokens per request (fact arrays can be long)
LMSTUDIO_NUM_CTX     = 8192  # advisory context hint (LM Studio honours its own loaded ctx)


# ─── GLM-OCR vision model (served by Ollama) ───────────────────────────────────
# Pull the model first, then set GLM_OCR_MODEL to the EXACT tag from `ollama list`:
#     ollama pull <glm-ocr-tag>
#     ollama list                 # ← copy the NAME column value here
# Run `python check_backends.py` to print the available tags.

GLM_OCR_MODEL       = "glm-ocr:latest"   # default tag (~2.2 GB) from `ollama pull glm-ocr`
GLM_OCR_NUM_CTX     = 4096
GLM_OCR_TEMPERATURE = 0.05
GLM_OCR_TIMEOUT     = 180         # seconds per vision request
