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

LMSTUDIO_TIMEOUT     = 240   # seconds — generous headroom for concurrent generation
LMSTUDIO_MAX_RETRIES = 2     # a timed-out call rarely succeeds on retry — don't burn 3×240s
LMSTUDIO_MAX_TOKENS  = 1536  # max generated tokens/request. Pages are now extracted
                             # in small chunks (settings.PAGE_EXTRACT_CHUNK_CHARS), so each
                             # call is light: prompt ~2100 tokens + this ~1536 ≈ 3700 tokens,
                             # which fits comfortably even at -c 8192 --parallel 2 (4096/slot).
                             # Smaller calls = bounded VRAM = no spill-to-RAM freeze, and you
                             # can run more parallel workers. Raise only if facts get cut off.
LMSTUDIO_NUM_CTX     = 8192  # advisory context hint (LM Studio honours its own loaded ctx)


# ─── Reasoning ("thinking") model for the validation judge ─────────────────────
# The validation judge (validators/context_validator.py) makes one bounded call
# per requirement that must (a) disambiguate context (design vs average speed) and
# (b) apply dimensional sanity (metres ≠ ratio, % ≠ length) CONSISTENTLY. A plain
# instruct model does this non-deterministically — on c2cd8556 two sibling ratio
# rules got opposite verdicts on the same metres-vs-ratio input. A reasoning model
# that thinks step-by-step before answering fixes that.
#
#   Download:  lmstudio-community/Qwen3-14B-GGUF  (Q4_K_M, ~9 GB — fits the 12 GB
#              RTX 5070 because validation runs solo, no GLM-OCR sharing).
#   Confirm the exact model id LM Studio exposes with `python check_backends.py`
#   and set LMSTUDIO_VALIDATION_MODEL to it.
#
# Thinking models emit a <think>…</think> block before the JSON answer; that block
# is stripped in lmstudio_client._extract_json, and the validation call gets the
# larger token budget below so the answer isn't truncated by the reasoning.
# Set USE_THINKING_FOR_VALIDATION=False to fall back to LMSTUDIO_TEXT_MODEL.
USE_THINKING_FOR_VALIDATION    = True
LMSTUDIO_VALIDATION_MODEL      = "qwen3-14b"
LMSTUDIO_VALIDATION_MAX_TOKENS = 4096   # reasoning eats output tokens before the JSON

# Reproducibility. temperature=0 alone is NOT deterministic on a parallel GPU server —
# two identical runs flipped 3/21 verdicts and swung the score 90.9%→100%. A fixed seed
# (+ greedy sampling) makes each judgment reproducible so prompt/rule changes can actually
# be measured instead of fighting noise. Set LMSTUDIO_SEED=None to restore sampling.
LMSTUDIO_SEED = 42

# Self-consistency votes per validation judgment. The judge's SEMANTIC fields
# (applicable / referent pick) are decided by majority over this many samples
# (seeds LMSTUDIO_SEED..+N-1) — reproducible across runs, robust to the thinking
# model's residual GPU nondeterminism. The numeric verdict itself is computed in
# code (validators/quantity.py) and is already deterministic. 1 = no voting.
# (LM Studio + qwen3 does NOT honor seed reliably — verified — so a few votes are
# needed to stabilise the majority; the deterministic core keeps the score sound.)
VALIDATION_VOTES = 5


# ─── GLM-OCR vision model (served by Ollama) ───────────────────────────────────
# Pull the model first, then set GLM_OCR_MODEL to the EXACT tag from `ollama list`:
#     ollama pull <glm-ocr-tag>
#     ollama list                 # ← copy the NAME column value here
# Run `python check_backends.py` to print the available tags.

# ─── Optional: route low-density pages to a small/fast model ───────────────────
# Low-density pages (short text, few engineering values) can be extracted by a
# small model served by Ollama (separate process from LM Studio, so no model
# swapping). OFF by default — on a 12GB GPU a 3B model competes for VRAM with the
# 14B (LM Studio) + GLM-OCR, and the page pre-filter already skips no-signal pages,
# so the win here is marginal/negative. Enable ONLY with VRAM headroom, after:
#   ollama pull qwen2.5:3b-instruct
USE_FAST_PAGE_ROUTING = False
FAST_PAGE_MODEL       = "qwen2.5:3b-instruct"   # Ollama small model for low-density pages


GLM_OCR_MODEL       = "glm-ocr:latest"   # default tag (~2.2 GB) from `ollama pull glm-ocr`
GLM_OCR_NUM_CTX     = 8192               # a rasterised page image is ~4k+ vision tokens —
                                         # 4096 overflows ("exceeds context size"); 8192 fits.
GLM_OCR_TEMPERATURE = 0.05
GLM_OCR_TIMEOUT     = 180         # seconds per vision request
