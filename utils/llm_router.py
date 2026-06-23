"""
utils/llm_router.py
-------------------
Single switchboard for LLM backends. Re-exports the SAME names that the
extractors / engines already import from utils.ollama_client, so switching a
file over is a one-line import change with nothing else to touch:

    -  from utils.ollama_client import generate_json
    +  from utils.llm_router    import generate_json

Which backend each name resolves to is decided ONCE here, from the toggles in
config/backend_settings.py:

    USE_LMSTUDIO_FOR_TEXT  → text gen (generate / generate_json / classify_sector
                             / get_model_for_task / TaskType) from LM Studio,
                             else from Ollama.
    USE_GLM_OCR_FOR_VISION → vision (vision_extract / vision_extract_table_json
                             / vision_describe_image) from GLM-OCR, else Ollama.

The two halves are independent. Leave both False and this module behaves exactly
like the legacy utils.ollama_client.

NOTE: embeddings (ollama.embed in kg_embeddings.py) are intentionally NOT routed
here — they stay on Ollama's mxbai-embed-large. Move them later if desired.
"""

from loguru import logger

from config.backend_settings import USE_LMSTUDIO_FOR_TEXT, USE_GLM_OCR_FOR_VISION

# ─── Text generation backend ───────────────────────────────────────────────────

if USE_LMSTUDIO_FOR_TEXT:
    logger.debug("[llm_router] text backend → LM Studio")
    from utils.lmstudio_client import (
        generate, generate_json, classify_sector,
        get_model_for_task, TaskType,
    )
else:
    logger.debug("[llm_router] text backend → Ollama")
    from utils.ollama_client import (
        generate, generate_json, classify_sector,
        get_model_for_task, TaskType,
    )


# ─── Vision backend ─────────────────────────────────────────────────────────────

if USE_GLM_OCR_FOR_VISION:
    logger.debug("[llm_router] vision backend → GLM-OCR")
    from utils.glm_ocr_client import (
        vision_extract, vision_extract_table_json, vision_describe_image,
    )
else:
    logger.debug("[llm_router] vision backend → Ollama (llama3.2-vision)")
    from utils.ollama_client import (
        vision_extract, vision_extract_table_json, vision_describe_image,
    )


__all__ = [
    "generate", "generate_json", "classify_sector",
    "get_model_for_task", "TaskType",
    "vision_extract", "vision_extract_table_json", "vision_describe_image",
]
