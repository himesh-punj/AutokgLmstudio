"""
utils/model_manager.py
----------------------
Automates LM Studio model load/unload with GPU offload so each pipeline stage
runs on the right model without manual `lms load` commands:

    extraction / engines  → text model      (Qwen2.5-14B-Instruct)  on GPU
    validation            → reasoning model  (Qwen3-14B)             on GPU

A 12 GB GPU can't hold both 14B models at once, so switching models unloads the
other first. Loading ALWAYS forces `--gpu max` (LM Studio otherwise sometimes
loads to CPU → ~9x slower). All calls are best-effort and never raise — if the
`lms` CLI isn't available the pipeline still runs against whatever is loaded.

Controlled by AUTO_MANAGE_MODELS in config/backend_settings.py. No-op when that
is False or when the text backend is Ollama (USE_LMSTUDIO_FOR_TEXT=False).
"""

import os
import subprocess

from loguru import logger

from config.backend_settings import (
    USE_LMSTUDIO_FOR_TEXT, AUTO_MANAGE_MODELS, LMSTUDIO_LMS_PATH,
    LMSTUDIO_GPU_OFFLOAD, LMSTUDIO_LOAD_CTX, LMSTUDIO_LOAD_PARALLEL,
)


def _enabled() -> bool:
    return bool(AUTO_MANAGE_MODELS and USE_LMSTUDIO_FOR_TEXT)


def _lms() -> str:
    if LMSTUDIO_LMS_PATH:
        return LMSTUDIO_LMS_PATH
    p = os.path.expanduser(os.path.join("~", ".lmstudio", "bin", "lms.exe"))
    return p if os.path.exists(p) else "lms"


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([_lms(), *args], capture_output=True, text=True, timeout=timeout)


def _ps() -> str:
    # `lms ps` prints to stdout when models are loaded but to stderr when none are
    # (and this can vary by version) — read both so is_loaded() is reliable.
    try:
        r = _run(["ps"], timeout=20)
        return (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:
        return ""


def is_loaded(model: str) -> bool:
    return bool(model) and model in _ps()


def unload_all() -> None:
    """Unload every resident LM Studio model (free VRAM)."""
    if not _enabled():
        return
    try:
        _run(["unload", "--all"], timeout=30)
        logger.info("[models] LM Studio: unloaded all")
    except Exception as e:
        logger.warning(f"[models] unload failed: {e}")


def ensure_loaded(model: str, ctx: int = None, parallel: int = None, gpu: str = None) -> bool:
    """Make `model` the single resident LM Studio model, loaded on GPU.

    Idempotent: if it's already loaded, do nothing; otherwise unload the others
    (two 14B models don't co-fit on 12 GB) and load it with --gpu max. Returns
    True if it's (now) loaded. No-op — returns True — when auto-management is off
    or the text backend isn't LM Studio.
    """
    if not _enabled():
        return True
    if is_loaded(model):
        logger.info(f"[models] '{model}' already loaded")
        return True

    unload_all()
    ctx = ctx or LMSTUDIO_LOAD_CTX
    parallel = parallel if parallel is not None else LMSTUDIO_LOAD_PARALLEL
    gpu = gpu or LMSTUDIO_GPU_OFFLOAD
    cmd = ["load", model, "--gpu", str(gpu), "-c", str(ctx), "-y"]
    if parallel:
        cmd += ["--parallel", str(parallel)]
    logger.info(f"[models] loading '{model}' (gpu={gpu}, ctx={ctx}, parallel={parallel}) …")
    try:
        r = _run(cmd, timeout=600)
        if r.returncode != 0:
            logger.warning(f"[models] load rc={r.returncode}: {(r.stderr or r.stdout or '')[:200]}")
    except Exception as e:
        logger.warning(f"[models] load failed: {e}")
        return False
    ok = is_loaded(model)
    logger.info(f"[models] '{model}' loaded" if ok else f"[models] '{model}' NOT confirmed loaded "
                f"(continuing against whatever is resident)")
    return ok
