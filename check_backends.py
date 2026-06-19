#!/usr/bin/env python3
"""
check_backends.py
-----------------
Connectivity / health check for the alternative LLM backends.
Run this BEFORE flipping the pipeline over to LM Studio + GLM-OCR.

    python check_backends.py

It verifies:
  1. LM Studio server is up, lists exposed model ids, and runs a tiny chat call.
  2. Ollama is up, lists tags, and confirms GLM_OCR_MODEL is pulled.
  3. Whether the configured model ids actually match what each server exposes.

Nothing here writes to Neo4j or touches the pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

from config.backend_settings import (
    USE_LMSTUDIO_FOR_TEXT, USE_GLM_OCR_FOR_VISION,
    LMSTUDIO_BASE_URL, LMSTUDIO_TEXT_MODEL, LMSTUDIO_FAST_MODEL,
    GLM_OCR_MODEL,
)

console = Console()


def check_lmstudio() -> bool:
    console.rule("[bold]LM Studio (text)[/bold]")
    from utils import lmstudio_client

    ok, model_ids = lmstudio_client.ping()
    if not ok:
        console.print(f"[red][X] Server not reachable at {LMSTUDIO_BASE_URL}[/red]")
        console.print("[yellow]  -> In LM Studio: Developer tab -> Start Server (port 1234)[/yellow]")
        console.print("[yellow]  -> Enable 'Just-In-Time Model Loading'[/yellow]")
        return False

    console.print(f"[green][OK] Server up[/green] at {LMSTUDIO_BASE_URL}")
    console.print(f"  Exposed models: {model_ids or '[dim]none loaded (JIT will load on request)[/dim]'}")

    for label, mid in (("text", LMSTUDIO_TEXT_MODEL), ("fast", LMSTUDIO_FAST_MODEL)):
        match = any(mid == m or mid in m for m in model_ids)
        flag = "[green]match[/green]" if match else "[yellow]not currently listed (JIT may still load it)[/yellow]"
        console.print(f"  {label} model id = [cyan]{mid}[/cyan] -> {flag}")

    # Tiny live generation test
    try:
        console.print("\n  Running a 1-line generation test...")
        out = lmstudio_client.generate(
            "Reply with exactly the word: OK", temperature=0.0
        )
        console.print(f"  Response: [green]{out[:80]}[/green]")
        return True
    except Exception as e:
        console.print(f"[red][X] Generation test failed: {e}[/red]")
        console.print("[yellow]  -> Check the model id matches an installed model in LM Studio.[/yellow]")
        return False


def check_glm_ocr() -> bool:
    console.rule("[bold]GLM-OCR (vision)[/bold]")
    from utils import glm_ocr_client

    ollama_ok, present, tags = glm_ocr_client.ping()
    if not ollama_ok:
        console.print("[red][X] Ollama not reachable. Start it with `ollama serve`.[/red]")
        return False

    console.print("[green][OK] Ollama up[/green]")
    console.print(f"  Installed tags: {tags or '[dim]none[/dim]'}")

    if present:
        console.print(f"  GLM-OCR model = [cyan]{GLM_OCR_MODEL}[/cyan] -> [green]present[/green]")
        return True

    console.print(f"  GLM-OCR model = [cyan]{GLM_OCR_MODEL}[/cyan] -> [red]NOT FOUND[/red]")
    console.print("[yellow]  -> Pull it, then set GLM_OCR_MODEL in config/backend_settings.py[/yellow]")
    console.print("[yellow]     to the exact NAME shown in `ollama list`.[/yellow]")
    return False


def main():
    console.print(Panel(
        f"USE_LMSTUDIO_FOR_TEXT  = {USE_LMSTUDIO_FOR_TEXT}\n"
        f"USE_GLM_OCR_FOR_VISION = {USE_GLM_OCR_FOR_VISION}",
        title="[bold]backend_settings toggles[/bold]", border_style="cyan",
    ))

    text_ok   = check_lmstudio() if USE_LMSTUDIO_FOR_TEXT else (console.print("[dim]LM Studio text: disabled[/dim]") or True)
    vision_ok = check_glm_ocr()  if USE_GLM_OCR_FOR_VISION else (console.print("[dim]GLM-OCR vision: disabled[/dim]") or True)

    console.rule("[bold]Summary[/bold]")
    console.print(f"  Text backend:   {'[green]READY[/green]' if text_ok else '[red]NOT READY[/red]'}")
    console.print(f"  Vision backend: {'[green]READY[/green]' if vision_ok else '[red]NOT READY[/red]'}")

    if text_ok and vision_ok:
        console.print(
            "\n[green]Both backends ready.[/green] To go live, swap the import line in each "
            "file that currently uses utils.ollama_client over to utils.llm_router."
        )
    else:
        console.print("\n[yellow]Fix the items above, then re-run this check.[/yellow]")


if __name__ == "__main__":
    main()
