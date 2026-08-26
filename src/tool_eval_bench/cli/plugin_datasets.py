"""Shared dataset loading with visible download progress.

Each accuracy plugin caches its dataset under ``data/<name>/`` and downloads it
from HuggingFace on first use.  GSM8K, MMLU, and IFEval carried three copies of
the same load-or-download flow, differing only in the benchmark's name, what it
calls its items, and whether a partial download can be resumed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console


def _download_method_hint() -> str:
    """Say which transport the loader will use, so a slow download is explicable."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return "via REST API"
    return "via datasets lib"


def load_dataset_with_progress(
    console: Console,
    *,
    name: str,
    noun: str,
    cache_path: Path,
    load: Callable[..., Any],
    cache_note: str,
    partial_path: Path | None = None,
) -> Any | None:
    """Load a cached dataset, downloading it with a progress spinner if absent.

    *noun* is what this benchmark calls its items ("questions", "prompts").
    *cache_note* is the path shown after a successful download.  Pass
    *partial_path* for a loader that can resume an interrupted download; it
    changes the wording and adds the "progress is saved" hint on failure.

    Returns the loaded items, or ``None`` when the download failed.  The caller
    should treat ``None`` as "stop, the user has already been told why".
    """
    if cache_path.exists():
        console.print(f"  [dim]Loading {name} from cache…[/]", end=" ")
        items = load()
        console.print(f"[bold green]✓[/] [dim]{len(items)} {noun}[/]")
        return items

    resuming = partial_path is not None and partial_path.exists()
    label = f"Resuming {name} download" if resuming else f"Downloading {name} dataset"
    console.print()
    with console.status(
        f"[bold]{label} from HuggingFace…[/] [dim]({_download_method_hint()})[/]",
        spinner="dots",
    ) as status:

        def on_download(downloaded: int, total: int) -> None:
            pct = downloaded / total * 100 if total else 0
            status.update(f"[bold]{label}…[/] [dim]{downloaded:,}/{total:,} {noun} ({pct:.0f}%)[/]")

        try:
            items = load(on_progress=on_download)
        except Exception as exc:
            resume_hint = (
                "  Progress is saved — re-run to resume from where it stopped.\n"
                if partial_path is not None
                else ""
            )
            console.print(
                f"\n  [bold red]✗[/] Failed to download {name} dataset: {exc}\n"
                "  [dim]This is usually caused by HuggingFace rate limiting.\n"
                f"{resume_hint}"
                "  Tip: pip install tool-eval-bench[hf] for rate-limit-free downloads.[/]"
            )
            return None

    console.print(
        f"  [bold green]✓[/] Downloaded [bold]{len(items)}[/] {noun} "
        f"[dim](cached to {cache_note})[/]"
    )
    return items
