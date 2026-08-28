"""Throughput benchmark runners for the CLI.

Extracted from the monolithic ``cli/bench.py``. The CLI runner invokes the
external llama-benchy benchmark and renders its progress and results.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rich.console import Console


def _measurement_run_key(event: dict[str, Any]) -> tuple[Any, ...]:
    """Stable key for one llama-benchy measurement run (not one HTTP request)."""
    return (
        event.get("prompt_size"),
        event.get("response_size"),
        event.get("context_size"),
        event.get("concurrency"),
        event.get("run_index"),
    )


class _BenchyProgressTracker:
    """Map per-request llama-benchy events onto measurement-run progress.

    llama-benchy emits ``request_start`` / ``request_end`` once per HTTP
    request.  At concurrency > 1 that means multiple ends for a single
    measurement run.  Counting raw ``request_end`` events makes the Rich
    bar climb past ``total_runs`` (e.g. 63/27 for the default sweep).
    """

    def __init__(self) -> None:
        self.current_test = ""
        self.completed_runs = 0
        self._req_to_run: dict[int, tuple[Any, ...]] = {}
        self._ends_remaining: dict[tuple[Any, ...], int] = {}
        self._completed_keys: set[tuple[Any, ...]] = set()

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "request_start":
            pp = event.get("prompt_size", "?")
            tg = event.get("response_size", "?")
            depth = event.get("context_size", 0)
            conc = event.get("concurrency", 1)
            run_idx = event.get("run_index", 0)
            self.current_test = f"pp{pp} tg{tg} @ d{depth} c{conc} run {run_idx}"
            request_id = event.get("request_id")
            if request_id is not None:
                run_key = _measurement_run_key(event)
                self._req_to_run[int(request_id)] = run_key
                expected = int(conc) if isinstance(conc, int) and conc > 0 else 1
                self._ends_remaining.setdefault(run_key, expected)
            return

        if event_type != "request_end":
            return

        request_id = event.get("request_id")
        matched_run: tuple[Any, ...] | None = None
        if request_id is not None:
            rid = int(request_id)
            if rid in self._req_to_run:
                matched_run = self._req_to_run.pop(rid)

        if matched_run is None:
            # Synthetic / legacy events without request correlation: one end
            # still means one completed measurement run.
            self.completed_runs += 1
            return

        remaining = self._ends_remaining.get(matched_run, 1) - 1
        self._ends_remaining[matched_run] = remaining
        if remaining <= 0 and matched_run not in self._completed_keys:
            self._completed_keys.add(matched_run)
            self.completed_runs = len(self._completed_keys)


def _probe_llamacpp_model_path(base_url: str) -> str | None:
    """Best-effort read of llama.cpp ``/props.model_path`` (the loaded GGUF)."""
    import httpx

    url = f"{base_url.rstrip('/').removesuffix('/v1')}/props"
    try:
        resp = httpx.get(url, timeout=3.0)
        if resp.status_code != 200:
            return None
        body = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    path = body.get("model_path")
    if not path:
        settings = body.get("default_generation_settings")
        if isinstance(settings, dict):
            path = settings.get("model")
    return path if isinstance(path, str) and path else None


def _resolve_benchy_tokenizer(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    explicit: str | None,
) -> str | None:
    """Find a local tokenizer for llama-benchy, reporting what was used.

    llama-benchy runs offline (``HF_HUB_OFFLINE=1``), so an empty HuggingFace
    cache makes it fail with a raw transformers traceback.  Resolving the path
    ourselves means users rarely need to pass ``--tokenizer`` by hand.
    """
    from tool_eval_bench.utils.tokenizers import resolve_tokenizer

    if explicit:
        return explicit

    resolution = resolve_tokenizer(model, model_root=display_name)
    if not resolution:
        model_path = _probe_llamacpp_model_path(base_url)
        if model_path:
            resolution = resolve_tokenizer(model, model_root=display_name, model_path=model_path)

    if resolution.source in ("hf-cache", "hf-cache-alias"):
        console.print(f"  [dim]🔤 Tokenizer: {resolution.detail} (HuggingFace cache)[/]")
    elif resolution.source == "model-path":
        console.print(f"  [dim]🔤 Tokenizer: {resolution.path}[/]")

    return resolution.path


def run_llama_benchy(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    *,
    pp: list[int],
    tg: list[int],
    depths: list[int],
    concurrency_levels: list[int],
    runs: int = 3,
    latency_mode: str = "generation",
    skip_coherence: bool = True,
    skip_warmup: bool = False,
    extra_args: list[str] | None = None,
    tokenizer: str | None = None,
    backend: str | None = None,
) -> list:
    """Run llama-benchy externally and display results.

    Returns a list of ThroughputSample objects for report persistence.
    """
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.runner.llama_benchy import (
        LlamaBenchyResult,
        is_available,
        run_llama_benchy,
    )

    if not is_available():
        console.print(
            "[bold red]Error:[/] llama-benchy is not available.\n"
            "Install it with: [bold cyan]pip install llama-benchy[/]\n"
            "Or ensure [bold cyan]uvx[/] is on PATH for zero-install usage."
        )
        sys.exit(1)

    tokenizer = _resolve_benchy_tokenizer(console, model, display_name, base_url, tokenizer)

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]pp={pp}  tg={tg}  depth={depths}  concurrency={concurrency_levels}  "
            f"runs={runs}  latency={latency_mode}[/]",
            title="[bold]⚡ llama-benchy Throughput Benchmark[/]",
            border_style="bright_cyan",
        )
    )
    console.print()

    total_test_points = len(pp) * len(tg) * len(depths) * len(concurrency_levels)
    total_runs = total_test_points * runs

    benchy_result: LlamaBenchyResult | None = None

    async def _run() -> None:
        nonlocal benchy_result

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task("Initializing…", total=total_runs)
            tracker = _BenchyProgressTracker()

            def on_progress(event: dict[str, Any]) -> None:
                event_type = event.get("type", "")
                if event_type == "bench_complete":
                    progress.update(task, completed=total_runs, description="[green]✓ Complete")
                    return
                tracker.handle(event)
                if event_type == "request_start":
                    progress.update(task, description=tracker.current_test)
                elif event_type == "request_end":
                    progress.update(task, completed=min(tracker.completed_runs, total_runs))

            benchy_result = await run_llama_benchy(
                base_url,
                model,
                api_key=api_key,
                pp=pp,
                tg=tg,
                depths=depths,
                concurrency_levels=concurrency_levels,
                runs=runs,
                latency_mode=latency_mode,
                skip_coherence=skip_coherence,
                skip_warmup=skip_warmup,
                backend=backend,
                extra_args=extra_args,
                tokenizer=tokenizer,
                on_progress=on_progress,
            )

            progress.update(task, completed=total_runs, description="[green]✓ Complete")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except RuntimeError as exc:
        console.print(f"\n[bold red]llama-benchy error:[/] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/]")
        sys.exit(1)

    if benchy_result is None:
        console.print("[bold red]No results from llama-benchy.[/]")
        return []

    if benchy_result.version:
        console.print(f"\n  [dim]llama-benchy {benchy_result.version}[/]")
    if benchy_result.latency_ms > 0:
        console.print(f"  [dim]Estimated latency: {benchy_result.latency_ms:.1f} ms[/]")

    ok_samples = [s for s in benchy_result.samples if not s.error]
    if ok_samples:
        console.print()

        labels: list[str] = []
        for s in ok_samples:
            labels.append(f"pp{s.label_pp} tg{s.tg_tokens} @ d{s.label_depth}")
        test_col_width = max(len(lbl) for lbl in labels)

        table = Table(
            title="[bold]llama-benchy Results[/]",
            show_header=True,
            header_style="bold",
            border_style="bright_cyan",
            expand=True,
        )
        table.add_column("Test", min_width=test_col_width, no_wrap=True)
        table.add_column("c", justify="center", width=4)
        table.add_column("pp t/s", justify="right", width=9)
        table.add_column("tg t/s", justify="right", width=9)
        table.add_column("TTFT (ms)", justify="right", width=10)
        table.add_column("Total (ms)", justify="right", width=10)
        table.add_column("Tokens", justify="right", width=10)

        for lbl, s in zip(labels, ok_samples, strict=False):
            table.add_row(
                lbl,
                f"c{s.concurrency}",
                f"{s.pp_tps:,.0f}",
                f"{s.tg_tps:,.1f}",
                f"{s.ttft_ms:,.0f}",
                f"{s.total_ms:,.0f}",
                f"{s.pp_tokens}+{s.tg_tokens}",
            )

        console.print(table)

    if ok_samples and ok_samples[0].calibration_confidence == "llama-benchy":
        console.print(
            "\n  [dim]ℹ Metrics sourced from llama-benchy — see "
            "[bold]https://github.com/eugr/llama-benchy[/] for methodology.[/]"
        )

    console.print()
    return ok_samples
