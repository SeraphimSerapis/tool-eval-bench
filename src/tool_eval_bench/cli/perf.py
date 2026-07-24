"""Throughput benchmark runners for the CLI.

Extracted from the monolithic ``cli/bench.py``. Both functions follow the
same pattern: display a header panel, run the benchmark with progress, render
a summary table. They are kept together because they share the same CLI
options (``--perf``, ``--perf-only``, ``--perf-legacy``) and similar
display conventions.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rich.console import Console


def run_throughput(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    *,
    pp: int,
    tg: int,
    depths: list[int],
    concurrency_levels: list[int],
) -> list:
    """Run llama-bench style throughput sweep and display results.

    Returns a list of ThroughputSample objects for report persistence.
    """
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.runner.throughput import ThroughputSample, run_throughput_matrix

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]pp={pp}  tg={tg}  depth={depths}  concurrency={concurrency_levels}[/]",
            title="[bold]⚡ Throughput Benchmark[/]",
            border_style="bright_cyan",
        )
    )

    completed: list[ThroughputSample] = []

    async def on_sample(sample: ThroughputSample, idx: int, total: int) -> None:
        completed.append(sample)
        label = f"pp{sample.label_pp} @ d{sample.label_depth} c{sample.concurrency}"
        if sample.error:
            console.print(f"  [red]✗[/] {label} — {sample.error}")
        else:
            console.print(
                f"  [green]✓[/] {label}  "
                f"[bold]{sample.pp_tps:,.0f}[/] pp t/s  "
                f"[bold]{sample.tg_tps:,.1f}[/] tg t/s  "
                f"[dim]ttft={sample.ttft_ms:,.0f}ms  total={sample.total_ms:,.0f}ms[/]"
            )

    matrix_result_holder: list[Any] = []

    async def _run() -> None:
        result = await run_throughput_matrix(
            base_url,
            model,
            pp=pp,
            tg=tg,
            depths=depths,
            concurrency_levels=concurrency_levels,
            api_key=api_key,
            on_sample=on_sample,
        )
        matrix_result_holder.append(result)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/]")
        sys.exit(1)

    # Summary table
    ok_samples = [s for s in completed if not s.error]
    if ok_samples:
        console.print()
        table = Table(
            title="[bold]Throughput Results[/]",
            show_header=True,
            header_style="bold",
            border_style="bright_cyan",
            expand=True,
        )
        table.add_column("Test", min_width=20, no_wrap=True)
        table.add_column("pp t/s", justify="right", width=10)
        table.add_column("tg t/s", justify="right", width=10)
        table.add_column("TTFT (ms)", justify="right", width=10)
        table.add_column("Total (ms)", justify="right", width=10)
        table.add_column("Tokens", justify="right", width=12)

        for s in ok_samples:
            conc_label = f"  c{s.concurrency}" if s.concurrency > 1 else ""
            label = f"pp{s.label_pp} tg{s.tg_tokens} @ d{s.label_depth}{conc_label}"
            table.add_row(
                label,
                f"{s.pp_tps:,.0f}",
                f"{s.tg_tps:,.1f}",
                f"{s.ttft_ms:,.0f}",
                f"{s.total_ms:,.0f}",
                f"{s.pp_tokens}+{s.tg_tokens}",
            )

        console.print(table)

    # Post-run hints
    matrix_result = matrix_result_holder[0] if matrix_result_holder else None
    if matrix_result is not None and matrix_result.spec_decoding_detected:
        method_label = (
            f" ({matrix_result.spec_decoding_method})" if matrix_result.spec_decoding_method else ""
        )
        console.print(
            Panel(
                f"[bold yellow]⚡ Speculative decoding detected{method_label}[/]\n"
                "Standard [cyan]tg t/s[/] under-reports real throughput for spec-decode models.\n"
                "Re-run with [bold cyan]--spec-bench[/] for acceptance rate (α) and effective t/s.",
                border_style="yellow",
            )
        )
    if ok_samples and ok_samples[0].calibration_confidence == "heuristic":
        console.print(
            "[dim yellow]⚠ Token counts use 4 chars/token heuristic — pp t/s may be "
            "inaccurate for non-English or multilingual models.[/]"
        )

    console.print()
    return completed


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
            current_test = ""
            completed_runs = 0

            def on_progress(event: dict[str, Any]) -> None:
                nonlocal current_test, completed_runs
                event_type = event.get("type", "")
                if event_type == "request_start":
                    pp = event.get("prompt_size", "?")
                    tg = event.get("response_size", "?")
                    depth = event.get("context_size", 0)
                    conc = event.get("concurrency", 1)
                    run_idx = event.get("run_index", 0)
                    current_test = f"pp{pp} tg{tg} @ d{depth} c{conc} run {run_idx}"
                    progress.update(task, description=current_test)
                elif event_type == "request_end":
                    completed_runs += 1
                    progress.update(task, completed=completed_runs)
                elif event_type == "bench_complete":
                    progress.update(task, completed=total_runs, description="[green]✓ Complete")

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
                extra_args=extra_args,
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
