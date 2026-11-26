"""Shared execution and finalization lifecycle for external plugins."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from rich.console import Console
from rich.markup import escape

from tool_eval_bench.application.mode_runs import ModeRun, finalize_mode_run
from tool_eval_bench.cli.headless import report_run_failed, report_run_saved
from tool_eval_bench.cli.plugin_progress import PluginProgressDisplay
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.redaction import redact_urls
from tool_eval_bench.storage.reports.mode import ModeReport

ProgressCallback = Callable[[int, int, dict[str, Any]], Awaitable[None]]


def items_per_minute(count: int, duration_seconds: float) -> float:
    """Return throughput in items/minute, or 0.0 when duration is non-positive."""
    return count / duration_seconds * 60 if duration_seconds > 0 else 0.0


def execute_plugin(
    console: Console,
    benchmark_name: str,
    run: Callable[[], Any],
    result_holder: list[Any],
) -> Any | None:
    """Execute a plugin coroutine with consistent failure handling."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        report_run_failed(console, "\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        report_run_failed(
            console,
            f"\n[bold red]{escape(benchmark_name)} error:[/] {escape(redact_urls(str(exc)))}",
        )
        sys.exit(1)

    if not result_holder:
        console.print(f"[bold red]No {escape(benchmark_name)} results.[/]")
        return None
    return result_holder[0]


def run_plugin_session(
    console: Console,
    *,
    benchmark_name: str,
    adapter: Any,
    total: int | Callable[[], int],
    run_plugin: Callable[[ProgressCallback], Awaitable[Any]],
    format_progress: Callable[[PluginProgressDisplay, int, dict[str, Any]], tuple[str, str]],
    format_finish: Callable[[Any], tuple[int, str]],
    record_item: Callable[[PluginProgressDisplay, dict[str, Any]], None] | None = None,
) -> Any | None:
    """Run one plugin with live progress display, adapter cleanup, and error handling."""
    result_holder: list[Any] = []

    async def run() -> None:
        try:
            resolved_total = total() if callable(total) else total
            with PluginProgressDisplay(console, total=resolved_total) as display:

                async def on_progress(
                    current: int, prog_total: int, item_info: dict[str, Any]
                ) -> None:
                    if record_item is not None:
                        record_item(display, item_info)
                    else:
                        display.tally.record(item_info)
                    stats, detail = format_progress(display, current, item_info)
                    display.advance(current, prog_total, stats=stats, detail=detail)

                result = await run_plugin(on_progress)
                result_holder.append(result)
                completed, final_stats = format_finish(result)
                display.finish(completed=completed, stats=final_stats)
        finally:
            if hasattr(adapter, "aclose"):
                await adapter.aclose()

    return execute_plugin(console, benchmark_name, run, result_holder)


def finalize_plugin_run(
    console: Console,
    *,
    mode: str,
    title: str,
    display_name: str,
    result: Any,
    config: dict[str, Any],
    report_metrics: list[str],
    report_lines: list[str],
    output_dir: str | None,
    label: str | None,
    run_context: RunContext | None,
) -> str:
    """Write and persist a completed plugin run, then print where the report went."""
    run = ModeRun(
        run_type=mode,
        config=config,
        scores={
            "final_score": round(result.score),
            "accuracy": result.score,
            "rating": result.rating,
            **result.details,
        },
        status="completed",
    )
    finalized = finalize_mode_run(
        run,
        ModeReport(
            title=f"{title} Benchmark",
            display_name=display_name,
            mode=mode,
            label=label,
            header=(*report_metrics, f"- **Rating**: {result.rating}"),
            body=tuple(report_lines),
        ),
        run_context=run_context,
        output_dir=output_dir,
    )
    report_run_saved(console, run, finalized)
    console.print(f"\n  [dim]Report saved to {escape(str(finalized.report_path))}[/]\n")
    return finalized.run_id
