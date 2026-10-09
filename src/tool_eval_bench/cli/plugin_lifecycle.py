"""Shared execution and finalization lifecycle for external plugins."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any

import httpx
from rich.console import Console

from tool_eval_bench.application.mode_runs import ModeRun, finalize_mode_run
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports.mode import ModeReport


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
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"\n[bold red]{benchmark_name} error:[/] {exc}")
        sys.exit(1)

    if not result_holder:
        console.print(f"[bold red]No {benchmark_name} results.[/]")
        return None
    return result_holder[0]


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
    finalized = finalize_mode_run(
        ModeRun(
            run_type=mode,
            config=config,
            scores={
                "final_score": round(result.score),
                "accuracy": result.score,
                "rating": result.rating,
                **result.details,
            },
            status="completed",
        ),
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
    console.print(f"\n  [dim]Report saved to {finalized.report_path}[/]\n")
    return finalized.run_id
