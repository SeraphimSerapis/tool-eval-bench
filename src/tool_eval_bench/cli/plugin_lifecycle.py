"""Shared execution and finalization lifecycle for external plugins."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from tool_eval_bench.application.finalization import finalize_completed_run


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
    *,
    mode: str,
    title: str,
    display_name: str,
    result: Any,
    config: dict[str, Any],
    report_metrics: list[str],
    report_lines: list[str],
    output_dir: str | None,
    run_context: Any | None,
    with_config_fingerprint: Callable[[dict[str, Any]], dict[str, Any]],
    persist_plugin_run: Callable[[dict[str, Any]], None],
    metadata_for_storage: Callable[[Any | None], dict[str, Any]],
) -> str:
    """Write and persist a completed plugin run through one invariant."""
    from tool_eval_bench.storage.reports import MarkdownReporter, markdown_label, report_filename
    from tool_eval_bench.utils.ids import build_run_id

    run_config = with_config_fingerprint(config)
    run_id = build_run_id(run_config)
    label = getattr(run_context, "label", None) if run_context is not None else None
    reporter = MarkdownReporter(root=output_dir)
    now = datetime.now(timezone.utc)
    folder = reporter.root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / report_filename(run_id, label)
    label_line = [f"- **Label**: {markdown_label(label)}"] if label else []
    markdown = [
        f"# {title} Benchmark — {display_name}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        f"- **Mode**: {mode}",
        *label_line,
        *report_metrics,
        f"- **Rating**: {result.rating}",
        "",
        *report_lines,
    ]
    run_data = {
        "run_id": run_id,
        "run_type": mode,
        "status": "completed",
        "config": run_config,
        "scores": {
            "final_score": round(result.score),
            "accuracy": result.score,
            "rating": result.rating,
            **result.details,
        },
        "metadata": metadata_for_storage(run_context),
    }

    def write_plugin_report() -> Path:
        path.write_text("\n".join(markdown), encoding="utf-8")
        return path

    finalize_completed_run(
        run_data,
        write_report=write_plugin_report,
        persist=persist_plugin_run,
    )
    return run_id
