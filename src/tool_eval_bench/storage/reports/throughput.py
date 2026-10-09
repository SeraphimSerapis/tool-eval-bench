"""Standalone throughput report for runs that skip the tool-call scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports._common import append_benchy_throughput_rows
from tool_eval_bench.storage.reports.mode import ModeReport, ReportContext, write_mode_report


def write_throughput_report(
    root: Path,
    run_id: str,
    model: str,
    throughput_samples: list[Any],
    *,
    run_context: RunContext | None = None,
) -> Path:
    """Write a standalone Markdown report for throughput-only runs."""
    label = run_context.label if run_context is not None else None
    report = throughput_report(model, throughput_samples, label=label)
    return write_mode_report(root, run_id, report, run_context)


def throughput_report(
    model: str,
    throughput_samples: list[Any],
    *,
    label: str | None,
) -> ModeReport:
    """Build the throughput-only report content."""
    md: list[str] = []
    ok_samples = [s for s in throughput_samples if not getattr(s, "error", None)]
    if ok_samples:
        md.extend(["## Results", ""])
        md.append("| Test | pp t/s | tg t/s | TTFT (ms) | Total (ms) | Tokens |")
        md.append("|---|---:|---:|---:|---:|---:|")
        append_benchy_throughput_rows(md, ok_samples)
    else:
        md.extend(["## Results", "", "No successful measurements recorded.", ""])

    err_samples = [s for s in throughput_samples if getattr(s, "error", None)]
    if err_samples:
        md.extend(["", "## Errors", ""])
        for s in err_samples:
            md.append(f"- `{s.error}`")
        md.append("")

    return ModeReport(
        title="Throughput Benchmark",
        display_name=model,
        mode="throughput-only",
        label=label,
        version_line=True,
        context=ReportContext.FULL,
        header=(),
        body=tuple(md),
    )
