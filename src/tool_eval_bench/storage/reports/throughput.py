"""Standalone throughput report for runs that skip the tool-call scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports._common import (
    _render_run_context,
    markdown_label,
    report_filename,
)


def write_throughput_report(
    root: Path,
    run_id: str,
    model: str,
    throughput_samples: list[Any],
    *,
    run_context: RunContext | None = None,
) -> Path:
    """Write a standalone Markdown report for throughput-only runs."""
    now = datetime.now(timezone.utc)
    folder = root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    label = run_context.label if run_context is not None else None
    path = folder / report_filename(run_id, label)

    # Version stamp
    version_str = ""
    if run_context:
        version_str = f"v{run_context.tool_version}"
        if run_context.git_sha:
            version_str += f" {run_context.git_sha}"

    md = [
        f"# Throughput Benchmark — {model}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        "- **Mode**: throughput-only",
    ]
    if version_str:
        md.append(f"- **tool-eval-bench**: `{version_str}`")
    if label:
        md.append(f"- **Label**: {markdown_label(label)}")
    md.append("")

    # Run Context section
    if run_context:
        md.extend(_render_run_context(run_context))

    ok_samples = [s for s in throughput_samples if not getattr(s, "error", None)]
    if ok_samples:
        md.extend(["## Results", ""])
        md.append("| Test | pp t/s | tg t/s | TTFT (ms) | Total (ms) | Tokens |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for s in ok_samples:
            conc_label = f" c{s.concurrency}" if s.concurrency > 1 else ""
            label = f"pp{s.label_pp} tg{s.tg_tokens} @ d{s.label_depth}{conc_label}"
            md.append(
                f"| {label} | {s.pp_tps:,.0f} | {s.tg_tps:,.1f} "
                f"| {s.ttft_ms:,.0f} | {s.total_ms:,.0f} "
                f"| {s.pp_tokens}+{s.tg_tokens} |"
            )
    else:
        md.extend(["## Results", "", "No successful measurements recorded.", ""])

    err_samples = [s for s in throughput_samples if getattr(s, "error", None)]
    if err_samples:
        md.extend(["", "## Errors", ""])
        for s in err_samples:
            md.append(f"- `{s.error}`")
        md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    return path
