"""Standalone throughput report for runs that skip the tool-call scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.redaction import redact_url
from tool_eval_bench.storage.reports._common import append_benchy_throughput_rows
from tool_eval_bench.storage.reports.mode import ModeReport, write_mode_report


def write_throughput_report(
    root: Path,
    run_id: str,
    model: str,
    throughput_samples: list[Any],
    *,
    run_context: RunContext | None = None,
) -> Path:
    """Write a standalone Markdown report for throughput-only runs.

    Without a dispatch target, the header facts come from ``run_context``.
    """
    ctx = run_context
    report = throughput_report(
        model,
        throughput_samples,
        label=ctx.label if ctx is not None else None,
        backend=ctx.backend if ctx is not None else None,
        server=ctx.base_url if ctx is not None else None,
        served_model=ctx.model if ctx is not None else None,
        model_root=ctx.server_model_root if ctx is not None else None,
    )
    return write_mode_report(root, run_id, report, run_context)


def throughput_report(
    model: str,
    throughput_samples: list[Any],
    *,
    label: str | None,
    backend: str | None,
    server: str | None,
    served_model: str | None,
    model_root: str | None,
) -> ModeReport:
    """Build the throughput-only report content.

    ``model`` is the display name in the title; ``served_model`` is the model
    ID sent to the API.  A ``None`` fact is left out of the header.  ``server``
    is redacted here, so no caller can put credentials in the report.
    """
    header: list[str] = []
    if backend is not None:
        header.append(f"- **Backend**: {backend}")
    if server is not None:
        header.append(f"- **Server**: {redact_url(server)}")
    if served_model is not None:
        header.append(f"- **Model (API)**: `{served_model}`")
    if model_root and model_root != served_model:
        header.append(f"- **Model (Root)**: `{model_root}`")
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
        header=tuple(header),
        body=tuple(md),
    )
