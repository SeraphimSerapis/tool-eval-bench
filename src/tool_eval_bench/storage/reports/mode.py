"""Shared report skeleton for the single-purpose benchmark modes.

Context-pressure sweeps, spec-bench, throughput-only runs and the external
plugins all write the same shape of artifact: a title, the run identity
bullets, mode-specific header bullets, the probed engine and host when a
`RunContext` exists, and a mode-specific body.  Each mode builds a `ModeReport`
and `write_mode_report` owns the parts they share, including where the file
lands.

These modes run with their own temperature, timeout and concurrency rather
than the CLI values in the context, so they render the engine table and not the
CLI parameter table.  A mode puts the parameters it really used in its header.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports._common import (
    _render_engine_context,
    markdown_label,
    report_filename,
    tool_version_line,
)


@dataclass(frozen=True)
class ModeReport:
    """Everything a mode contributes to its Markdown artifact.

    ``label`` is the run's ``--label``.  It names the report file, renders in
    the header, and is stored in the run's metadata.
    """

    title: str
    display_name: str
    mode: str
    label: str | None
    header: tuple[str, ...]
    body: tuple[str, ...]


def write_mode_report(
    root: Path,
    run_id: str,
    report: ModeReport,
    run_context: RunContext | None,
) -> Path:
    """Write a mode report under ``root/YYYY/MM`` and return its path."""
    now = datetime.now(timezone.utc)
    folder = root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / report_filename(run_id, report.label)
    lines = [
        f"# {report.title} — {report.display_name}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        f"- **Mode**: {report.mode}",
    ]
    if run_context is not None:
        lines.append(tool_version_line(run_context))
    if report.label:
        lines.append(f"- **Label**: {markdown_label(report.label)}")
    lines.extend(report.header)
    lines.append("")
    if run_context is not None:
        lines.extend(_render_engine_context(run_context))
    lines.extend(report.body)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
