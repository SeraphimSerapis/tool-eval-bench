"""Finalization for the single-purpose benchmark modes.

Context-pressure sweeps, spec-bench, throughput-only runs and the external
plugins each produce one report and one stored run.  `finalize_mode_run` gives
all of them the same stored shape: a redacted config carrying its fingerprint,
a run ID derived from it, `RunContext` metadata, and the report path.  The
report is written before the run is persisted, so a failed write never leaves
a completed run without its artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tool_eval_bench.application import run_queries
from tool_eval_bench.application.finalization import finalize_completed_run
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.storage.reports.mode import ModeReport, write_mode_report
from tool_eval_bench.utils.fingerprint import with_config_fingerprint
from tool_eval_bench.utils.ids import build_run_id


@dataclass(frozen=True)
class ModeRun:
    """The stored half of a mode run; the report half is a `ModeReport`."""

    run_type: str
    config: Mapping[str, Any]
    scores: dict[str, Any]
    status: Literal["completed", "failed"]


@dataclass(frozen=True)
class FinalizedRun:
    run_id: str
    report_path: Path


def finalize_mode_run(
    run: ModeRun,
    report: ModeReport,
    *,
    run_context: RunContext | None,
    output_dir: str | None,
) -> FinalizedRun:
    """Write the report, then persist the run, and return where it landed."""
    config = with_config_fingerprint(run.config)
    run_id = build_run_id(config)
    metadata = run_context.to_dict() if run_context is not None else {}
    if report.label:
        metadata["label"] = report.label
    run_data: dict[str, Any] = {
        "run_id": run_id,
        "run_type": run.run_type,
        "status": run.status,
        "config": config,
        "scores": run.scores,
        "metadata": metadata,
    }
    root = MarkdownReporter(root=output_dir).root
    finalize_completed_run(
        run_data,
        write_report=lambda: write_mode_report(root, run_id, report, run_context),
        # Looked up per call so a patched ``run_queries.persist_run`` is honoured.
        persist=run_queries.persist_run,
    )
    return FinalizedRun(run_id=run_id, report_path=Path(run_data["report_path"]))
