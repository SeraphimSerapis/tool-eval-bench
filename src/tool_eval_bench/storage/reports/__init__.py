"""Markdown report writers for benchmark runs.

`MarkdownReporter` is the public entry point and stays the only thing callers
need.  The five report types have nothing in common beyond an output directory,
so each lives in its own module and the reporter delegates to it.

Every completed run must produce a Markdown artifact under `runs/YYYY/MM/`, so
these writers are part of the run contract rather than a convenience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import ModelScoreSummary, ScenarioReportMetadata
from tool_eval_bench.storage.reports._common import (
    _default_reports_root,
    _render_run_context,
    markdown_label,
    report_filename,
    safe_label_text,
    slugify_label,
)
from tool_eval_bench.storage.reports.pressure import write_pressure_sweep_report
from tool_eval_bench.storage.reports.scenario import write_scenario_report
from tool_eval_bench.storage.reports.spec_decode import write_spec_decode_report
from tool_eval_bench.storage.reports.summary import write_summary_report
from tool_eval_bench.storage.reports.throughput import write_throughput_report

__all__ = [
    "MarkdownReporter",
    "_default_reports_root",
    "_render_run_context",
    "markdown_label",
    "report_filename",
    "safe_label_text",
    "slugify_label",
]


class MarkdownReporter:
    """Writes run artifacts under a reports root.

    A thin facade over the per-report modules.  It owns only the output
    directory; each writer owns its own layout.
    """

    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or _default_reports_root())

    def write_pressure_sweep_report(
        self,
        *,
        run_id: str,
        model: str,
        backend: str,
        display_url: str,
        context_size: int,
        level_results: list[dict[str, Any]],
        breaking_point: float | None,
        first_degradation: float | None,
        label: str | None = None,
    ) -> Path:
        """Write a trace-complete artifact for a context-pressure sweep."""
        return write_pressure_sweep_report(
            self.root,
            run_id=run_id,
            model=model,
            backend=backend,
            display_url=display_url,
            context_size=context_size,
            level_results=level_results,
            breaking_point=breaking_point,
            first_degradation=first_degradation,
            label=label,
        )

    def write_scenario_report(
        self,
        run_id: str,
        model: str,
        summary: ModelScoreSummary,
        *,
        throughput_samples: list[Any] | None = None,
        context_pressure_config: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
        scenario_metadata: Mapping[str, ScenarioReportMetadata] | None = None,
        scenario_packs: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Write the Markdown report for a scenario-based benchmark run."""
        return write_scenario_report(
            self.root,
            run_id,
            model,
            summary,
            throughput_samples=throughput_samples,
            context_pressure_config=context_pressure_config,
            run_context=run_context,
            scenario_metadata=scenario_metadata,
            scenario_packs=scenario_packs,
        )

    def write_throughput_report(
        self,
        run_id: str,
        model: str,
        throughput_samples: list[Any],
        *,
        run_context: RunContext | None = None,
    ) -> Path:
        """Write a standalone Markdown report for throughput-only runs."""
        return write_throughput_report(
            self.root, run_id, model, throughput_samples, run_context=run_context
        )

    def write_summary_report(
        self,
        run_id: str,
        model: str,
        summaries: list[ModelScoreSummary],
        agg: dict,
        *,
        throughput_samples: list[Any] | None = None,
        report_paths: list[str] | None = None,
        run_context: RunContext | None = None,
    ) -> Path:
        """Write a consolidated cross-trial summary report."""
        return write_summary_report(
            self.root,
            run_id,
            model,
            summaries,
            agg,
            throughput_samples=throughput_samples,
            report_paths=report_paths,
            run_context=run_context,
        )

    def write_spec_decode_report(
        self,
        run_id: str,
        model: str,
        spec_samples: list[Any],
        label: str | None = None,
    ) -> Path:
        """Write a Markdown report for speculative decoding benchmark results."""
        return write_spec_decode_report(self.root, run_id, model, spec_samples, label)
