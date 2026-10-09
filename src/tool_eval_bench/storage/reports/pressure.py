"""Context-pressure sweep report.

One section per pressure level, with the full trace for every scenario run at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.redaction import redact_url
from tool_eval_bench.storage.reports._common import _trace_block
from tool_eval_bench.storage.reports.mode import ModeReport, write_mode_report


def write_pressure_sweep_report(
    root: Path,
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
    run_context: RunContext | None = None,
) -> Path:
    """Write a trace-complete artifact for a context-pressure sweep.

    *display_url* is redacted before it is written, whatever the caller passes.
    """
    report = pressure_sweep_report(
        model=model,
        backend=backend,
        server=display_url,
        context_size=context_size,
        level_results=level_results,
        breaking_point=breaking_point,
        first_degradation=first_degradation,
        label=label,
    )
    return write_mode_report(root, run_id, report, run_context)


def pressure_sweep_report(
    *,
    model: str,
    backend: str,
    server: str,
    context_size: int,
    level_results: list[dict[str, Any]],
    breaking_point: float | None,
    first_degradation: float | None,
    label: str | None,
) -> ModeReport:
    """Build the context-pressure sweep report content.

    *server* is redacted here, so no caller can put credentials in the report.
    """
    header = [
        f"- **Backend**: {backend}",
        f"- **Server**: {redact_url(server)}",
        f"- **Context Window**: {context_size:,} tokens",
        f"- **Executed Levels**: {len(level_results)}",
        (
            f"- **Breaking Point**: {breaking_point:.0%}"
            if breaking_point is not None
            else "- **Breaking Point**: none"
        ),
        (
            f"- **First Degradation**: {first_degradation:.0%}"
            if first_degradation is not None
            else "- **First Degradation**: none"
        ),
    ]
    markdown: list[str] = []
    for index, level in enumerate(level_results, start=1):
        markdown.extend(
            [
                f"## Level {index} — {level['ratio']:.0%}",
                "",
                f"- **Fill Tokens**: {level['fill_tokens']:,}",
                f"- **Pass Rate**: {level['score_pct']:.1f}%",
            ]
        )
        if level.get("error"):
            markdown.append(f"- **Level Error**: {level['error']}")
        markdown.append("")
        for scenario in level["scenario_results"]:
            markdown.extend(
                [
                    f"### {scenario['scenario_id']}",
                    "",
                    f"- **Status**: {scenario['status']}",
                    f"- **Points**: {scenario['points']} / 2",
                    f"- **Summary**: {scenario.get('summary') or ''}",
                    f"- **Expected**: {scenario.get('expected_behavior') or ''}",
                    "- **Tool Calls**: "
                    + (", ".join(scenario.get("tool_calls_made") or []) or "none"),
                    "",
                    "#### Full trace",
                    "",
                    *_trace_block(scenario.get("raw_log", "")),
                    "",
                ]
            )
    return ModeReport(
        title="Context Pressure Sweep",
        display_name=model,
        mode="context-pressure-sweep",
        label=label,
        header=tuple(header),
        body=tuple(markdown),
    )
