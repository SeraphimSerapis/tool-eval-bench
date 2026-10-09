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
    planned_levels: int | None = None,
    interrupted: bool = False,
    stop_reason: str | None = None,
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
        planned_levels=planned_levels,
        interrupted=interrupted,
        stop_reason=stop_reason,
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
    planned_levels: int | None = None,
    interrupted: bool = False,
    stop_reason: str | None = None,
) -> ModeReport:
    """Build the context-pressure sweep report content.

    *server* is redacted here, so no caller can put credentials in the report.
    """
    if interrupted:
        planned = planned_levels if planned_levels is not None else "?"
        breaking_line = (
            f"- **Breaking Point**: withheld (interrupted after {len(level_results)} "
            f"of {planned} levels)"
        )
    elif breaking_point is not None:
        breaking_line = f"- **Breaking Point**: {breaking_point:.0%}"
    elif level_results and all(level["score_pct"] is None for level in level_results):
        breaking_line = "- **Breaking Point**: n/a (no level was scored)"
    else:
        breaking_line = "- **Breaking Point**: none"
    header = [
        f"- **Backend**: {backend}",
        f"- **Server**: {redact_url(server)}",
        f"- **Context Window**: {context_size:,} tokens",
        f"- **Executed Levels**: {len(level_results)}",
        breaking_line,
        (
            f"- **First Degradation**: {first_degradation:.0%}"
            if first_degradation is not None
            else "- **First Degradation**: none"
        ),
    ]
    if stop_reason is not None:
        header.append(f"- **Stopped Early**: {stop_reason}")
    markdown: list[str] = []
    for index, level in enumerate(level_results, start=1):
        fill_line = f"- **Fill Tokens**: {level['fill_tokens']:,}"
        if level.get("fill_tokens_estimated"):
            fill_line += " (target; not measured, the server has no compatible /tokenize)"
        score = level["score_pct"]
        excluded = set(level.get("excluded_scenarios") or ())
        markdown.extend(
            [
                f"## Level {index} — {level['ratio']:.0%}",
                "",
                fill_line,
                (
                    f"- **Pass Rate**: {score:.1f}%"
                    if score is not None
                    else "- **Pass Rate**: n/a (no scenario was scored)"
                ),
            ]
        )
        if level.get("excluded_count"):
            markdown.append(
                f"- **Excluded**: {level['excluded_count']} "
                "(infrastructure failures, not counted in the pass rate)"
            )
        if level.get("error"):
            markdown.append(f"- **Level Error**: {level['error']}")
        markdown.append("")
        for scenario in level["scenario_results"]:
            status = str(scenario["status"])
            if scenario["scenario_id"] in excluded:
                status += " (excluded from scoring)"
            markdown.extend(
                [
                    f"### {scenario['scenario_id']}",
                    "",
                    f"- **Status**: {status}",
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
