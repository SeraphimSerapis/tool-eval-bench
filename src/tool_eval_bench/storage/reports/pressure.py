"""Context-pressure sweep report.

One section per pressure level, with the full trace for every scenario run at it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tool_eval_bench.storage.reports._common import (
    _trace_block,
    markdown_label,
    report_filename,
)


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
) -> Path:
    """Write a trace-complete artifact for a context-pressure sweep."""
    now = datetime.now(timezone.utc)
    folder = root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / report_filename(run_id, label)
    label_line = [f"- **Label**: {markdown_label(label)}"] if label else []
    markdown = [
        f"# Context Pressure Sweep — {model}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        "- **Mode**: context-pressure-sweep",
        *label_line,
        f"- **Backend**: {backend}",
        f"- **Server**: {display_url}",
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
        "",
    ]
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
    path.write_text("\n".join(markdown), encoding="utf-8")
    return path
