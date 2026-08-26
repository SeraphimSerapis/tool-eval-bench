"""Per-run scenario report.

The primary artifact: per-scenario verdicts, category scores, and full traces.
Scenarios from a held-out pack keep their status and points but withhold
titles, summaries, and traces.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import (
    Category,
    ModelScoreSummary,
    ScenarioReportMetadata,
    ScenarioStatus,
)
from tool_eval_bench.storage.reports._common import (
    _HELD_OUT_LABEL,
    _markdown_heading,
    _markdown_table_cell,
    _render_held_out_note,
    _render_run_context,
    _trace_block,
    report_filename,
)


def write_scenario_report(
    root: Path,
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
    """Write a Markdown report for a scenario-based benchmark run.

    Scenarios marked held-out in ``scenario_metadata`` have their titles,
    summaries, and traces withheld: the report is the artifact people
    publish, and publishing a held-out scenario destroys its value.
    """
    now = datetime.now(timezone.utc)
    folder = root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    label = run_context.label if run_context is not None else None
    path = folder / report_filename(run_id, label)

    status_emoji = {
        ScenarioStatus.PASS: "✅",
        ScenarioStatus.PARTIAL: "⚠️",
        ScenarioStatus.FAIL: "❌",
    }

    # Version stamp from RunContext or fallback.  Reports are the artifact
    # people compare across machines, so the code identity has to travel with
    # them — a score is meaningless without knowing which evaluators produced
    # it.  The version string is derived from git, so it identifies the commit
    # even when no RunContext (and therefore no explicit SHA) is available.
    version_str = ""
    if run_context:
        version_str = f" (v{run_context.tool_version}"
        if run_context.git_sha:
            version_str += f" {run_context.git_sha}"
        version_str += ")"
    else:
        from tool_eval_bench import __version__

        version_str = f" (v{__version__})"

    md = [
        f"# Tool-Call Benchmark — {model}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        f"- **tool-eval-bench**: `{version_str.strip(' ()')}`" if version_str else "",
        f"- **Final Score**: **{summary.final_score}** / 100",
        f"- **Total Points**: {summary.total_points} / {summary.max_points}",
        f"- **Rating**: {summary.rating}",
    ]
    if summary.weighted_score is not None:
        md.append(
            f"- **Weighted Score**: **{summary.weighted_score}** / 100 _(difficulty-weighted)_"
        )
    if summary.excluded_scenarios:
        excluded = ", ".join(f"`{sid}`" for sid in summary.excluded_scenarios)
        md.append(
            f"- **Completion Rate**: {summary.completion_rate}% — "
            f"{len(summary.excluded_scenarios)} scenario(s) excluded from scoring "
            f"due to infrastructure failures (timeout / connection / 5xx): {excluded}"
        )
    # Filter empty lines from conditional version stamp
    md = [line for line in md if line is not None and line != ""] + [""]

    # Tool definition token overhead estimate (PERF-03)
    # Check if any category-L (Toolset Scale) scenarios were included —
    # those use the large toolset instead of UNIVERSAL_TOOLS.
    scenario_metadata = scenario_metadata or {}
    has_large_toolset = any(
        scenario_metadata.get(r.scenario_id)
        and scenario_metadata[r.scenario_id].category == Category.L
        for r in summary.scenario_results
    )
    if has_large_toolset:
        import json as _json

        from tool_eval_bench.domain.tools_large import LARGE_TOOLSET

        tool_json_chars = len(_json.dumps(LARGE_TOOLSET))
        est_tokens = tool_json_chars // 4  # ~4 chars per token heuristic
        md.append(
            f"- **Tool Definition Overhead**: ~{est_tokens:,} tokens ({len(LARGE_TOOLSET)} tools, {tool_json_chars:,} chars)"
        )
    else:
        import json as _json

        from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS

        tool_json_chars = len(_json.dumps(UNIVERSAL_TOOLS))
        est_tokens = tool_json_chars // 4
        md.append(
            f"- **Tool Definition Overhead**: ~{est_tokens:,} tokens ({len(UNIVERSAL_TOOLS)} tools, {tool_json_chars:,} chars)"
        )

    # Deployability composite (only when latency data is present)
    if summary.deployability is not None:
        med_s = (summary.median_turn_ms or 0) / 1000
        md.extend(
            [
                f"- **Deployability**: **{summary.deployability}** / 100 (α={summary.alpha})",
                f"- **Quality**: {summary.final_score} / 100",
                f"- **Responsiveness**: {summary.responsiveness} / 100 (median turn: {med_s:.1f}s)",
            ]
        )

    md.append("")

    # Context pressure info
    if context_pressure_config:
        ratio = context_pressure_config.get("ratio", 0)
        fill_tokens = context_pressure_config.get("fill_tokens", 0)
        ctx_size = context_pressure_config.get("context_size", 0)
        pct = int(ratio * 100)
        md.insert(
            -1,
            f"- **Context Pressure**: {pct}% (~{fill_tokens:,} tokens prefilled of {ctx_size:,} context)",
        )

    # Safety warnings
    if summary.safety_warnings:
        md.extend(
            [
                "> [!WARNING]",
                f"> **{len(summary.safety_warnings)} safety-critical failure(s) detected:**",
            ]
        )
        for w in summary.safety_warnings:
            md.append(f"> - {_markdown_heading(w)}")
        md.append("")

    # Run Context section (issue #6)
    if run_context:
        md.extend(_render_run_context(run_context))

    md.extend(
        [
            "## Category Scores",
            "",
            "| Category | Earned | Max | Percent |",
            "|---|---|---|---|",
        ]
    )

    for cs in summary.category_scores:
        md.append(
            f"| {_markdown_table_cell(cs.label)} | {cs.earned} | {cs.max_points} | {cs.percent}% |"
        )

    md.extend(["", "## Scenario Results", ""])
    md.append("| ID | Title | Diff | Status | Points | Failure | Summary |")
    md.append("|---|---|:---:|---:|---:|---|---|")

    _diff_labels = {1: "★", 2: "★★", 3: "★★★", 4: "★★★★", 5: "★★★★★"}

    held_out_ids = {sid for sid, meta in scenario_metadata.items() if meta.held_out}

    for r in summary.scenario_results:
        emoji = status_emoji.get(r.status, "?")
        metadata = scenario_metadata.get(r.scenario_id)
        diff = metadata.difficulty if metadata else None
        diff_str = _diff_labels.get(diff, "?") if diff else "?"
        failure = r.failure_kind or "—"
        if r.scenario_id in held_out_ids:
            title = f"_{_HELD_OUT_LABEL}_"
            detail = f"_{_HELD_OUT_LABEL}_"
        else:
            title = metadata.title if metadata else r.scenario_id
            note = f" ({r.note})" if r.note else ""
            detail = f"{r.summary}{note}"
        md.append(
            f"| {_markdown_table_cell(r.scenario_id)} | {_markdown_table_cell(title)} | {diff_str} | "
            f"{emoji} {_markdown_table_cell(r.status.value)} | {r.points}/2 | "
            f"{_markdown_table_cell(failure)} | {_markdown_table_cell(detail)} |"
        )

    if held_out_ids:
        md.extend(_render_held_out_note(sorted(held_out_ids), scenario_packs))

    # Difficulty distribution summary
    from collections import Counter

    diff_pass: Counter[int] = Counter()
    diff_total: Counter[int] = Counter()
    for r in summary.scenario_results:
        metadata = scenario_metadata.get(r.scenario_id)
        d = metadata.difficulty if metadata else None
        if d:
            diff_total[d] += 1
            if r.status == ScenarioStatus.PASS:
                diff_pass[d] += 1
    if diff_total:
        _tier_names = {1: "Trivial", 2: "Easy", 3: "Moderate", 4: "Hard", 5: "Very Hard"}
        md.extend(["", "## Performance by Difficulty", ""])
        md.append("| Tier | Scenarios | Passed | Rate |")
        md.append("|---|:---:|:---:|:---:|")
        for d in sorted(diff_total):
            total = diff_total[d]
            passed = diff_pass[d]
            pct = round(passed / total * 100) if total else 0
            md.append(f"| {_tier_names.get(d, '?')} ({d}) | {total} | {passed} | {pct}% |")

    # Throughput section
    ok_samples = [s for s in (throughput_samples or []) if not getattr(s, "error", None)]
    if ok_samples:
        md.extend(["", "## Throughput Metrics", ""])
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

    diagnostic_results = [
        r for r in summary.scenario_results if r.parallel_tool_turns or r.state_checkpoints
    ]
    if diagnostic_results:
        md.extend(["", "## Hard Mode Diagnostics", ""])
        for r in diagnostic_results:
            details: list[str] = []
            if r.parallel_tool_turns:
                turns = ", ".join(str(turn) for turn in r.parallel_tool_turns)
                details.append(f"parallel tool turns: {turns}")
            details.extend(r.state_checkpoints)
            md.append(f"- **{r.scenario_id}**: {'; '.join(details)}")

    # Trace section
    md.extend(["", "## Traces", ""])
    for r in summary.scenario_results:
        md.append(f"### {_markdown_heading(r.scenario_id)}")
        md.append("")
        if r.scenario_id in held_out_ids:
            md.append(
                f"_{_HELD_OUT_LABEL} — the trace would disclose the scenario's "
                "prompt and expected tool calls._"
            )
            md.append("")
            continue
        md.extend(_trace_block(r.raw_log))
        md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    return path
