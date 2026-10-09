"""Cross-trial summary report.

Synthesises N trial reports into one document with reliability metrics,
per-scenario variance, and failure analysis.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import (
    ModelScoreSummary,
    ScenarioReportMetadata,
    ScenarioStatus,
)
from tool_eval_bench.storage.reports._common import (
    HELD_OUT_CELL,
    _markdown_table_cell,
    _render_held_out_note,
    _render_run_context,
    append_benchy_throughput_rows,
    held_out_ids,
    report_filename,
)


def _relative_report_path(report_path: str, folder: Path) -> str:
    """Point at a trial report relative to the summary, never by absolute path.

    The default reports root is absolute, and the summary is meant to be
    shared, so an absolute path would publish the user's home directory.
    Trials can straddle a month boundary, so a bare file name is not enough.
    """
    try:
        return Path(os.path.relpath(report_path, folder)).as_posix()
    except ValueError:  # different drives on Windows
        return Path(report_path).name


def write_summary_report(
    root: Path,
    run_id: str,
    model: str,
    summaries: list[ModelScoreSummary],
    agg: dict,
    *,
    throughput_samples: list[Any] | None = None,
    report_paths: list[str] | None = None,
    run_context: RunContext | None = None,
    scenario_metadata: Mapping[str, ScenarioReportMetadata] | None = None,
    scenario_packs: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a consolidated cross-trial summary report.

    This synthesizes N individual trial reports into a single document with
    reliability metrics, per-scenario variance, and failure analysis.
    Scenarios marked held-out in ``scenario_metadata`` keep their statuses but
    have their summaries withheld, as in the per-trial report, and the
    held-out note attests to ``scenario_packs``.
    """
    held_out = held_out_ids(scenario_metadata)
    now = datetime.now(timezone.utc)
    folder = root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    label = run_context.label if run_context is not None else None
    path = folder / report_filename(run_id, label, suffix="_summary")

    n = agg.get("trials", len(summaries))

    # Status emoji lookup
    status_emoji = {
        ScenarioStatus.PASS: "✅",
        ScenarioStatus.PARTIAL: "⚠️",
        ScenarioStatus.FAIL: "❌",
    }
    status_short = {
        ScenarioStatus.PASS: "pass",
        ScenarioStatus.PARTIAL: "partial",
        ScenarioStatus.FAIL: "fail",
    }

    # Version stamp
    version_line = ""
    if run_context:
        version_line = f"- **tool-eval-bench**: `v{run_context.tool_version}"
        if run_context.git_sha:
            version_line += f" {run_context.git_sha}"
        version_line += "`"

    md = [
        f"# Cross-Trial Summary — {model}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
    ]
    if version_line:
        md.append(version_line)
    md.extend(
        [
            f"- **Trials**: {n}",
            "",
        ]
    )

    # Run Context section (issue #6)
    if run_context:
        md.extend(_render_run_context(run_context))

    # ── Headline numbers ──
    md.extend(
        [
            "## Headline Scores",
            "",
            "| Metric | " + " | ".join(f"Trial {i + 1}" for i in range(n)) + " | Mean ± σ |",
            "|---|" + "".join(":---:|" for _ in range(n)) + ":---:|",
        ]
    )

    scores = [s.final_score for s in summaries]
    points = [s.total_points for s in summaries]
    ratings = [s.rating for s in summaries]

    md.append(
        "| **Final Score** | "
        + " | ".join(str(s) for s in scores)
        + f" | **{agg['final_score_mean']:.1f} ± {agg['final_score_stddev']:.1f}** |"
    )
    md.append(
        "| **Total Points** | "
        + " | ".join(f"{p}/{summaries[0].max_points}" for p in points)
        + f" | **{agg['total_points_mean']:.1f} ± {agg['total_points_stddev']:.1f}** |"
    )
    # Ratings do not average; name the shared one or say they differ.
    rating_agg = ratings[0] if len(set(ratings)) == 1 else "varies"
    md.append("| **Rating** | " + " | ".join(ratings) + f" | {rating_agg} |")
    num_warnings = [len(s.safety_warnings) for s in summaries]
    md.append("| **Safety Warnings** | " + " | ".join(str(w) for w in num_warnings) + " | — |")
    md.append("")

    # ── Reliability metrics ──
    pass_at = agg.get("pass_at_k", 0)
    pass_hat = agg.get("pass_hat_k", 0)
    gap = agg.get("reliability_gap", 0)
    ci_lo, ci_hi = agg.get("final_score_ci95", (0, 0))

    md.extend(
        [
            "## Reliability Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| **Pass@{n}** (capability ceiling) | {pass_at:.1f}% |",
            f"| **Pass^{n}** (reliability floor) | {pass_hat:.1f}% |",
            f"| **Reliability Gap** | {gap:.1f}pp |",
            f"| **95% CI** | [{ci_lo:.1f}, {ci_hi:.1f}] |",
            "",
        ]
    )

    if gap > 20:
        md.extend(
            [
                "> [!WARNING]",
                f"> **{gap:.0f}pp reliability gap is very high.** The model *can* solve "
                f"{pass_at:.0f}% of scenarios but only *reliably* solves {pass_hat:.0f}%.",
                "",
            ]
        )
    elif gap > 5:
        md.extend(
            [
                "> [!NOTE]",
                f"> **{gap:.0f}pp reliability gap** — moderate consistency variance across trials.",
                "",
            ]
        )

    # ── Per-scenario cross-trial table ──
    scenario_ids = [r.scenario_id for r in summaries[0].scenario_results]
    per_scenario = agg.get("per_scenario", {})

    md.extend(
        [
            "## Per-Scenario Results",
            "",
            "| Scenario | " + " | ".join(f"T{i + 1}" for i in range(n)) + " | Pass@k | Pass^k |",
            "|---|" + "".join(":---:|" for _ in range(n)) + ":---:|:---:|",
        ]
    )

    never_pass = []
    flaky = []
    consistent_partial = []

    for sid in scenario_ids:
        row_cells = []
        statuses = []
        for s in summaries:
            r = next((r for r in s.scenario_results if r.scenario_id == sid), None)
            if r:
                emoji = status_emoji.get(r.status, "?")
                row_cells.append(emoji)
                statuses.append(r.status)
            else:
                row_cells.append("—")

        stats = per_scenario.get(sid, {})
        pass_k = "✓" if stats.get("pass_at_k") else "✗"
        pass_hat_k = "✓" if stats.get("pass_hat_k") else "**✗**"

        md.append(
            f"| {_markdown_table_cell(sid)} | "
            + " | ".join(row_cells)
            + f" | {pass_k} | {pass_hat_k} |"
        )

        # Classify scenarios
        if all(st == ScenarioStatus.FAIL for st in statuses):
            summary_t1 = next(
                (r.summary for r in summaries[0].scenario_results if r.scenario_id == sid), ""
            )
            never_pass.append((sid, summary_t1))
        elif any(st == ScenarioStatus.FAIL for st in statuses) and any(
            st != ScenarioStatus.FAIL for st in statuses
        ):
            flaky.append((sid, [status_short.get(st, "?") for st in statuses]))
        elif all(st == ScenarioStatus.PARTIAL for st in statuses):
            summary_t1 = next(
                (r.summary for r in summaries[0].scenario_results if r.scenario_id == sid), ""
            )
            consistent_partial.append((sid, summary_t1))

    md.append("")

    # ── Category variance ──
    cat_stats = agg.get("per_category", {})
    if cat_stats:
        md.extend(
            [
                "## Category Variance",
                "",
                "| Category | " + " | ".join(f"T{i + 1}" for i in range(n)) + " | Variance |",
                "|---|" + "".join(":---:|" for _ in range(n)) + ":---|",
            ]
        )

        for cs in summaries[0].category_scores:
            cat_key = cs.category.value
            stats = cat_stats.get(cat_key, {})
            percents = []
            for s in summaries:
                c = next((c for c in s.category_scores if c.category == cs.category), None)
                percents.append(f"{c.percent:.0f}%" if c else "—")

            stddev = stats.get("stddev_percent", 0)
            if stddev > 15:
                variance = f"⚠️ **{stddev:.0f}pp swing**"
            elif stddev == 0:
                variance = "**Zero variance**"
            else:
                variance = f"{stddev:.1f}pp"

            md.append(
                f"| {stats.get('label', cat_key)} | " + " | ".join(percents) + f" | {variance} |"
            )

        md.append("")

    # ── Failure analysis ──
    if never_pass or flaky or consistent_partial:
        md.extend(["## Failure Analysis", ""])

    if never_pass:
        md.extend(
            [
                "### ❌ Never Passes (0/N trials)",
                "",
                "| Scenario | Issue |",
                "|---|---|",
            ]
        )
        for sid, summary in never_pass:
            md.append(
                f"| **{_markdown_table_cell(sid)}** | {_issue_cell(sid, summary, held_out)} |"
            )
        md.append("")

    if flaky:
        md.extend(
            [
                "### 🔀 Flaky (passes in some trials, fails in others)",
                "",
                "| Scenario | Results |",
                "|---|---|",
            ]
        )
        for sid, statuses_list in flaky:
            results_str = ", ".join(statuses_list)
            md.append(f"| **{_markdown_table_cell(sid)}** | {results_str} |")
        md.append("")

    if consistent_partial:
        md.extend(
            [
                "### ⚠️ Consistently Partial",
                "",
                "| Scenario | Issue |",
                "|---|---|",
            ]
        )
        for sid, summary in consistent_partial:
            md.append(f"| {_markdown_table_cell(sid)} | {_issue_cell(sid, summary, held_out)} |")
        md.append("")

    shown_held_out = sorted(held_out & set(scenario_ids))
    if shown_held_out:
        md.extend(_render_held_out_note(shown_held_out, scenario_packs))
        md.append("")

    # ── Deployability (from first summary with data) ──
    deploy_summary = next((s for s in summaries if s.deployability is not None), None)
    if deploy_summary:
        md.extend(
            [
                "## Deployability",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Quality | {deploy_summary.final_score} / 100 |",
                f"| Responsiveness | {deploy_summary.responsiveness} / 100 |",
                f"| Deployability | **{deploy_summary.deployability}** / 100 (α={deploy_summary.alpha}) |",
                f"| Median Turn | {(deploy_summary.median_turn_ms or 0) / 1000:.1f}s |",
                "",
            ]
        )

    # ── Throughput ──
    ok_samples = [s for s in (throughput_samples or []) if not getattr(s, "error", None)]
    if ok_samples:
        md.extend(["## Throughput Metrics", ""])
        md.append("| Test | pp t/s | tg t/s | TTFT (ms) | Total (ms) | Tokens |")
        md.append("|---|---:|---:|---:|---:|---:|")
        append_benchy_throughput_rows(md, ok_samples)
        md.append("")

    # ── Links to individual trial reports ──
    if report_paths:
        md.extend(["## Individual Trial Reports", ""])
        for i, rp in enumerate(report_paths):
            md.append(f"- Trial {i + 1}: `{_relative_report_path(rp, folder)}`")
        md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    return path


def _issue_cell(scenario_id: str, summary: str, held_out: set[str]) -> str:
    """Render an evaluator summary, which for a pack scenario can name the answer."""
    if scenario_id in held_out:
        return HELD_OUT_CELL
    return _markdown_table_cell(summary)
