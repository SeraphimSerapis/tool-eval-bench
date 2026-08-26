"""Cross-trial summary report.

Synthesises N trial reports into one document with reliability metrics,
per-scenario variance, and failure analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import (
    ModelScoreSummary,
    ScenarioStatus,
)
from tool_eval_bench.storage.reports._common import (
    _render_run_context,
    report_filename,
)


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
) -> Path:
    """Write a consolidated cross-trial summary report.

    This synthesizes N individual trial reports into a single document with
    reliability metrics, per-scenario variance, and failure analysis.
    """
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
    md.append("| **Rating** | " + " | ".join(ratings) + f" | {ratings[0]} |")
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

        md.append(f"| {sid} | " + " | ".join(row_cells) + f" | {pass_k} | {pass_hat_k} |")

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
            md.append(f"| **{sid}** | {summary} |")
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
            md.append(f"| **{sid}** | {results_str} |")
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
            md.append(f"| {sid} | {summary} |")
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
        for s in ok_samples:
            conc_label = f" c{s.concurrency}" if s.concurrency > 1 else ""
            label = f"pp{s.label_pp} tg{s.tg_tokens} @ d{s.label_depth}{conc_label}"
            md.append(
                f"| {label} | {s.pp_tps:,.0f} | {s.tg_tps:,.1f} "
                f"| {s.ttft_ms:,.0f} | {s.total_ms:,.0f} "
                f"| {s.pp_tokens}+{s.tg_tokens} |"
            )
        md.append("")

    # ── Links to individual trial reports ──
    if report_paths:
        md.extend(["## Individual Trial Reports", ""])
        for i, rp in enumerate(report_paths):
            md.append(f"- Trial {i + 1}: `{rp}`")
        md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    return path
