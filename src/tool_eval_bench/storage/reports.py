"""Markdown report writer for agentic tool-call benchmark runs.

Generates human-readable reports with scenario results, category scores,
throughput metrics, and full traces for inspection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import ModelScoreSummary, ScenarioStatus


def _default_reports_root() -> str:
    """Resolve default reports root relative to the project root.

    Same pattern as db.py — ensures reports go to the same location
    regardless of the working directory when invoking the CLI.
    """
    # Walk up from storage/reports.py → storage → tool_eval_bench → src → project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return str(project_root / "runs")


class MarkdownReporter:
    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or _default_reports_root())

    def write_scenario_report(
        self,
        run_id: str,
        model: str,
        summary: ModelScoreSummary,
        *,
        throughput_samples: list[Any] | None = None,
        context_pressure_config: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
    ) -> Path:
        """Write a Markdown report for a scenario-based benchmark run."""
        now = datetime.now(timezone.utc)
        folder = self.root / f"{now.year:04d}" / f"{now.month:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{run_id}.md"

        status_emoji = {
            ScenarioStatus.PASS: "✅",
            ScenarioStatus.PARTIAL: "⚠️",
            ScenarioStatus.FAIL: "❌",
        }

        # Version stamp from RunContext or fallback
        version_str = ""
        if run_context:
            version_str = f" (v{run_context.tool_version}"
            if run_context.git_sha:
                version_str += f" {run_context.git_sha}"
            version_str += ")"
        elif not run_context:
            try:
                from tool_eval_bench import __version__
                version_str = f" (v{__version__})"
            except ImportError:
                pass

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
        # Filter empty lines from conditional version stamp
        md = [line for line in md if line is not None and line != ""] + [""]

        # Tool definition token overhead estimate (PERF-03)
        # Use the scenario IDs to determine which toolset was used
        has_large_toolset = any(
            r.scenario_id.startswith("TC-3") or r.scenario_id.startswith("TC-4")
            for r in summary.scenario_results
            if int(r.scenario_id.split("-")[1]) >= 37
        )
        if has_large_toolset:
            import json as _json
            from tool_eval_bench.domain.tools_large import LARGE_TOOLSET
            tool_json_chars = len(_json.dumps(LARGE_TOOLSET))
            est_tokens = tool_json_chars // 4  # ~4 chars per token heuristic
            md.append(f"- **Tool Definition Overhead**: ~{est_tokens:,} tokens ({len(LARGE_TOOLSET)} tools, {tool_json_chars:,} chars)")
        else:
            import json as _json
            from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS
            tool_json_chars = len(_json.dumps(UNIVERSAL_TOOLS))
            est_tokens = tool_json_chars // 4
            md.append(f"- **Tool Definition Overhead**: ~{est_tokens:,} tokens ({len(UNIVERSAL_TOOLS)} tools, {tool_json_chars:,} chars)")

        # Deployability composite (only when latency data is present)
        if summary.deployability is not None:
            med_s = (summary.median_turn_ms or 0) / 1000
            md.extend([
                f"- **Deployability**: **{summary.deployability}** / 100 (α={summary.alpha})",
                f"- **Quality**: {summary.final_score} / 100",
                f"- **Responsiveness**: {summary.responsiveness} / 100 (median turn: {med_s:.1f}s)",
            ])

        md.append("")

        # Context pressure info
        if context_pressure_config:
            ratio = context_pressure_config.get("ratio", 0)
            fill_tokens = context_pressure_config.get("fill_tokens", 0)
            ctx_size = context_pressure_config.get("context_size", 0)
            pct = int(ratio * 100)
            md.insert(-1, f"- **Context Pressure**: {pct}% (~{fill_tokens:,} tokens prefilled of {ctx_size:,} context)")

        # Safety warnings
        if summary.safety_warnings:
            md.extend([
                "> [!WARNING]",
                f"> **{len(summary.safety_warnings)} safety-critical failure(s) detected:**",
            ])
            for w in summary.safety_warnings:
                md.append(f"> - {w}")
            md.append("")

        # Run Context section (issue #6)
        if run_context:
            md.extend(_render_run_context(run_context))

        md.extend([
            "## Category Scores",
            "",
            "| Category | Earned | Max | Percent |",
            "|---|---|---|---|",
        ])

        for cs in summary.category_scores:
            md.append(f"| {cs.label} | {cs.earned} | {cs.max_points} | {cs.percent}% |")

        md.extend(["", "## Scenario Results", ""])
        md.append("| ID | Title | Status | Points | Summary |")
        md.append("|---|---|---|---|---|")

        for r in summary.scenario_results:
            emoji = status_emoji.get(r.status, "?")
            note = f" ({r.note})" if r.note else ""
            md.append(f"| {r.scenario_id} | {r.summary.split('.')[0]} | {emoji} {r.status.value} | {r.points}/2 | {r.summary}{note} |")

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

        # Trace section
        md.extend(["", "## Traces", ""])
        for r in summary.scenario_results:
            md.append(f"### {r.scenario_id}")
            md.append("")
            md.append("```text")
            md.append(r.raw_log)
            md.append("```")
            md.append("")

        path.write_text("\n".join(md), encoding="utf-8")
        return path

    def write_throughput_report(
        self,
        run_id: str,
        model: str,
        throughput_samples: list[Any],
        *,
        run_context: RunContext | None = None,
    ) -> Path:
        """Write a standalone Markdown report for throughput-only runs."""
        now = datetime.now(timezone.utc)
        folder = self.root / f"{now.year:04d}" / f"{now.month:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{run_id}.md"

        # Version stamp
        version_str = ""
        if run_context:
            version_str = f"v{run_context.tool_version}"
            if run_context.git_sha:
                version_str += f" {run_context.git_sha}"

        md = [
            f"# Throughput Benchmark — {model}",
            "",
            f"- **Run ID**: `{run_id}`",
            f"- **Date**: `{now.isoformat()}`",
            "- **Mode**: throughput-only",
        ]
        if version_str:
            md.append(f"- **tool-eval-bench**: `{version_str}`")
        md.append("")

        # Run Context section
        if run_context:
            md.extend(_render_run_context(run_context))

        ok_samples = [s for s in throughput_samples if not getattr(s, "error", None)]
        if ok_samples:
            md.extend(["## Results", ""])
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
        else:
            md.extend(["## Results", "", "No successful measurements recorded.", ""])

        err_samples = [s for s in throughput_samples if getattr(s, "error", None)]
        if err_samples:
            md.extend(["", "## Errors", ""])
            for s in err_samples:
                md.append(f"- `{s.error}`")
            md.append("")

        path.write_text("\n".join(md), encoding="utf-8")
        return path

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
        """Write a consolidated cross-trial summary report.

        This synthesizes N individual trial reports into a single document with
        reliability metrics, per-scenario variance, and failure analysis.
        """
        now = datetime.now(timezone.utc)
        folder = self.root / f"{now.year:04d}" / f"{now.month:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{run_id}_summary.md"

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
        md.extend([
            f"- **Trials**: {n}",
            "",
        ])

        # Run Context section (issue #6)
        if run_context:
            md.extend(_render_run_context(run_context))

        # ── Headline numbers ──
        md.extend([
            "## Headline Scores",
            "",
            "| Metric | " + " | ".join(f"Trial {i+1}" for i in range(n)) + " | Mean ± σ |",
            "|---|" + "".join(":---:|" for _ in range(n)) + ":---:|",
        ])

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
        md.append(
            "| **Rating** | "
            + " | ".join(ratings)
            + f" | {ratings[0]} |"
        )
        num_warnings = [len(s.safety_warnings) for s in summaries]
        md.append(
            "| **Safety Warnings** | "
            + " | ".join(str(w) for w in num_warnings)
            + " | — |"
        )
        md.append("")

        # ── Reliability metrics ──
        pass_at = agg.get("pass_at_k", 0)
        pass_hat = agg.get("pass_hat_k", 0)
        gap = agg.get("reliability_gap", 0)
        ci_lo, ci_hi = agg.get("final_score_ci95", (0, 0))

        md.extend([
            "## Reliability Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| **Pass@{n}** (capability ceiling) | {pass_at:.1f}% |",
            f"| **Pass^{n}** (reliability floor) | {pass_hat:.1f}% |",
            f"| **Reliability Gap** | {gap:.1f}pp |",
            f"| **95% CI** | [{ci_lo:.1f}, {ci_hi:.1f}] |",
            "",
        ])

        if gap > 20:
            md.extend([
                "> [!WARNING]",
                f"> **{gap:.0f}pp reliability gap is very high.** The model *can* solve "
                f"{pass_at:.0f}% of scenarios but only *reliably* solves {pass_hat:.0f}%.",
                "",
            ])
        elif gap > 5:
            md.extend([
                "> [!NOTE]",
                f"> **{gap:.0f}pp reliability gap** — moderate consistency variance across trials.",
                "",
            ])

        # ── Per-scenario cross-trial table ──
        scenario_ids = [r.scenario_id for r in summaries[0].scenario_results]
        per_scenario = agg.get("per_scenario", {})

        md.extend([
            "## Per-Scenario Results",
            "",
            "| Scenario | " + " | ".join(f"T{i+1}" for i in range(n)) + " | Pass@k | Pass^k |",
            "|---|" + "".join(":---:|" for _ in range(n)) + ":---:|:---:|",
        ])

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
                summary_t1 = next((r.summary for r in summaries[0].scenario_results if r.scenario_id == sid), "")
                never_pass.append((sid, summary_t1))
            elif any(st == ScenarioStatus.FAIL for st in statuses) and any(st != ScenarioStatus.FAIL for st in statuses):
                flaky.append((sid, [status_short.get(st, "?") for st in statuses]))
            elif all(st == ScenarioStatus.PARTIAL for st in statuses):
                summary_t1 = next((r.summary for r in summaries[0].scenario_results if r.scenario_id == sid), "")
                consistent_partial.append((sid, summary_t1))

        md.append("")

        # ── Category variance ──
        cat_stats = agg.get("per_category", {})
        if cat_stats:
            md.extend([
                "## Category Variance",
                "",
                "| Category | " + " | ".join(f"T{i+1}" for i in range(n)) + " | Variance |",
                "|---|" + "".join(":---:|" for _ in range(n)) + ":---|",
            ])

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

                md.append(f"| {stats.get('label', cat_key)} | " + " | ".join(percents) + f" | {variance} |")

            md.append("")

        # ── Failure analysis ──
        if never_pass or flaky or consistent_partial:
            md.extend(["## Failure Analysis", ""])

        if never_pass:
            md.extend([
                "### ❌ Never Passes (0/N trials)",
                "",
                "| Scenario | Issue |",
                "|---|---|",
            ])
            for sid, summary in never_pass:
                md.append(f"| **{sid}** | {summary} |")
            md.append("")

        if flaky:
            md.extend([
                "### 🔀 Flaky (passes in some trials, fails in others)",
                "",
                "| Scenario | Results |",
                "|---|---|",
            ])
            for sid, statuses_list in flaky:
                results_str = ", ".join(statuses_list)
                md.append(f"| **{sid}** | {results_str} |")
            md.append("")

        if consistent_partial:
            md.extend([
                "### ⚠️ Consistently Partial",
                "",
                "| Scenario | Issue |",
                "|---|---|",
            ])
            for sid, summary in consistent_partial:
                md.append(f"| {sid} | {summary} |")
            md.append("")

        # ── Deployability (from first summary with data) ──
        deploy_summary = next((s for s in summaries if s.deployability is not None), None)
        if deploy_summary:
            md.extend([
                "## Deployability",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Quality | {deploy_summary.final_score} / 100 |",
                f"| Responsiveness | {deploy_summary.responsiveness} / 100 |",
                f"| Deployability | **{deploy_summary.deployability}** / 100 (α={deploy_summary.alpha}) |",
                f"| Median Turn | {(deploy_summary.median_turn_ms or 0) / 1000:.1f}s |",
                "",
            ])

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
                md.append(f"- Trial {i+1}: `{rp}`")
            md.append("")

        path.write_text("\n".join(md), encoding="utf-8")
        return path

    def write_spec_decode_report(
        self,
        run_id: str,
        model: str,
        spec_samples: list[Any],
    ) -> Path:
        """Write a Markdown report for speculative decoding benchmark results."""
        now = datetime.now(timezone.utc)
        folder = self.root / f"{now.year:04d}" / f"{now.month:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{run_id}.md"

        md = [
            f"# Speculative Decoding Benchmark — {model}",
            "",
            f"- **Run ID**: `{run_id}`",
            f"- **Date**: `{now.isoformat()}`",
            "- **Mode**: spec-bench",
        ]

        # Detect method
        methods = {s.spec_method for s in spec_samples if hasattr(s, "spec_method")}
        method_str = ", ".join(sorted(methods)) if methods else "unknown"
        md.append(f"- **Spec Method**: {method_str}")

        # Check if acceptance rate is available
        has_ar = any(
            getattr(s, "acceptance_rate", None) is not None for s in spec_samples
        )
        if not has_ar:
            md.extend([
                "",
                "> [!NOTE]",
                "> Acceptance rate metrics were not available from the server.",
                "> Effective t/s (wall-clock based) still captures the real benefit",
                "> of MTP / speculative decoding.",
            ])

        md.append("")

        # Results table
        md.extend(["## Results", ""])

        if has_ar:
            md.append(
                "| Prompt | Depth | Eff t/s | Stream t/s | α (accept) | τ (length) "
                "| Speedup | TTFT (ms) | Total (ms) | Tokens |"
            )
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        else:
            md.append(
                "| Prompt | Depth | Eff t/s | Stream t/s "
                "| Speedup | TTFT (ms) | Total (ms) | Tokens |"
            )
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        for s in spec_samples:
            eff = getattr(s, "effective_tg_tps", 0)
            tg = getattr(s, "tg_tps", 0)
            ar = getattr(s, "acceptance_rate", None)
            al = getattr(s, "acceptance_length", None)
            sp = getattr(s, "speedup_ratio", None)
            ttft = getattr(s, "ttft_ms", 0)
            total = getattr(s, "total_ms", 0)
            tg_tok = getattr(s, "tg_tokens", 0)
            prompt_type = getattr(s, "prompt_type", "?")
            depth = getattr(s, "depth", 0)

            sp_str = f"{sp:.2f}x" if sp is not None else "—"

            if has_ar:
                ar_str = f"{ar * 100:.1f}%" if ar is not None else "—"
                al_str = f"{al:.1f}" if al is not None else "—"
                md.append(
                    f"| {prompt_type} | {depth} | {eff:,.1f} | {tg:,.1f} "
                    f"| {ar_str} | {al_str} | {sp_str} "
                    f"| {ttft:,.0f} | {total:,.0f} | {tg_tok} |"
                )
            else:
                md.append(
                    f"| {prompt_type} | {depth} | {eff:,.1f} | {tg:,.1f} "
                    f"| {sp_str} "
                    f"| {ttft:,.0f} | {total:,.0f} | {tg_tok} |"
                )

        md.append("")

        # Acceptance rate bar chart (if available)
        if has_ar:
            md.extend(["## Acceptance Rate by Prompt Type", ""])
            md.append("```")
            for s in spec_samples:
                ar = getattr(s, "acceptance_rate", None)
                pt = getattr(s, "prompt_type", "?")
                depth = getattr(s, "depth", 0)
                if ar is not None:
                    bar_len = int(ar * 40)
                    bar = "█" * bar_len + "░" * (40 - bar_len)
                    md.append(f"  {pt:>10} d{depth:<5} {bar} {ar * 100:.1f}%")
                else:
                    md.append(f"  {pt:>10} d{depth:<5} {'·' * 40} n/a")
            md.append("```")
            md.append("")

        # Per-prompt-type summary
        prompt_groups: dict[str, list] = {}
        for s in spec_samples:
            pt = getattr(s, "prompt_type", "?")
            prompt_groups.setdefault(pt, []).append(s)

        if len(prompt_groups) > 1:
            md.extend(["## Per-Prompt-Type Summary", ""])
            md.append("| Prompt Type | Avg Eff t/s | Avg Stream t/s | Avg α |")
            md.append("|---|---:|---:|---:|")

            for pt, group in sorted(prompt_groups.items()):
                avg_eff = sum(s.effective_tg_tps for s in group) / len(group)
                avg_tg = sum(s.tg_tps for s in group) / len(group)
                ars = [s.acceptance_rate for s in group if s.acceptance_rate is not None]
                avg_ar = f"{sum(ars) / len(ars) * 100:.1f}%" if ars else "—"
                md.append(f"| {pt} | {avg_eff:,.1f} | {avg_tg:,.1f} | {avg_ar} |")

            md.append("")

        # Interpretation guide
        md.extend([
            "## Interpretation Guide",
            "",
            "- **Eff t/s** (Effective t/s): Output tokens ÷ wall-clock generation time. "
            "This is what users experience. Higher is better.",
            "- **Stream t/s**: Token generation rate measured from SSE stream timing. "
            "For standard decoding, this matches Eff t/s. For spec decode, Eff t/s "
            "is typically higher.",
            "- **α (accept)**: Acceptance rate — % of draft tokens accepted by the verifier. "
            "Higher means the draft model/MTP heads predict well for this workload.",
            "- **τ (length)**: Average acceptance length — tokens accepted per speculative step. "
            "Higher means more tokens generated per verification pass.",
            "- **Speedup**: Effective t/s ÷ baseline t/s. Values > 1.0x indicate spec decode "
            "is providing a benefit.",
            "",
            "> [!TIP]",
            "> Acceptance rates vary significantly by prompt type. Code and structured tasks",
            "> typically show higher acceptance rates than creative/open-ended generation",
            "> because future tokens are more predictable.",
            "",
        ])

        path.write_text("\n".join(md), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Run context rendering (shared by all report types)
# ---------------------------------------------------------------------------

def _render_run_context(ctx: RunContext) -> list[str]:
    """Render RunContext as Markdown tables for embedding in reports."""
    md: list[str] = []

    # -- Run Context table (Tier 2: CLI parameters) --
    md.extend([
        "## Run Context",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Backend | {ctx.backend} |",
        f"| Server | `{ctx.base_url}` |",
        f"| Model (API) | `{ctx.model}` |",
    ])
    if ctx.server_model_root and ctx.server_model_root != ctx.model:
        md.append(f"| Model (Root) | `{ctx.server_model_root}` |")
    md.extend([
        f"| Temperature | {ctx.temperature} |",
        f"| Seed | {ctx.seed if ctx.seed is not None else '—'} |",
        f"| Max Turns | {ctx.max_turns} |",
        f"| Timeout | {ctx.timeout_seconds}s |",
        f"| Scenarios | {ctx.scenario_selector} |",
        f"| Parallel | {ctx.parallel} {'(sequential)' if ctx.parallel <= 1 else ''} |",
        f"| Error Rate | {ctx.error_rate} |",
        f"| Thinking | {'enabled' if ctx.thinking_enabled else 'disabled'} |",
    ])
    if ctx.context_pressure is not None:
        md.append(f"| Context Pressure | {ctx.context_pressure:.0%} |")
    if ctx.extra_params:
        import json as _json
        md.append(f"| Extra Params | `{_json.dumps(ctx.extra_params)}` |")
    md.append("")

    # -- Inference Engine table (Tier 3: best-effort) --
    has_engine_info = any([
        ctx.engine_name, ctx.engine_version, ctx.max_model_len,
        ctx.quantization, ctx.gpu_count, ctx.spec_decoding,
    ])
    if has_engine_info:
        md.extend([
            "## Inference Engine",
            "",
            "| Property | Value |",
            "|---|---|",
        ])
        if ctx.engine_name:
            version_str = f" {ctx.engine_version}" if ctx.engine_version else ""
            md.append(f"| Engine | {ctx.engine_name}{version_str} |")
        if ctx.max_model_len:
            md.append(f"| Max Model Length | {ctx.max_model_len:,} |")
        if ctx.quantization:
            md.append(f"| Quantization | {ctx.quantization} |")
        if ctx.gpu_count:
            md.append(f"| GPU Count | {ctx.gpu_count} |")
        if ctx.spec_decoding:
            md.append(f"| Spec Decoding | {ctx.spec_decoding} |")
        md.extend([
            f"| Host | `{ctx.hostname}` |",
            f"| Platform | `{ctx.platform_info}` |",
            f"| Python | {ctx.python_version} |",
            "",
        ])
    else:
        # Minimal environment info even without engine probes
        md.extend([
            "## Environment",
            "",
            "| Property | Value |",
            "|---|---|",
            f"| Host | `{ctx.hostname}` |",
            f"| Platform | `{ctx.platform_info}` |",
            f"| Python | {ctx.python_version} |",
            "",
        ])

    return md
