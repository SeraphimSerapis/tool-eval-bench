"""Tests for MarkdownReporter — verifies report structure and content.

TEST-03 from critical review: the report generator produces complex Markdown
output that should be verified for structural correctness.
"""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import (
    Category,
    CategoryScore,
    ModelScoreSummary,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.storage.reports import MarkdownReporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_summary(
    *,
    score: int = 75,
    safety_warnings: list[str] | None = None,
    num_results: int = 3,
) -> ModelScoreSummary:
    """Build a ModelScoreSummary with predictable test data."""
    results = []
    for i in range(num_results):
        results.append(ScenarioResult(
            scenario_id=f"TC-{i + 1:02d}",
            status=[ScenarioStatus.PASS, ScenarioStatus.PARTIAL, ScenarioStatus.FAIL][i % 3],
            points=[2, 1, 0][i % 3],
            summary=f"Scenario {i + 1} summary.",
            raw_log=f"[Turn 1] User: test\\n[Turn 1] Model: response for TC-{i + 1:02d}",
        ))
    return ModelScoreSummary(
        scenario_results=results,
        category_scores=[
            CategoryScore(Category.A, "A Tool Selection", 4, 6, 66.7),
            CategoryScore(Category.B, "B Parameter Precision", 5, 6, 83.3),
        ],
        final_score=score,
        total_points=sum(r.points for r in results),
        max_points=num_results * 2,
        rating="★★★★ Good",
        safety_warnings=safety_warnings or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarkdownReporter:
    def test_basic_report_structure(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary()
        path = reporter.write_scenario_report("run_test_001", "test-model", summary)

        assert path.exists()
        content = path.read_text()

        # Title
        assert "# Tool-Call Benchmark — test-model" in content
        # Metadata
        assert "run_test_001" in content
        assert "75" in content  # final score
        assert "★★★★ Good" in content

    def test_report_includes_category_table(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary()
        path = reporter.write_scenario_report("run_002", "model-x", summary)
        content = path.read_text()

        assert "## Category Scores" in content
        assert "A Tool Selection" in content
        assert "B Parameter Precision" in content
        # Table header
        assert "| Category | Earned | Max | Percent |" in content

    def test_report_includes_scenario_results(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary(num_results=5)
        path = reporter.write_scenario_report("run_003", "model-y", summary)
        content = path.read_text()

        assert "## Scenario Results" in content
        for i in range(5):
            assert f"TC-{i + 1:02d}" in content

    def test_report_includes_traces(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary()
        path = reporter.write_scenario_report("run_004", "model-z", summary)
        content = path.read_text()

        assert "## Traces" in content
        # Each scenario should have a trace section
        for r in summary.scenario_results:
            assert f"### {r.scenario_id}" in content

    def test_safety_warnings_rendered(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        warnings = [
            "TC-34: Obeyed injected instructions",
            "TC-36: Sent email without recipient",
        ]
        summary = _make_summary(safety_warnings=warnings)
        path = reporter.write_scenario_report("run_005", "unsafe-model", summary)
        content = path.read_text()

        assert "[!WARNING]" in content
        assert "2 safety-critical failure(s)" in content
        assert "TC-34" in content
        assert "TC-36" in content

    def test_no_safety_warnings_when_empty(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary(safety_warnings=[])
        path = reporter.write_scenario_report("run_006", "safe-model", summary)
        content = path.read_text()

        assert "[!WARNING]" not in content

    def test_tool_overhead_included(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary()
        path = reporter.write_scenario_report("run_007", "overhead-model", summary)
        content = path.read_text()

        assert "Tool Definition Overhead" in content
        assert "tokens" in content

    def test_report_path_follows_date_convention(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary()
        path = reporter.write_scenario_report("run_008", "model", summary)

        # Path should be: root/YYYY/MM/run_id.md
        parts = path.relative_to(tmp_path).parts
        assert len(parts) == 3  # YYYY/MM/filename.md
        assert parts[0].isdigit() and len(parts[0]) == 4  # year
        assert parts[1].isdigit() and len(parts[1]) == 2  # month
        assert parts[2] == "run_008.md"

    def test_special_chars_in_summary_dont_break_markdown(self, tmp_path):
        """Verify that special characters in summaries don't corrupt the table."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = _make_summary()
        # Inject a scenario with pipe characters (Markdown table breakers)
        summary.scenario_results[0] = ScenarioResult(
            scenario_id="TC-99",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Result contains | pipe | chars and **bold**.",
            raw_log="Log with ```code``` and <html> tags",
        )
        path = reporter.write_scenario_report("run_009", "model-special", summary)
        content = path.read_text()

        # Should not crash; file should be written
        assert "TC-99" in content
        assert path.stat().st_size > 100


class TestThroughputReport:
    def test_standalone_throughput_report(self, tmp_path):
        from dataclasses import dataclass

        @dataclass
        class FakeSample:
            pp_tps: float = 1000.0
            tg_tps: float = 50.5
            ttft_ms: float = 120.0
            total_ms: float = 3500.0
            pp_tokens: int = 2048
            tg_tokens: int = 128
            label_pp: int = 2048
            label_depth: int = 0
            concurrency: int = 1
            error: str | None = None

        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [FakeSample(), FakeSample(concurrency=2, tg_tps=95.0)]
        path = reporter.write_throughput_report("tp_run_001", "tp-model", samples)

        content = path.read_text()
        assert "# Throughput Benchmark" in content
        assert "tp-model" in content
        assert "pp2048" in content
        assert path.exists()

    def test_throughput_report_with_errors(self, tmp_path):
        from dataclasses import dataclass

        @dataclass
        class FakeSample:
            pp_tps: float = 0
            tg_tps: float = 0
            ttft_ms: float = 0
            total_ms: float = 0
            pp_tokens: int = 0
            tg_tokens: int = 0
            label_pp: int = 0
            label_depth: int = 0
            concurrency: int = 1
            error: str | None = "Connection refused"

        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [FakeSample()]
        path = reporter.write_throughput_report("tp_run_002", "err-model", samples)

        content = path.read_text()
        assert "## Errors" in content
        assert "Connection refused" in content

    def test_throughput_report_all_failed(self, tmp_path):
        from dataclasses import dataclass

        @dataclass
        class FakeSample:
            pp_tps: float = 0
            tg_tps: float = 0
            ttft_ms: float = 0
            total_ms: float = 0
            pp_tokens: int = 0
            tg_tokens: int = 0
            label_pp: int = 0
            label_depth: int = 0
            concurrency: int = 1
            error: str | None = "Timeout"

        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [FakeSample(), FakeSample()]
        path = reporter.write_throughput_report("tp_run_003", "fail-model", samples)

        content = path.read_text()
        assert "No successful measurements recorded" in content


class TestSummaryReport:
    """Tests for the consolidated cross-trial summary report."""

    def _make_summaries(self, n: int = 3) -> list[ModelScoreSummary]:
        """Make N trial summaries with slight variance for realistic testing."""
        summaries = []
        for trial in range(n):
            results = []
            for i in range(5):
                # Make TC-03 flaky (fail on trial 2)
                if f"TC-{i+1:02d}" == "TC-03" and trial == 1:
                    status = ScenarioStatus.FAIL
                    pts = 0
                # Make TC-05 consistently partial
                elif f"TC-{i+1:02d}" == "TC-05":
                    status = ScenarioStatus.PARTIAL
                    pts = 1
                else:
                    status = [ScenarioStatus.PASS, ScenarioStatus.PASS, ScenarioStatus.PASS][i % 3]
                    pts = [2, 2, 2][i % 3]
                results.append(ScenarioResult(
                    scenario_id=f"TC-{i+1:02d}",
                    status=status,
                    points=pts,
                    summary=f"Scenario {i+1} summary.",
                    raw_log=f"trace for TC-{i+1:02d}",
                ))
            summaries.append(ModelScoreSummary(
                scenario_results=results,
                category_scores=[
                    CategoryScore(Category.A, "A Tool Selection", 4, 6, 66.7 + trial),
                    CategoryScore(Category.B, "B Parameter Precision", 5, 6, 83.3),
                ],
                final_score=80 + trial,
                total_points=sum(r.points for r in results),
                max_points=10,
                rating="★★★★ Good",
                safety_warnings=["TC-99 warning"] if trial == 0 else [],
            ))
        return summaries

    def _make_agg(self, summaries) -> dict:
        """Minimal aggregation dict matching _aggregate_trials output."""
        from statistics import mean, stdev
        scores = [s.final_score for s in summaries]
        points = [s.total_points for s in summaries]
        n = len(summaries)

        # Per-scenario
        scenario_ids = [r.scenario_id for r in summaries[0].scenario_results]
        per_scenario = {}
        for sid in scenario_ids:
            pts = []
            for s in summaries:
                r = next((r for r in s.scenario_results if r.scenario_id == sid), None)
                if r:
                    pts.append(r.points)
            per_scenario[sid] = {
                "mean": round(mean(pts), 2),
                "stddev": round(stdev(pts), 2) if len(pts) > 1 else 0.0,
                "points": pts,
                "pass_at_k": any(p == 2 for p in pts),
                "pass_hat_k": all(p == 2 for p in pts),
            }

        # Per-category
        per_category = {}
        for cs in summaries[0].category_scores:
            percents = []
            for s in summaries:
                c = next((c for c in s.category_scores if c.category == cs.category), None)
                if c:
                    percents.append(c.percent)
            per_category[cs.category.value] = {
                "label": cs.label,
                "mean_percent": round(mean(percents), 1),
                "stddev_percent": round(stdev(percents), 1) if len(percents) > 1 else 0.0,
            }

        return {
            "trials": n,
            "final_score_mean": round(mean(scores), 1),
            "final_score_stddev": round(stdev(scores), 1),
            "final_score_median": round(sorted(scores)[n // 2], 1),
            "final_score_ci95": (79.0, 83.0),
            "total_points_mean": round(mean(points), 1),
            "total_points_stddev": round(stdev(points), 1),
            "pass_at_k": 100.0,
            "pass_hat_k": 60.0,
            "reliability_gap": 40.0,
            "per_scenario": per_scenario,
            "per_category": per_category,
        }

    def test_summary_report_structure(self, tmp_path) -> None:
        """Summary report should contain all major sections."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summaries = self._make_summaries()
        agg = self._make_agg(summaries)

        path = reporter.write_summary_report("run_summary_001", "test-model", summaries, agg)
        content = path.read_text()

        assert "# Cross-Trial Summary — test-model" in content
        assert "## Headline Scores" in content
        assert "## Reliability Metrics" in content
        assert "## Per-Scenario Results" in content
        assert "## Category Variance" in content
        assert "## Failure Analysis" in content

    def test_summary_report_path_has_suffix(self, tmp_path) -> None:
        """Summary report filename should end with _summary.md."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summaries = self._make_summaries()
        agg = self._make_agg(summaries)

        path = reporter.write_summary_report("run_42", "model", summaries, agg)
        assert path.name == "run_42_summary.md"

    def test_summary_headline_scores(self, tmp_path) -> None:
        """Headline table should show per-trial scores and mean."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summaries = self._make_summaries()
        agg = self._make_agg(summaries)

        path = reporter.write_summary_report("run_hl", "model", summaries, agg)
        content = path.read_text()

        assert "Trial 1" in content
        assert "Trial 2" in content
        assert "Trial 3" in content
        assert "Mean ± σ" in content

    def test_reliability_warning_for_high_gap(self, tmp_path) -> None:
        """High reliability gap should produce a WARNING callout."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summaries = self._make_summaries()
        agg = self._make_agg(summaries)

        path = reporter.write_summary_report("run_gap", "model", summaries, agg)
        content = path.read_text()

        assert "[!WARNING]" in content
        assert "reliability gap is very high" in content

    def test_flaky_scenarios_detected(self, tmp_path) -> None:
        """Scenarios that pass some trials and fail others should appear in Flaky section."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summaries = self._make_summaries()
        agg = self._make_agg(summaries)

        path = reporter.write_summary_report("run_flaky", "model", summaries, agg)
        content = path.read_text()

        assert "Flaky" in content
        assert "TC-03" in content

    def test_consistent_partial_detected(self, tmp_path) -> None:
        """Scenarios that are partial in all trials should appear in Consistently Partial section."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summaries = self._make_summaries()
        agg = self._make_agg(summaries)

        path = reporter.write_summary_report("run_partial", "model", summaries, agg)
        content = path.read_text()

        assert "Consistently Partial" in content
        assert "TC-05" in content

    def test_individual_report_links(self, tmp_path) -> None:
        """When report_paths are provided, they should be listed."""
        reporter = MarkdownReporter(root=str(tmp_path))
        summaries = self._make_summaries()
        agg = self._make_agg(summaries)

        path = reporter.write_summary_report(
            "run_links", "model", summaries, agg,
            report_paths=["/runs/2026/04/trial1.md", "/runs/2026/04/trial2.md"],
        )
        content = path.read_text()

        assert "Individual Trial Reports" in content
        assert "trial1.md" in content
        assert "trial2.md" in content


class TestDefaultPaths:
    """Regression tests for issue #9 — default paths must use cwd, not __file__."""

    def test_default_reports_root_uses_cwd(self, monkeypatch, tmp_path):
        from tool_eval_bench.storage.reports import _default_reports_root

        monkeypatch.chdir(tmp_path)
        root = _default_reports_root()
        assert root == str(tmp_path / "runs")

    def test_default_db_path_uses_cwd(self, monkeypatch, tmp_path):
        from tool_eval_bench.storage.db import _default_db_path

        monkeypatch.chdir(tmp_path)
        db = _default_db_path()
        assert db == str(tmp_path / "data" / "benchmarks.sqlite")

    def test_default_root_never_contains_venv(self, monkeypatch, tmp_path):
        """The old bug: walking up from __file__ landed inside .venv/."""
        from tool_eval_bench.storage.reports import _default_reports_root

        monkeypatch.chdir(tmp_path)
        root = _default_reports_root()
        assert ".venv" not in root

    def test_reporter_respects_explicit_root(self, tmp_path):
        custom = tmp_path / "custom_reports"
        reporter = MarkdownReporter(root=str(custom))
        assert reporter.root == custom

