"""Tests for CLI display module: BenchmarkDisplay, _print_category_scores, _print_final_panel, print_final_report.

Covers cli/display.py which was at 27% coverage. Uses constructed ModelScoreSummary
objects and captured console output to verify formatting.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from tool_eval_bench.cli.display import (
    BenchmarkDisplay,
    _print_category_scores,
    _print_final_panel,
    print_final_report,
)
from tool_eval_bench.evals.scenarios import SCENARIOS

from tool_eval_bench.domain.scenarios import (
    Category,
    CategoryScore,
    ModelScoreSummary,
    ScenarioResult,
    ScenarioStatus,
)


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_summary(
    *,
    score: int = 75,
    rating: str = "★★★★ Good",
    total_points: int = 90,
    max_points: int = 138,
    safety_warnings: list[str] | None = None,
    deployability: int | None = 72,
    responsiveness: int | None = 68,
    median_turn_ms: float | None = 2500.0,
    alpha: float = 0.7,
    worst_category: str | None = "Tool Selection",
    worst_category_percent: int | None = 60,
    total_tokens: int = 50000,
    token_efficiency: float | None = 1.8,
    num_results: int = 6,
) -> ModelScoreSummary:
    """Build a ModelScoreSummary with predictable test data."""
    results = []
    statuses = [ScenarioStatus.PASS, ScenarioStatus.PARTIAL, ScenarioStatus.FAIL]
    for i in range(num_results):
        results.append(ScenarioResult(
            scenario_id=f"TC-{i + 1:02d}",
            status=statuses[i % 3],
            points=[2, 1, 0][i % 3],
            summary=f"Scenario {i + 1} result.",
            tool_calls_made=["get_weather"] if i % 3 == 0 else [],
            expected_behavior="Expected behavior for TC-01",
            duration_seconds=10.0 + i,
            ttft_ms=50.0 + i * 10,
            turn_count=1 + (i % 3),
            prompt_tokens=1000,
            completion_tokens=500,
        ))

    category_scores = [
        CategoryScore(Category.A, "A Tool Selection", 4, 6, 66.7),
        CategoryScore(Category.B, "B Parameter Precision", 5, 6, 83.3),
        CategoryScore(Category.C, "C Multi-Step Chains", 6, 6, 100.0),
    ]

    return ModelScoreSummary(
        scenario_results=results,
        category_scores=category_scores,
        final_score=score,
        total_points=total_points,
        max_points=max_points,
        rating=rating,
        safety_warnings=safety_warnings or [],
        deployability=deployability,
        responsiveness=responsiveness,
        median_turn_ms=median_turn_ms,
        alpha=alpha,
        worst_category=worst_category,
        worst_category_percent=worst_category_percent,
        total_tokens=total_tokens,
        token_efficiency=token_efficiency,
    )


# ===========================================================================
# _print_category_scores
# ===========================================================================


class TestPrintCategoryScores:
    """Tests for the _print_category_scores static function."""

    def test_shows_all_categories(self) -> None:
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_category_scores(console, summary)

        output = console.file.getvalue()
        assert "A Tool Selection" in output
        assert "B Parameter Precision" in output
        assert "C Multi-Step Chains" in output

    def test_shows_percentages(self) -> None:
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_category_scores(console, summary)

        output = console.file.getvalue()
        assert "66.7%" in output
        assert "83.3%" in output
        assert "100.0%" in output

    def test_shows_earned_max(self) -> None:
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_category_scores(console, summary)

        output = console.file.getvalue()
        assert "4/6" in output
        assert "5/6" in output
        assert "6/6" in output


# ===========================================================================
# _print_final_panel
# ===========================================================================


class TestPrintFinalPanel:
    """Tests for the _print_final_panel static function."""

    def test_shows_model_and_score(self) -> None:
        summary = _make_summary(score=85)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "test-model" in output
        assert "85 / 100" in output

    def test_shows_rating(self) -> None:
        summary = _make_summary(score=85, rating="★★★★ Good")
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "★★★★ Good" in output

    def test_shows_pass_partial_fail_counts(self) -> None:
        summary = _make_summary(num_results=6)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "passed" in output
        assert "partial" in output
        assert "failed" in output

    def test_shows_points_total(self) -> None:
        summary = _make_summary(total_points=90, max_points=138)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "90/138" in output

    def test_shows_deployability_when_latency_present(self) -> None:
        summary = _make_summary(deployability=72, responsiveness=68, median_turn_ms=2500.0)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Deployability:" in output
        assert "Responsiveness:" in output
        assert "72" in output
        assert "68" in output

    def test_shows_median_turn_ms(self) -> None:
        summary = _make_summary(median_turn_ms=2500.0)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "2.5s" in output

    def test_shows_alpha(self) -> None:
        summary = _make_summary(alpha=0.8)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "α=0.8" in output

    def test_hides_weakest_when_all_100(self) -> None:
        summary = _make_summary(worst_category_percent=100)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Weakest:" not in output

    def test_shows_weakest_when_below_100(self) -> None:
        summary = _make_summary(worst_category="Tool Selection", worst_category_percent=60)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Weakest:" in output
        assert "Tool Selection" in output

    def test_shows_token_usage_when_present(self) -> None:
        summary = _make_summary(total_tokens=50000, token_efficiency=1.8)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Token Usage" in output
        assert "50,000" in output
        assert "1.8" in output

    def test_shows_safety_warnings(self) -> None:
        summary = _make_summary(
            safety_warnings=["Category K scored below threshold"],
        )
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "SAFETY WARNINGS" in output
        assert "Category K" in output

    def test_shows_elapsed_time(self) -> None:
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=45.3)

        output = console.file.getvalue()
        assert "45.3s" in output

    def test_shows_scoring_methodology(self) -> None:
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "How this score is calculated" in output
        assert "pass=2pt" in output
        assert "partial=1pt" in output
        assert "fail=0pt" in output

    def test_shows_deployability_methodology_when_present(self) -> None:
        summary = _make_summary(deployability=72, alpha=0.7)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Deployability:" in output
        assert "0.7×quality" in output

    def test_no_deployability_when_no_latency(self) -> None:
        summary = _make_summary(deployability=None, responsiveness=None, median_turn_ms=None)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Deployability:" not in output
        assert "Responsiveness:" not in output

    def test_no_token_usage_when_zero(self) -> None:
        summary = _make_summary(total_tokens=0, token_efficiency=None)
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Token Usage" not in output

    def test_no_safety_warnings_when_empty(self) -> None:
        summary = _make_summary(safety_warnings=[])
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "SAFETY WARNINGS" not in output


# ===========================================================================
# print_final_report
# ===========================================================================


class TestPrintFinalReport:
    """Tests for the print_final_report static function."""

    def test_shows_category_scores(self) -> None:
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        print_final_report(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Category Breakdown" in output
        assert "A Tool Selection" in output

    def test_shows_scenario_details_table(self) -> None:
        summary = _make_summary(num_results=3)
        console = Console(file=StringIO(), width=200, no_color=True)
        print_final_report(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Scenario Details" in output
        assert "TC-01" in output
        assert "TC-02" in output
        assert "TC-03" in output

    def test_shows_expected_for_failures(self) -> None:
        summary = _make_summary(num_results=3)
        console = Console(file=StringIO(), width=200, no_color=True)
        print_final_report(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Expected" in output

    def test_shows_tool_calls_made_for_non_pass(self) -> None:
        summary = _make_summary(num_results=3)
        console = Console(file=StringIO(), width=200, no_color=True)
        print_final_report(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Called:" in output

    def test_shows_final_panel(self) -> None:
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        print_final_report(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Benchmark Complete" in output
        assert "test-model" in output


# ===========================================================================
# STATUS_LABELS
# ===========================================================================


class TestStatusLabels:
    """Tests for the STATUS_LABELS constant."""

    def test_pass_label(self) -> None:
        from tool_eval_bench.cli.display import STATUS_LABELS
        from tool_eval_bench.domain.scenarios import ScenarioStatus

        assert STATUS_LABELS[ScenarioStatus.PASS] == ("✅ PASS", "green")

    def test_partial_label(self) -> None:
        from tool_eval_bench.cli.display import STATUS_LABELS
        from tool_eval_bench.domain.scenarios import ScenarioStatus

        assert STATUS_LABELS[ScenarioStatus.PARTIAL] == ("⚠️  PARTIAL", "yellow")

    def test_fail_label(self) -> None:
        from tool_eval_bench.cli.display import STATUS_LABELS
        from tool_eval_bench.domain.scenarios import ScenarioStatus

        assert STATUS_LABELS[ScenarioStatus.FAIL] == ("❌ FAIL", "red")


# ===========================================================================
# CATEGORY_COLORS
# ===========================================================================


class TestCategoryColors:
    """Tests for the CATEGORY_COLORS constant."""

    def test_all_categories_have_colors(self) -> None:
        from tool_eval_bench.cli.display import CATEGORY_COLORS
        from tool_eval_bench.domain.scenarios import Category

        for cat in Category:
            assert cat in CATEGORY_COLORS

    def test_specific_colors(self) -> None:
        from tool_eval_bench.cli.display import CATEGORY_COLORS
        from tool_eval_bench.domain.scenarios import Category

        assert CATEGORY_COLORS[Category.A] == "cyan"
        assert CATEGORY_COLORS[Category.K] == "bright_red"
        assert CATEGORY_COLORS[Category.O] == "orchid1"


# ===========================================================================
# RATING_COLORS
# ===========================================================================


class TestRatingColors:
    """Tests for the RATING_COLORS constant."""

    def test_excellent_color(self) -> None:
        from tool_eval_bench.cli.display import RATING_COLORS
        assert RATING_COLORS["★★★★★ Excellent"] == "bold green"

    def test_good_color(self) -> None:
        from tool_eval_bench.cli.display import RATING_COLORS
        assert RATING_COLORS["★★★★ Good"] == "bold cyan"

    def test_adequate_color(self) -> None:
        from tool_eval_bench.cli.display import RATING_COLORS
        assert RATING_COLORS["★★★ Adequate"] == "bold yellow"

    def test_safety_capped_color(self) -> None:
        from tool_eval_bench.cli.display import RATING_COLORS
        assert RATING_COLORS["★★★ Adequate (safety-capped)"] == "bold red"

    def test_weak_color(self) -> None:
        from tool_eval_bench.cli.display import RATING_COLORS
        assert RATING_COLORS["★★ Weak"] == "bold red"

    def test_poor_color(self) -> None:
        from tool_eval_bench.cli.display import RATING_COLORS
        assert RATING_COLORS["★ Poor"] == "bold red"


# ===========================================================================
# BenchmarkDisplay._format_result_line
# ===========================================================================


class TestFormatResultLine:
    """Tests for the BenchmarkDisplay._format_result_line method."""

    def _make_display(self) -> MagicMock:
        """Create a MagicMock for BenchmarkDisplay with _format_result_line accessible."""


        display = MagicMock(spec=BenchmarkDisplay)
        display.scenarios = SCENARIOS
        return display

    def test_pass_result_shows_green(self) -> None:
        display = self._make_display()


        scenario = SCENARIOS[0]
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Correct tool selection.",
            duration_seconds=5.3,
            ttft_ms=45.0,
            turn_count=1,
        )

        # Call the actual method
        line = BenchmarkDisplay._format_result_line(display, scenario, result)

        assert "TC-01" in line
        assert "✅ PASS" in line
        assert "2[/]/2" in line or "2/2" in line
        assert "5.3s" in line
        assert "45ms" in line

    def test_fail_result_shows_red_and_summary(self) -> None:
        display = self._make_display()


        scenario = SCENARIOS[0]
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Wrong tool called.",
            duration_seconds=3.1,
            ttft_ms=120.0,
            turn_count=1,
        )

        line = BenchmarkDisplay._format_result_line(display, scenario, result)

        assert "❌ FAIL" in line
        assert "Wrong tool called" in line

    def test_partial_result_shows_yellow(self) -> None:
        display = self._make_display()


        scenario = SCENARIOS[0]
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PARTIAL,
            points=1,
            summary="Partial match.",
            duration_seconds=4.0,
            turn_count=1,
        )

        line = BenchmarkDisplay._format_result_line(display, scenario, result)

        assert "⚠️  PARTIAL" in line
        assert "1[/]/2" in line or "1/2" in line

    def test_multi_turn_shows_turn_count(self) -> None:
        display = self._make_display()


        scenario = SCENARIOS[0]
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Correct.",
            duration_seconds=10.0,
            ttft_ms=50.0,
            turn_count=3,
        )

        line = BenchmarkDisplay._format_result_line(display, scenario, result)

        assert "t3" in line

    def test_no_ttft_shows_no_latency(self) -> None:
        display = self._make_display()


        scenario = SCENARIOS[0]
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Correct.",
            duration_seconds=5.0,
            ttft_ms=None,
            turn_count=1,
        )

        line = BenchmarkDisplay._format_result_line(display, scenario, result)

        assert "ttft=" not in line


# ===========================================================================
# BenchmarkDisplay._build_footer
# ===========================================================================


class TestBuildFooter:
    """Tests for the BenchmarkDisplay._build_footer method."""

    def test_initial_state_shows_waiting(self) -> None:


        display = BenchmarkDisplay(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8080",
            scenarios=SCENARIOS,
        )
        display.started_at = 0  # fixed time for test
        display.results = {}
        display.active_scenario = None

        footer = display._build_footer()
        text = str(footer)
        assert "Waiting" in text

    def test_complete_shows_checkmark(self) -> None:


        display = BenchmarkDisplay(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8080",
            scenarios=SCENARIOS,
        )
        display.started_at = 0
        display.results = {"TC-01": MagicMock()}
        display.active_scenario = None

        # Pretend we've completed all scenarios
        original_scenarios = display.scenarios
        display.scenarios = SCENARIOS[:1]
        display.results = {"TC-01": MagicMock()}

        footer = display._build_footer()
        text = str(footer)
        assert "Complete" in text

        display.scenarios = original_scenarios

    def test_active_scenario_shows_scenario_name(self) -> None:


        display = BenchmarkDisplay(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8080",
            scenarios=SCENARIOS,
        )
        display.started_at = 0
        display.results = {}
        display.active_scenario = "TC-05"

        footer = display._build_footer()
        text = str(footer)
        assert "TC-05" in text

    def test_progress_bar_shows_count(self) -> None:


        display = BenchmarkDisplay(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8080",
            scenarios=SCENARIOS,
        )
        display.started_at = 0
        display.results = {"TC-01": MagicMock(), "TC-02": MagicMock()}
        display.active_scenario = None

        footer = display._build_footer()
        text = str(footer)
        assert "2/" in text  # 2 done out of total


# ===========================================================================
# print_final_report with throughput_samples
# ===========================================================================


class TestPrintFinalReportThroughput:
    """Tests for print_final_report with throughput data."""

    def test_shows_throughput_highlights(self) -> None:
        """When throughput_samples are provided, throughput section should appear."""
        from unittest.mock import MagicMock

        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)

        # Create mock throughput samples
        single_sample = MagicMock()
        single_sample.pp_tps = 100.0
        single_sample.tg_tps = 50.0
        single_sample.ttft_ms = 100.0
        single_sample.concurrency = 1
        single_sample.error = None

        concurrent_sample = MagicMock()
        concurrent_sample.pp_tps = 200.0
        concurrent_sample.tg_tps = 100.0
        concurrent_sample.concurrency = 2
        concurrent_sample.error = None
        print_final_report(
            console, "test-model", summary, elapsed=120.5,
            throughput_samples=[single_sample, concurrent_sample],
        )

        output = console.file.getvalue()
        assert "Throughput" in output
        assert "pp t/s" in output
        assert "tg t/s" in output

    def test_shows_concurrency_levels(self) -> None:
        from unittest.mock import MagicMock

        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)

        c1 = MagicMock(pp_tps=100, tg_tps=50, ttft_ms=100, concurrency=1, error=None)
        c2 = MagicMock(pp_tps=200, tg_tps=100, ttft_ms=200, concurrency=2, error=None)
        c4 = MagicMock(pp_tps=300, tg_tps=150, ttft_ms=300, concurrency=4, error=None)
        print_final_report(
            console, "test-model", summary, elapsed=120.5,
            throughput_samples=[c1, c2, c4],
        )

        output = console.file.getvalue()
        assert "Single:" in output  # concurrency=1 renders as "Single:"
        assert "c2:" in output
        assert "c4:" in output
