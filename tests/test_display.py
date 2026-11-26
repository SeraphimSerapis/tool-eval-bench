"""Tests for CLI display module: BenchmarkDisplay, _print_category_scores, _print_final_panel, print_final_report.

Covers cli/display.py which was at 27% coverage. Uses constructed ModelScoreSummary
objects and captured console output to verify formatting.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from tool_eval_bench.cli.display import (
    BenchmarkDisplay,
    _print_category_scores,
    _print_final_panel,
    print_final_report,
)
from tool_eval_bench.domain.scenarios import (
    Category,
    CategoryScore,
    ModelScoreSummary,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.evals.scenarios import SCENARIOS

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
        results.append(
            ScenarioResult(
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
            )
        )

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

    def test_shows_benchmark_version_in_summary(self) -> None:
        from tool_eval_bench import __version__

        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=45.3)

        output = console.file.getvalue()
        assert "Benchmark:" in output
        assert f"tool-eval-bench v{__version__}" in output

    def test_benchmark_version_precedes_pass_counts(self) -> None:
        """The version line lives in the primary summary block, not buried
        near the completion time at the bottom of the panel."""
        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        _print_final_panel(console, "test-model", summary, elapsed=45.3)

        output = console.file.getvalue()
        assert output.index("Benchmark:") < output.index("passed")
        assert output.index("Benchmark:") < output.index("Completed in")

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

    def test_shows_hardmode_scenario_title(self) -> None:
        summary = _make_summary(num_results=0)
        summary.scenario_results = [
            ScenarioResult(
                scenario_id="TC-84",
                status=ScenarioStatus.PARTIAL,
                points=1,
                summary="partial",
            )
        ]
        console = Console(file=StringIO(), width=200, no_color=True)
        print_final_report(console, "test-model", summary, elapsed=120.5)

        output = console.file.getvalue()
        assert "Long-Horizon Recovery" in output


# ===========================================================================
# CATEGORY_COLORS
# ===========================================================================


class TestCategoryColors:
    """Tests for the CATEGORY_COLORS completeness invariant."""

    def test_all_categories_have_colors(self) -> None:
        from tool_eval_bench.cli.display import CATEGORY_COLORS
        from tool_eval_bench.domain.scenarios import Category

        for cat in Category:
            assert cat in CATEGORY_COLORS


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
            console,
            "test-model",
            summary,
            elapsed=120.5,
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
            console,
            "test-model",
            summary,
            elapsed=120.5,
            throughput_samples=[c1, c2, c4],
        )

        output = console.file.getvalue()
        assert "Single:" in output  # concurrency=1 renders as "Single:"
        assert "c2:" in output
        assert "c4:" in output

    def test_marks_estimated_single_stream_prefill(self) -> None:
        from unittest.mock import MagicMock

        from tool_eval_bench.storage.reports._common import PP_ESTIMATED_NOTE

        summary = _make_summary()
        console = Console(file=StringIO(), width=200, no_color=True)
        estimated = MagicMock(
            pp_tps=754.0,
            tg_tps=65.4,
            ttft_ms=2717.0,
            concurrency=1,
            error=None,
            pp_estimated=True,
        )
        print_final_report(
            console,
            "test-model",
            summary,
            elapsed=120.5,
            throughput_samples=[estimated],
        )
        output = console.file.getvalue()
        assert "754* pp t/s" in output
        # The panel wraps the note, so match the phrase that stays on one line.
        assert "response chunk before the first content token" in output
        assert PP_ESTIMATED_NOTE.split(" llama-benchy", maxsplit=1)[0] in output


# ===========================================================================
# Markup escaping and interactive URL redaction
# ===========================================================================


def test_rich_markup_brackets_and_url_redaction_do_not_crash_or_leak(
    monkeypatch, tmp_path, capsys
) -> None:
    import json
    from typing import Any

    import httpx
    import pytest

    from tool_eval_bench.cli import model_probe, plugin_datasets, plugin_lifecycle, probe
    from tool_eval_bench.cli.headless import HeadlessConsole
    from tool_eval_bench.cli.plugin_progress import truncate

    real_client = httpx.AsyncClient

    # 1. plugin_progress.truncate escapes square-bracket markup tokens.
    buf = StringIO()
    console = Console(file=buf, width=120, no_color=True)
    escaped = truncate("Wrap answer in [/] and [bold red]brackets[/]")
    console.print(f"[dim italic]{escaped}[/]")
    assert "[/]" in buf.getvalue()

    # 2. plugin_lifecycle.execute_plugin survives unclosed [/] in both normal and HeadlessConsole.
    def boom() -> Any:
        async def _coro() -> None:
            raise RuntimeError("bad bracket [/] from http://secret.internal:8000/v1")

        return _coro()

    with pytest.raises(SystemExit) as exc_plain:
        plugin_lifecycle.execute_plugin(console, "GSM8K[/]", boom, [])
    assert exc_plain.value.code == 1

    with pytest.raises(SystemExit) as exc_json:
        plugin_lifecycle.execute_plugin(HeadlessConsole(), "GSM8K[/]", boom, [])
    assert exc_json.value.code == 1
    headless_err = json.loads(capsys.readouterr().err.strip())
    assert headless_err["error"] == "run_failed"
    assert "secret.internal" not in headless_err["message"]

    # 3. plugin_datasets.load_dataset_with_progress escapes [hf], name, noun, and exception text.
    ds_buf = StringIO()
    ds_console = Console(file=ds_buf, width=120, no_color=True)
    items = plugin_datasets.load_dataset_with_progress(
        ds_console,
        name="IFEval[/]",
        noun="prompts[/]",
        cache_path=tmp_path / "missing.jsonl",
        load=lambda **_kw: (_ for _ in ()).throw(RuntimeError("rate limit [/] [bad")),
        cache_note="data/ifeval/test[/].jsonl",
    )
    assert items is None
    assert "tool-eval-bench[hf]" in ds_buf.getvalue()
    assert "rate limit [/]" in ds_buf.getvalue()

    # 4. probe.preflight_model_check redacts secret URL in 500 body when display_url is masked.
    def mock_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, text="upstream http://secret-upstream.internal:8000/v1 failed [/] [bad"
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(mock_500), **kw),
    )
    probe_buf = StringIO()
    probe_console = Console(file=probe_buf, width=120, no_color=True)
    with pytest.raises(SystemExit) as probe_exc:
        probe.preflight_model_check(
            probe_console,
            "http://secret.internal:8000",
            "model[/]",
            None,
            display_url="http://***:8000",
        )
    assert probe_exc.value.code == 3
    assert "[/]" in probe_buf.getvalue()
    assert "secret-upstream.internal" not in probe_buf.getvalue()

    # 5. model_probe._detect_model redacts URL in non-JSON body when display_url is masked.
    def mock_html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text="login at http://sso.secret.internal:8443/auth [/]",
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(mock_html), **kw),
    )
    mp_buf = StringIO()
    mp_console = Console(file=mp_buf, width=160, no_color=True)
    with pytest.raises(SystemExit) as mp_exc:
        model_probe._detect_model(
            "http://secret.internal:8000",
            None,
            mp_console,
            display_url="http://***:8000",
        )
    assert mp_exc.value.code == 2
    assert "sso.secret.internal" not in mp_buf.getvalue()


@pytest.mark.asyncio
async def test_display_and_final_report_escape_dynamic_markup_and_preserve_title_padding() -> None:
    from dataclasses import replace

    from tool_eval_bench.domain.models import RunContext
    from tool_eval_bench.domain.scenarios import CapabilityScore, FailureKind
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

    tc01 = next(s for s in ALL_SCENARIOS if s.id == "TC-01")
    tc02 = next(s for s in ALL_SCENARIOS if s.id == "TC-02")
    sc_bracket = replace(tc01, title="Short [/] Title")
    sc_plain = replace(tc02, title="Plain Title")

    res_pass = ScenarioResult(
        scenario_id=sc_bracket.id,
        status=ScenarioStatus.PASS,
        points=2,
        summary="Passed with [/] and [bad token",
        duration_seconds=1.1,
    )
    res_plain = ScenarioResult(
        scenario_id=sc_plain.id,
        status=ScenarioStatus.PASS,
        points=2,
        summary="Plain summary",
        duration_seconds=1.1,
    )
    res_fail = ScenarioResult(
        scenario_id="TC-03[/]",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Failed [/] [bad output",
        tool_calls_made=["lookup([/])", "run([bad)"],
        expected_behavior="Call safe_tool([/]) [bad",
        duration_seconds=2.3,
    )
    res_infra = ScenarioResult(
        scenario_id="TC-04",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="",
        note="Request timed out [/] [bad",
        failure_kind=FailureKind.TIMEOUT,
        duration_seconds=120.0,
    )

    summary = ModelScoreSummary(
        scenario_results=[res_pass, res_fail, res_infra],
        category_scores=[
            CategoryScore(
                category=Category.A,
                label="Direct [/] [bad",
                earned=2,
                max_points=4,
                percent=50.0,
                pass_count=1,
                fail_count=1,
            )
        ],
        capability_scores=[
            CapabilityScore(
                capability="policy",
                label="Policy [/] [bad",
                earned=2,
                max_points=4,
                percent=50.0,
                scenario_ids=["TC-01[/]", "TC-03[bad"],
            )
        ],
        final_score=50,
        total_points=2,
        max_points=4,
        rating="★★★ Adequate [/]",
        excluded_scenarios=["TC-04[/]"],
        completion_rate=66.7,
        worst_category="Direct [/] [bad",
        worst_category_percent=50,
        safety_warnings=["TC-03: leaked [/] [bad"],
    )
    run_ctx = RunContext(
        tool_version="0.1.0",
        git_sha=None,
        hostname="host",
        platform_info="linux",
        python_version="3.13",
        model="qwen[/][bad",
        backend="vllm[/]",
        base_url="http://***:8000/[/]",
        timeout_seconds=120.0,
        server_model_root="/models/[bad]/qwen[/]",
        engine_name="vLLM[/]",
        engine_version="0.8[bad",
        quantization="AWQ[/][bad",
    )

    buf = StringIO()
    console = Console(file=buf, width=160, no_color=True)
    disp = BenchmarkDisplay(
        model="qwen[/][bad",
        backend="vllm[/]",
        base_url="http://***:8000/[/]",
        scenarios=[sc_bracket, sc_plain],
        run_context=run_ctx,
    )
    disp.console = console

    # Alignment check: padding before escape keeps identical status column offset.
    line_bracket_buf = StringIO()
    line_plain_buf = StringIO()
    Console(file=line_bracket_buf, width=200, no_color=True).print(
        disp._format_result_line(sc_bracket, res_pass)
    )
    Console(file=line_plain_buf, width=200, no_color=True).print(
        disp._format_result_line(sc_plain, res_plain)
    )
    assert line_bracket_buf.getvalue().index("PASS") == line_plain_buf.getvalue().index("PASS")

    disp.start()
    await disp.on_scenario_start(sc_bracket, 0, 2)
    footer = disp._build_footer()
    assert "Short [/] Title" in footer.plain
    await disp.on_scenario_result(sc_bracket, res_pass, 0, 2)
    await disp.on_scenario_result(sc_plain, res_infra, 1, 2)
    disp.set_finished(summary)

    print_final_report(
        console,
        "qwen[/][bad",
        summary,
        3.4,
        run_context=run_ctx,
    )

    rendered = buf.getvalue()
    assert "qwen[/][bad" in rendered
    assert "vLLM[/] 0.8[bad" in rendered
    assert "AWQ[/][bad" in rendered
    assert "/models/[bad]/qwen[/]" in rendered
    assert "TC-03: leaked [/] [bad" in rendered
    assert "lookup([/])" in rendered
    assert "Call safe_tool([/]) [bad" in rendered


def test_plugin_lifecycle_and_spec_bench_redact_urls_on_interactive_console_and_escape_paths(
    monkeypatch, tmp_path
) -> None:
    from typing import Any
    from unittest.mock import AsyncMock

    import httpx
    import pytest

    from tool_eval_bench.cli import plugin_lifecycle, spec_bench
    from tool_eval_bench.runner.speculative import SpecDecodeSample

    db_path = str(tmp_path / "benchmarks.sqlite")
    monkeypatch.setattr("tool_eval_bench.storage.db._default_db_path", lambda: db_path)

    # 1. Interactive Console in plugin_lifecycle redacts URLs and escapes benchmark_name / paths.
    plain_buf = StringIO()
    plain_console = Console(file=plain_buf, width=140, no_color=True)

    def fail_coro() -> Any:
        async def _inner() -> None:
            raise httpx.ConnectError(
                "failed connecting to http://secret-plugin.internal:8000/v1?key=sk-plugin [/] [bad"
            )

        return _inner()

    with pytest.raises(SystemExit) as exc_info:
        plugin_lifecycle.execute_plugin(plain_console, "GSM8K[/]", fail_coro, [])
    assert exc_info.value.code == 1
    out = plain_buf.getvalue()
    assert "GSM8K[/] error:" in out
    assert "secret-plugin.internal" not in out
    assert "sk-plugin" not in out
    assert "[/]" in out

    empty_buf = StringIO()
    empty_console = Console(file=empty_buf, width=140, no_color=True)
    assert plugin_lifecycle.execute_plugin(empty_console, "MMLU[/][bad", AsyncMock(), []) is None
    assert "No MMLU[/][bad results." in empty_buf.getvalue()

    fake_res = MagicMock(score=87.5, rating="★★★★ Good", details={"correct": 7, "total": 8})
    out_dir = tmp_path / "plugin[bold]out"
    fin_buf = StringIO()
    fin_console = Console(file=fin_buf, width=160, no_color=True)
    plugin_lifecycle.finalize_plugin_run(
        fin_console,
        mode="gsm8k",
        title="GSM8K",
        display_name="Qwen[/]",
        result=fake_res,
        config={"model": "qwen", "backend": "vllm", "base_url": "http://***:8000"},
        report_metrics=["- **Accuracy**: 87.5%"],
        report_lines=["ok"],
        output_dir=str(out_dir),
        label=None,
        run_context=None,
    )
    assert "plugin[bold]out" in fin_buf.getvalue().replace("\n", "")

    # 2. Interactive Console in spec_bench redacts URLs on exception and escapes prompt_type / paths.
    async def fail_spec(*_a: Any, **_kw: Any) -> list[SpecDecodeSample]:
        raise httpx.ConnectError(
            "spec failed at http://secret-spec.internal:8090/v1?api_key=sk-spec-secret [/] [bad"
        )

    monkeypatch.setattr("tool_eval_bench.runner.speculative.run_spec_bench", fail_spec)
    spec_err_buf = StringIO()
    spec_err_console = Console(file=spec_err_buf, width=160, no_color=True)
    with pytest.raises(SystemExit) as spec_exc:
        spec_bench.run_spec_bench(
            spec_err_console,
            "qwen",
            "Qwen[/]",
            "http://secret-spec.internal:8090",
            None,
            pp=128,
            tg=64,
            depths=[0],
            output_dir=str(tmp_path / "spec_err"),
        )
    assert spec_exc.value.code == 1
    spec_err_text = spec_err_buf.getvalue()
    assert "secret-spec.internal" not in spec_err_text
    assert "sk-spec-secret" not in spec_err_text
    assert "[/]" in spec_err_text

    async def ok_spec(*_a: Any, on_sample: Any = None, **_kw: Any) -> list[SpecDecodeSample]:
        s_ok = SpecDecodeSample(
            pp_tokens=128,
            tg_tokens=64,
            depth=0,
            ttft_ms=12.0,
            total_ms=500.0,
            pp_tps=900.0,
            tg_tps=120.0,
            acceptance_rate=0.75,
            acceptance_length=3.2,
            draft_tokens_delta=80,
            accepted_tokens_delta=60,
            num_drafts_delta=20,
            acceptance_source="response",
            prompt_type="code[/]",
        )
        s_err = SpecDecodeSample(
            depth=0,
            prompt_type="bad[p]",
            error="cell error from http://cell-secret.internal:8000/v1 [/]",
        )
        if on_sample is not None:
            await on_sample(s_ok, 0, 2)
            await on_sample(s_err, 1, 2)
        return [s_ok, s_err]

    monkeypatch.setattr("tool_eval_bench.runner.speculative.run_spec_bench", ok_spec)
    spec_ok_dir = tmp_path / "spec[bold]dir"
    spec_ok_buf = StringIO()
    spec_ok_console = Console(file=spec_ok_buf, width=160, no_color=True)
    samples = spec_bench.run_spec_bench(
        spec_ok_console,
        "qwen",
        "Qwen[/]",
        "http://localhost:8000",
        None,
        pp=128,
        tg=64,
        depths=[0],
        prompt_types=["code[/]", "bad[p]"],
        output_dir=str(spec_ok_dir),
    )
    assert len(samples) == 2
    spec_ok_text = spec_ok_buf.getvalue()
    assert "code[/]" in spec_ok_text
    assert "spec[bold]dir" in spec_ok_text.replace("\n", "")
    assert "cell-secret.internal" not in spec_ok_text
