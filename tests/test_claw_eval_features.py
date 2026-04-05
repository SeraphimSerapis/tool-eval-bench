"""Tests for the new Claw-Eval–inspired features: error injection and Pass@k metrics."""

import random

from tool_eval_bench.runner.orchestrator import _INJECTED_ERRORS, _maybe_inject_error


class TestErrorInjection:
    """Verify the controlled error injection for robustness testing."""

    def test_no_injection_at_zero_rate(self) -> None:
        """error_rate=0.0 should never inject errors."""
        original = {"temperature": 14, "condition": "Cloudy"}
        for _ in range(100):
            result = _maybe_inject_error(original, 0.0)
            assert result is original

    def test_always_injects_at_full_rate(self) -> None:
        """error_rate=1.0 should always inject errors."""
        original = {"temperature": 14, "condition": "Cloudy"}
        for _ in range(20):
            result = _maybe_inject_error(original, 1.0)
            assert result in _INJECTED_ERRORS

    def test_injected_errors_have_status(self) -> None:
        """Every injected error has an 'error' message and 'status' code."""
        for err in _INJECTED_ERRORS:
            assert "error" in err
            assert "status" in err
            assert err["status"] in (429, 500, 503)

    def test_probabilistic_injection(self) -> None:
        """At error_rate=0.5, roughly half the results should be errors."""
        random.seed(42)
        original = {"ok": True}
        results = [_maybe_inject_error(original, 0.5) for _ in range(1000)]
        error_count = sum(1 for r in results if r in _INJECTED_ERRORS)
        # Should be roughly 500 ± 50 (very generous band)
        assert 350 < error_count < 650, f"Got {error_count} errors out of 1000"

    def test_negative_rate_no_injection(self) -> None:
        """Negative error rate should behave like 0."""
        original = {"data": "safe"}
        result = _maybe_inject_error(original, -0.5)
        assert result is original


class TestPassAtKMetrics:
    """Verify Pass@k / Pass^k computation in trial aggregation."""

    def test_pass_at_k_computation(self) -> None:
        """Test that Pass@k correctly identifies scenarios that passed at least once."""
        from tool_eval_bench.domain.scenarios import (
            CategoryScore,
            ModelScoreSummary,
            ScenarioResult,
            ScenarioStatus,
        )
        from tool_eval_bench.cli.bench import _aggregate_trials

        # Create 2 trials with 3 scenarios each
        # Scenario A: passes both times → Pass@k=True, Pass^k=True
        # Scenario B: passes once → Pass@k=True, Pass^k=False
        # Scenario C: never passes → Pass@k=False, Pass^k=False
        from tool_eval_bench.domain.scenarios import Category

        def make_summary(results: list[tuple[str, int]]) -> ModelScoreSummary:
            return ModelScoreSummary(
                final_score=50.0,
                total_points=sum(p for _, p in results),
                max_points=6,
                rating="★★ Below Average",
                scenario_results=[
                    ScenarioResult(scenario_id=sid, status=ScenarioStatus.PASS if pts == 2 else ScenarioStatus.FAIL, points=pts, summary="")
                    for sid, pts in results
                ],
                category_scores=[
                    CategoryScore(category=Category.A, label="A", earned=sum(p for _, p in results), max_points=6, percent=50.0),
                ],
            )

        trial_1 = make_summary([("SC-A", 2), ("SC-B", 2), ("SC-C", 0)])
        trial_2 = make_summary([("SC-A", 2), ("SC-B", 0), ("SC-C", 0)])

        agg = _aggregate_trials([trial_1, trial_2])

        assert agg["pass_at_k"] > 0  # At least SC-A and SC-B
        assert agg["pass_hat_k"] > 0  # At least SC-A

        # SC-A: pass@k=True, pass^k=True
        assert agg["per_scenario"]["SC-A"]["pass_at_k"] is True
        assert agg["per_scenario"]["SC-A"]["pass_hat_k"] is True

        # SC-B: pass@k=True, pass^k=False
        assert agg["per_scenario"]["SC-B"]["pass_at_k"] is True
        assert agg["per_scenario"]["SC-B"]["pass_hat_k"] is False

        # SC-C: pass@k=False, pass^k=False
        assert agg["per_scenario"]["SC-C"]["pass_at_k"] is False
        assert agg["per_scenario"]["SC-C"]["pass_hat_k"] is False

        # Overall rates: 2/3 pass@k, 1/3 pass^k
        assert agg["pass_at_k"] == round(100 * 2 / 3, 1)
        assert agg["pass_hat_k"] == round(100 * 1 / 3, 1)
        assert agg["reliability_gap"] == round(agg["pass_at_k"] - agg["pass_hat_k"], 1)

    def test_single_trial_returns_empty(self) -> None:
        """_aggregate_trials returns {} for a single trial (no stats)."""
        from tool_eval_bench.cli.bench import _aggregate_trials

        assert _aggregate_trials([object()]) == {}


class TestResponsivenessCurve:
    """Verify the logistic responsiveness mapping from latency to score."""

    def test_zero_or_negative_returns_100(self) -> None:
        from tool_eval_bench.domain.scenarios import responsiveness_score
        assert responsiveness_score(0) == 100
        assert responsiveness_score(-100) == 100

    def test_instant_response_near_100(self) -> None:
        from tool_eval_bench.domain.scenarios import responsiveness_score
        assert responsiveness_score(500) >= 90

    def test_3s_is_inflection_at_50(self) -> None:
        from tool_eval_bench.domain.scenarios import responsiveness_score
        assert responsiveness_score(3000) == 50

    def test_slow_response_low_score(self) -> None:
        from tool_eval_bench.domain.scenarios import responsiveness_score
        assert responsiveness_score(10000) < 25
        assert responsiveness_score(30000) < 10

    def test_monotonically_decreasing(self) -> None:
        from tool_eval_bench.domain.scenarios import responsiveness_score
        latencies = [100, 500, 1000, 2000, 3000, 5000, 10000, 30000]
        scores = [responsiveness_score(ms) for ms in latencies]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Not monotonic at {latencies[i]}ms"


class TestDeployabilityComposite:
    """Verify the composite quality x speed score."""

    def test_no_latency_returns_none(self) -> None:
        from tool_eval_bench.domain.scenarios import compute_deployability
        d, r, m = compute_deployability(90, None)
        assert d is None and r is None and m is None

    def test_zero_latency_returns_none(self) -> None:
        from tool_eval_bench.domain.scenarios import compute_deployability
        d, r, m = compute_deployability(90, 0.0)
        assert d is None

    def test_fast_high_quality(self) -> None:
        """Fast + high quality -> high deployability."""
        from tool_eval_bench.domain.scenarios import compute_deployability
        d, r, m = compute_deployability(95, 500.0, alpha=0.7)
        assert d is not None
        assert d >= 90
        assert r >= 90

    def test_slow_high_quality(self) -> None:
        """Slow + high quality -> moderate deployability."""
        from tool_eval_bench.domain.scenarios import compute_deployability
        d, r, m = compute_deployability(95, 10000.0, alpha=0.7)
        assert d is not None
        assert d < 80
        assert d > 60

    def test_fast_low_quality(self) -> None:
        """Fast but low quality -> moderate deployability."""
        from tool_eval_bench.domain.scenarios import compute_deployability
        d, r, m = compute_deployability(40, 500.0, alpha=0.7)
        assert d is not None
        assert d < 60

    def test_alpha_weight_effect(self) -> None:
        """Higher alpha -> quality matters more."""
        from tool_eval_bench.domain.scenarios import compute_deployability
        d_high, _, _ = compute_deployability(90, 10000.0, alpha=0.9)
        d_low, _, _ = compute_deployability(90, 10000.0, alpha=0.3)
        assert d_high > d_low

    def test_score_results_with_latency(self) -> None:
        """score_results computes deployability when latency data is present."""
        from tool_eval_bench.domain.scenarios import (
            Category, ScenarioDefinition, ScenarioResult, ScenarioStatus,
        )
        from tool_eval_bench.runner.orchestrator import score_results

        scenarios = [ScenarioDefinition(
            id="TC-01", title="Test", category=Category.A,
            user_message="test", description="test",
            handle_tool_call=lambda s, r: {},
            evaluate=lambda s: ("pass", "ok"),
        )]
        results = [ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS,
            points=2, summary="ok",
            turn_latencies_ms=[1500.0, 2000.0, 1800.0],
        )]
        summary = score_results(results, scenarios)
        assert summary.deployability is not None
        assert summary.responsiveness is not None
        assert summary.median_turn_ms == 1800.0

    def test_score_results_without_latency(self) -> None:
        """score_results skips deployability when no latency data."""
        from tool_eval_bench.domain.scenarios import (
            Category, ScenarioDefinition, ScenarioResult, ScenarioStatus,
        )
        from tool_eval_bench.runner.orchestrator import score_results

        scenarios = [ScenarioDefinition(
            id="TC-01", title="Test", category=Category.A,
            user_message="test", description="test",
            handle_tool_call=lambda s, r: {},
            evaluate=lambda s: ("pass", "ok"),
        )]
        results = [ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS,
            points=2, summary="ok",
        )]
        summary = score_results(results, scenarios)
        assert summary.deployability is None
        assert summary.responsiveness is None
