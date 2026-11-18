"""Tests for responsiveness scoring and deployability composite metrics."""


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
            Category,
            ScenarioDefinition,
            ScenarioResult,
            ScenarioStatus,
        )
        from tool_eval_bench.runner.orchestrator import score_results

        scenarios = [
            ScenarioDefinition(
                id="TC-01",
                title="Test",
                category=Category.A,
                user_message="test",
                description="test",
                handle_tool_call=lambda s, r: {},
                evaluate=lambda s: ("pass", "ok"),
            )
        ]
        results = [
            ScenarioResult(
                scenario_id="TC-01",
                status=ScenarioStatus.PASS,
                points=2,
                summary="ok",
                turn_latencies_ms=[1500.0, 2000.0, 1800.0],
            )
        ]
        summary = score_results(results, scenarios)
        assert summary.deployability is not None
        assert summary.responsiveness is not None
        assert summary.median_turn_ms == 1800.0

    def test_score_results_without_latency(self) -> None:
        """score_results skips deployability when no latency data."""
        from tool_eval_bench.domain.scenarios import (
            Category,
            ScenarioDefinition,
            ScenarioResult,
            ScenarioStatus,
        )
        from tool_eval_bench.runner.orchestrator import score_results

        scenarios = [
            ScenarioDefinition(
                id="TC-01",
                title="Test",
                category=Category.A,
                user_message="test",
                description="test",
                handle_tool_call=lambda s, r: {},
                evaluate=lambda s: ("pass", "ok"),
            )
        ]
        results = [
            ScenarioResult(
                scenario_id="TC-01",
                status=ScenarioStatus.PASS,
                points=2,
                summary="ok",
            )
        ]
        summary = score_results(results, scenarios)
        assert summary.deployability is None
        assert summary.responsiveness is None
