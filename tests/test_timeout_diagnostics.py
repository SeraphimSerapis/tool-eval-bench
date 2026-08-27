"""Infrastructure failures: how they render, what they advise, how they are bounded.

Covers the three halves of the timeout problem. A scenario dropped from scoring
must not render as a verdict, the run must say what to change, and a turn that
is not streamed must be bounded by what the model actually demonstrated rather
than by a number calibrated for a streamed turn.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from tool_eval_bench.cli.display import BenchmarkDisplay
from tool_eval_bench.cli.timeout_advice import (
    infrastructure_note,
    suggested_timeout,
    timeout_advice,
)
from tool_eval_bench.domain.scenarios import (
    FailureKind,
    ModelScoreSummary,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.domain.timeouts import unstreamed_turn_timeout
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE


def _result(
    *,
    kind: str | None = FailureKind.TIMEOUT,
    status: ScenarioStatus = ScenarioStatus.FAIL,
    summary: str = "",
    latencies: list[float] | None = None,
    points: int = 0,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="TC-88",
        status=status,
        points=points,
        summary=summary,
        failure_kind=kind,
        duration_seconds=271.3,
        ttft_ms=490.2,
        turn_count=1,
        turn_latencies_ms=latencies if latencies is not None else [151199.4],
    )


def _summary(results: list[ScenarioResult]) -> ModelScoreSummary:
    return ModelScoreSummary(
        scenario_results=results,
        category_scores=[],
        final_score=0,
        total_points=0,
        max_points=0,
        rating="★ Poor",
        excluded_scenarios=[r.scenario_id for r in results if r.is_infrastructure_failure],
    )


# ---------------------------------------------------------------------------
# Result line
# ---------------------------------------------------------------------------


class TestResultLine:
    def _line(self, result: ScenarioResult) -> str:
        scenario = next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == "TC-88")
        display = BenchmarkDisplay("m", "vllm", "http://x", scenarios=[scenario])
        console = Console(record=True, width=200)
        console.print(display._format_result_line(scenario, result))
        return console.export_text()

    def test_timeout_is_not_labelled_a_failure(self) -> None:
        line = self._line(_result())
        assert "TIMEOUT" in line
        assert "FAIL" not in line

    def test_timeout_does_not_claim_zero_points(self) -> None:
        # 0/2 would say the scenario scored zero. It left the denominator too.
        line = self._line(_result())
        assert "0/2" not in line
        assert "–/2" in line

    def test_timeout_states_why_the_line_is_there(self) -> None:
        assert "excluded from scoring" in self._line(_result())

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (FailureKind.TIMEOUT, "TIMEOUT"),
            (FailureKind.CONNECTION_ERROR, "NO CONN"),
            (FailureKind.SERVER_ERROR, "SERVER"),
        ],
    )
    def test_each_infrastructure_kind_has_its_own_label(self, kind: str, expected: str) -> None:
        assert expected in self._line(_result(kind=kind))

    def test_model_failure_still_renders_as_a_failure(self) -> None:
        line = self._line(
            _result(
                kind=FailureKind.MISSING_STEP, summary="Returned extra text.", latencies=[286.5]
            )
        )
        assert "FAIL" in line
        assert "0/2" in line
        assert "Returned extra text." in line

    def test_pass_is_untouched(self) -> None:
        line = self._line(
            _result(kind=None, status=ScenarioStatus.PASS, points=2, summary="Did the thing.")
        )
        assert "PASS" in line
        assert "2/2" in line


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------


class TestInfrastructureNote:
    def test_falls_back_to_a_reason_when_there_is_no_summary(self) -> None:
        assert infrastructure_note(_result()) == "excluded from scoring (request timed out)"

    def test_a_real_summary_wins(self) -> None:
        assert infrastructure_note(_result(summary="Server said no.")) == "Server said no."

    def test_unknown_kind_still_gets_a_note(self) -> None:
        assert "excluded from scoring" in infrastructure_note(_result(kind="something_new"))


class TestTimeoutAdvice:
    def test_silent_when_nothing_timed_out(self) -> None:
        clean = _result(kind=None, status=ScenarioStatus.PASS, points=2, summary="ok")
        assert timeout_advice(_summary([clean]), timeout_seconds=120) == []

    def test_silent_for_a_connection_error(self) -> None:
        # Raising --timeout would not help, so do not suggest it.
        conn = _result(kind=FailureKind.CONNECTION_ERROR)
        assert timeout_advice(_summary([conn]), timeout_seconds=120) == []

    def test_names_the_observed_latency_and_current_timeout(self) -> None:
        text = " ".join(timeout_advice(_summary([_result()]), timeout_seconds=120))
        assert "151s" in text
        assert "currently 120" in text

    def test_suggests_a_timeout_that_clears_the_slowest_turn(self) -> None:
        text = " ".join(timeout_advice(_summary([_result()]), timeout_seconds=120))
        assert "--timeout 420" in text

    def test_offers_a_way_to_shorten_generation(self) -> None:
        text = " ".join(timeout_advice(_summary([_result()]), timeout_seconds=120))
        assert "--no-think" in text
        assert "max_tokens" in text

    def test_explains_the_streaming_asymmetry(self) -> None:
        text = " ".join(timeout_advice(_summary([_result()]), timeout_seconds=120))
        assert "first turn is streamed" in text

    def test_works_without_a_known_timeout(self) -> None:
        text = " ".join(timeout_advice(_summary([_result()]), timeout_seconds=None))
        assert "higher --timeout" in text

    def test_works_without_latency_data(self) -> None:
        text = " ".join(timeout_advice(_summary([_result(latencies=[])]), timeout_seconds=120))
        assert text
        assert "--timeout" in text

    def test_counts_every_timed_out_scenario(self) -> None:
        text = " ".join(timeout_advice(_summary([_result(), _result()]), timeout_seconds=120))
        assert "2 scenario(s) timed out" in text


class TestSuggestedTimeout:
    def test_clears_the_observed_turn(self) -> None:
        assert suggested_timeout(151_199, 120) >= 151 * 2

    def test_rounds_to_a_typeable_number(self) -> None:
        assert suggested_timeout(151_199, 120) % 60 == 0

    def test_never_suggests_lowering_the_configured_value(self) -> None:
        assert suggested_timeout(1_000, 600) >= 600


# ---------------------------------------------------------------------------
# Unstreamed turn budget
# ---------------------------------------------------------------------------


class TestUnstreamedTurnTimeout:
    def test_slow_first_turn_widens_the_budget(self) -> None:
        # 151s on turn 1 must not leave turn 2 bounded by 120s.
        assert unstreamed_turn_timeout(120, 151_199) == pytest.approx(151.199 * 2.5)

    def test_fast_first_turn_keeps_the_configured_timeout(self) -> None:
        assert unstreamed_turn_timeout(120, 300) == 120

    def test_never_narrows_the_configured_timeout(self) -> None:
        assert unstreamed_turn_timeout(600, 1_000) == 600

    @pytest.mark.parametrize("first_turn_ms", [0.0, -1.0])
    def test_no_measurement_leaves_the_timeout_alone(self, first_turn_ms: float) -> None:
        assert unstreamed_turn_timeout(120, first_turn_ms) == 120

    def test_the_reported_case_now_fits(self) -> None:
        # All three GLM-5.3-Flash runs: turn 1 between 131s and 180s, turn 2
        # killed at exactly the 120s default.
        for turn_one_seconds in (131.4, 151.2, 179.9):
            budget = unstreamed_turn_timeout(120, turn_one_seconds * 1000)
            assert budget > turn_one_seconds, "turn 2 must get more room than turn 1 took"
