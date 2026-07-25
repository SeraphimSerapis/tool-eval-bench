"""Infrastructure failures must not be scored as model incompetence.

A timeout, connection error, or 5xx from the serving endpoint says nothing
about the model's tool-calling ability.  These scenarios are dropped from both
the numerator and the denominator of the quality score and reported separately
as a completion rate.

Also covers the adapter's retry/backoff behaviour for transient statuses.
"""

from __future__ import annotations

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import (
    OpenAICompatibleAdapter,
    _backoff_delay,
    _retry_after_seconds,
)
from tool_eval_bench.domain.adapters import RETRYABLE_STATUS_CODES
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.runner.orchestrator import _classify_runtime_error, score_results


def _scenario(sid: str, category: Category = Category.A) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=sid,
        title=sid,
        category=category,
        user_message="",
        description="",
        handle_tool_call=lambda state, call: None,
        evaluate=lambda state: None,  # type: ignore[arg-type,return-value]
    )


def _result(
    sid: str,
    status: ScenarioStatus,
    points: int,
    failure_kind: str | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        status=status,
        points=points,
        summary="",
        failure_kind=failure_kind,
    )


class TestInfrastructureExclusion:
    def test_timeout_is_dropped_from_numerator_and_denominator(self) -> None:
        scenarios = [_scenario("S1"), _scenario("S2"), _scenario("S3")]
        results = [
            _result("S1", ScenarioStatus.PASS, 2),
            _result("S2", ScenarioStatus.PASS, 2),
            _result("S3", ScenarioStatus.FAIL, 0, FailureKind.TIMEOUT),
        ]

        summary = score_results(results, scenarios=scenarios)

        # 4/4 rather than 4/6 — the timed-out scenario is not graded at all.
        assert summary.max_points == 4
        assert summary.total_points == 4
        assert summary.final_score == 100
        assert summary.excluded_scenarios == ["S3"]
        assert summary.completion_rate == pytest.approx(66.7)

    @pytest.mark.parametrize(
        "kind",
        [FailureKind.TIMEOUT, FailureKind.CONNECTION_ERROR, FailureKind.SERVER_ERROR],
    )
    def test_every_infra_kind_is_excluded(self, kind: str) -> None:
        summary = score_results(
            [_result("S1", ScenarioStatus.FAIL, 0, kind)],
            scenarios=[_scenario("S1")],
        )
        assert summary.excluded_scenarios == ["S1"]
        assert summary.max_points == 0
        assert summary.final_score == 0

    @pytest.mark.parametrize(
        "kind",
        [
            FailureKind.WRONG_TOOL,
            FailureKind.WRONG_ARGS,
            FailureKind.MISSING_STEP,
            FailureKind.FORBIDDEN_ACTION,
            FailureKind.BUDGET_EXCEEDED,
            FailureKind.MODEL_CRASH,
            None,
        ],
    )
    def test_model_failures_are_still_scored_as_zero(self, kind: str | None) -> None:
        scenarios = [_scenario("S1"), _scenario("S2")]
        results = [
            _result("S1", ScenarioStatus.PASS, 2),
            _result("S2", ScenarioStatus.FAIL, 0, kind),
        ]
        summary = score_results(results, scenarios=scenarios)

        assert summary.excluded_scenarios == []
        assert summary.max_points == 4
        assert summary.final_score == 50
        assert summary.completion_rate == 100.0

    def test_category_denominator_shrinks_for_excluded_scenarios(self) -> None:
        scenarios = [
            _scenario("A1", Category.A),
            _scenario("A2", Category.A),
            _scenario("B1", Category.B),
        ]
        results = [
            _result("A1", ScenarioStatus.PASS, 2),
            _result("A2", ScenarioStatus.FAIL, 0, FailureKind.SERVER_ERROR),
            _result("B1", ScenarioStatus.PASS, 2),
        ]
        summary = score_results(results, scenarios=scenarios)

        by_cat = {cs.category: cs for cs in summary.category_scores}
        assert by_cat[Category.A].max_points == 2
        assert by_cat[Category.A].percent == 100.0
        assert by_cat[Category.A].fail_count == 0

    def test_excluded_results_are_still_reported(self) -> None:
        """Transparency: the run report must still show what was skipped."""
        scenarios = [_scenario("S1"), _scenario("S2")]
        results = [
            _result("S1", ScenarioStatus.PASS, 2),
            _result("S2", ScenarioStatus.FAIL, 0, FailureKind.TIMEOUT),
        ]
        summary = score_results(results, scenarios=scenarios)

        assert [r.scenario_id for r in summary.scenario_results] == ["S1", "S2"]
        assert summary.to_dict()["excluded_scenarios"] == ["S2"]
        assert summary.to_dict()["completion_rate"] == pytest.approx(50.0)

    def test_clean_run_omits_exclusion_keys(self) -> None:
        summary = score_results(
            [_result("S1", ScenarioStatus.PASS, 2)], scenarios=[_scenario("S1")]
        )
        assert "excluded_scenarios" not in summary.to_dict()
        assert "completion_rate" not in summary.to_dict()

    def test_timed_out_latencies_do_not_skew_responsiveness(self) -> None:
        scenarios = [_scenario("S1"), _scenario("S2")]
        fast = _result("S1", ScenarioStatus.PASS, 2)
        fast.turn_latencies_ms = [100.0]
        slow = _result("S2", ScenarioStatus.FAIL, 0, FailureKind.TIMEOUT)
        slow.turn_latencies_ms = [120_000.0]

        summary = score_results([fast, slow], scenarios=scenarios)

        assert summary.median_turn_ms == 100.0

    def test_partial_status_is_never_treated_as_infra(self) -> None:
        """Only a FAIL can be an infrastructure exclusion."""
        partial = _result("S1", ScenarioStatus.PARTIAL, 1, FailureKind.TIMEOUT)
        assert partial.is_infrastructure_failure is False


class TestErrorClassification:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS_CODES))
    def test_retryable_statuses_classify_as_server_error(self, status: int) -> None:
        exc = httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("POST", "http://x/v1/chat/completions"),
            response=httpx.Response(status),
        )
        assert _classify_runtime_error(exc) == FailureKind.SERVER_ERROR

    def test_client_error_still_classifies_as_wrong_args(self) -> None:
        exc = httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("POST", "http://x/v1/chat/completions"),
            response=httpx.Response(422),
        )
        assert _classify_runtime_error(exc) == FailureKind.WRONG_ARGS

    @pytest.mark.parametrize(
        "exc",
        [httpx.ReadError("reset"), httpx.RemoteProtocolError("truncated")],
    )
    def test_transport_errors_classify_as_connection_error(self, exc: Exception) -> None:
        assert _classify_runtime_error(exc) == FailureKind.CONNECTION_ERROR


class TestRetryBackoff:
    def test_default_timeout_is_shared_and_generous(self) -> None:
        from tool_eval_bench.runner.orchestrator import DEFAULT_TIMEOUT_SECONDS

        assert DEFAULT_TIMEOUT_SECONDS == DEFAULT_REQUEST_TIMEOUT_SECONDS
        assert DEFAULT_REQUEST_TIMEOUT_SECONDS >= 120.0

    def test_retry_after_header_is_honored_when_sane(self) -> None:
        resp = httpx.Response(429, headers={"Retry-After": "1.5"})
        assert _retry_after_seconds(resp) == 1.5
        assert _backoff_delay(0, resp) == 1.5

    @pytest.mark.parametrize("raw", ["Wed, 21 Oct 2015 07:28:00 GMT", "-1", "9999", "abc"])
    def test_absurd_or_unparseable_retry_after_falls_back_to_backoff(self, raw: str) -> None:
        resp = httpx.Response(429, headers={"Retry-After": raw})
        assert _retry_after_seconds(resp) is None
        assert 0.0 <= _backoff_delay(0, resp) <= 0.5

    def test_backoff_is_bounded_and_grows(self) -> None:
        assert 0.0 <= _backoff_delay(0) <= 0.5
        assert 0.0 <= _backoff_delay(1) <= 1.0
        assert 0.0 <= _backoff_delay(20) <= 8.0

    @pytest.mark.asyncio
    async def test_transient_503_is_retried_then_succeeds(self, monkeypatch) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(503, text="overloaded")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hi", "role": "assistant"}}]},
            )

        adapter = OpenAICompatibleAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            "tool_eval_bench.adapters.openai_compat._backoff_delay", lambda *a, **k: 0.0
        )

        result = await adapter.chat_completion(
            model="m", messages=[], base_url="http://x", tools=None
        )

        assert result.content == "hi"
        assert attempts["n"] == 3
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_persistent_503_exhausts_budget_and_propagates(self, monkeypatch) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503, text="overloaded")

        adapter = OpenAICompatibleAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            "tool_eval_bench.adapters.openai_compat._backoff_delay", lambda *a, **k: 0.0
        )

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await adapter.chat_completion(model="m", messages=[], base_url="http://x", tools=None)

        # Classified as infrastructure, so the scenario is excluded rather than failed.
        assert _classify_runtime_error(excinfo.value) == FailureKind.SERVER_ERROR
        assert attempts["n"] == 3
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_rate_limit_is_not_fed_back_to_the_model_as_content(self, monkeypatch) -> None:
        """A 429 used to surface as assistant content, scoring the model down."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="slow down")

        adapter = OpenAICompatibleAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            "tool_eval_bench.adapters.openai_compat._backoff_delay", lambda *a, **k: 0.0
        )

        with pytest.raises(httpx.HTTPStatusError):
            await adapter.chat_completion(model="m", messages=[], base_url="http://x", tools=None)
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_client_error_is_still_returned_gracefully(self) -> None:
        """400/422 stay graceful — they are usually the model's malformed args."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, text="bad tool arguments")

        adapter = OpenAICompatibleAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = await adapter.chat_completion(
            model="m", messages=[], base_url="http://x", tools=None
        )

        assert "[server error 422]" in result.content
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_timeouts_are_not_retried(self) -> None:
        """Retrying a spent timeout budget would multiply run wall-clock time."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            raise httpx.ReadTimeout("too slow", request=request)

        adapter = OpenAICompatibleAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with pytest.raises(httpx.ReadTimeout):
            await adapter.chat_completion(model="m", messages=[], base_url="http://x", tools=None)

        assert attempts["n"] == 1
        await adapter.aclose()
