"""Rate-limit handling in the OpenAI-compatible adapter.

Covers the 429-specific retry budget, Retry-After handling across a full quota
window, and the adaptive pacing that keeps a benchmark run alive against a
hosted endpoint with per-minute quotas.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tool_eval_bench.adapters.http_retry import _daily_quota_exhaustion
from tool_eval_bench.adapters.openai_compat import (
    DEFAULT_MAX_RATE_LIMIT_RETRIES,
    DEFAULT_MAX_RETRIES,
    RateLimitCoordinator,
    _rate_limit_delay,
    _retry_after_seconds,
)
from tool_eval_bench.adapters.openai_compat import (
    OpenAICompatibleAdapter as Adapter,
)

_OK = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

# A real response body from Google's Gemini API for a daily free-tier quota.
_DAILY_QUOTA_BODY = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota... Please retry in 36.36s.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaDimensions": {"location": "global", "model": "gemini-3.7-flash"},
                        "quotaValue": "20",
                    }
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s"},
        ],
    }
}

# A per-minute quota — the ordinary, retryable case.
_PER_MINUTE_QUOTA_BODY = {
    "error": {
        "code": 429,
        "message": "Quota exceeded.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                        "quotaValue": "15",
                    }
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"},
        ],
    }
}


def _adapter(handler, *, paced: bool = False, **kwargs) -> Adapter:
    """Adapter over a mock transport, with pacing disabled unless asked for.

    Pacing sleeps for real; tests that care about it drive the coordinator
    directly rather than paying its wall-clock cost on every retry test.
    """
    adapter = Adapter(**kwargs)
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if not paced:
        adapter._rate_limits = RateLimitCoordinator(
            max_interval_seconds=0.0,
            step_seconds=0.0,
            observer=kwargs.get("rate_limit_observer"),
        )
    return adapter


@pytest.fixture
def no_sleep(monkeypatch) -> None:
    """Make backoff instant so tests exercise control flow, not wall clock."""
    monkeypatch.setattr(
        "tool_eval_bench.adapters.http_retry._rate_limit_delay", lambda *a, **k: 0.0
    )
    monkeypatch.setattr("tool_eval_bench.adapters.http_retry._backoff_delay", lambda *a, **k: 0.0)


class TestRateLimitBackoff:
    def test_rate_limit_budget_is_larger_than_generic_budget(self) -> None:
        """A per-minute quota outlasts the generic transient window."""
        assert DEFAULT_MAX_RATE_LIMIT_RETRIES > DEFAULT_MAX_RETRIES

    def test_retry_after_beyond_generic_cap_is_honored_for_rate_limits(self) -> None:
        resp = httpx.Response(429, headers={"Retry-After": "30"})
        assert _retry_after_seconds(resp) is None  # generic cap rejects it
        assert _retry_after_seconds(resp, cap=60.0) == 30.0
        assert _rate_limit_delay(0, resp) == 30.0

    def test_absurd_retry_after_falls_back_to_computed_backoff(self) -> None:
        resp = httpx.Response(429, headers={"Retry-After": "9999"})
        assert 1.0 <= _rate_limit_delay(0, resp) <= 2.0

    def test_backoff_grows_and_is_capped(self) -> None:
        assert 1.0 <= _rate_limit_delay(0) <= 2.0
        assert 2.0 <= _rate_limit_delay(1) <= 4.0
        assert 30.0 <= _rate_limit_delay(20) <= 60.0

    def test_backoff_never_returns_a_useless_near_zero_sleep(self) -> None:
        assert all(_rate_limit_delay(0) >= 1.0 for _ in range(50))

    def test_retry_delay_parsed_from_gemini_body_when_no_header(self) -> None:
        """Gemini sends RetryInfo.retryDelay in the JSON body, not Retry-After."""
        resp = httpx.Response(429, json=_PER_MINUTE_QUOTA_BODY)
        assert _retry_after_seconds(resp, cap=60.0) == 2.0
        assert _rate_limit_delay(0, resp) == 2.0

    def test_header_wins_over_body_when_both_present(self) -> None:
        resp = httpx.Response(429, headers={"Retry-After": "5"}, json=_PER_MINUTE_QUOTA_BODY)
        assert _retry_after_seconds(resp, cap=60.0) == 5.0

    def test_malformed_body_does_not_crash_retry_parsing(self) -> None:
        resp = httpx.Response(429, content=b"not json")
        assert _retry_after_seconds(resp, cap=60.0) is None


class TestDailyQuotaExhaustion:
    def test_daily_quota_is_detected(self) -> None:
        resp = httpx.Response(429, json=_DAILY_QUOTA_BODY)
        reason = _daily_quota_exhaustion(resp)
        assert reason is not None
        assert "gemini-3.7-flash" in reason
        assert "20" in reason

    def test_per_minute_quota_is_not_daily_exhaustion(self) -> None:
        resp = httpx.Response(429, json=_PER_MINUTE_QUOTA_BODY)
        assert _daily_quota_exhaustion(resp) is None

    def test_non_quota_429_is_not_daily_exhaustion(self) -> None:
        resp = httpx.Response(429, text="slow down")
        assert _daily_quota_exhaustion(resp) is None

    def test_ordinary_openai_error_body_is_not_daily_exhaustion(self) -> None:
        resp = httpx.Response(429, json={"error": {"message": "rate limited", "type": "..."}})
        assert _daily_quota_exhaustion(resp) is None

    @pytest.mark.asyncio
    async def test_daily_quota_fails_fast_without_burning_the_retry_budget(self, no_sleep) -> None:
        """A daily quota cannot recover within any retry budget; retrying wastes minutes
        and would otherwise surface as an unexplained scenario timeout."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(429, json=_DAILY_QUOTA_BODY)

        adapter = _adapter(handler, max_rate_limit_retries=6)
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert excinfo.value.response.status_code == 429
        assert attempts["n"] == 1  # no retries at all
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_per_minute_quota_still_retries_normally(self, no_sleep) -> None:
        """Only the daily case should fail fast; ordinary rate limits still recover."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, json=_PER_MINUTE_QUOTA_BODY)
            return httpx.Response(200, json=_OK)

        adapter = _adapter(handler)
        result = await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert result.content == "hi"
        assert attempts["n"] == 2
        await adapter.aclose()


class TestRateLimitRetries:
    @pytest.mark.asyncio
    async def test_run_survives_a_burst_of_rate_limits(self, no_sleep) -> None:
        """The generic budget (2) would have failed this scenario; 429s get more."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] <= 4:
                return httpx.Response(429, text="quota exceeded")
            return httpx.Response(200, json=_OK)

        adapter = _adapter(handler)
        result = await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert result.content == "hi"
        assert attempts["n"] == 5
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_persistent_rate_limit_still_propagates(self, no_sleep) -> None:
        """A permanently throttled endpoint is infrastructure, not a model fail."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(429, text="quota exceeded")

        adapter = _adapter(handler, max_rate_limit_retries=3)
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert excinfo.value.response.status_code == 429
        assert attempts["n"] == 4
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_rate_limits_do_not_consume_the_generic_budget(self, no_sleep) -> None:
        """429s then a 503 — the 503 must still get its own retries."""
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(len(seen))
            if len(seen) <= 3:
                return httpx.Response(429, text="slow down")
            if len(seen) <= 5:
                return httpx.Response(503, text="overloaded")
            return httpx.Response(200, json=_OK)

        adapter = _adapter(handler)
        result = await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert result.content == "hi"
        assert len(seen) == 6
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_streamed_rate_limit_is_retried(self, no_sleep) -> None:
        attempts = {"n": 0}
        body = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, text="slow down")
            return httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            )

        adapter = _adapter(handler)
        result = await adapter.chat_completion(
            model="m", messages=[], base_url="http://x", stream=True
        )

        assert result.content == "hi"
        assert attempts["n"] == 2
        await adapter.aclose()


# `asyncio.sleep` schedules against the event loop clock and can return
# fractionally early. On Windows that clock ticks about every 15.6ms, so a run
# of N sleeps measures up to N ticks short of the nominal total: the pacing
# assertion below saw 0.187 against a nominal 0.2 on every windows-latest run,
# because the CI matrix pins that runner's seed.
#
# These two assertions are about whether pacing happened at all. Without it
# both measure roughly zero, so allowing one tick per sleep costs them nothing.
_CLOCK_SLACK_SECONDS = 0.016


class TestAdaptivePacing:
    @pytest.mark.asyncio
    async def test_pacing_stays_off_until_a_rate_limit_is_seen(self) -> None:
        """Local endpoints must not pay for hosted-quota handling."""
        adapter = _adapter(lambda request: httpx.Response(200, json=_OK), paced=True)
        for _ in range(3):
            await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert adapter._rate_limits.min_interval == 0.0
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_pacing_widens_on_each_rate_limit(self) -> None:
        coordinator = RateLimitCoordinator(max_interval_seconds=4.0, step_seconds=0.5)

        await coordinator.on_rate_limited(0.0)
        assert coordinator.min_interval == 0.5
        await coordinator.on_rate_limited(0.0)
        assert coordinator.min_interval == 1.0

        for _ in range(10):
            await coordinator.on_rate_limited(0.0)
        assert coordinator.min_interval == 4.0  # capped

    @pytest.mark.asyncio
    async def test_pacing_decays_after_sustained_success(self) -> None:
        coordinator = RateLimitCoordinator(step_seconds=0.5)
        await coordinator.on_rate_limited(0.0)
        await coordinator.on_rate_limited(0.0)
        assert coordinator.min_interval == 1.0

        for _ in range(8):
            await coordinator.on_success()
        assert coordinator.min_interval == 0.5

        for _ in range(8):
            await coordinator.on_success()
        assert coordinator.min_interval == 0.0  # below one step → unthrottled

    @pytest.mark.asyncio
    async def test_pause_holds_concurrent_requests_together(self) -> None:
        """One request's 429 must slow the whole run, not just that request."""
        coordinator = RateLimitCoordinator()
        await coordinator.on_rate_limited(0.25)

        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(coordinator.acquire() for _ in range(4)))
        waited = asyncio.get_running_loop().time() - started

        assert waited >= 0.25 - _CLOCK_SLACK_SECONDS

    @pytest.mark.asyncio
    async def test_paced_requests_are_spaced_out(self) -> None:
        coordinator = RateLimitCoordinator(step_seconds=0.1)
        await coordinator.on_rate_limited(0.0)  # min_interval = 0.1

        started = asyncio.get_running_loop().time()
        for _ in range(3):
            await coordinator.acquire()
        elapsed = asyncio.get_running_loop().time() - started

        # First acquire is free, the next two wait one step each. Two sleeps, so
        # the measurement can fall two ticks short of the nominal 0.2.
        assert elapsed >= 0.2 - 2 * _CLOCK_SLACK_SECONDS


class TestDisplayReporting:
    """Throttling belongs in the progress footer, not in log lines between results."""

    @pytest.mark.asyncio
    async def test_observer_receives_status_updates(self, no_sleep) -> None:
        seen: list[tuple[int, float]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if len(seen) < 2:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json=_OK)

        adapter = _adapter(handler)
        adapter._rate_limits = RateLimitCoordinator(max_interval_seconds=0.01, step_seconds=0.001)
        adapter.set_rate_limit_observer(lambda s: seen.append((s.retries, s.pacing_seconds)))

        await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert [retries for retries, _ in seen] == [1, 2]
        assert seen[-1][1] > seen[0][1]  # pacing widened
        assert adapter.rate_limit_status.retries == 2
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_a_broken_observer_never_breaks_the_run(self, no_sleep) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json=_OK)

        def boom(status) -> None:
            raise RuntimeError("display exploded")

        adapter = _adapter(handler, rate_limit_observer=boom)
        result = await adapter.chat_completion(model="m", messages=[], base_url="http://x")

        assert result.content == "hi"
        await adapter.aclose()

    def test_display_shows_throttling_in_footer_and_summary(self) -> None:
        from tool_eval_bench.adapters.openai_compat import RateLimitStatus
        from tool_eval_bench.cli.display import BenchmarkDisplay

        display = BenchmarkDisplay("m", "vllm", "http://x", scenarios=[])
        assert "rate limit" not in display._build_footer().plain.lower()

        display.on_rate_limit(RateLimitStatus(retries=3, total_wait_seconds=12.0, pacing_seconds=1))
        footer = display._build_footer().plain

        assert "3 rate limits" in footer
        assert "pacing 1.0s" in footer
