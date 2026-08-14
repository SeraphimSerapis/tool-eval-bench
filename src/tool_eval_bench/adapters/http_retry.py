"""Shared HTTP retry, backoff, and rate-limit machinery for backend adapters.

Both wire formats the benchmark speaks — OpenAI-compatible
(``/v1/chat/completions``) and native Gemini (``:generateContent``) — hit the
same operational realities: transient gateway errors and, on hosted endpoints,
per-minute quotas.  The retry policy, the adaptive pacing, and the HTTP client
lifecycle live here so both adapters behave identically under pressure.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from tool_eval_bench.domain.adapters import RETRYABLE_STATUS_CODES, ChatCompletionResult
from tool_eval_bench.utils.urls import redact_url as _redact_url

logger = logging.getLogger(__name__)

# Retries apply to fast failures only.  Read timeouts are deliberately NOT
# retried: the budget is already spent, and a retry would multiply the run's
# wall-clock time.  Timeouts are instead excluded from quality scoring — see
# domain.scenarios.INFRASTRUCTURE_FAILURE_KINDS.
DEFAULT_MAX_RETRIES = 2
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0

# Rate limits get their own, much more patient budget.  A hosted endpoint with
# a per-minute quota (Gemini, OpenAI, Anthropic free/low tiers) will 429 an
# otherwise healthy benchmark run, and the generic transient budget above
# burns through in under two seconds — far shorter than the quota window.
DEFAULT_MAX_RATE_LIMIT_RETRIES = 6
_RATE_LIMIT_BASE_SECONDS = 2.0
_RATE_LIMIT_CAP_SECONDS = 60.0

# Adaptive pacing.  Backing off only for the failed request is not enough when
# the limit is a request *rate*: the next request runs straight back into it.
# After a 429 the adapter spaces subsequent requests out, then relaxes again
# once the endpoint stays healthy.
_PACING_STEP_SECONDS = 0.5
_PACING_MAX_SECONDS = 10.0
_PACING_DECAY_AFTER_SUCCESSES = 8


def _error_body(response: httpx.Response) -> dict | None:
    """Best-effort parse of a JSON error body; None if there isn't one."""
    try:
        data = response.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _retry_delay_from_body(response: httpx.Response) -> float | None:
    """Parse Google's ``RetryInfo.retryDelay`` (e.g. ``"2s"``) from an error body.

    Only the native Gemini API sends this — OpenAI-compatible endpoints send
    ``Retry-After`` as a header instead — but checking a body that doesn't have
    it is a harmless no-op, so both wire formats can share this helper.
    """
    data = _error_body(response)
    if data is None:
        return None
    for detail in ((data.get("error") or {}).get("details")) or []:
        if not isinstance(detail, dict):
            continue
        if not str(detail.get("@type", "")).endswith("RetryInfo"):
            continue
        raw = detail.get("retryDelay")
        if isinstance(raw, str) and raw.endswith("s"):
            try:
                return float(raw[:-1])
            except ValueError:
                return None
    return None


def _retry_after_seconds(
    response: httpx.Response,
    cap: float = _BACKOFF_CAP_SECONDS,
) -> float | None:
    """Parse a retry hint from the ``Retry-After`` header or a JSON error body."""
    raw = response.headers.get("Retry-After")
    delay: float | None = None
    if raw:
        try:
            delay = float(raw.strip())
        except ValueError:
            delay = None  # HTTP-date form; fall back to computed backoff
    if delay is None:
        delay = _retry_delay_from_body(response)
    if delay is None:
        return None
    if delay < 0 or delay > cap:
        return None
    return delay


def _daily_quota_exhaustion(response: httpx.Response) -> str | None:
    """Describe a 429 as unrecoverable if it names a *daily* quota, else None.

    Google's Gemini API reports both per-minute and per-day quotas as a plain
    HTTP 429 with ``RESOURCE_EXHAUSTED``, and even attaches a ``RetryInfo``
    delay to the daily case — but that delay only reflects request pacing
    within the window, not when the daily quota itself resets. Retrying (for
    up to a few minutes, per the rate-limit budget) cannot succeed until then,
    so it is better to fail the request immediately with a clear reason than
    to silently burn the retry budget and surface as a generic timeout.
    """
    data = _error_body(response)
    if data is None:
        return None
    error = data.get("error") or {}
    if error.get("status") != "RESOURCE_EXHAUSTED":
        return None
    for detail in error.get("details") or []:
        if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith(
            "QuotaFailure"
        ):
            continue
        for violation in detail.get("violations") or []:
            quota_id = str(violation.get("quotaId", ""))
            if "PerDay" in quota_id:
                dims = violation.get("quotaDimensions") or {}
                model = dims.get("model")
                limit = violation.get("quotaValue")
                where = f" for {model}" if model else ""
                how_many = f" (limit: {limit}/day)" if limit else ""
                return f"Daily quota exhausted{where}{how_many}"
    return None


def _backoff_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Exponential backoff with full jitter, honoring a sane Retry-After."""
    if response is not None:
        hinted = _retry_after_seconds(response)
        if hinted is not None:
            return hinted
    window = min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_CAP_SECONDS)
    return random.uniform(0.0, window)


def _rate_limit_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Backoff for HTTP 429, honoring Retry-After up to a full quota window.

    Uses half jitter rather than full jitter: a near-zero sleep is wasted on a
    rate limit, but some spread is still needed so concurrent scenarios do not
    retry in lockstep.
    """
    if response is not None:
        hinted = _retry_after_seconds(response, cap=_RATE_LIMIT_CAP_SECONDS)
        if hinted is not None:
            return hinted
    window = min(_RATE_LIMIT_BASE_SECONDS * (2**attempt), _RATE_LIMIT_CAP_SECONDS)
    return window / 2 + random.uniform(0.0, window / 2)


@dataclass(frozen=True)
class RateLimitStatus:
    """Snapshot of rate-limit pressure, for progress display and reporting."""

    retries: int
    total_wait_seconds: float
    pacing_seconds: float


RateLimitObserver = Callable[[RateLimitStatus], None]


class RateLimitCoordinator:
    """Shared rate-limit state for all requests issued by one adapter.

    Two mechanisms, both no-ops until the endpoint actually returns a 429:

    * a **pause** — every in-flight request waits out a rate-limit window
      observed by any one of them, instead of each discovering it separately;
    * **adaptive pacing** — a minimum interval between request starts that
      doubles on each rate limit and decays after sustained success.

    An optional observer receives a status snapshot whenever the pressure
    changes, so the CLI can show throttling in its progress footer instead of
    interleaving log lines with benchmark results.
    """

    def __init__(
        self,
        *,
        max_interval_seconds: float = _PACING_MAX_SECONDS,
        step_seconds: float = _PACING_STEP_SECONDS,
        observer: RateLimitObserver | None = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._max_interval = max_interval_seconds
        self._step = step_seconds
        self._observer = observer
        self._pause_until = 0.0
        self._min_interval = 0.0
        self._next_slot = 0.0
        self._successes = 0
        self._retries = 0
        self._total_wait = 0.0

    @property
    def min_interval(self) -> float:
        """Current spacing between request starts (0.0 when unthrottled)."""
        return self._min_interval

    def set_observer(self, observer: RateLimitObserver | None) -> None:
        self._observer = observer

    def status(self) -> RateLimitStatus:
        return RateLimitStatus(
            retries=self._retries,
            total_wait_seconds=self._total_wait,
            pacing_seconds=self._min_interval,
        )

    def _notify(self) -> None:
        if self._observer is None:
            return
        try:
            self._observer(self.status())
        except Exception:  # pragma: no cover - display must never break a run
            logger.debug("Rate-limit observer raised", exc_info=True)

    async def acquire(self) -> None:
        """Wait for this request's turn under the current pacing and pause."""
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._pause_until, self._next_slot)
            self._next_slot = start + self._min_interval
        delay = start - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        # A pause may have been declared after this request reserved its slot.
        while (remaining := self._pause_until - time.monotonic()) > 0:
            await asyncio.sleep(remaining)

    async def on_rate_limited(self, delay: float) -> None:
        """Record a 429: pause everyone and widen the spacing."""
        async with self._lock:
            self._pause_until = max(self._pause_until, time.monotonic() + delay)
            self._min_interval = min(max(self._min_interval * 2, self._step), self._max_interval)
            self._successes = 0
            self._retries += 1
            self._total_wait += delay
        logger.debug("Rate limited; pacing requests %.2fs apart", self._min_interval)
        self._notify()

    async def on_success(self) -> None:
        """Record a healthy response, relaxing pacing after a quiet stretch."""
        if self._min_interval <= 0.0:
            return
        async with self._lock:
            self._successes += 1
            if self._successes < _PACING_DECAY_AFTER_SUCCESSES:
                return
            self._successes = 0
            relaxed = self._min_interval / 2
            self._min_interval = 0.0 if relaxed < self._step else relaxed
        logger.debug("Rate limit eased; pacing now %.2fs apart", self._min_interval)
        self._notify()


class RetryingHTTPAdapter:
    """HTTP client lifecycle plus the shared retry / rate-limit policy.

    Concrete adapters implement the wire format; this base owns connection
    reuse, the retry budgets, and the rate-limit coordinator.
    """

    def __init__(
        self,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_rate_limit_retries: int = DEFAULT_MAX_RATE_LIMIT_RETRIES,
        rate_limit_observer: RateLimitObserver | None = None,
    ) -> None:
        self._client: httpx.AsyncClient | None = None
        self._max_retries = max_retries
        self._max_rate_limit_retries = max_rate_limit_retries
        self._rate_limits = RateLimitCoordinator(observer=rate_limit_observer)

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily on first access.

        The client is created WITHOUT a fixed timeout — callers pass
        per-request timeouts to avoid mismatch between warm-up (60s),
        throughput (180s), and scenario requests.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),  # generous default; overridden per-request
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client (releases connections)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def rate_limit_status(self) -> RateLimitStatus:
        """Cumulative rate-limit pressure seen by this adapter."""
        return self._rate_limits.status()

    def set_rate_limit_observer(self, observer: RateLimitObserver | None) -> None:
        """Route rate-limit status changes to a progress display."""
        self._rate_limits.set_observer(observer)

    async def _with_retries(
        self,
        attempt: Callable[[], Awaitable[ChatCompletionResult]],
        *,
        url: str,
        max_retries: int | None = None,
        max_rate_limit_retries: int | None = None,
    ) -> ChatCompletionResult:
        """Retry transient rate-limit / gateway failures with jittered backoff.

        Retries only conditions that fail fast and are plausibly self-healing.
        Rate limits (429) draw on a separate, larger budget than other transient
        failures, because a per-minute quota outlasts the generic backoff window.
        The final failure is re-raised so the orchestrator can classify it as an
        infrastructure failure rather than a model error.
        """
        budget = self._max_retries if max_retries is None else max_retries
        rate_limit_budget = (
            self._max_rate_limit_retries
            if max_rate_limit_retries is None
            else max_rate_limit_retries
        )
        retry_index = 0
        rate_limit_retries = 0
        while True:
            rate_limited = False
            try:
                result = await attempt()
                await self._rate_limits.on_success()
                return result
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in RETRYABLE_STATUS_CODES:
                    raise
                rate_limited = status == 429
                if rate_limited:
                    exhaustion = _daily_quota_exhaustion(exc.response)
                    if exhaustion is not None:
                        logger.warning(
                            "%s from %s; not retrying (would not resolve within the retry budget)",
                            exhaustion,
                            _redact_url(url),
                        )
                        raise
                    if rate_limit_retries >= rate_limit_budget:
                        raise
                    rate_limit_retries += 1
                    delay = _rate_limit_delay(rate_limit_retries - 1, exc.response)
                    attempt_label = (rate_limit_retries, rate_limit_budget)
                else:
                    if retry_index >= budget:
                        raise
                    delay = _backoff_delay(retry_index, exc.response)
                    retry_index += 1
                    attempt_label = (retry_index + 1, budget + 1)
                reason = f"HTTP {status}"
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                if retry_index >= budget:
                    raise
                delay = _backoff_delay(retry_index)
                retry_index += 1
                attempt_label = (retry_index + 1, budget + 1)
                reason = type(exc).__name__

            if rate_limited:
                # Hold every other in-flight request too, and widen the spacing
                # between requests so the retry does not walk into the same wall.
                await self._rate_limits.on_rate_limited(delay)
                # Logged at debug: throttling is surfaced through the progress
                # display instead, so it does not interleave with result lines.
                logger.debug(
                    "Rate limited by %s; waiting %.2fs before retry %d/%d",
                    _redact_url(url),
                    delay,
                    *attempt_label,
                )
            else:
                logger.info(
                    "Transient failure from %s (%s); retrying in %.2fs (attempt %d/%d)",
                    _redact_url(url),
                    reason,
                    delay,
                    *attempt_label,
                )
                await asyncio.sleep(delay)
