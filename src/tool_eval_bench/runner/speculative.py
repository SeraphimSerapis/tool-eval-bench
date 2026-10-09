"""Speculative decoding / MTP throughput benchmarking.

Measures the *real-world effectiveness* of speculative decoding techniques
(multi-token prediction, draft models, n-gram matching) that standard
t/s metrics fail to capture.

Key metrics:
- Effective t/s:      tokens after the first ÷ wall-clock time after the first token
- Acceptance rate (α): % of draft tokens accepted by the verifier
- Acceptance length:   avg output tokens per speculative step, including the verifier token
- Speedup ratio:       effective t/s ÷ baseline t/s
- Goodput:             accepted tokens ÷ wall-clock time

Data sources:
- vLLM:     ``metrics.speculative_decoding`` in the response when the server
            runs with ``--per-request-spec-decode-metrics``; otherwise
            Prometheus counter deltas at /metrics
- llama.cpp: ``timings.draft_n`` / ``draft_n_accepted`` in the response; the
            /metrics step counter (``--metrics``) adds acceptance length when
            no other request drafted during the measurement
- Strata:   ``timings.draft_n`` / ``draft_n_accepted`` in the response;
            otherwise Prometheus counter deltas at /metrics
- TensorFold: request-local counts in the response; otherwise Prometheus
- SGLang:   live gauges are available to spec-live, but are not request-local here
- Fallback: wall-clock effective t/s only (always available)
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from tool_eval_bench.domain.measurement import MeasurementClient, MeasurementClientFactory
from tool_eval_bench.domain.spec_decode import per_position_acceptance
from tool_eval_bench.runner.spec_detection import (
    SpecDecodeCounters as SpecDecodeCounters,
)
from tool_eval_bench.runner.spec_detection import (
    SpecDecodeInfo as SpecDecodeInfo,
)
from tool_eval_bench.runner.spec_detection import (
    detect_spec_decoding as detect_spec_decoding,
)
from tool_eval_bench.runner.spec_detection import (
    parse_prometheus_spec_metrics as parse_prometheus_spec_metrics,
)
from tool_eval_bench.runner.spec_detection import (
    scrape_spec_metrics as scrape_spec_metrics,
)
from tool_eval_bench.runner.throughput import (
    ThroughputSample,
    TokenizerConfig,
    _build_depth_system_text,
    _build_messages,
    _stream_one,
    calibrate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vLLM per-request metrics
# ---------------------------------------------------------------------------


@dataclass
class PerRequestSpecMetrics:
    """Request-local speculative counters, with an optional step count.

    ``summary`` mode gives the three counters and ``num_spec_tokens``;
    ``detailed`` mode adds the per-step arrays. vLLM marks the shape as
    experimental, so the parser accepts only what it can verify. TensorFold
    provides draft and accepted counts but no speculative-step count.
    """

    num_draft_tokens: int
    num_accepted_draft_tokens: int
    num_spec_steps: int | None
    num_spec_tokens: int | None = None
    per_step_accepted: list[int] | None = None
    per_step_drafted: list[int] | None = None


def _int_list(value: object) -> list[int] | None:
    if not isinstance(value, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        return None
    return list(value)


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def parse_per_request_spec_metrics(raw: object) -> PerRequestSpecMetrics | None:
    """Parse ``metrics.speculative_decoding`` from a vLLM response.

    Returns ``None`` when the object is missing any of the three counters or
    they are not non-negative integers. Per-step arrays are dropped, not
    rejected, when they are malformed or disagree with each other in length.
    """
    if not isinstance(raw, dict):
        return None
    drafted = _non_negative_int(raw.get("num_draft_tokens"))
    accepted = _non_negative_int(raw.get("num_accepted_draft_tokens"))
    steps = _non_negative_int(raw.get("num_spec_steps"))
    if drafted is None or accepted is None or steps is None:
        return None
    per_step_accepted = _int_list(raw.get("per_step_accepted"))
    per_step_drafted = _int_list(raw.get("per_step_drafted"))
    if (
        per_step_accepted is None
        or per_step_drafted is None
        or len(per_step_accepted) != len(per_step_drafted)
    ):
        per_step_accepted = per_step_drafted = None
    return PerRequestSpecMetrics(
        num_draft_tokens=drafted,
        num_accepted_draft_tokens=accepted,
        num_spec_steps=steps,
        num_spec_tokens=_non_negative_int(raw.get("num_spec_tokens")),
        per_step_accepted=per_step_accepted,
        per_step_drafted=per_step_drafted,
    )


def parse_tensorfold_spec_metrics(raw: object) -> PerRequestSpecMetrics | None:
    """Read request-local draft counts; engine rounds are not spec steps."""
    if not isinstance(raw, dict):
        return None
    drafted = _non_negative_int(raw.get("drafted"))
    accepted = _non_negative_int(raw.get("accepted"))
    if drafted is None or accepted is None or accepted > drafted:
        return None
    return PerRequestSpecMetrics(drafted, accepted, num_spec_steps=None)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpecDecodeSample:
    """Result of a single speculative decoding benchmark measurement.

    Extends ThroughputSample concept with spec-decode-specific metrics.
    """

    # Base throughput data (from _stream_one)
    pp_tokens: int = 0
    tg_tokens: int = 0
    depth: int = 0
    concurrency: int = 1
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    pp_tps: float = 0.0
    tg_tps: float = 0.0  # standard tg t/s from stream timing
    error: str | None = None

    # Spec-decode-specific metrics
    acceptance_rate: float | None = None  # 0.0–1.0
    acceptance_length: float | None = None  # avg tokens per spec step
    draft_tokens_delta: int | None = None  # draft tokens in this measurement
    accepted_tokens_delta: int | None = None  # accepted tokens in this measurement
    num_drafts_delta: int | None = None  # spec steps in this measurement
    # Where the acceptance counters came from: "response" (request-local
    # metrics, exact), "prometheus" (server-wide counter deltas, subject to
    # cross-talk), "timings" (llama.cpp and Strata response timings, exact),
    # or None when unavailable.
    acceptance_source: str | None = None
    # Configured draft length, when the server reports it directly.
    num_spec_tokens: int | None = None
    # Per-step arrays from vLLM ``detailed`` mode; None in ``summary`` mode.
    per_step_accepted: list[int] | None = None
    per_step_drafted: list[int] | None = None

    # Derived metrics
    spec_method: str = "unknown"  # mtp / draft_model / ngram / eagle
    baseline_tg_tps: float | None = None  # stored baseline for comparison

    # Prompt type used
    prompt_type: str = "filler"  # filler / code / structured / custom label

    # Repeated measurements pooled into this sample (see pool_spec_samples).
    runs: int = 1
    # Runs of this cell that failed and were left out of the pool, so a cell
    # averaged over fewer runs than requested says so.
    failed_runs: int = 0
    acceptance_rate_range: tuple[float, float] | None = None

    @property
    def effective_tg_tps(self) -> float:
        """Decode rate after the first token, the metric users actually feel.

        The first token arrives at TTFT, so the window after it holds the
        other N − 1 tokens. That is the throughput baseline's formula, so a
        run without speculation reports a 1.00x speedup over its own baseline,
        and dividing by verify steps per second gives τ.
        """
        if self.total_ms > 0 and self.tg_tokens > 0:
            # Subtract TTFT to measure generation phase only
            gen_ms = self.total_ms - self.ttft_ms if self.ttft_ms > 0 else self.total_ms
            tokens = (
                self.tg_tokens - 1 if self.ttft_ms > 0 and self.tg_tokens > 1 else self.tg_tokens
            )
            if gen_ms > 0:
                return tokens / (gen_ms / 1000)
        return 0.0

    @property
    def goodput(self) -> float | None:
        """Accepted draft tokens per second of wall-clock generation time.

        None when the sample has no acceptance counters (SGLang, a server
        without ``/metrics``, a proxy): output tokens are not accepted draft
        tokens, so effective t/s is not a stand-in.
        """
        if self.accepted_tokens_delta is None:
            return None
        if self.total_ms > 0:
            gen_ms = self.total_ms - self.ttft_ms if self.ttft_ms > 0 else self.total_ms
            if gen_ms > 0:
                return self.accepted_tokens_delta / (gen_ms / 1000)
        return 0.0

    @property
    def speedup_ratio(self) -> float | None:
        """Speedup vs baseline (effective_tg_tps / baseline_tg_tps)."""
        if self.baseline_tg_tps and self.baseline_tg_tps > 0:
            return self.effective_tg_tps / self.baseline_tg_tps
        return None

    @property
    def draft_tps(self) -> float | None:
        """Drafted tokens per second of wall-clock generation time.

        Shows how fast the draft model runs, regardless of acceptance.
        Compare with goodput to see how much draft compute is wasted.
        """
        if (
            self.draft_tokens_delta is not None
            and self.draft_tokens_delta > 0
            and self.total_ms > 0
        ):
            gen_ms = self.total_ms - self.ttft_ms if self.ttft_ms > 0 else self.total_ms
            if gen_ms > 0:
                return self.draft_tokens_delta / (gen_ms / 1000)
        return None

    @property
    def waste_ratio(self) -> float | None:
        """Fraction of drafted tokens rejected by the verifier (0.0–1.0).

        Lower is better. A value of 0.82 means 82% of draft compute is
        discarded — the draft model is poorly aligned with the target.
        """
        if self.acceptance_rate is not None:
            return 1.0 - self.acceptance_rate
        return None

    @property
    def verify_steps_per_s(self) -> float | None:
        """Target-model verification passes per second of generation time.

        Without speculation every forward pass yields exactly one token, and a
        verification pass costs at least as much as a plain decode pass plus
        the drafting in between, so this is a lower bound on the no-spec
        decode rate. Effective t/s divided by it is therefore a ceiling on the
        real speedup, and it equals τ. ``--baseline-tgs`` measures the real
        thing.
        """
        if self.num_drafts_delta is not None and self.num_drafts_delta > 0 and self.total_ms > 0:
            gen_ms = self.total_ms - self.ttft_ms if self.ttft_ms > 0 else self.total_ms
            if gen_ms > 0:
                return self.num_drafts_delta / (gen_ms / 1000)
        return None

    @property
    def draft_window(self) -> float | None:
        """Average tokens drafted per speculative step.

        This reveals the configured draft window size. If draft_window=15
        but acceptance_length=3.5, only 2.5 drafted tokens per step are
        accepted (τ counts the verifier's bonus token), so positions 4–15 are
        mostly wasted. ``domain.spec_decode.window_utilization`` turns the
        pair into the fraction of drafted positions accepted.
        """
        if (
            self.draft_tokens_delta is not None
            and self.num_drafts_delta is not None
            and self.num_drafts_delta > 0
        ):
            return self.draft_tokens_delta / self.num_drafts_delta
        return None

    @property
    def per_position_acceptance(self) -> list[float] | None:
        """Acceptance rate at each draft position for this request.

        Only available from vLLM ``detailed`` per-request metrics. Shows where
        in the draft window acceptance falls off, which is what decides the
        right ``num_speculative_tokens``.
        """
        if self.per_step_accepted is None or self.per_step_drafted is None:
            return None
        rates = per_position_acceptance([(self.per_step_accepted, self.per_step_drafted)])
        return rates or None

    def to_result(self) -> dict[str, Any]:
        """The measurements a spec-bench run stores for this sample.

        Derived metrics are included so history and export readers need not
        reimplement them. Floats are raw, not the rounded figures the report
        prints. The per-step arrays are left out: vLLM ``detailed`` mode reports
        one entry per speculative step, thousands per request, and
        ``per_position_acceptance`` keeps what they are used for.
        """
        range_ = self.acceptance_rate_range
        return {
            "prompt_type": self.prompt_type,
            "depth": self.depth,
            "runs": self.runs,
            "failed_runs": self.failed_runs,
            "spec_method": self.spec_method,
            "pp_tokens": self.pp_tokens,
            "tg_tokens": self.tg_tokens,
            "concurrency": self.concurrency,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "pp_tps": self.pp_tps,
            "tg_tps": self.tg_tps,
            "effective_tg_tps": self.effective_tg_tps,
            "goodput": self.goodput,
            "baseline_tg_tps": self.baseline_tg_tps,
            "speedup_ratio": self.speedup_ratio,
            "acceptance_source": self.acceptance_source,
            "acceptance_rate": self.acceptance_rate,
            "acceptance_rate_range": list(range_) if range_ is not None else None,
            "acceptance_length": self.acceptance_length,
            "draft_tokens_delta": self.draft_tokens_delta,
            "accepted_tokens_delta": self.accepted_tokens_delta,
            "num_drafts_delta": self.num_drafts_delta,
            "num_spec_tokens": self.num_spec_tokens,
            "draft_window": self.draft_window,
            "draft_tps": self.draft_tps,
            "waste_ratio": self.waste_ratio,
            "verify_steps_per_s": self.verify_steps_per_s,
            "per_position_acceptance": self.per_position_acceptance,
        }

    def apply_per_request_metrics(self, metrics: PerRequestSpecMetrics) -> None:
        """Fill acceptance fields from verified request-local counters."""
        self.draft_tokens_delta = metrics.num_draft_tokens
        self.accepted_tokens_delta = metrics.num_accepted_draft_tokens
        self.num_drafts_delta = metrics.num_spec_steps
        self.num_spec_tokens = metrics.num_spec_tokens
        self.per_step_accepted = metrics.per_step_accepted
        self.per_step_drafted = metrics.per_step_drafted
        self.acceptance_source = "response"
        if metrics.num_draft_tokens > 0:
            self.acceptance_rate = metrics.num_accepted_draft_tokens / metrics.num_draft_tokens
        if metrics.num_spec_steps is not None and metrics.num_spec_steps > 0:
            self.acceptance_length = (
                1.0 + metrics.num_accepted_draft_tokens / metrics.num_spec_steps
            )

    @classmethod
    def from_throughput_sample(
        cls,
        sample: ThroughputSample,
        *,
        spec_method: str = "unknown",
        prompt_type: str = "filler",
    ) -> SpecDecodeSample:
        """Create a SpecDecodeSample from a base ThroughputSample."""
        return cls(
            pp_tokens=sample.pp_tokens,
            tg_tokens=sample.tg_tokens,
            depth=sample.depth,
            concurrency=sample.concurrency,
            ttft_ms=sample.ttft_ms,
            total_ms=sample.total_ms,
            pp_tps=sample.pp_tps,
            tg_tps=sample.tg_tps,
            error=sample.error,
            spec_method=spec_method,
            prompt_type=prompt_type,
        )


def pool_spec_samples(samples: list[SpecDecodeSample]) -> SpecDecodeSample:
    """Pool repeated measurements of one sweep cell into a single sample.

    Every extensive quantity (tokens, time, counters) becomes its per-run
    mean, so each ratio property reads as the token-weighted pooled value
    while the displayed TTFT, total time, and token count stay per-request.
    Per-step arrays are concatenated because per-position rates are counts.
    Failed runs are dropped and counted in ``failed_runs``; when every run
    failed the first failure is returned as-is.
    """
    ok = [s for s in samples if s.error is None]
    if not ok:
        return samples[0]
    failed = len(samples) - len(ok)
    if len(ok) == 1:
        return replace(ok[0], failed_runs=failed) if failed else ok[0]
    n = len(ok)
    first = ok[0]

    def mean(values: list[float]) -> float:
        return sum(values) / n

    def mean_counter(values: list[int | None]) -> int | None:
        present = [v for v in values if v is not None]
        return round(sum(present) / len(present)) if present else None

    pooled = SpecDecodeSample(
        pp_tokens=round(mean([s.pp_tokens for s in ok])),
        tg_tokens=round(mean([s.tg_tokens for s in ok])),
        depth=first.depth,
        concurrency=first.concurrency,
        ttft_ms=mean([s.ttft_ms for s in ok]),
        total_ms=mean([s.total_ms for s in ok]),
        pp_tps=mean([s.pp_tps for s in ok]),
        tg_tps=mean([s.tg_tps for s in ok]),
        draft_tokens_delta=mean_counter([s.draft_tokens_delta for s in ok]),
        accepted_tokens_delta=mean_counter([s.accepted_tokens_delta for s in ok]),
        num_drafts_delta=mean_counter([s.num_drafts_delta for s in ok]),
        acceptance_source=first.acceptance_source,
        num_spec_tokens=first.num_spec_tokens,
        spec_method=first.spec_method,
        baseline_tg_tps=first.baseline_tg_tps,
        prompt_type=first.prompt_type,
        runs=n,
        failed_runs=failed,
    )
    # Rates from the pooled counters, not the mean of per-run rates, so a
    # short run does not weigh as much as a long one.
    drafted = sum(s.draft_tokens_delta or 0 for s in ok)
    accepted = sum(s.accepted_tokens_delta or 0 for s in ok)
    if drafted > 0:
        pooled.acceptance_rate = accepted / drafted
    # τ, draft window, and Steps/s need a step count, and under cross-traffic
    # only some runs get one (measure_spec_single borrows it only from a clean
    # delta). Dividing every run's accepted tokens by the steps of a few would
    # overstate τ, so these come from the runs that have steps.
    stepped = [s for s in ok if s.num_drafts_delta is not None]
    steps = sum(s.num_drafts_delta or 0 for s in stepped)
    if steps > 0:
        stepped_accepted = sum(s.accepted_tokens_delta or 0 for s in stepped)
        stepped_drafted = sum(s.draft_tokens_delta or 0 for s in stepped)
        pooled.acceptance_length = 1.0 + stepped_accepted / steps
        if len(stepped) < n:
            # draft_window divides the all-run draft mean by this, so scale the
            # step mean to keep the stepped runs' drafted-per-step ratio.
            pooled.num_drafts_delta = (
                round(pooled.draft_tokens_delta * steps / stepped_drafted)
                if stepped_drafted > 0 and pooled.draft_tokens_delta is not None
                else None
            )
    rates = [s.acceptance_rate for s in ok if s.acceptance_rate is not None]
    if rates:
        pooled.acceptance_rate_range = (min(rates), max(rates))
    if all(s.per_step_accepted is not None and s.per_step_drafted is not None for s in ok):
        pooled.per_step_accepted = [v for s in ok for v in (s.per_step_accepted or [])]
        pooled.per_step_drafted = [v for s in ok for v in (s.per_step_drafted or [])]
    return pooled


# Callback type
OnSpecSample = Callable[[SpecDecodeSample, int, int], Awaitable[None]]


# ---------------------------------------------------------------------------
# Prompt types for varied workloads
# ---------------------------------------------------------------------------

_CODE_PROMPT = (
    "Implement a Python function that takes a list of integers and returns the "
    "longest increasing subsequence. Use dynamic programming with O(n log n) "
    "complexity. Include type hints, docstring, and handle edge cases like empty "
    "lists and single elements. Then write comprehensive unit tests using pytest "
    "that cover normal cases, edge cases, and performance with large inputs.\n\n"
    "```python\n"
    "from typing import List\n\n"
    "def longest_increasing_subsequence(nums: List[int]) -> List[int]:\n"
    "```"
)

_STRUCTURED_PROMPT = (
    "Parse the following semi-structured log entries and extract a JSON array of "
    "events. Each event should have: timestamp (ISO 8601), level (INFO/WARN/ERROR), "
    "service name, message, and any key=value metadata pairs.\n\n"
    "```\n"
    "2025-03-15T14:23:01.445Z INFO  [auth-service] User login successful uid=12345 ip=192.168.1.100 duration_ms=45\n"
    "2025-03-15T14:23:02.112Z WARN  [api-gateway] Rate limit approaching threshold client_id=abc-789 current=950 max=1000\n"
    "2025-03-15T14:23:02.890Z ERROR [payment-service] Transaction failed tx_id=TX-2025-0315-001 amount=299.99 currency=EUR\n"
    "2025-03-15T14:23:03.201Z INFO  [notification-service] Email queued recipient=user@example.com template=payment_failed retry=0\n"
    "2025-03-15T14:23:04.567Z WARN  [auth-service] Failed login attempt uid=99999 ip=10.0.0.55 attempts=3 lockout=false\n"
    "```\n\n"
    "Output the JSON array:"
)


def _get_prompt_for_type(
    prompt_type: str, custom_prompts: dict[str, str] | None = None
) -> str | None:
    """Return a fixed prompt for a given type, or None for filler."""
    if custom_prompts and prompt_type in custom_prompts:
        return custom_prompts[prompt_type]
    if prompt_type == "code":
        return _CODE_PROMPT
    elif prompt_type == "structured":
        return _STRUCTURED_PROMPT
    return None


# ---------------------------------------------------------------------------
# Single spec-decode measurement
# ---------------------------------------------------------------------------


async def measure_spec_single(
    client: MeasurementClient,
    base_url: str,
    model: str,
    *,
    pp: int = 2048,
    tg: int = 128,
    depth: int = 0,
    api_key: str | None = None,
    tok_cfg: TokenizerConfig | None = None,
    spec_info: SpecDecodeInfo | None = None,
    baseline_tg_tps: float | None = None,
    prompt_type: str = "filler",
    metrics_url: str | None = None,
    temperature: float = 0.0,
    custom_prompts: dict[str, str] | None = None,
) -> SpecDecodeSample:
    """Measure throughput with speculative decoding awareness.

    If Prometheus metrics are available, scrapes counters before and after
    the generation to compute per-request acceptance rate.
    """
    tok_cfg = tok_cfg or TokenizerConfig()
    spec_info = spec_info or SpecDecodeInfo()

    # Build messages — use typed prompt if specified.  A fixed prompt is the
    # user turn; depth is filled with context in the system turn, exactly as
    # for the filler workload, so every prompt type can be swept over depth.
    fixed_prompt = _get_prompt_for_type(prompt_type, custom_prompts)
    if fixed_prompt:
        system_text = (
            await _build_depth_system_text(client, base_url, model, depth, api_key, tok_cfg)
            if depth > 0
            else "You are a helpful assistant."
        )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": fixed_prompt},
        ]
    else:
        messages = await _build_messages(
            client,
            base_url,
            model,
            pp,
            depth,
            api_key,
            tok_cfg,
        )

    # Scrape counters BEFORE generation, unless the server has already shown
    # that it reports exact per-request metrics in the response body.
    counters_before: SpecDecodeCounters | None = None
    if spec_info.has_prometheus and not spec_info.has_per_request_metrics:
        counters_before = await scrape_spec_metrics(
            client, base_url, api_key, metrics_url=metrics_url
        )

    # Run the generation
    sample = await _stream_one(
        client, base_url, model, messages, tg, api_key, tok_cfg, temperature=temperature
    )
    sample.depth = depth
    sample.concurrency = 1
    sample.requested_depth = depth
    if fixed_prompt is None:
        sample.requested_pp = pp

    # Convert to SpecDecodeSample
    spec_sample = SpecDecodeSample.from_throughput_sample(
        sample,
        spec_method=spec_info.method,
        prompt_type=prompt_type,
    )
    spec_sample.baseline_tg_tps = baseline_tg_tps

    # vLLM --per-request-spec-decode-metrics and TensorFold's counts: exact
    # and request-scoped, so they win over the server-wide counter delta
    # whenever they are present.
    if sample.error:
        return spec_sample
    per_request = parse_per_request_spec_metrics(sample.spec_decode_metrics)
    if per_request is None:
        per_request = parse_tensorfold_spec_metrics(sample.tensorfold_spec_metrics)
    if per_request is not None:
        spec_sample.apply_per_request_metrics(per_request)
        return spec_sample

    # Scrape counters AFTER generation and compute deltas
    prometheus_delta: tuple[int, int, int | None] | None = None
    if spec_info.has_prometheus and counters_before is not None:
        counters_after = await scrape_spec_metrics(
            client, base_url, api_key, metrics_url=metrics_url
        )
        if counters_after is not None:
            steps_delta = (
                int(counters_after.num_drafts - counters_before.num_drafts)
                if counters_before.has_num_drafts and counters_after.has_num_drafts
                else None
            )
            prometheus_delta = (
                int(counters_after.draft_tokens - counters_before.draft_tokens),
                int(counters_after.accepted_tokens - counters_before.accepted_tokens),
                steps_delta,
            )

    # llama.cpp and Strata report this request's draft counts in ``timings``
    # (draft_n / draft_n_accepted), extracted by _stream_one(). They are
    # request-scoped, so they win over the server-wide delta. Timings carry no
    # step count. The delta lends its step count only when its draft and
    # accepted counts equal the request's: any other request that drafted
    # during the window would have added draft tokens and broken the match.
    draft_n = sample.draft_n
    # llama.cpp's server_slot_stats::to_json (tools/server/server-common.cpp)
    # writes draft_n only when n_draft_tokens > 0, so on llama.cpp a timings
    # object without it is an exact zero. Strata and unknown servers can omit
    # the field for "not reported", so they keep the delta fallback. Detection
    # sets has_per_request_timings from EngineProfile.absent_draft_n_is_zero.
    if draft_n is None and sample.timings_present and spec_info.has_per_request_timings:
        draft_n = 0
    if draft_n is not None:
        accepted = sample.draft_n_accepted or 0
        steps: int | None = None
        if prometheus_delta is not None and prometheus_delta[:2] == (draft_n, accepted):
            steps = prometheus_delta[2]
        spec_sample.apply_per_request_metrics(
            PerRequestSpecMetrics(draft_n, accepted, num_spec_steps=steps)
        )
        spec_sample.acceptance_source = "timings"
        return spec_sample

    if prometheus_delta is not None:
        drafted, accepted_delta, steps_delta = prometheus_delta
        spec_sample.draft_tokens_delta = drafted
        spec_sample.accepted_tokens_delta = accepted_delta
        spec_sample.num_drafts_delta = steps_delta
        spec_sample.acceptance_source = "prometheus"
        if drafted > 0:
            spec_sample.acceptance_rate = accepted_delta / drafted
        if steps_delta and steps_delta > 0:
            spec_sample.acceptance_length = 1.0 + accepted_delta / steps_delta

    return spec_sample


# ---------------------------------------------------------------------------
# Full spec-decode benchmark
# ---------------------------------------------------------------------------


async def run_spec_bench(
    base_url: str,
    model: str,
    *,
    pp: int = 2048,
    tg: int = 128,
    depths: list[int] | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    client_factory: MeasurementClientFactory,
    spec_method: str = "auto",
    baseline_tg_tps: float | None = None,
    prompt_types: list[str] | None = None,
    on_sample: OnSpecSample | None = None,
    metrics_url: str | None = None,
    runs: int = 1,
    temperature: float = 0.0,
    custom_prompts: dict[str, str] | None = None,
) -> list[SpecDecodeSample]:
    """Run speculative decoding benchmark sweep.

    Measures effective throughput across different prompt types and context
    depths. If Prometheus metrics are available, also reports acceptance
    rate and acceptance length.

    Args:
        base_url: Server base URL.
        model: Model name/alias.
        pp: Prompt tokens for filler prompts.
        tg: Max generation tokens.
        depths: Context depth sweep (default: [0]).
        api_key: Optional API key.
        timeout: Request timeout.
        spec_method: Spec decode method hint (auto/mtp/draft/ngram/eagle).
        baseline_tg_tps: Known baseline tg t/s for speedup calculation.
        prompt_types: Prompt types to test (default: [filler, code, structured]).
        on_sample: Progress callback.
        metrics_url: Optional direct URL to the Prometheus /metrics endpoint.
            Useful when the API is behind a proxy (e.g. LiteLLM) and /metrics
            lives on a different host/port.
        runs: Measurements per depth × prompt cell, pooled into one sample.
            One request of a hundred tokens is only a few dozen speculative
            steps, so a single run gives a noisy acceptance rate.
        temperature: Sampling temperature. Rejection sampling accepts more at
            low temperature, so 0.0 reports a ceiling for sampled workloads.
        custom_prompts: Extra prompt types, label to prompt text, selectable
            through ``prompt_types``.

    Returns:
        List of SpecDecodeSample results.
    """
    depths = depths or [0]
    prompt_types = prompt_types or ["filler", "code", "structured"]

    async with client_factory(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_connections=10,
        max_keepalive_connections=5,
    ) as client:
        # Calibrate tokenizer
        tok_cfg = await calibrate(client, base_url, model, api_key)

        # Detect spec decode configuration
        spec_info = await detect_spec_decoding(
            client,
            base_url,
            api_key,
            backend_hint=spec_method,
            metrics_url=metrics_url,
        )

        if spec_info.has_per_request_timings:
            logger.info(
                "llama.cpp backend detected — spec decode metrics will be "
                "extracted from per-request timings (draft_n/draft_n_accepted). "
                "Per-request stats are exact for each measurement.",
            )

        # Build sweep: depth × prompt_type
        combos = [(d, pt) for d in depths for pt in prompt_types]
        total = len(combos)
        samples: list[SpecDecodeSample] = []

        for idx, (depth, prompt_type) in enumerate(combos):
            cell: list[SpecDecodeSample] = []
            for _ in range(max(1, runs)):
                cell.append(
                    await measure_spec_single(
                        client,
                        base_url,
                        model,
                        pp=pp,
                        tg=tg,
                        depth=depth,
                        api_key=api_key,
                        tok_cfg=tok_cfg,
                        spec_info=spec_info,
                        baseline_tg_tps=baseline_tg_tps,
                        prompt_type=prompt_type,
                        metrics_url=metrics_url,
                        temperature=temperature,
                        custom_prompts=custom_prompts,
                    )
                )
                # Learn the source from the very first response so the
                # remaining runs of this cell already skip the scrape.
                if cell[-1].acceptance_source == "response":
                    spec_info.has_per_request_metrics = True
                    if (cell[-1].draft_tokens_delta or 0) > 0:
                        spec_info.active = True
            spec_sample = pool_spec_samples(cell)
            samples.append(spec_sample)

            # The source is only known once a response has come back, so the
            # cross-talk caveat waits for the first sample instead of firing
            # on every server that happens to expose /metrics.
            if idx == 0 and spec_sample.acceptance_source == "response":
                logger.info(
                    "Server returns per-request spec decode metrics in the response body. "
                    "Prometheus deltas are not needed, so concurrent traffic does not "
                    "affect these measurements.",
                )
            elif idx == 0 and spec_sample.acceptance_source == "prometheus":
                logger.warning(
                    "Prometheus /metrics acceptance-rate counters are server-wide aggregates. "
                    "If other models are serving concurrent traffic on this endpoint, "
                    "per-request acceptance rate measurements will be inaccurate. "
                    "For clean measurements: use a single-model server with no concurrent "
                    "load, or start vLLM with --per-request-spec-decode-metrics summary.",
                )

            if on_sample:
                await on_sample(spec_sample, idx, total)

    return samples
