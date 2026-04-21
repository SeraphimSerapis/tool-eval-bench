"""Speculative decoding / MTP throughput benchmarking.

Measures the *real-world effectiveness* of speculative decoding techniques
(multi-token prediction, draft models, n-gram matching) that standard
t/s metrics fail to capture.

Key metrics:
- Effective t/s:      output tokens ÷ wall-clock time (user-perceived speed)
- Acceptance rate (α): % of draft tokens accepted by the verifier
- Acceptance length:   avg tokens accepted per speculative step
- Speedup ratio:       effective t/s ÷ baseline t/s
- Goodput:             accepted tokens ÷ wall-clock time

Data sources:
- vLLM:     Prometheus counters at /metrics
- llama.cpp: /metrics endpoint (if --metrics flag enabled)
- SGLang:   Prometheus counters at /metrics
- Fallback: wall-clock effective t/s only (always available)
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from tool_eval_bench.runner.throughput import (
    TokenizerConfig,
    ThroughputSample,
    _build_messages,
    _headers,
    _stream_one,
    calibrate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus metric parsing
# ---------------------------------------------------------------------------

@dataclass
class SpecDecodeCounters:
    """Snapshot of speculative decoding counters from Prometheus /metrics."""
    accepted_tokens: float = 0.0
    draft_tokens: float = 0.0
    num_drafts: float = 0.0
    timestamp: float = 0.0

    @property
    def acceptance_rate(self) -> float | None:
        """Draft token acceptance rate (0.0–1.0)."""
        if self.draft_tokens > 0:
            return self.accepted_tokens / self.draft_tokens
        return None

    @property
    def acceptance_length(self) -> float | None:
        """Average tokens accepted per speculative step."""
        if self.num_drafts > 0:
            return self.accepted_tokens / self.num_drafts
        return None


# Regex patterns for Prometheus counter lines
# Note: vLLM includes labels like {engine="0",model_name="..."} between the
# metric name and the value.  The (?:\{[^}]*\})? group handles this optional
# label block so we match both bare and labelled counter lines.
_PROM_PATTERNS = {
    # vLLM metrics
    "accepted_tokens": re.compile(
        r"^(?:vllm:)?spec_decode_num_accepted_tokens(?:_total)?(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)",
        re.MULTILINE,
    ),
    "draft_tokens": re.compile(
        r"^(?:vllm:)?spec_decode_num_draft_tokens(?:_total)?(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)",
        re.MULTILINE,
    ),
    "num_drafts": re.compile(
        r"^(?:vllm:)?spec_decode_num_drafts(?:_total)?(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)",
        re.MULTILINE,
    ),
}


def _metrics_url(base_url: str) -> str:
    """Build the /metrics URL (Prometheus endpoint, NOT under /v1)."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return f"{b}/metrics"


def parse_prometheus_spec_metrics(text: str) -> SpecDecodeCounters:
    """Parse speculative decoding counters from Prometheus text format.

    Works with vLLM, SGLang, and any server exposing counters with
    the ``spec_decode_`` prefix.
    """
    counters = SpecDecodeCounters(timestamp=time.time())

    for field_name, pattern in _PROM_PATTERNS.items():
        match = pattern.search(text)
        if match:
            setattr(counters, field_name, float(match.group(1)))

    return counters


async def scrape_spec_metrics(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None = None,
    metrics_url: str | None = None,
) -> SpecDecodeCounters | None:
    """Scrape speculative decoding counters from /metrics endpoint.

    Returns None if the endpoint is unavailable or doesn't contain
    spec decode metrics.
    """
    url = metrics_url or _metrics_url(base_url)
    try:
        resp = await client.get(url, headers=_headers(api_key), timeout=5.0)
        if resp.status_code != 200:
            return None
        counters = parse_prometheus_spec_metrics(resp.text)
        # Only return if at least one spec decode counter is non-zero
        # (meaning spec decode is actually active on the server)
        if counters.draft_tokens > 0 or counters.accepted_tokens > 0:
            return counters
        # Also return if counters are all zero but the metric names are present
        # (server has spec decode but hasn't processed any requests yet)
        if "spec_decode" in resp.text:
            return counters
        return None
    except Exception as exc:
        logger.debug("Could not scrape /metrics: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Spec decode detection
# ---------------------------------------------------------------------------

@dataclass
class SpecDecodeInfo:
    """Information about the server's speculative decoding configuration."""
    active: bool = False
    method: str = "unknown"  # mtp, draft_model, ngram, eagle, unknown
    has_prometheus: bool = False
    detail: str = ""


async def detect_spec_decoding(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None = None,
    backend_hint: str = "auto",
    metrics_url: str | None = None,
) -> SpecDecodeInfo:
    """Probe whether speculative decoding is active on the server.

    Detection strategy:
    1. Check /metrics for spec_decode counters (vLLM / SGLang)
    2. Accept user hint via backend_hint
    """
    info = SpecDecodeInfo()

    # Try Prometheus endpoint
    url = metrics_url or _metrics_url(base_url)
    try:
        resp = await client.get(url, headers=_headers(api_key), timeout=5.0)
        if resp.status_code == 200 and "spec_decode" in resp.text:
            info.active = True
            info.has_prometheus = True
            info.detail = "Detected via Prometheus /metrics (spec_decode counters present)"

            # Try to infer method from metric names
            text = resp.text
            if "eagle" in text.lower():
                info.method = "eagle"
            elif "ngram" in text.lower():
                info.method = "ngram"
            elif "mtp" in text.lower() or "multi_token" in text.lower():
                info.method = "mtp"
            else:
                info.method = "draft_model"  # generic default for vLLM spec decode
    except Exception:
        pass

    # Accept user hint
    if backend_hint not in ("auto", ""):
        if backend_hint in ("mtp", "draft", "ngram", "eagle"):
            info.method = backend_hint
            if not info.active:
                info.active = True
                info.detail = f"Assumed active via --spec-method={backend_hint}"

    return info


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
    acceptance_rate: float | None = None      # 0.0–1.0 (from Prometheus deltas)
    acceptance_length: float | None = None    # avg tokens per spec step
    draft_tokens_delta: int | None = None     # draft tokens in this measurement
    accepted_tokens_delta: int | None = None  # accepted tokens in this measurement
    num_drafts_delta: int | None = None       # spec steps in this measurement

    # Derived metrics
    spec_method: str = "unknown"              # mtp / draft_model / ngram / eagle
    baseline_tg_tps: float | None = None      # stored baseline for comparison

    # Prompt type used
    prompt_type: str = "filler"               # filler / code / structured

    @property
    def effective_tg_tps(self) -> float:
        """Output tokens ÷ wall-clock time — the metric users actually feel."""
        if self.total_ms > 0 and self.tg_tokens > 0:
            # Subtract TTFT to measure generation phase only
            gen_ms = self.total_ms - self.ttft_ms if self.ttft_ms > 0 else self.total_ms
            if gen_ms > 0:
                return self.tg_tokens / (gen_ms / 1000)
        return 0.0

    @property
    def goodput(self) -> float:
        """Accepted tokens per second of wall-clock generation time."""
        if self.accepted_tokens_delta is not None and self.total_ms > 0:
            gen_ms = self.total_ms - self.ttft_ms if self.ttft_ms > 0 else self.total_ms
            if gen_ms > 0:
                return self.accepted_tokens_delta / (gen_ms / 1000)
        # Fall back to effective t/s (all output tokens are "accepted" from user perspective)
        return self.effective_tg_tps

    @property
    def speedup_ratio(self) -> float | None:
        """Speedup vs baseline (effective_tg_tps / baseline_tg_tps)."""
        if self.baseline_tg_tps and self.baseline_tg_tps > 0:
            return self.effective_tg_tps / self.baseline_tg_tps
        return None

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


def _get_prompt_for_type(prompt_type: str) -> str | None:
    """Return a fixed prompt for a given type, or None for filler."""
    if prompt_type == "code":
        return _CODE_PROMPT
    elif prompt_type == "structured":
        return _STRUCTURED_PROMPT
    return None


# ---------------------------------------------------------------------------
# Single spec-decode measurement
# ---------------------------------------------------------------------------

async def measure_spec_single(
    client: httpx.AsyncClient,
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
) -> SpecDecodeSample:
    """Measure throughput with speculative decoding awareness.

    If Prometheus metrics are available, scrapes counters before and after
    the generation to compute per-request acceptance rate.
    """
    tok_cfg = tok_cfg or TokenizerConfig()
    spec_info = spec_info or SpecDecodeInfo()

    # Build messages — use typed prompt if specified
    fixed_prompt = _get_prompt_for_type(prompt_type)
    if fixed_prompt:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": fixed_prompt},
        ]
    else:
        messages = await _build_messages(
            client, base_url, model, pp, depth, api_key, tok_cfg,
        )

    # Scrape counters BEFORE generation
    counters_before: SpecDecodeCounters | None = None
    if spec_info.has_prometheus:
        counters_before = await scrape_spec_metrics(client, base_url, api_key, metrics_url=metrics_url)

    # Run the generation
    sample = await _stream_one(client, base_url, model, messages, tg, api_key)
    sample.depth = depth
    sample.concurrency = 1
    sample.requested_pp = pp
    sample.requested_depth = depth

    # Convert to SpecDecodeSample
    spec_sample = SpecDecodeSample.from_throughput_sample(
        sample,
        spec_method=spec_info.method,
        prompt_type=prompt_type,
    )
    spec_sample.baseline_tg_tps = baseline_tg_tps

    # Scrape counters AFTER generation and compute deltas
    if spec_info.has_prometheus and counters_before is not None:
        counters_after = await scrape_spec_metrics(client, base_url, api_key, metrics_url=metrics_url)
        if counters_after is not None:
            spec_sample.draft_tokens_delta = int(
                counters_after.draft_tokens - counters_before.draft_tokens
            )
            spec_sample.accepted_tokens_delta = int(
                counters_after.accepted_tokens - counters_before.accepted_tokens
            )
            spec_sample.num_drafts_delta = int(
                counters_after.num_drafts - counters_before.num_drafts
            )

            # Compute rates from deltas
            dt = spec_sample.draft_tokens_delta
            at = spec_sample.accepted_tokens_delta
            nd = spec_sample.num_drafts_delta
            if dt and dt > 0:
                spec_sample.acceptance_rate = at / dt if at is not None else None
            if nd and nd > 0 and at is not None:
                spec_sample.acceptance_length = at / nd

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
    spec_method: str = "auto",
    baseline_tg_tps: float | None = None,
    prompt_types: list[str] | None = None,
    on_sample: OnSpecSample | None = None,
    metrics_url: str | None = None,
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

    Returns:
        List of SpecDecodeSample results.
    """
    depths = depths or [0]
    prompt_types = prompt_types or ["filler", "code", "structured"]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    ) as client:
        # Calibrate tokenizer
        tok_cfg = await calibrate(client, base_url, model, api_key)

        # Detect spec decode configuration
        spec_info = await detect_spec_decoding(
            client, base_url, api_key, backend_hint=spec_method,
            metrics_url=metrics_url,
        )

        if spec_info.has_prometheus:
            logger.warning(
                "Prometheus /metrics acceptance-rate counters are server-wide aggregates. "
                "If other models are serving concurrent traffic on this endpoint, "
                "per-request acceptance rate measurements will be inaccurate. "
                "For clean measurements: use a single-model server with no concurrent load.",
            )

        # Build sweep: depth × prompt_type
        combos = [(d, pt) for d in depths for pt in prompt_types]
        total = len(combos)
        samples: list[SpecDecodeSample] = []

        for idx, (depth, prompt_type) in enumerate(combos):
            spec_sample = await measure_spec_single(
                client, base_url, model,
                pp=pp, tg=tg, depth=depth,
                api_key=api_key, tok_cfg=tok_cfg,
                spec_info=spec_info,
                baseline_tg_tps=baseline_tg_tps,
                prompt_type=prompt_type,
                metrics_url=metrics_url,
            )
            samples.append(spec_sample)

            if on_sample:
                await on_sample(spec_sample, idx, total)

    return samples
