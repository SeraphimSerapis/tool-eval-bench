"""Live speculative decoding monitor.

Polls the Prometheus /metrics endpoint of a vLLM (or compatible) server
and maintains a rolling window of speculative decoding statistics for
real-time terminal visualization.

Usage:
    tool-eval-bench --spec-live
    tool-eval-bench --spec-live --metrics-url http://host:8000/metrics

Design note:
    vLLM's Prometheus counters update every ~10 seconds (its internal log
    interval), not every second.  If we compute deltas between consecutive
    1-second polls, 9 out of 10 will be zero — making the dashboard appear
    dead.  To work around this, we:

    1. Compute a *cumulative* acceptance rate (total accepted / total drafted)
       which is always meaningful regardless of poll frequency.
    2. Track the *last interval where counters actually changed* and display
       those rates as "instantaneous" metrics.
    3. Only append to sparkline history when there was real activity, so the
       history charts show actual behavior rather than flat zeros.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus metric patterns (extended from speculative.py)
# ---------------------------------------------------------------------------

# Prometheus numeric value pattern — handles plain (523754.0) and scientific
# notation (1.378852e+06) which vLLM uses for large cumulative counters.
_NUM = r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

_COUNTER_PATTERNS: dict[str, re.Pattern[str]] = {
    # Spec decode counters (vLLM / SGLang)
    "accepted_tokens": re.compile(
        rf"^(?:vllm[:_])?spec_decode_num_accepted_tokens(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "draft_tokens": re.compile(
        rf"^(?:vllm[:_])?spec_decode_num_draft_tokens(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "num_drafts": re.compile(
        rf"^(?:vllm[:_])?spec_decode_num_drafts(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    # Engine throughput gauges (deprecated in vLLM ≥0.8, but still present in
    # older versions — we fall back to counter-derived rates when these are 0)
    "prompt_tps": re.compile(
        rf"^(?:vllm[:_])?avg_prompt_throughput_toks_per_s(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "generation_tps": re.compile(
        rf"^(?:vllm[:_])?avg_generation_throughput_toks_per_s(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    # KV cache — old name (gpu_cache_usage_perc) and new name (kv_cache_usage_perc)
    "gpu_cache_usage": re.compile(
        rf"^(?:vllm[:_])?gpu_cache_usage_perc(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "kv_cache_usage": re.compile(
        rf"^(?:vllm[:_])?kv_cache_usage_perc(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    # Requests
    "running_reqs": re.compile(
        rf"^(?:vllm[:_])?num_requests_running(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "waiting_reqs": re.compile(
        rf"^(?:vllm[:_])?num_requests_waiting(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    # Prefix cache — old gauge and new counters
    "prefix_cache_hit": re.compile(
        rf"^(?:vllm[:_])?(?:gpu_)?prefix_cache_hit_rate(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "prefix_cache_queries": re.compile(
        rf"^(?:vllm[:_])?prefix_cache_queries(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "prefix_cache_hits": re.compile(
        rf"^(?:vllm[:_])?prefix_cache_hits(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    # Token counts (cumulative) — primary source for throughput in vLLM ≥0.8
    "prompt_tokens_total": re.compile(
        rf"^(?:vllm[:_])?prompt_tokens_total(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "generation_tokens_total": re.compile(
        rf"^(?:vllm[:_])?generation_tokens_total(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    # -- llama.cpp counters (llamacpp: prefix) --
    # These provide basic throughput stats; spec decode data comes per-request.
    "llamacpp_prompt_tokens_total": re.compile(
        rf"^llamacpp:prompt_tokens_total\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_predicted_tokens_total": re.compile(
        rf"^llamacpp:tokens_predicted_total\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_prompt_tokens_seconds": re.compile(
        rf"^llamacpp:prompt_tokens_seconds\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_predicted_tokens_seconds": re.compile(
        rf"^llamacpp:predicted_tokens_seconds\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_requests_processing": re.compile(
        rf"^llamacpp:requests_processing\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_requests_deferred": re.compile(
        rf"^llamacpp:requests_deferred\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_kv_cache_usage_ratio": re.compile(
        rf"^llamacpp:kv_cache_usage_ratio\s+{_NUM}",
        re.MULTILINE,
    ),
}

# Per-position acceptance rate pattern (vLLM specific)
_PER_POSITION_PATTERN = re.compile(
    r"^(?:vllm[:_])?spec_decode_per_position_acceptance_rate"
    rf'\{{[^}}]*position="(\d+)"[^}}]*\}}\s+{_NUM}',
    re.MULTILINE,
)


@dataclass
class MetricsSnapshot:
    """Single point-in-time scrape of server metrics."""

    timestamp: float = 0.0

    # Spec decode counters (cumulative) — vLLM / SGLang
    accepted_tokens: float = 0.0
    draft_tokens: float = 0.0
    num_drafts: float = 0.0

    # Engine gauges (vLLM)
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    gpu_cache_usage: float | None = None   # legacy: gpu_cache_usage_perc
    kv_cache_usage: float | None = None    # current: kv_cache_usage_perc
    running_reqs: float = 0.0
    waiting_reqs: float = 0.0
    prefix_cache_hit: float = 0.0  # legacy gauge (0–1)
    prefix_cache_queries: float = 0.0  # new counter
    prefix_cache_hits: float = 0.0     # new counter
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0

    # Per-position acceptance rates (position → rate)
    per_position_rates: dict[int, float] = field(default_factory=dict)

    # -- llama.cpp metrics --
    llamacpp_prompt_tokens_total: float = 0.0
    llamacpp_predicted_tokens_total: float = 0.0
    llamacpp_prompt_tokens_seconds: float = 0.0
    llamacpp_predicted_tokens_seconds: float = 0.0
    llamacpp_requests_processing: float = 0.0
    llamacpp_requests_deferred: float = 0.0
    llamacpp_kv_cache_usage_ratio: float | None = None

    @property
    def has_spec_decode(self) -> bool:
        return self.draft_tokens > 0 or self.accepted_tokens > 0

    @property
    def has_llamacpp_metrics(self) -> bool:
        """True if this snapshot contains llama.cpp backend metrics."""
        return (
            self.llamacpp_predicted_tokens_seconds > 0
            or self.llamacpp_prompt_tokens_total > 0
            or self.llamacpp_predicted_tokens_total > 0
        )


@dataclass
class SpecLiveDelta:
    """Computed delta between two snapshots — the interesting stuff.

    Fields are split into three categories:
    - **Cumulative rates**: computed from total counters, always meaningful
    - **Interval rates**: computed from the *last interval that had activity*
    - **Instantaneous gauges**: read directly from the current snapshot
    """

    elapsed_s: float = 0.0

    # --- Cumulative rates (always available once counters > 0) ---
    cumulative_acceptance_rate: float | None = None  # total_accepted / total_drafted
    cumulative_acceptance_length: float | None = None  # total_accepted / total_drafts
    cumulative_draft_window: float | None = None  # total_drafted / total_drafts

    # --- Interval rates (from the last interval with counter changes) ---
    acceptance_rate: float | None = None  # 0.0–1.0
    acceptance_length: float | None = None  # avg tokens per draft step
    draft_window: float | None = None  # avg drafted per step
    waste_ratio: float | None = None  # 1 - acceptance_rate

    # Throughput from deltas
    accepted_tps: float = 0.0  # accepted tokens / elapsed
    drafted_tps: float = 0.0  # drafted tokens / elapsed

    # Whether counters changed in this interval
    had_activity: bool = False

    # --- Instantaneous gauges (from current snapshot) ---
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    gpu_cache_pct: float = 0.0
    running_reqs: int = 0
    waiting_reqs: int = 0
    prefix_cache_hit_pct: float = 0.0

    # Per-position rates snapshot (vLLM gauge — already a rolling average)
    per_position_rates: dict[int, float] = field(default_factory=dict)

    # Cumulative totals
    total_accepted: int = 0
    total_drafted: int = 0
    total_drafts: int = 0


def _parse_snapshot(text: str) -> MetricsSnapshot:
    """Parse Prometheus text into a MetricsSnapshot."""
    snap = MetricsSnapshot(timestamp=time.time())

    for name, pattern in _COUNTER_PATTERNS.items():
        m = pattern.search(text)
        if m:
            setattr(snap, name, float(m.group(1)))

    # Per-position acceptance rates
    for m in _PER_POSITION_PATTERN.finditer(text):
        pos = int(m.group(1))
        rate = float(m.group(2))
        snap.per_position_rates[pos] = rate

    return snap


def compute_delta(prev: MetricsSnapshot, curr: MetricsSnapshot) -> SpecLiveDelta:
    """Compute a delta between two consecutive snapshots."""
    dt = curr.timestamp - prev.timestamp
    if dt <= 0:
        dt = 1.0  # avoid division by zero

    d_accepted = curr.accepted_tokens - prev.accepted_tokens
    d_drafted = curr.draft_tokens - prev.draft_tokens
    d_drafts = curr.num_drafts - prev.num_drafts

    had_activity = d_drafted > 0 or d_accepted > 0

    # Throughput gauges — prefer the Prometheus gauge values when available,
    # but fall back to counter-derived rates when they are 0 (deprecated in
    # vLLM ≥0.8 where avg_*_throughput_toks_per_s gauges were removed).
    gen_tps = curr.generation_tps
    prompt_tps_val = curr.prompt_tps

    if gen_tps == 0 and dt > 0:
        d_gen_tokens = curr.generation_tokens_total - prev.generation_tokens_total
        if d_gen_tokens > 0:
            gen_tps = d_gen_tokens / dt

    if prompt_tps_val == 0 and dt > 0:
        d_prompt_tokens = curr.prompt_tokens_total - prev.prompt_tokens_total
        if d_prompt_tokens > 0:
            prompt_tps_val = d_prompt_tokens / dt

    # llama.cpp fallback: use llamacpp:predicted_tokens_seconds gauge directly
    if gen_tps == 0 and curr.llamacpp_predicted_tokens_seconds > 0:
        gen_tps = curr.llamacpp_predicted_tokens_seconds
    if prompt_tps_val == 0 and curr.llamacpp_prompt_tokens_seconds > 0:
        prompt_tps_val = curr.llamacpp_prompt_tokens_seconds

    # llama.cpp counter-derived fallback for throughput
    if gen_tps == 0 and dt > 0:
        d_lc_gen = curr.llamacpp_predicted_tokens_total - prev.llamacpp_predicted_tokens_total
        if d_lc_gen > 0:
            gen_tps = d_lc_gen / dt
    if prompt_tps_val == 0 and dt > 0:
        d_lc_prompt = curr.llamacpp_prompt_tokens_total - prev.llamacpp_prompt_tokens_total
        if d_lc_prompt > 0:
            prompt_tps_val = d_lc_prompt / dt

    # Running / waiting requests: merge vLLM and llama.cpp
    running = curr.running_reqs
    waiting = curr.waiting_reqs
    if running == 0 and curr.llamacpp_requests_processing > 0:
        running = curr.llamacpp_requests_processing
    if waiting == 0 and curr.llamacpp_requests_deferred > 0:
        waiting = curr.llamacpp_requests_deferred

    # KV cache — prefer new kv_cache_usage_perc when present (even if 0.0,
    # which is a valid reading when idle), fall back to legacy gpu_cache_usage_perc,
    # then to llama.cpp kv_cache_usage_ratio
    if curr.kv_cache_usage is not None:
        cache_frac = curr.kv_cache_usage
    elif curr.gpu_cache_usage is not None:
        cache_frac = curr.gpu_cache_usage
    elif curr.llamacpp_kv_cache_usage_ratio is not None:
        cache_frac = curr.llamacpp_kv_cache_usage_ratio
    else:
        cache_frac = 0.0

    # Prefix cache — prefer legacy gauge, fall back to counter-derived rate
    prefix_hit_rate = curr.prefix_cache_hit
    if prefix_hit_rate == 0 and curr.prefix_cache_queries > 0:
        # Compute session hit rate from cumulative counters
        prefix_hit_rate = curr.prefix_cache_hits / curr.prefix_cache_queries

    delta = SpecLiveDelta(
        elapsed_s=dt,
        had_activity=had_activity,
        # Throughput (gauge or counter-derived fallback)
        prompt_tps=prompt_tps_val,
        generation_tps=gen_tps,
        # Instantaneous gauges — always from current snapshot (merged vLLM + llama.cpp)
        gpu_cache_pct=cache_frac * 100,
        running_reqs=int(running),
        waiting_reqs=int(waiting),
        prefix_cache_hit_pct=prefix_hit_rate * 100,
        # Per-position rates are vLLM gauges (rolling averages, always current)
        per_position_rates=dict(curr.per_position_rates),
        # Cumulative totals
        total_accepted=int(curr.accepted_tokens),
        total_drafted=int(curr.draft_tokens),
        total_drafts=int(curr.num_drafts),
    )

    # --- Cumulative rates (always computed from totals) ---
    if curr.draft_tokens > 0:
        delta.cumulative_acceptance_rate = curr.accepted_tokens / curr.draft_tokens
    if curr.num_drafts > 0:
        delta.cumulative_acceptance_length = curr.accepted_tokens / curr.num_drafts
        delta.cumulative_draft_window = curr.draft_tokens / curr.num_drafts

    # --- Interval rates (only when counters actually changed) ---
    if d_drafted > 0:
        delta.acceptance_rate = d_accepted / d_drafted
        delta.waste_ratio = 1.0 - delta.acceptance_rate
    if d_drafts > 0:
        delta.acceptance_length = d_accepted / d_drafts
        delta.draft_window = d_drafted / d_drafts
    if dt > 0:
        delta.accepted_tps = d_accepted / dt
        delta.drafted_tps = d_drafted / dt

    return delta


def metrics_url_from_base(base_url: str) -> str:
    """Build the /metrics URL from a base URL."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return f"{b}/metrics"


async def scrape_snapshot(
    client: httpx.AsyncClient,
    url: str,
    api_key: str | None = None,
) -> MetricsSnapshot | None:
    """Scrape metrics endpoint and return a snapshot, or None on failure."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = await client.get(url, headers=headers, timeout=5.0)
        if resp.status_code != 200:
            return None
        return _parse_snapshot(resp.text)
    except Exception as exc:
        logger.debug("Scrape failed: %s", exc)
        return None
