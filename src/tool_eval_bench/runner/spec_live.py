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

from tool_eval_bench.utils.urls import metrics_url as _metrics_url_from_base

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus metric patterns (extended from speculative.py)
# ---------------------------------------------------------------------------

# Prometheus numeric value pattern — handles plain and scientific notation.
# Counters are non-negative, but gauges such as a ratio can legitimately be
# rendered with a sign by a compatible exporter.
_NUM = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_NUM_VALUE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

_COUNTER_PATTERNS: dict[str, re.Pattern[str]] = {
    # Spec decode counters (vLLM).  The parser sums every matching series,
    # because vLLM emits one counter set per engine.
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
        rf"^llamacpp:kv_cache_usage_ratio(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    # llama.cpp now exports the same cumulative speculative counters as vLLM.
    "llamacpp_accepted_tokens": re.compile(
        rf"^llamacpp:spec_decode_num_accepted_tokens(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_draft_tokens": re.compile(
        rf"^llamacpp:spec_decode_num_draft_tokens(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "llamacpp_num_drafts": re.compile(
        rf"^llamacpp:spec_decode_num_drafts(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
}

# Per-position metrics.  The current vLLM and llama.cpp exporters use
# cumulative counters, while older vLLM builds exposed a gauge.
_PER_POSITION_RATE_METRIC_NAMES = (
    "vllm:spec_decode_per_position_acceptance_rate",
    "vllm_spec_decode_per_position_acceptance_rate",
)
_PER_POSITION_COUNTER_METRIC_NAMES = (
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "vllm_spec_decode_num_accepted_tokens_per_pos",
    "vllm_spec_decode_num_accepted_tokens_per_pos_total",
)
_LLAMACPP_PER_POSITION_COUNTER_METRIC_NAMES = (
    "llamacpp:spec_decode_num_accepted_tokens_per_pos",
    "llamacpp:spec_decode_num_accepted_tokens_per_pos_total",
)

# These are the method values accepted by current vLLM's SpeculativeConfig.
# Method detection only trusts an explicit method/config label.  Generic
# ``spec_decode_*`` counters prove that speculation is active, not whether it
# uses a draft model, MTP, EAGLE, or another proposer.
_SUPPORTED_SPEC_METHODS = frozenset(
    {
        "draft_model",
        "eagle",
        "eagle3",
        "extract_hidden_states",
        "mtp",
        "ngram",
        "ngram_gpu",
        "medusa",
        "mlp_speculator",
        "suffix",
        "dflash",
        "dspark",
        "custom_class",
    }
)
_METHOD_LABEL_PATTERN = re.compile(
    r"(?:spec_method|speculative_method|method)\s*[:=]\s*[\"']?"
    r"([A-Za-z0-9_.+-]+)",
    re.IGNORECASE,
)

# Extract model_name labels from spec_decode metrics — used to detect draft model identity
_MODEL_NAME_LABEL = re.compile(
    r'model_name="([^"]+)"',
)


# `[^"]` also matches a backslash, so `(?:\\.|[^"])*` gave the engine two ways
# to consume every escape and exponential backtracking to work through on a
# label that never closes its quote. Excluding the backslash from the negated
# class leaves exactly one parse. Metrics text comes off the wire from whatever
# server the run points at, so this is reachable input.
_LABEL_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _parse_labels(raw_labels: str | None) -> dict[str, str]:
    """Parse the simple quoted labels emitted by Prometheus text format."""
    if not raw_labels:
        return {}
    return {
        name: value.replace(r"\"", '"').replace(r"\\", "\\")
        for name, value in _LABEL_PATTERN.findall(raw_labels)
    }


def _metric_series(text: str, metric_name: str) -> list[tuple[dict[str, str], float]]:
    """Return all ``(labels, value)`` series for one exact metric name."""
    pattern = re.compile(
        rf"^{re.escape(metric_name)}"
        rf"(?:\{{(?P<labels>[^}}]*)\}})?[ \t]+"
        rf"(?P<value>{_NUM_VALUE})(?:[ \t]+.*)?$",
        re.MULTILINE,
    )
    return [
        (_parse_labels(match.group("labels")), float(match.group("value")))
        for match in pattern.finditer(text)
    ]


def _is_rank_label(name: str) -> bool:
    """Identify labels that can distinguish replicated engine gauges."""
    return name == "rank" or name == "engine" or name.endswith("_rank")


def _gauge_series_key(item: tuple[dict[str, str], float]) -> tuple[object, ...]:
    """Prefer rank-zero gauges, then use labels for deterministic selection."""
    labels, _ = item
    rank_labels = sorted((name, value) for name, value in labels.items() if _is_rank_label(name))
    if rank_labels:
        non_zero = sum(value not in {"0", "0.0"} for _, value in rank_labels)
        return (0, non_zero, tuple(rank_labels), tuple(sorted(labels.items())))
    return (1, 0, (), tuple(sorted(labels.items())))


def _select_gauge_series(
    series: list[tuple[dict[str, str], float]],
) -> float | None:
    """Select one representative from replicated gauge series.

    Gauges describe the current state of one scheduler/rank, so summing them
    would inflate acceptance rates and utilization.  Rank zero is the stable
    representative when rank labels exist; otherwise selection is
    deterministic and leaves the value unchanged.
    """
    if not series:
        return None
    return min(series, key=_gauge_series_key)[1]


def _sum_pattern_values(pattern: re.Pattern[str], text: str) -> tuple[float | None, int]:
    """Sum all numeric matches for a metric pattern and return count too."""
    matches = list(pattern.finditer(text))
    if not matches:
        return None, 0
    return sum(float(match.group(1)) for match in matches), len(matches)


def counter_delta(previous: float, current: float) -> float:
    """Return a non-negative counter delta, treating a decrease as a reset."""
    return current - previous if current >= previous else max(0.0, current)


def _canonical_spec_method(value: str) -> str | None:
    """Return a supported method name, preserving explicit variants."""
    method = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "draft": "draft_model",
        "draftmodel": "draft_model",
        "standalone": "draft_model",
        "draft_flash": "dflash",
        "multi_token_prediction": "mtp",
        "nextn": "mtp",
        "prompt_lookup": "ngram",
        "ngram_gpu": "ngram_gpu",
        "custom": "custom_class",
    }
    method = aliases.get(method, method)
    if method in _SUPPORTED_SPEC_METHODS:
        return method

    # Parallel drafting and batch-size schedules are config variants, not
    # separate upstream methods.  Keep an explicit suffix visible when a
    # provider chooses to expose it in a label.
    for suffix in ("_parallel", "_dynamic"):
        base = method.removesuffix(suffix)
        if base in _SUPPORTED_SPEC_METHODS:
            return method
    return None


def _detect_spec_method(text: str) -> str:
    """Detect a method only when the metrics contain an explicit method hint.

    Current vLLM, SGLang, and llama.cpp metric names do not identify the
    proposer.  Returning ``unknown`` for generic counters is deliberate: a
    dashboard must not turn proof of speculative decoding into a false claim
    about the configured method.
    """
    for line in text.splitlines():
        if not (
            "spec_decode" in line.lower()
            or "spec_method" in line.lower()
            or "speculative_method" in line.lower()
            or "sglang:spec_" in line.lower()
        ):
            continue
        for match in _METHOD_LABEL_PATTERN.finditer(line):
            method = _canonical_spec_method(match.group(1))
            if method is not None:
                return method
    return "unknown"


def _extract_model_names(text: str) -> set[str]:
    """Extract all unique model_name label values from Prometheus text."""
    return set(_MODEL_NAME_LABEL.findall(text))


_SGLANG_SPEC_METRICS = {
    "acceptance_length": "sglang:spec_accept_length",
    "acceptance_rate": "sglang:spec_accept_rate",
    "cap_length": "sglang:spec_cap_length",
    "block_accept_length": "sglang:spec_block_accept_length",
    "num_steps": "sglang:spec_num_steps",
    "num_draft_tokens": "sglang:spec_num_draft_tokens",
}
_SGLANG_ENGINE_METRICS = {
    "generation_tps": "sglang:gen_throughput",
    "cache_usage": "sglang:token_usage",
    "running_reqs": "sglang:num_running_reqs",
    "waiting_reqs": "sglang:num_queue_reqs",
    "prefix_cache_hit": "sglang:cache_hit_rate",
}
_SUM_COUNTER_KEYS = frozenset(
    {
        "accepted_tokens",
        "draft_tokens",
        "num_drafts",
        "prefix_cache_queries",
        "prefix_cache_hits",
        "prompt_tokens_total",
        "generation_tokens_total",
        "llamacpp_prompt_tokens_total",
        "llamacpp_predicted_tokens_total",
        "llamacpp_accepted_tokens",
        "llamacpp_draft_tokens",
        "llamacpp_num_drafts",
    }
)


@dataclass
class MetricsSnapshot:
    """Single point-in-time scrape of server metrics."""

    timestamp: float = 0.0

    # Spec decode counters (cumulative) — vLLM, or llama.cpp aliases when it
    # is the only backend represented by the scrape.
    accepted_tokens: float = 0.0
    draft_tokens: float = 0.0
    num_drafts: float = 0.0

    # Direct SGLang gauges.  These are instantaneous scheduler values, not
    # counters, and are therefore kept separate from cumulative totals.
    sglang_acceptance_rate: float | None = None
    sglang_acceptance_length: float | None = None
    sglang_cap_length: float | None = None
    sglang_block_accept_length: float | None = None
    sglang_num_steps: float | None = None
    sglang_num_draft_tokens: float | None = None

    # Engine gauges (vLLM)
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    gpu_cache_usage: float | None = None  # legacy: gpu_cache_usage_perc
    kv_cache_usage: float | None = None  # current: kv_cache_usage_perc
    running_reqs: float = 0.0
    waiting_reqs: float = 0.0
    prefix_cache_hit: float = 0.0  # legacy gauge (0–1)
    prefix_cache_queries: float = 0.0  # new counter
    prefix_cache_hits: float = 0.0  # new counter
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0

    # Per-position acceptance rates (position → rate 0.0–1.0)
    per_position_rates: dict[int, float] = field(default_factory=dict)

    # Per-position accepted token counters (position → cumulative count)
    per_position_counters: dict[int, float] = field(default_factory=dict)

    # Detected speculative decoding method (dflash, mtp, eagle, etc.)
    spec_method: str = "unknown"

    # Model names found in Prometheus metric labels
    model_names: set[str] = field(default_factory=set)

    # -- llama.cpp metrics --
    llamacpp_prompt_tokens_total: float = 0.0
    llamacpp_predicted_tokens_total: float = 0.0
    llamacpp_prompt_tokens_seconds: float = 0.0
    llamacpp_predicted_tokens_seconds: float = 0.0
    llamacpp_requests_processing: float = 0.0
    llamacpp_requests_deferred: float = 0.0
    llamacpp_kv_cache_usage_ratio: float | None = None
    llamacpp_accepted_tokens: float = 0.0
    llamacpp_draft_tokens: float = 0.0
    llamacpp_num_drafts: float = 0.0
    llamacpp_per_position_counters: dict[int, float] = field(default_factory=dict)

    # Presence flags distinguish a legitimate zero gauge/counter from an
    # exporter that does not implement the metric family.
    vllm_spec_metrics_present: bool = False
    sglang_spec_metrics_present: bool = False
    llamacpp_spec_metrics_present: bool = False
    llamacpp_metrics_present: bool = False

    # ``vllm`` / ``sglang`` / ``llamacpp`` when known, otherwise ``unknown``.
    spec_backend: str = "unknown"

    @property
    def has_spec_decode(self) -> bool:
        return (
            self.vllm_spec_metrics_present
            or self.sglang_spec_metrics_present
            or self.llamacpp_spec_metrics_present
            or self.draft_tokens > 0
            or self.accepted_tokens > 0
            or self.num_drafts > 0
        )

    @property
    def has_counter_spec_decode(self) -> bool:
        """Whether this scrape has cumulative spec counters."""
        return (
            self.vllm_spec_metrics_present
            or self.llamacpp_spec_metrics_present
            or self.draft_tokens > 0
            or self.accepted_tokens > 0
            or self.num_drafts > 0
            or bool(self.per_position_counters)
        )

    @property
    def has_sglang_metrics(self) -> bool:
        """True if this snapshot contains a known SGLang metric family."""
        return self.sglang_spec_metrics_present or (
            self.spec_backend == "sglang"
            and (self.generation_tps > 0 or self.running_reqs > 0 or self.waiting_reqs > 0)
        )

    @property
    def has_llamacpp_metrics(self) -> bool:
        """True if this snapshot contains llama.cpp backend metrics."""
        return (
            self.llamacpp_metrics_present
            or self.llamacpp_spec_metrics_present
            or self.llamacpp_predicted_tokens_seconds > 0
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

    # Detected speculative decoding method
    spec_method: str = "unknown"

    # Model names found in Prometheus metric labels
    model_names: set[str] = field(default_factory=set)

    # Inferred num_speculative_tokens (from cumulative draft window)
    num_spec_tokens: int | None = None

    # Direct gauges have no cumulative token totals to report.
    counter_metrics_available: bool = True
    spec_metrics_source: str = "unknown"

    # SGLang DSpark fields, when exposed by the server.
    spec_cap_length: float | None = None
    spec_block_accept_length: float | None = None
    spec_num_steps: int | None = None


def _parse_snapshot(text: str) -> MetricsSnapshot:
    """Parse Prometheus text into a MetricsSnapshot."""
    snap = MetricsSnapshot(timestamp=time.time())

    for name, pattern in _COUNTER_PATTERNS.items():
        value, count = _sum_pattern_values(pattern, text)
        if value is None:
            continue
        # Cumulative counters are additive across vLLM engines and llama.cpp
        # workers.  Gauges are state for one engine and must not be summed.
        if name in _SUM_COUNTER_KEYS:
            setattr(snap, name, value)
        else:
            first = pattern.search(text)
            if first is not None:
                setattr(snap, name, float(first.group(1)))
        if name in {"accepted_tokens", "draft_tokens", "num_drafts"}:
            snap.vllm_spec_metrics_present = True
        if name.startswith("llamacpp_"):
            snap.llamacpp_metrics_present = True
        if name in {"llamacpp_accepted_tokens", "llamacpp_draft_tokens", "llamacpp_num_drafts"}:
            snap.llamacpp_spec_metrics_present = True

    # SGLang exposes direct gauges.  Every scheduler/rank can emit a series;
    # select a rank-zero representative rather than summing replicated state.
    for field_name, metric_name in _SGLANG_SPEC_METRICS.items():
        value = _select_gauge_series(_metric_series(text, metric_name))
        if value is not None:
            setattr(snap, f"sglang_{field_name}", value)
            snap.sglang_spec_metrics_present = True
    if snap.sglang_spec_metrics_present:
        snap.spec_backend = "sglang"

    # SGLang's ordinary engine gauges use the same units as the dashboard's
    # vLLM fields.  Keep the generic fields populated for a pure SGLang scrape.
    sglang_engine_values: dict[str, float] = {}
    for field_name, metric_name in _SGLANG_ENGINE_METRICS.items():
        value = _select_gauge_series(_metric_series(text, metric_name))
        if value is not None:
            sglang_engine_values[field_name] = value
    if sglang_engine_values:
        snap.spec_backend = "sglang"
        if snap.generation_tps == 0:
            snap.generation_tps = sglang_engine_values.get("generation_tps", 0.0)
        if snap.kv_cache_usage is None and snap.gpu_cache_usage is None:
            snap.kv_cache_usage = sglang_engine_values.get("cache_usage")
        if snap.running_reqs == 0:
            snap.running_reqs = sglang_engine_values.get("running_reqs", 0.0)
        if snap.waiting_reqs == 0:
            snap.waiting_reqs = sglang_engine_values.get("waiting_reqs", 0.0)
        if snap.prefix_cache_hit == 0:
            snap.prefix_cache_hit = sglang_engine_values.get("prefix_cache_hit", 0.0)

    # Current vLLM and llama.cpp per-position counters are additive across
    # engines.  Group by position before calculating rates.
    for metric_name in _PER_POSITION_RATE_METRIC_NAMES:
        by_position: dict[int, list[tuple[dict[str, str], float]]] = {}
        for labels, value in _metric_series(text, metric_name):
            position_text = labels.get("position")
            if position_text is not None:
                by_position.setdefault(int(position_text), []).append((labels, value))
        for position_index, series in by_position.items():
            selected = _select_gauge_series(series)
            if selected is not None:
                snap.per_position_rates[position_index] = selected

    for metric_name in (
        *_PER_POSITION_COUNTER_METRIC_NAMES,
        *_LLAMACPP_PER_POSITION_COUNTER_METRIC_NAMES,
    ):
        for labels, value in _metric_series(text, metric_name):
            position = labels.get("position")
            if position is None:
                continue
            position_index = int(position)
            if metric_name in _LLAMACPP_PER_POSITION_COUNTER_METRIC_NAMES:
                snap.llamacpp_per_position_counters[position_index] = (
                    snap.llamacpp_per_position_counters.get(position_index, 0.0) + value
                )
                snap.llamacpp_spec_metrics_present = True
            else:
                snap.per_position_counters[position_index] = (
                    snap.per_position_counters.get(position_index, 0.0) + value
                )
                snap.vllm_spec_metrics_present = True

    # Use llama.cpp spec counters as the generic counter view when no vLLM
    # counter family is present.  This keeps existing dashboard consumers
    # backend-neutral while retaining explicit fields for mixed scrapes.
    if snap.llamacpp_spec_metrics_present and not snap.vllm_spec_metrics_present:
        snap.spec_backend = "llamacpp"
        snap.accepted_tokens = snap.llamacpp_accepted_tokens
        snap.draft_tokens = snap.llamacpp_draft_tokens
        snap.num_drafts = snap.llamacpp_num_drafts
        snap.per_position_counters = dict(snap.llamacpp_per_position_counters)
    elif snap.vllm_spec_metrics_present:
        snap.spec_backend = "vllm"

    if snap.llamacpp_metrics_present and snap.spec_backend == "unknown":
        snap.spec_backend = "llamacpp"

    # If we have per-position counters but no rate gauges, compute rates
    # from counters: rate[pos] = counter[pos] / num_drafts
    if not snap.per_position_rates and snap.per_position_counters and snap.num_drafts > 0:
        for position_index, counter_value in snap.per_position_counters.items():
            snap.per_position_rates[position_index] = counter_value / snap.num_drafts

    # Detect speculative decoding method from raw text
    snap.spec_method = _detect_spec_method(text)

    # Extract model names from metric labels
    snap.model_names = _extract_model_names(text)

    return snap


def compute_delta(prev: MetricsSnapshot, curr: MetricsSnapshot) -> SpecLiveDelta:
    """Compute a delta between two consecutive snapshots."""
    dt = curr.timestamp - prev.timestamp
    if dt <= 0:
        dt = 1.0  # avoid division by zero

    def _counter_values(snapshot: MetricsSnapshot) -> tuple[float, float, float]:
        """Return accepted, drafted, and draft-step counters for a snapshot."""
        if snapshot.llamacpp_spec_metrics_present and not snapshot.vllm_spec_metrics_present:
            return (
                snapshot.llamacpp_accepted_tokens,
                snapshot.llamacpp_draft_tokens,
                snapshot.llamacpp_num_drafts,
            )
        return snapshot.accepted_tokens, snapshot.draft_tokens, snapshot.num_drafts

    def _is_sglang(snapshot: MetricsSnapshot) -> bool:
        return snapshot.sglang_spec_metrics_present or any(
            value is not None
            for value in (
                snapshot.sglang_acceptance_rate,
                snapshot.sglang_acceptance_length,
                snapshot.sglang_num_steps,
                snapshot.sglang_num_draft_tokens,
            )
        )

    prev_accepted, prev_drafted, prev_drafts = _counter_values(prev)
    curr_accepted, curr_drafted, curr_drafts = _counter_values(curr)
    is_sglang = _is_sglang(curr)

    d_accepted = counter_delta(prev_accepted, curr_accepted)
    d_drafted = counter_delta(prev_drafted, curr_drafted)
    d_drafts = counter_delta(prev_drafts, curr_drafts)

    had_activity = (
        any(
            value is not None
            for value in (
                curr.sglang_acceptance_rate,
                curr.sglang_acceptance_length,
                curr.sglang_num_steps,
                curr.sglang_num_draft_tokens,
            )
        )
        if is_sglang
        else d_drafted > 0 or d_accepted > 0
    )

    # Throughput gauges — prefer the Prometheus gauge values when available,
    # but fall back to counter-derived rates when they are 0 (deprecated in
    # vLLM ≥0.8 where avg_*_throughput_toks_per_s gauges were removed).
    gen_tps = curr.generation_tps
    prompt_tps_val = curr.prompt_tps

    if gen_tps == 0 and dt > 0:
        d_gen_tokens = counter_delta(prev.generation_tokens_total, curr.generation_tokens_total)
        if d_gen_tokens > 0:
            gen_tps = d_gen_tokens / dt

    if prompt_tps_val == 0 and dt > 0:
        d_prompt_tokens = counter_delta(prev.prompt_tokens_total, curr.prompt_tokens_total)
        if d_prompt_tokens > 0:
            prompt_tps_val = d_prompt_tokens / dt

    # llama.cpp fallback: use llamacpp:predicted_tokens_seconds gauge directly
    if gen_tps == 0 and curr.llamacpp_predicted_tokens_seconds > 0:
        gen_tps = curr.llamacpp_predicted_tokens_seconds
    if prompt_tps_val == 0 and curr.llamacpp_prompt_tokens_seconds > 0:
        prompt_tps_val = curr.llamacpp_prompt_tokens_seconds

    # llama.cpp counter-derived fallback for throughput
    if gen_tps == 0 and dt > 0:
        d_lc_gen = counter_delta(
            prev.llamacpp_predicted_tokens_total,
            curr.llamacpp_predicted_tokens_total,
        )
        if d_lc_gen > 0:
            gen_tps = d_lc_gen / dt
    if prompt_tps_val == 0 and dt > 0:
        d_lc_prompt = counter_delta(
            prev.llamacpp_prompt_tokens_total,
            curr.llamacpp_prompt_tokens_total,
        )
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
        total_accepted=int(curr_accepted),
        total_drafted=int(curr_drafted),
        total_drafts=int(curr_drafts),
        # Spec decode method
        spec_method=curr.spec_method,
        # Model names from Prometheus labels
        model_names=set(curr.model_names),
        counter_metrics_available=not is_sglang,
        spec_metrics_source=(
            "sglang"
            if is_sglang
            else "llamacpp"
            if curr.llamacpp_spec_metrics_present and not curr.vllm_spec_metrics_present
            else "vllm"
            if curr.vllm_spec_metrics_present
            else "unknown"
        ),
    )

    if is_sglang:
        # SGLang's gauges already expose the semantic values.  In particular,
        # spec_accept_length includes the verifier's bonus token.
        delta.cumulative_acceptance_rate = curr.sglang_acceptance_rate
        delta.acceptance_rate = curr.sglang_acceptance_rate
        if curr.sglang_acceptance_rate is not None:
            delta.waste_ratio = 1.0 - curr.sglang_acceptance_rate
        delta.cumulative_acceptance_length = curr.sglang_acceptance_length
        delta.acceptance_length = curr.sglang_acceptance_length
        delta.spec_cap_length = curr.sglang_cap_length
        delta.spec_block_accept_length = curr.sglang_block_accept_length
        if curr.sglang_num_draft_tokens is not None:
            # SGLang documents these as independent active configuration
            # values.  With top-k drafting, draft tokens are not steps times
            # one, so do not divide the configured window by num_steps.
            delta.cumulative_draft_window = curr.sglang_num_draft_tokens
            delta.draft_window = curr.sglang_num_draft_tokens
        if curr.sglang_num_steps is not None:
            delta.spec_num_steps = round(curr.sglang_num_steps)
        if curr.sglang_num_draft_tokens is not None:
            delta.num_spec_tokens = round(curr.sglang_num_draft_tokens)
    else:
        # vLLM and llama.cpp count accepted draft tokens separately from the
        # verifier's bonus token.  Their published mean acceptance length is
        # therefore 1 + accepted_draft_tokens / verification_steps.
        if curr_drafted > 0:
            delta.cumulative_acceptance_rate = curr_accepted / curr_drafted
        if curr_drafts > 0:
            delta.cumulative_acceptance_length = 1.0 + curr_accepted / curr_drafts
            delta.cumulative_draft_window = curr_drafted / curr_drafts
            # Infer num_speculative_tokens from the draft window.  This equals
            # the configured value when draft windows are uniform.
            delta.num_spec_tokens = round(curr_drafted / curr_drafts)

        # --- Interval rates (only when counters actually changed) ---
        if d_drafted > 0:
            delta.acceptance_rate = d_accepted / d_drafted
            delta.waste_ratio = 1.0 - delta.acceptance_rate
        if d_drafts > 0:
            delta.acceptance_length = 1.0 + d_accepted / d_drafts
            delta.draft_window = d_drafted / d_drafts
    if dt > 0:
        delta.accepted_tps = d_accepted / dt
        delta.drafted_tps = d_drafted / dt

    return delta


def metrics_url_from_base(base_url: str) -> str:
    """Build the /metrics URL from a base URL."""
    return _metrics_url_from_base(base_url)


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


@dataclass
class ServerSpecInfo:
    """Information about speculative decoding gathered from server APIs.

    Populated at startup by probing /v1/models and /version rather than
    relying on keyword scanning of Prometheus text (which rarely works).
    """

    spec_method: str | None = None  # e.g. "draft_model", "mtp", "dflash"
    draft_model_name: str | None = None  # e.g. "Qwen/Qwen3-0.6B"
    target_model_name: str | None = None
    num_speculative_tokens: int | None = None


async def probe_server_spec_info(
    base_url: str,
    *,
    api_key: str | None = None,
    primary_model: str = "unknown",
) -> ServerSpecInfo:
    """Probe the inference server for speculative decoding configuration.

    Strategy (in priority order):
    1. GET /v1/models — record the served target model only.  A second model
       ID is not evidence that it is a drafter, since gateways commonly list
       several independently addressable models.
    2. GET /version (vLLM-specific) — consume ``speculative_config`` only when
       the server explicitly exposes it.

    This is called once at startup, not on every poll.
    """
    info = ServerSpecInfo()
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    b = base_url.rstrip("/")
    if not b.endswith("/v1"):
        b = f"{b}/v1"

    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        # --- 1. Probe /v1/models ---
        try:
            resp = await client.get(f"{b}/models", headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                model_ids: list[str] = []
                if isinstance(body, dict) and "data" in body:
                    for entry in body["data"]:
                        if isinstance(entry, dict) and "id" in entry:
                            model_ids.append(entry["id"])

                if model_ids:
                    primary_norm = primary_model.lower().replace("/", "").replace("-", "")
                    info.target_model_name = next(
                        (
                            model_id
                            for model_id in model_ids
                            if primary_norm in model_id.lower().replace("/", "").replace("-", "")
                            or model_id.lower().replace("/", "").replace("-", "") in primary_norm
                        ),
                        model_ids[0],
                    )
                    # Do not infer a draft model from a second /v1/models item.
                    # vLLM's draft model is an engine-internal configuration.
        except Exception as exc:
            logger.debug("/v1/models probe failed: %s", exc)

        # --- 2. Probe /version (vLLM-specific) ---
        version_url = base_url.rstrip("/")
        if version_url.endswith("/v1"):
            version_url = version_url[:-3]
        try:
            resp = await client.get(f"{version_url}/version", headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict):
                    # Some vLLM builds expose speculative_config in /version
                    spec_cfg = body.get("speculative_config")
                    if isinstance(spec_cfg, dict):
                        method = spec_cfg.get("method")
                        if isinstance(method, str):
                            method_name = _canonical_spec_method(method)
                            if method_name is not None:
                                info.spec_method = method_name
                        draft = spec_cfg.get("model")
                        if isinstance(draft, str) and draft:
                            info.draft_model_name = draft
                        nst = spec_cfg.get("num_speculative_tokens")
                        if isinstance(nst, int):
                            info.num_speculative_tokens = nst
                        logger.info(
                            "Detected spec config from /version: method=%s, draft=%s, k=%s",
                            info.spec_method,
                            info.draft_model_name,
                            info.num_speculative_tokens,
                        )
        except Exception as exc:
            logger.debug("/version probe failed: %s", exc)

    return info
