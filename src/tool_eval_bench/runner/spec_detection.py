"""Speculative decoding detection and Prometheus counters, independent of benchmarks."""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass

from tool_eval_bench.domain.engines import (
    FLARE,
    LLAMACPP,
    SPEC_COUNTER_ORDER,
    SPLASH,
    EngineProfile,
    metrics_namespace_present,
)
from tool_eval_bench.domain.measurement import MeasurementClient

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
    has_num_drafts: bool = True

    @property
    def acceptance_rate(self) -> float | None:
        """Draft token acceptance rate (0.0–1.0)."""
        if self.draft_tokens > 0:
            return self.accepted_tokens / self.draft_tokens
        return None

    @property
    def acceptance_length(self) -> float | None:
        """Average output tokens per speculative step, including the verifier token."""
        if self.num_drafts > 0:
            return 1.0 + self.accepted_tokens / self.num_drafts
        return None


# Regex patterns for Prometheus counter lines
# Note: vLLM includes labels like {engine="0",model_name="..."} between the
# metric name and the value.  The (?:\{[^}]*\})? group handles this optional
# label block so we match both bare and labelled counter lines.
# Prometheus numeric value — handles plain and scientific notation (1.378e+06)
_NUM = r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

_PROM_PATTERNS = {
    # vLLM metrics (both prefix variants) and current llama.cpp counters.
    "accepted_tokens": re.compile(
        rf"^(?:vllm[:_]|llamacpp:|tensorfold:)?spec_decode_num_accepted_tokens(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "draft_tokens": re.compile(
        rf"^(?:vllm[:_]|llamacpp:|tensorfold:)?spec_decode_num_draft_tokens(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "num_drafts": re.compile(
        rf"^(?:vllm[:_]|llamacpp:|tensorfold:)?spec_decode_num_drafts(?:_total)?(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
}

# Flare exposes cumulative speculative counters under its own ``flare_``
# namespace rather than the ``spec_decode_`` naming family.
_FLARE_PROM_PATTERNS = {
    "accepted_tokens": re.compile(
        rf"^flare_accepted_draft_tokens_total(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "draft_tokens": re.compile(
        rf"^flare_verification_draft_tokens_total(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "num_drafts": re.compile(
        rf"^flare_draft_request_rounds_total(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
}


# Splash renders its lifetime decode-batch draft counters under ``splash_``
# (server/metrics.py PROMETHEUS_SERIES). It exports no draft-cycle counter, so
# a Splash scrape never carries ``num_drafts``.
_SPLASH_PROM_PATTERNS = {
    "accepted_tokens": re.compile(
        rf"^splash_accepted_draft_tokens_total(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
    "draft_tokens": re.compile(
        rf"^splash_drafted_tokens_total(?:\{{[^}}]*\}})?\s+{_NUM}",
        re.MULTILINE,
    ),
}

# Engines whose speculative counters carry their own names instead of the
# ``spec_decode_`` family, in the order a mixed scrape is read.
_NAMESPACED_SPEC_PATTERNS: tuple[tuple[EngineProfile, dict[str, re.Pattern[str]]], ...] = (
    (FLARE, _FLARE_PROM_PATTERNS),
    (SPLASH, _SPLASH_PROM_PATTERNS),
)


def _has_namespaced_spec_counters(text: str, profile: EngineProfile | None = None) -> bool:
    """Whether *text* holds a finite counter sample of an engine-named family.

    *profile* restricts the check to that engine's family; ``None`` checks all.
    Finite, as ``parse_prometheus_spec_metrics`` reads them, so detection never
    names a family whose only samples the parser discards.
    """
    return any(
        metrics_namespace_present(text, owner) and _parse_counter_family(text, patterns)[1]
        for owner, patterns in _NAMESPACED_SPEC_PATTERNS
        if profile is None or owner is profile
    )


def _parse_counter_family(
    text: str, patterns: dict[str, re.Pattern[str]]
) -> tuple[SpecDecodeCounters, bool]:
    counters = SpecDecodeCounters(timestamp=time.time(), has_num_drafts=False)
    matched = False
    for field_name, pattern in patterns.items():
        finite = [
            val for match in pattern.finditer(text) if math.isfinite(val := float(match.group(1)))
        ]
        if finite:
            matched = True
            setattr(counters, field_name, sum(finite))
            if field_name == "num_drafts":
                counters.has_num_drafts = True
    return counters, matched


def parse_prometheus_spec_metrics(text: str) -> SpecDecodeCounters:
    """Parse speculative decoding counters from Prometheus text format.

    Works with vLLM, llama.cpp, TensorFold, Strata (``spec_decode_``
    counters), Flare (``flare_*`` draft counters) and Splash (``splash_*``
    draft counters, without a step count). Only one counter family is summed
    so a mixed scrape never double-counts.
    """
    counters, matched = _parse_counter_family(text, _PROM_PATTERNS)
    if matched:
        return counters
    for owner, patterns in _NAMESPACED_SPEC_PATTERNS:
        if metrics_namespace_present(text, owner):
            owned, owned_matched = _parse_counter_family(text, patterns)
            if owned_matched:
                return owned
    return counters


# ---------------------------------------------------------------------------
# Speculative method labels
# ---------------------------------------------------------------------------

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

# `[^"]` also matches a backslash, so `(?:\\.|[^"])*` gave the engine two ways
# to consume every escape and exponential backtracking to work through on a
# label that never closes its quote. Excluding the backslash from the negated
# class leaves exactly one parse. Metrics text comes off the wire from whatever
# server the run points at, so this is reachable input.
_LABEL_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')

_LABELLED_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)\{(?P<labels>[^}]*)\}",
    re.MULTILINE,
)

# ``method`` is generic enough to mean something else on an unrelated metric,
# so it only counts on a speculative series. The longer names are unambiguous.
_SPEC_SERIES_METHOD_LABELS = ("spec_method", "speculative_method", "method")
_ANY_SERIES_METHOD_LABELS = ("spec_method", "speculative_method")


def _parse_labels(raw_labels: str | None) -> dict[str, str]:
    """Parse the simple quoted labels emitted by Prometheus text format."""
    if not raw_labels:
        return {}
    return {
        name: value.replace(r"\"", '"').replace(r"\\", "\\")
        for name, value in _LABEL_PATTERN.findall(raw_labels)
    }


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


def canonical_spec_method_hint(value: str) -> str | None:
    """The method name to report and store for a ``--spec-method`` value.

    Aliases such as ``draft``, ``standalone``, and ``nextn`` become
    ``draft_model`` and ``mtp``, so spec-bench, spec-live, and stored rows use
    one name per method. Returns None for ``auto`` (or an empty value).
    """
    if value in ("auto", ""):
        return None
    return _canonical_spec_method(value) or value


def _detect_spec_method(text: str, profile: EngineProfile | None = None) -> str:
    """Detect a method only when a sample carries an explicit method label.

    No vLLM, SGLang, or llama.cpp exporter puts a method label on its
    speculative series, so for them this returns ``unknown``. Only a label
    whose name says it holds the method counts. A model name, a path, or HELP
    text that happens to contain ``eagle`` or ``mtp`` says nothing about how
    the server drafts.

    When no label names a method, *profile*'s ``fixed_spec_method`` does.
    Detection and spec-live both resolve through here so they cannot disagree.
    """
    method = _labelled_spec_method(text)
    if method == "unknown" and profile is not None and profile.fixed_spec_method:
        return profile.fixed_spec_method
    return method


def _labelled_spec_method(text: str) -> str:
    for match in _LABELLED_SAMPLE.finditer(text):
        name = match.group("name").lower()
        is_spec_series = (
            "spec_decode" in name
            or name.startswith("sglang:spec_")
            or name.startswith("flare_speculative_")
            or name
            in {
                "flare_accepted_draft_tokens_total",
                "flare_verification_draft_tokens_total",
                "flare_draft_request_rounds_total",
            }
        )
        label_names = _SPEC_SERIES_METHOD_LABELS if is_spec_series else _ANY_SERIES_METHOD_LABELS
        labels = _parse_labels(match.group("labels"))
        for label_name in label_names:
            value = labels.get(label_name)
            if value is None:
                continue
            method = _canonical_spec_method(value)
            if method is not None:
                return method
    return "unknown"


async def scrape_spec_metrics(
    client: MeasurementClient,
    base_url: str,
    api_key: str | None = None,
    metrics_url: str | None = None,
) -> SpecDecodeCounters | None:
    """Scrape speculative decoding counters from /metrics endpoint.

    Returns None if the endpoint is unavailable or doesn't contain
    spec decode metrics.
    """
    try:
        measurement = client
        resp = await measurement.metrics(metrics_url=metrics_url)
        if resp.status_code != 200:
            return None
        counters = parse_prometheus_spec_metrics(resp.text)
        # Only return if at least one spec decode counter is non-zero
        # (meaning spec decode is actually active on the server)
        if counters.draft_tokens > 0 or counters.accepted_tokens > 0:
            return counters
        # Also return if counters are all zero but the metric names are present
        # (server has spec decode but hasn't processed any requests yet)
        if "spec_decode" in resp.text or _has_namespaced_spec_counters(resp.text):
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
    has_per_request_timings: bool = False  # llama.cpp: draft_n in response timings
    # vLLM --per-request-spec-decode-metrics. Learned from the first response
    # that carries it; once set, the Prometheus scrape around each request is
    # skipped because the response body is exact and request-scoped.
    has_per_request_metrics: bool = False
    detail: str = ""
    # Display name of the engine whose counter rules detection applied, or
    # None when no engine-specific rule matched. Log hints name it.
    engine_name: str | None = None


async def detect_spec_decoding(
    client: MeasurementClient,
    base_url: str,
    api_key: str | None = None,
    backend_hint: str = "auto",
    metrics_url: str | None = None,
) -> SpecDecodeInfo:
    """Probe whether speculative decoding is active on the server.

    Detection strategy:
    1. Check /metrics for spec_decode counters (vLLM / SGLang)
    2. Check /metrics for llamacpp: prefix (llama.cpp — per-request timings)
    3. Accept user hint via backend_hint
    """
    info = SpecDecodeInfo()

    # Try Prometheus endpoint
    try:
        measurement = client
        resp = await measurement.metrics(metrics_url=metrics_url)
        if resp.status_code == 200:
            text = resp.text

            # vLLM, Flare, Splash, and compatible exporters: look for spec counters.
            if "spec_decode" in text or _has_namespaced_spec_counters(text):
                info.active = True
                info.has_prometheus = True
                info.detail = "Detected via Prometheus /metrics (spec_decode counters present)"

                # Only trust an explicit method token. Generic counters prove
                # activity, but do not identify the configured proposer.
                # Namespaces are matched at line start so a label value such
                # as model_name="acme/strata:7b" does not name the engine.
                # Every engine in SPEC_COUNTER_ORDER renders its counters
                # whether or not it drafts, so only drafted tokens prove activity.
                _, has_standard_counters = _parse_counter_family(text, _PROM_PATTERNS)
                namespaced = {owner for owner, _ in _NAMESPACED_SPEC_PATTERNS}
                # parse_prometheus_spec_metrics reads an engine-named family only
                # when no spec_decode_ family is present. The rules applied here
                # must belong to the family it reads, so a load-only namespace
                # (llamacpp:, strata:) never claims another engine's counters.
                reads_namespaced = not has_standard_counters and _has_namespaced_spec_counters(text)
                for profile in SPEC_COUNTER_ORDER:
                    if profile in namespaced:
                        if not reads_namespaced or not _has_namespaced_spec_counters(text, profile):
                            continue
                    elif reads_namespaced:
                        continue
                    if metrics_namespace_present(text, profile):
                        info.active = parse_prometheus_spec_metrics(text).draft_tokens > 0
                        info.has_per_request_timings = profile.absent_draft_n_is_zero
                        info.method = _detect_spec_method(text, profile)
                        info.detail = profile.spec_counter_detail or info.detail
                        info.engine_name = profile.display_name
                        break
                else:
                    info.method = _detect_spec_method(text)

            # llama.cpp: no spec_decode counters, but we can detect the backend
            # and know that draft stats will come from per-request timings.
            # Anchored like the check above, so "llamacpp:" inside a label
            # value or HELP text does not make another server llama.cpp.
            elif metrics_namespace_present(text, LLAMACPP):
                info.has_per_request_timings = True
                info.engine_name = LLAMACPP.display_name
                info.detail = (
                    "llama.cpp detected — spec decode metrics available "
                    "via per-request timings (draft_n/draft_n_accepted)"
                )
                # Don't set active=True yet — we'll confirm per-request
    except Exception as exc:
        logger.debug("Spec decode detection probe failed: %s", exc)

    # Accept user hint
    if backend_hint not in ("auto", ""):
        canonical = _canonical_spec_method(backend_hint)
        if canonical is not None:
            info.method = canonical
            if not info.active:
                info.active = True
                # Only assume per-request timings if we positively identified
                # a llama.cpp backend from /metrics.  When /metrics is simply
                # unreachable (e.g. vLLM behind a proxy), we don't know the
                # backend and shouldn't claim per-request timings support.
                if not info.has_per_request_timings:
                    # Unknown backend — Prometheus may just be unreachable.
                    # Spec bench will try Prometheus first, then per-request fallback.
                    pass
                info.detail = f"Assumed active via --spec-method={backend_hint}"

    return info
