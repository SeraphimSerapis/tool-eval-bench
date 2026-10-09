"""Speculative decoding detection and Prometheus counters, independent of benchmarks."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from tool_eval_bench.domain.engines import SPEC_COUNTER_ORDER, metrics_namespace_present
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


def parse_prometheus_spec_metrics(text: str) -> SpecDecodeCounters:
    """Parse speculative decoding counters from Prometheus text format.

    Works with vLLM and compatible servers exposing counters with
    the ``spec_decode_`` prefix.
    """
    counters = SpecDecodeCounters(timestamp=time.time())

    for field_name, pattern in _PROM_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            setattr(counters, field_name, sum(float(match.group(1)) for match in matches))

    counters.has_num_drafts = bool(_PROM_PATTERNS["num_drafts"].search(text))
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


def _detect_spec_method(text: str) -> str:
    """Detect a method only when a sample carries an explicit method label.

    No vLLM, SGLang, or llama.cpp exporter puts a method label on its
    speculative series, so for them this returns ``unknown``. Only a label
    whose name says it holds the method counts. A model name, a path, or HELP
    text that happens to contain ``eagle`` or ``mtp`` says nothing about how
    the server drafts.
    """
    for match in _LABELLED_SAMPLE.finditer(text):
        name = match.group("name").lower()
        is_spec_series = "spec_decode" in name or name.startswith("sglang:spec_")
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
    has_per_request_timings: bool = False  # llama.cpp: draft_n in response timings
    # vLLM --per-request-spec-decode-metrics. Learned from the first response
    # that carries it; once set, the Prometheus scrape around each request is
    # skipped because the response body is exact and request-scoped.
    has_per_request_metrics: bool = False
    detail: str = ""


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

            # vLLM and compatible exporters: look for spec_decode counters.
            if "spec_decode" in text:
                info.active = True
                info.has_prometheus = True
                info.detail = "Detected via Prometheus /metrics (spec_decode counters present)"

                # Only trust an explicit method token. Generic counters prove
                # activity, but do not identify the configured proposer.
                # Namespaces are matched at line start so a label value such
                # as model_name="acme/strata:7b" does not name the engine.
                # Every engine in SPEC_COUNTER_ORDER renders its counters on
                # every scrape, so only drafted tokens prove activity.
                for profile in SPEC_COUNTER_ORDER:
                    if metrics_namespace_present(text, profile):
                        info.active = parse_prometheus_spec_metrics(text).draft_tokens > 0
                        info.has_per_request_timings = profile.absent_draft_n_is_zero
                        info.method = profile.fixed_spec_method or "unknown"
                        info.detail = profile.spec_counter_detail or info.detail
                        break
                else:
                    info.method = _detect_spec_method(text)

            # llama.cpp: no spec_decode counters, but we can detect the backend
            # and know that draft stats will come from per-request timings.
            # Unlike the check above this is a substring test, so a label value
            # containing "llamacpp:" also matches.
            elif "llamacpp:" in text:
                info.has_per_request_timings = True
                info.detail = (
                    "llama.cpp detected — spec decode metrics available "
                    "via per-request timings (draft_n/draft_n_accepted)"
                )
                # Don't set active=True yet — we'll confirm per-request
    except Exception as exc:
        logger.debug("Spec decode detection probe failed: %s", exc)

    # Accept user hint
    if backend_hint not in ("auto", ""):
        if backend_hint in {
            "mtp",
            "nextn",
            "draft",
            "standalone",
            "dflash",
            "dspark",
            "ngram",
            "ngram_gpu",
            "eagle",
            "eagle3",
            "medusa",
            "mlp_speculator",
            "suffix",
            "custom_class",
        }:
            info.method = backend_hint
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
