"""Speculative decoding detection and Prometheus counters, independent of benchmarks."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

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
                if "tensorfold:" in text:
                    # Counters exist even with drafting off, and mtp_*
                    # aliases count all proposers, not just MTP.
                    info.active = parse_prometheus_spec_metrics(text).draft_tokens > 0
                    info.method = "unknown"
                    info.detail = (
                        "TensorFold draft counters available; proposer configuration unknown"
                    )
                elif "eagle" in text.lower():
                    info.method = "eagle"
                elif "ngram" in text.lower():
                    info.method = "ngram"
                elif "mtp" in text.lower() or "multi_token" in text.lower():
                    info.method = "mtp"
                else:
                    info.method = "unknown"

            # llama.cpp: no spec_decode counters, but we can detect the backend
            # and know that draft stats will come from per-request timings
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
