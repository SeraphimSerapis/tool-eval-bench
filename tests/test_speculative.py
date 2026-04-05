"""Tests for speculative decoding / MTP benchmarking module."""

from __future__ import annotations

import pytest

from tool_eval_bench.runner.speculative import (
    SpecDecodeCounters,
    SpecDecodeSample,
    SpecDecodeInfo,
    parse_prometheus_spec_metrics,
)
from tool_eval_bench.runner.throughput import ThroughputSample


# ---------------------------------------------------------------------------
# SpecDecodeCounters
# ---------------------------------------------------------------------------


class TestSpecDecodeCounters:
    def test_acceptance_rate_basic(self):
        c = SpecDecodeCounters(accepted_tokens=75, draft_tokens=100)
        assert c.acceptance_rate == pytest.approx(0.75)

    def test_acceptance_rate_zero_drafts(self):
        c = SpecDecodeCounters(accepted_tokens=0, draft_tokens=0)
        assert c.acceptance_rate is None

    def test_acceptance_length_basic(self):
        c = SpecDecodeCounters(accepted_tokens=120, num_drafts=40)
        assert c.acceptance_length == pytest.approx(3.0)

    def test_acceptance_length_zero_drafts(self):
        c = SpecDecodeCounters(accepted_tokens=50, num_drafts=0)
        assert c.acceptance_length is None

    def test_all_metrics_together(self):
        c = SpecDecodeCounters(accepted_tokens=200, draft_tokens=300, num_drafts=80)
        assert c.acceptance_rate == pytest.approx(200 / 300)
        assert c.acceptance_length == pytest.approx(200 / 80)


# ---------------------------------------------------------------------------
# Prometheus parsing
# ---------------------------------------------------------------------------


class TestParsePrometheusMetrics:
    def test_vllm_format(self):
        """Parse vLLM-style Prometheus metrics."""
        text = """\
# HELP vllm:spec_decode_num_accepted_tokens_total Total accepted tokens
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total 1542

# HELP vllm:spec_decode_num_draft_tokens_total Total draft tokens
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total 2100

# HELP vllm:spec_decode_num_drafts_total Total draft steps
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total 525
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == pytest.approx(1542)
        assert counters.draft_tokens == pytest.approx(2100)
        assert counters.num_drafts == pytest.approx(525)
        assert counters.acceptance_rate == pytest.approx(1542 / 2100)
        assert counters.acceptance_length == pytest.approx(1542 / 525)

    def test_without_vllm_prefix(self):
        """Parse metrics without the vllm: prefix."""
        text = """\
spec_decode_num_accepted_tokens 800
spec_decode_num_draft_tokens 1200
spec_decode_num_drafts 300
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == pytest.approx(800)
        assert counters.draft_tokens == pytest.approx(1200)
        assert counters.num_drafts == pytest.approx(300)

    def test_with_total_suffix(self):
        """Parse metrics with _total suffix (standard Prometheus convention)."""
        text = """\
spec_decode_num_accepted_tokens_total 500.0
spec_decode_num_draft_tokens_total 750.0
spec_decode_num_drafts_total 200.0
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == pytest.approx(500.0)
        assert counters.draft_tokens == pytest.approx(750.0)
        assert counters.num_drafts == pytest.approx(200.0)

    def test_empty_metrics(self):
        """Parse empty or irrelevant metrics text."""
        text = """\
# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 0
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == 0
        assert counters.draft_tokens == 0
        assert counters.num_drafts == 0
        assert counters.acceptance_rate is None

    def test_partial_metrics(self):
        """Handle partial metrics (some counters present, others missing)."""
        text = "spec_decode_num_accepted_tokens 100\nspec_decode_num_draft_tokens 200\n"
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == pytest.approx(100)
        assert counters.draft_tokens == pytest.approx(200)
        assert counters.num_drafts == 0  # missing
        assert counters.acceptance_rate == pytest.approx(0.5)
        assert counters.acceptance_length is None


# ---------------------------------------------------------------------------
# SpecDecodeSample
# ---------------------------------------------------------------------------


class TestSpecDecodeSample:
    def test_effective_tg_tps(self):
        """Effective t/s = output tokens / (wall time - TTFT)."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,  # 2 seconds total
            ttft_ms=500,    # 0.5s TTFT → 1.5s gen time
        )
        # 100 tokens / 1.5s = 66.67 t/s
        assert s.effective_tg_tps == pytest.approx(100 / 1.5, rel=0.01)

    def test_effective_tg_tps_no_ttft(self):
        """When TTFT is 0, use total time."""
        s = SpecDecodeSample(tg_tokens=50, total_ms=1000, ttft_ms=0)
        assert s.effective_tg_tps == pytest.approx(50.0)

    def test_effective_tg_tps_no_tokens(self):
        """Zero tokens → 0 effective t/s."""
        s = SpecDecodeSample(tg_tokens=0, total_ms=1000)
        assert s.effective_tg_tps == 0.0

    def test_goodput_with_accepted_tokens(self):
        """Goodput uses accepted tokens when available."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,
            ttft_ms=500,
            accepted_tokens_delta=80,
        )
        # 80 accepted / 1.5s gen time = 53.33 t/s
        assert s.goodput == pytest.approx(80 / 1.5, rel=0.01)

    def test_goodput_falls_back_to_effective(self):
        """Without accepted tokens, goodput = effective t/s."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,
            ttft_ms=500,
        )
        assert s.goodput == s.effective_tg_tps

    def test_speedup_ratio(self):
        """Speedup ratio = effective / baseline."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,
            ttft_ms=500,
            baseline_tg_tps=40.0,
        )
        effective = 100 / 1.5  # ~66.67
        assert s.speedup_ratio == pytest.approx(effective / 40.0, rel=0.01)

    def test_speedup_ratio_none_without_baseline(self):
        """No baseline → no speedup ratio."""
        s = SpecDecodeSample(tg_tokens=100, total_ms=2000, ttft_ms=500)
        assert s.speedup_ratio is None

    def test_from_throughput_sample(self):
        """Construct from a base ThroughputSample."""
        ts = ThroughputSample(
            pp_tokens=2048,
            tg_tokens=128,
            depth=0,
            concurrency=1,
            ttft_ms=200,
            total_ms=5000,
            pp_tps=10000,
            tg_tps=25.0,
        )
        spec = SpecDecodeSample.from_throughput_sample(ts, spec_method="mtp")
        assert spec.pp_tokens == 2048
        assert spec.tg_tokens == 128
        assert spec.tg_tps == 25.0
        assert spec.spec_method == "mtp"
        assert spec.acceptance_rate is None
        assert spec.effective_tg_tps > 0


# ---------------------------------------------------------------------------
# SpecDecodeInfo
# ---------------------------------------------------------------------------


class TestSpecDecodeInfo:
    def test_default_not_active(self):
        info = SpecDecodeInfo()
        assert info.active is False
        assert info.has_prometheus is False
        assert info.method == "unknown"

    def test_active_with_prometheus(self):
        info = SpecDecodeInfo(active=True, has_prometheus=True, method="mtp")
        assert info.active is True
        assert info.method == "mtp"


# ---------------------------------------------------------------------------
# ThroughputSample.effective_tg_tps (added property)
# ---------------------------------------------------------------------------


class TestThroughputSampleEffective:
    def test_effective_matches_wall_clock(self):
        """Effective t/s should use wall-clock time, not stream timing."""
        s = ThroughputSample(
            tg_tokens=128,
            total_ms=4000,   # 4s total
            ttft_ms=1000,    # 1s TTFT → 3s gen
            tg_tps=30.0,     # stream-measured (could differ)
        )
        # 128 / 3.0 = 42.67 — different from stream tg_tps
        assert s.effective_tg_tps == pytest.approx(128 / 3.0, rel=0.01)

    def test_effective_zero_tokens(self):
        s = ThroughputSample(tg_tokens=0, total_ms=1000)
        assert s.effective_tg_tps == 0.0

    def test_effective_vs_stream(self):
        """For standard decoding, effective ≈ stream. For spec-decode, effective > stream."""
        # Simulate spec-decode scenario: 128 tokens in 2s wall clock
        # but stream measured 30 t/s (because it sees individual chunks)
        s = ThroughputSample(
            tg_tokens=128,
            total_ms=2500,
            ttft_ms=500,
            tg_tps=30.0,
        )
        # effective = 128 / 2.0 = 64 t/s — 2x the stream measurement
        assert s.effective_tg_tps == pytest.approx(64.0)
        assert s.effective_tg_tps > s.tg_tps


# ---------------------------------------------------------------------------
# Regression: run_spec_bench attribute access
# ---------------------------------------------------------------------------


class TestRunSpecBenchAttributes:
    """Regression test for attribute name consistency in run_spec_bench."""

    def test_spec_decode_info_has_prometheus_attribute(self):
        """Ensure SpecDecodeInfo uses 'has_prometheus' consistently.

        Regression: run_spec_bench previously accessed 'prometheus_available'
        which doesn't exist, causing an AttributeError at runtime.
        """
        info = SpecDecodeInfo(active=True, has_prometheus=True, method="mtp")
        # Verify the attribute exists and is accessible
        assert hasattr(info, "has_prometheus")
        assert info.has_prometheus is True
        # Verify the old wrong name does NOT exist
        assert not hasattr(info, "prometheus_available")

    def test_spec_decode_info_without_prometheus(self):
        info = SpecDecodeInfo(active=True, has_prometheus=False, method="mtp")
        assert info.has_prometheus is False

