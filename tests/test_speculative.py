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

    def test_vllm_labelled_format(self):
        """Parse vLLM metrics with {engine,model_name} labels (regression)."""
        text = """\
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Qwen3.6-35B"} 6086.0
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Qwen3.6-35B"} 7216.0
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="Qwen3.6-35B"} 3608.0
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == pytest.approx(6086.0)
        assert counters.draft_tokens == pytest.approx(7216.0)
        assert counters.num_drafts == pytest.approx(3608.0)
        assert counters.acceptance_rate == pytest.approx(6086.0 / 7216.0)
        assert counters.acceptance_length == pytest.approx(6086.0 / 3608.0)

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

    def test_scientific_notation_values(self):
        """vLLM reports large counters in scientific notation (e.g. 1.378e+06).

        Regression: the old regex only captured the mantissa, dropping the
        exponent and causing wrong counter values.
        """
        text = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 1.5e+04
vllm:spec_decode_num_draft_tokens_total{engine="0"} 2.1e+04
vllm:spec_decode_num_drafts_total{engine="0"} 5.25e+03
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == pytest.approx(15000.0)
        assert counters.draft_tokens == pytest.approx(21000.0)
        assert counters.num_drafts == pytest.approx(5250.0)
        assert counters.acceptance_rate == pytest.approx(15000 / 21000)


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

    # -- draft_tps --

    def test_draft_tps_basic(self):
        """Draft t/s = draft_tokens_delta / gen_time."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,
            ttft_ms=500,
            draft_tokens_delta=300,
        )
        # 300 draft tokens / 1.5s gen time = 200 draft t/s
        assert s.draft_tps == pytest.approx(200.0, rel=0.01)

    def test_draft_tps_none_without_deltas(self):
        """No draft data → None."""
        s = SpecDecodeSample(tg_tokens=100, total_ms=2000)
        assert s.draft_tps is None

    def test_draft_tps_zero_draft_tokens(self):
        """Zero draft tokens → None."""
        s = SpecDecodeSample(
            tg_tokens=100, total_ms=2000, draft_tokens_delta=0,
        )
        assert s.draft_tps is None

    # -- waste_ratio --

    def test_waste_ratio_basic(self):
        """Waste = 1 - acceptance_rate."""
        s = SpecDecodeSample(acceptance_rate=0.25)
        assert s.waste_ratio == pytest.approx(0.75)

    def test_waste_ratio_high_acceptance(self):
        """High acceptance → low waste."""
        s = SpecDecodeSample(acceptance_rate=0.90)
        assert s.waste_ratio == pytest.approx(0.10)

    def test_waste_ratio_none_without_acceptance(self):
        """No acceptance rate → None."""
        s = SpecDecodeSample()
        assert s.waste_ratio is None

    # -- draft_window --

    def test_draft_window_basic(self):
        """Window = draft_tokens / num_drafts."""
        s = SpecDecodeSample(
            draft_tokens_delta=315,
            num_drafts_delta=21,
        )
        # 315 / 21 = 15 tokens per draft step
        assert s.draft_window == pytest.approx(15.0)

    def test_draft_window_none_without_num_drafts(self):
        """Missing num_drafts → None."""
        s = SpecDecodeSample(draft_tokens_delta=300)
        assert s.draft_window is None

    def test_draft_window_zero_num_drafts(self):
        """Zero draft steps → None (avoid division by zero)."""
        s = SpecDecodeSample(
            draft_tokens_delta=300,
            num_drafts_delta=0,
        )
        assert s.draft_window is None

    def test_draft_window_vs_acceptance_length(self):
        """Window and τ together reveal utilization."""
        s = SpecDecodeSample(
            draft_tokens_delta=315,   # 15 tokens drafted per step
            accepted_tokens_delta=70, # ~3.33 accepted per step
            num_drafts_delta=21,
            acceptance_rate=70 / 315,
            acceptance_length=70 / 21,  # set by measure_spec_single
        )
        assert s.draft_window == pytest.approx(15.0)
        assert s.acceptance_length == pytest.approx(70 / 21, rel=0.01)
        # Window utilization: τ/window = 3.33/15 = 22% — poor
        utilization = s.acceptance_length / s.draft_window
        assert utilization == pytest.approx(0.222, rel=0.01)


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


# ---------------------------------------------------------------------------
# vLLM non-regression: detect_spec_decoding must NOT set has_per_request_timings
# ---------------------------------------------------------------------------


class TestDetectSpecDecodingVLLMNonRegression:
    """Verify detect_spec_decoding never claims per-request timings for vLLM.

    These tests cover all four vLLM scenarios to ensure the llama.cpp additions
    don't accidentally contaminate the vLLM detection path.
    """

    @pytest.mark.asyncio
    async def test_vllm_spec_decode_detected(self):
        """vLLM with spec_decode in /metrics → active, has_prometheus, NOT has_per_request_timings."""
        import httpx
        body = "spec_decode_num_accepted_tokens 100\nspec_decode_num_draft_tokens 200\n"
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding
            info = await detect_spec_decoding(client, "http://host:8000/v1")
        assert info.active is True
        assert info.has_prometheus is True
        assert info.has_per_request_timings is False  # CRITICAL: vLLM → no per-request

    @pytest.mark.asyncio
    async def test_vllm_spec_decode_with_hint(self):
        """vLLM with spec_decode + --spec-method=mtp → active, has_prometheus, NOT has_per_request_timings."""
        import httpx
        body = "spec_decode_mtp_tokens 100\nspec_decode_num_draft_tokens 200\n"
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding
            info = await detect_spec_decoding(client, "http://host:8000/v1", backend_hint="mtp")
        assert info.active is True
        assert info.has_prometheus is True
        assert info.has_per_request_timings is False  # CRITICAL
        assert info.method == "mtp"

    @pytest.mark.asyncio
    async def test_vllm_behind_proxy_with_hint(self):
        """vLLM behind proxy (/metrics unreachable) + --spec-method=mtp → active, NOT has_per_request_timings.

        Regression: previously this path set has_per_request_timings=True,
        which would incorrectly claim llama.cpp-style timings for vLLM.
        """
        import httpx
        transport = httpx.MockTransport(lambda r: httpx.Response(404))
        async with httpx.AsyncClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding
            info = await detect_spec_decoding(client, "http://host:8000/v1", backend_hint="mtp")
        assert info.active is True
        assert info.has_prometheus is False
        assert info.has_per_request_timings is False  # CRITICAL: don't assume llama.cpp
        assert info.method == "mtp"

    @pytest.mark.asyncio
    async def test_vllm_no_spec_decode(self):
        """vLLM without speculative decoding → nothing active."""
        import httpx
        # vLLM metrics without spec_decode counters
        body = "vllm:prompt_tokens_total 50000\nvllm:generation_tokens_total 12000\n"
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding
            info = await detect_spec_decoding(client, "http://host:8000/v1")
        assert info.active is False
        assert info.has_prometheus is False
        assert info.has_per_request_timings is False

    @pytest.mark.asyncio
    async def test_llamacpp_with_hint(self):
        """llama.cpp with llamacpp: metrics + --spec-method=mtp → active, has_per_request_timings."""
        import httpx
        body = "llamacpp:prompt_tokens_total 19345\nllamacpp:tokens_predicted_total 1157\n"
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding
            info = await detect_spec_decoding(client, "http://host:8000/v1", backend_hint="mtp")
        assert info.active is True
        assert info.has_prometheus is False
        assert info.has_per_request_timings is True  # llama.cpp confirmed
        assert info.method == "mtp"

    @pytest.mark.asyncio
    async def test_llamacpp_without_hint(self):
        """llama.cpp without hint → NOT active (can't confirm spec decode from /metrics alone)."""
        import httpx
        body = "llamacpp:prompt_tokens_total 19345\nllamacpp:tokens_predicted_total 1157\n"
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding
            info = await detect_spec_decoding(client, "http://host:8000/v1")
        assert info.active is False  # can't confirm spec decode without hint
        assert info.has_per_request_timings is True  # but we know the backend


class TestLlamaCppTimings:
    """Tests for llama.cpp speculative decoding via per-request timings."""

    def test_spec_decode_info_has_per_request_timings(self):
        """SpecDecodeInfo tracks llama.cpp per-request timings flag."""
        info = SpecDecodeInfo(has_per_request_timings=True)
        assert info.has_per_request_timings is True
        assert info.has_prometheus is False
        assert info.active is False  # not auto-detected from /metrics

    def test_spec_decode_info_default_no_per_request(self):
        """Default SpecDecodeInfo has no per-request timings."""
        info = SpecDecodeInfo()
        assert info.has_per_request_timings is False

    def test_throughput_sample_draft_fields(self):
        """ThroughputSample has draft_n and draft_n_accepted from llama.cpp."""
        ts = ThroughputSample(
            pp_tokens=2048,
            tg_tokens=128,
            draft_n=156,
            draft_n_accepted=120,
        )
        assert ts.draft_n == 156
        assert ts.draft_n_accepted == 120

    def test_throughput_sample_draft_fields_default_none(self):
        """draft_n fields default to None."""
        ts = ThroughputSample()
        assert ts.draft_n is None
        assert ts.draft_n_accepted is None

    def test_spec_sample_from_llamacpp_timings(self):
        """SpecDecodeSample populated from llama.cpp per-request timings.

        When Prometheus counters are absent, measure_spec_single() should
        fall back to ThroughputSample.draft_n / draft_n_accepted.
        """
        # Simulate what _stream_one returns for a llama.cpp server
        ts = ThroughputSample(
            pp_tokens=2048,
            tg_tokens=128,
            depth=0,
            concurrency=1,
            ttft_ms=200,
            total_ms=3000,
            pp_tps=10000,
            tg_tps=42.0,
            draft_n=156,
            draft_n_accepted=120,
        )
        spec = SpecDecodeSample.from_throughput_sample(ts, spec_method="mtp")

        # Simulate the fallback path in measure_spec_single
        # (no Prometheus data → use per-request timings)
        if spec.draft_tokens_delta is None and ts.draft_n is not None:
            spec.draft_tokens_delta = ts.draft_n
            spec.accepted_tokens_delta = ts.draft_n_accepted or 0
            if ts.draft_n > 0:
                spec.acceptance_rate = (ts.draft_n_accepted or 0) / ts.draft_n

        assert spec.draft_tokens_delta == 156
        assert spec.accepted_tokens_delta == 120
        assert spec.acceptance_rate == pytest.approx(120 / 156)
        assert spec.waste_ratio == pytest.approx(1.0 - 120 / 156)
        # acceptance_length and draft_window remain None (llama.cpp doesn't expose num_drafts)
        assert spec.acceptance_length is None
        assert spec.draft_window is None
        # effective t/s still works
        assert spec.effective_tg_tps > 0

    def test_spec_sample_llamacpp_zero_drafts(self):
        """When draft_n=0, acceptance_rate should not be set."""
        ts = ThroughputSample(
            tg_tokens=128,
            total_ms=3000,
            ttft_ms=200,
            draft_n=0,
            draft_n_accepted=0,
        )
        spec = SpecDecodeSample.from_throughput_sample(ts, spec_method="mtp")

        # Simulate fallback: draft_n=0 should not set acceptance_rate
        if spec.draft_tokens_delta is None and ts.draft_n is not None:
            spec.draft_tokens_delta = ts.draft_n
            spec.accepted_tokens_delta = ts.draft_n_accepted or 0
            if ts.draft_n > 0:
                spec.acceptance_rate = (ts.draft_n_accepted or 0) / ts.draft_n

        assert spec.draft_tokens_delta == 0
        assert spec.acceptance_rate is None  # no drafts → no rate

    def test_spec_sample_llamacpp_all_accepted(self):
        """Perfect acceptance: draft_n == draft_n_accepted."""
        ts = ThroughputSample(
            tg_tokens=128,
            total_ms=3000,
            ttft_ms=200,
            draft_n=100,
            draft_n_accepted=100,
        )
        spec = SpecDecodeSample.from_throughput_sample(ts, spec_method="draft")

        if spec.draft_tokens_delta is None and ts.draft_n is not None:
            spec.draft_tokens_delta = ts.draft_n
            spec.accepted_tokens_delta = ts.draft_n_accepted or 0
            if ts.draft_n > 0:
                spec.acceptance_rate = (ts.draft_n_accepted or 0) / ts.draft_n

        assert spec.acceptance_rate == pytest.approx(1.0)
        assert spec.waste_ratio == pytest.approx(0.0)

    def test_prometheus_takes_precedence_over_timings(self):
        """When Prometheus deltas are available, they should win over timings."""
        ts = ThroughputSample(
            tg_tokens=128,
            total_ms=3000,
            ttft_ms=200,
            draft_n=100,        # llama.cpp timings present
            draft_n_accepted=80,
        )
        spec = SpecDecodeSample.from_throughput_sample(ts, spec_method="mtp")

        # Simulate Prometheus data being available (vLLM path)
        spec.draft_tokens_delta = 200      # from Prometheus
        spec.accepted_tokens_delta = 150   # from Prometheus
        spec.acceptance_rate = 150 / 200

        # Now the fallback check should NOT override
        if spec.draft_tokens_delta is None and ts.draft_n is not None:
            spec.draft_tokens_delta = ts.draft_n
            spec.accepted_tokens_delta = ts.draft_n_accepted or 0

        # Prometheus values should persist
        assert spec.draft_tokens_delta == 200
        assert spec.accepted_tokens_delta == 150
        assert spec.acceptance_rate == pytest.approx(0.75)
