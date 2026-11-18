"""Tests for speculative decoding / MTP benchmarking module."""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import MeasurementTestClient
from tests.coverage_helpers import MagicAsyncClient
from tests.coverage_helpers import throughput_sample as _cov_sample
from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.domain.spec_decode import suggested_draft_window, window_utilization
from tool_eval_bench.runner.speculative import (
    SpecDecodeCounters,
    SpecDecodeInfo,
    SpecDecodeSample,
    detect_spec_decoding,
    parse_prometheus_spec_metrics,
    pool_spec_samples,
    scrape_spec_metrics,
)
from tool_eval_bench.runner.throughput import ThroughputSample, TokenizerConfig

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
        assert c.acceptance_length == pytest.approx(4.0)

    def test_acceptance_length_zero_drafts(self):
        c = SpecDecodeCounters(accepted_tokens=50, num_drafts=0)
        assert c.acceptance_length is None

    def test_all_metrics_together(self):
        c = SpecDecodeCounters(accepted_tokens=200, draft_tokens=300, num_drafts=80)
        assert c.acceptance_rate == pytest.approx(200 / 300)
        assert c.acceptance_length == pytest.approx(1 + 200 / 80)


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
        assert counters.acceptance_length == pytest.approx(1 + 1542 / 525)

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
        assert counters.acceptance_length == pytest.approx(1 + 6086.0 / 3608.0)

    def test_vllm_sums_multiple_engine_series(self):
        text = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 40
vllm:spec_decode_num_accepted_tokens_total{engine="1"} 60
vllm:spec_decode_num_draft_tokens_total{engine="0"} 50
vllm:spec_decode_num_draft_tokens_total{engine="1"} 100
vllm:spec_decode_num_drafts_total{engine="0"} 10
vllm:spec_decode_num_drafts_total{engine="1"} 20
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.accepted_tokens == pytest.approx(100)
        assert counters.draft_tokens == pytest.approx(150)
        assert counters.num_drafts == pytest.approx(30)
        assert counters.acceptance_length == pytest.approx(1 + 100 / 30)

    def test_current_llamacpp_counter_names(self):
        text = """\
llamacpp:spec_decode_num_accepted_tokens_total 90
llamacpp:spec_decode_num_draft_tokens_total 300
llamacpp:spec_decode_num_drafts_total 30
"""
        counters = parse_prometheus_spec_metrics(text)
        assert counters.acceptance_rate == pytest.approx(0.3)
        assert counters.acceptance_length == pytest.approx(4.0)

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
        """Effective t/s = (output tokens - 1) / (wall time - TTFT)."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,  # 2 seconds total
            ttft_ms=500,  # 0.5s TTFT → 1.5s gen time
        )
        # The first token arrived at TTFT, so 99 tokens / 1.5s = 66.0 t/s
        assert s.effective_tg_tps == pytest.approx(99 / 1.5)

    def test_effective_tg_tps_matches_stream_rate_formula(self):
        """With no speculation gain, effective t/s equals the N-1 stream rate."""
        s = SpecDecodeSample(tg_tokens=101, total_ms=2100, ttft_ms=100, tg_tps=50.0)
        s.baseline_tg_tps = 50.0
        assert s.effective_tg_tps == pytest.approx(50.0)
        assert s.speedup_ratio == pytest.approx(1.0)

    def test_effective_tg_tps_single_token_keeps_full_count(self):
        """One token with a TTFT has no inter-token interval; keep N, not 0."""
        s = SpecDecodeSample(tg_tokens=1, total_ms=1000, ttft_ms=500)
        assert s.effective_tg_tps == pytest.approx(2.0)

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

    def test_goodput_is_none_without_acceptance_counters(self):
        """Without accepted tokens there is no goodput; output t/s is not a stand-in."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,
            ttft_ms=500,
        )
        assert s.goodput is None

    def test_goodput_zero_accepted_is_zero(self):
        """A measured zero acceptance is 0 t/s, distinct from unavailable."""
        s = SpecDecodeSample(tg_tokens=100, total_ms=2000, ttft_ms=500, accepted_tokens_delta=0)
        assert s.goodput == 0.0

    def test_goodput_zero_total_ms(self):
        s = SpecDecodeSample(tg_tokens=50, total_ms=0, accepted_tokens_delta=40)
        assert s.goodput == 0.0

    def test_speedup_ratio(self):
        """Speedup ratio = effective / baseline."""
        s = SpecDecodeSample(
            tg_tokens=100,
            total_ms=2000,
            ttft_ms=500,
            baseline_tg_tps=40.0,
        )
        effective = 99 / 1.5  # 66.0
        assert s.speedup_ratio == pytest.approx(effective / 40.0)

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

    @pytest.mark.asyncio
    async def test_fixed_prompt_reports_requested_context_depth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Code/structured prompts carry the requested depth as system-turn context."""
        import tool_eval_bench.runner.speculative as speculative

        async def fake_stream_one(*args, **kwargs):
            return ThroughputSample(pp_tokens=42, tg_tokens=4, total_ms=100, ttft_ms=10)

        monkeypatch.setattr(speculative, "_stream_one", fake_stream_one)
        sample = await speculative.measure_spec_single(
            None,  # type: ignore[arg-type]
            "http://localhost:8000/v1",
            "model",
            pp=8192,
            tg=4,
            depth=4096,
            prompt_type="code",
        )

        assert sample.depth == 4096
        assert sample.pp_tokens == 42

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
            tg_tokens=100,
            total_ms=2000,
            draft_tokens_delta=0,
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
            draft_tokens_delta=315,  # 15 tokens drafted per step
            accepted_tokens_delta=70,  # ~3.33 accepted drafts per step
            num_drafts_delta=21,
            acceptance_rate=70 / 315,
            acceptance_length=1 + 70 / 21,  # verifier token + accepted drafts
        )
        assert s.draft_window == pytest.approx(15.0)
        assert s.acceptance_length == pytest.approx(1 + 70 / 21, rel=0.01)
        # Window utilization: (τ - 1)/window = 3.33/15 = 22%, which equals α
        utilization = window_utilization(s.acceptance_length, s.draft_window)
        assert utilization == pytest.approx(70 / 315)
        assert utilization == pytest.approx(s.acceptance_rate)


class TestWindowUtilization:
    def test_full_acceptance_is_exactly_one(self):
        """Every drafted position accepted: τ = W + 1, utilization 100%, not above."""
        assert window_utilization(5.0, 4.0) == pytest.approx(1.0)

    def test_zero_acceptance_is_zero(self):
        """τ = 1 means no draft was accepted, so utilization is 0, not 1/W."""
        assert window_utilization(1.0, 4.0) == 0.0

    def test_empty_window_is_none(self):
        assert window_utilization(2.0, 0.0) is None

    def test_suggestion_never_matches_a_window_of_two(self):
        """A window of 2 is already the floor, so there is nothing to reduce to."""
        assert suggested_draft_window(1.2, 2.0) is None

    def test_suggestion_fires_for_an_oversized_window(self):
        assert suggested_draft_window(3.0, 15.0) == 3

    def test_suggestion_none_when_window_is_already_small_enough(self):
        assert suggested_draft_window(3.0, 3.0) is None


# ---------------------------------------------------------------------------
# ThroughputSample.effective_tg_tps (added property)
# ---------------------------------------------------------------------------


class TestThroughputSampleEffective:
    def test_effective_matches_wall_clock(self):
        """Effective t/s should use wall-clock time, not stream timing."""
        s = ThroughputSample(
            tg_tokens=128,
            total_ms=4000,  # 4s total
            ttft_ms=1000,  # 1s TTFT → 3s gen
            tg_tps=30.0,  # stream-measured (could differ)
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
        async with MeasurementTestClient(transport=transport) as client:
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
        async with MeasurementTestClient(transport=transport) as client:
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
        async with MeasurementTestClient(transport=transport) as client:
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
        async with MeasurementTestClient(transport=transport) as client:
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
        async with MeasurementTestClient(transport=transport) as client:
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
        async with MeasurementTestClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding

            info = await detect_spec_decoding(client, "http://host:8000/v1")
        assert info.active is False  # can't confirm spec decode without hint
        assert info.has_per_request_timings is True  # but we know the backend

    @staticmethod
    def _current_llamacpp_metrics(drafted: int, accepted: int, steps: int) -> str:
        # llama-server's to_metrics() renders these counters on every scrape.
        return (
            "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed\n"
            "llamacpp:prompt_tokens_total 212548\n"
            "# HELP llamacpp:spec_decode_num_draft_tokens_total Speculative: Total draft tokens\n"
            f"llamacpp:spec_decode_num_draft_tokens_total {drafted}\n"
            f"llamacpp:spec_decode_num_accepted_tokens_total {accepted}\n"
            f"llamacpp:spec_decode_num_drafts_total {steps}\n"
        )

    @pytest.mark.asyncio
    async def test_current_llamacpp_with_drafts_is_active_with_timings(self):
        import httpx

        body = self._current_llamacpp_metrics(5017, 1787, 1261)
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with MeasurementTestClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding

            info = await detect_spec_decoding(client, "http://host:8080/v1")
        assert info.active is True
        assert info.has_prometheus is True
        assert info.has_per_request_timings is True

    @pytest.mark.asyncio
    async def test_current_llamacpp_without_drafts_is_not_active(self):
        """The counters exist at zero on a server with no draft model."""
        import httpx

        body = self._current_llamacpp_metrics(0, 0, 0)
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with MeasurementTestClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding

            info = await detect_spec_decoding(client, "http://host:8080/v1")
        assert info.active is False
        assert info.has_prometheus is True
        assert info.has_per_request_timings is True

    @pytest.mark.asyncio
    async def test_llamacpp_in_a_label_value_does_not_claim_timings(self):
        import httpx

        body = 'vllm:spec_decode_num_draft_tokens_total{model_name="llamacpp:gemma"} 0\n'
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
        async with MeasurementTestClient(transport=transport) as client:
            from tool_eval_bench.runner.speculative import detect_spec_decoding

            info = await detect_spec_decoding(client, "http://host:8000/v1")
        assert info.active is True
        assert info.has_per_request_timings is False


async def _detect_from_metrics(body: str) -> SpecDecodeInfo:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with MeasurementTestClient(transport=transport) as client:
        return await detect_spec_decoding(client, "http://host:8000/v1")


_VLLM_SPEC_LINES = """\
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{{engine="0",model_name="{model}"}} 50.0
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="{model}"}} 200.0
vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="{model}"}} 100.0
"""


@pytest.mark.asyncio
async def test_scrape_spec_metrics_states() -> None:
    ok_body = (
        "spec_decode_num_accepted_tokens 100\n"
        "spec_decode_num_draft_tokens 200\n"
        "spec_decode_num_drafts 50\n"
    )
    async with MeasurementTestClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=ok_body))
    ) as client:
        c = await scrape_spec_metrics(client, "http://host:8000/v1")
    assert c is not None and c.accepted_tokens == pytest.approx(100)

    async with MeasurementTestClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, text="spec_decode_num_accepted_tokens 0\nspec_decode_num_draft_tokens 0\n"
            )
        )
    ) as client:
        zero_c = await scrape_spec_metrics(client, "http://host:8000/v1")
    assert zero_c is not None

    for transport in (
        httpx.MockTransport(lambda r: httpx.Response(200, text="some_other_metric 42\n")),
        httpx.MockTransport(lambda r: httpx.Response(404)),
        httpx.MockTransport(lambda r: (_ for _ in ()).throw(ConnectionError("down"))),
    ):
        async with MeasurementTestClient(transport=transport) as client:
            assert await scrape_spec_metrics(client, "http://host:8000/v1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "acme/eagle-7b",
        "/models/ngram-lookup-test",
        "deepseek-ai/DeepSeek-V3-MTP",
        "qwen-multi_token-preview",
        'org/method=\\"eagle\\"',
    ],
)
async def test_detect_spec_decoding_ignores_method_words_in_model_name(model: str) -> None:
    info = await _detect_from_metrics(_VLLM_SPEC_LINES.format(model=model))

    assert info.active is True
    assert info.method == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "# HELP vllm:spec_decode_num_draft_tokens_total EAGLE draft tokens.\n"
        "vllm:spec_decode_num_draft_tokens_total 200\n",
        "spec_decode_eagle_accepted 100\nspec_decode_num_draft_tokens 200\n",
        'vllm:spec_decode_num_draft_tokens_total 200\nhttp_requests_total{method="ngram"} 1\n',
    ],
    ids=["help-text", "metric-name", "method-label-off-spec-series"],
)
async def test_detect_spec_decoding_ignores_method_words_outside_labels(body: str) -> None:
    info = await _detect_from_metrics(body)

    assert info.active is True
    assert info.method == "unknown"


@pytest.mark.asyncio
async def test_detect_spec_decoding_reads_llamacpp_scrape_as_unknown_method() -> None:
    body = (
        "# HELP llamacpp:spec_decode_num_draft_tokens_total Speculative: Total draft tokens\n"
        "# TYPE llamacpp:spec_decode_num_draft_tokens_total counter\n"
        "llamacpp:spec_decode_num_draft_tokens_total 5017\n"
        "llamacpp:spec_decode_num_accepted_tokens_total 1787\n"
        "llamacpp:spec_decode_num_drafts_total 1261\n"
        'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 737\n'
    )

    info = await _detect_from_metrics(body)

    assert info.active is True
    assert info.method == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ('model_name="acme/ngram-7b",spec_method="eagle3"', "eagle3"),
        ('speculative_method="NEXTN"', "mtp"),
        ('method="ngram",model_name="acme/eagle-7b"', "ngram"),
    ],
)
async def test_detect_spec_decoding_reads_explicit_method_label(labels: str, expected: str) -> None:
    body = f"vllm:spec_decode_num_draft_tokens_total{{{labels}}} 200\n"

    info = await _detect_from_metrics(body)

    assert info.method == expected


@pytest.mark.asyncio
async def test_detect_spec_decoding_reads_spec_method_label_on_any_series() -> None:
    body = (
        "vllm:spec_decode_num_draft_tokens_total 200\n"
        'exporter:speculative_config{spec_method="suffix"} 1\n'
    )

    info = await _detect_from_metrics(body)

    assert info.method == "suffix"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("mtp", "mtp"),
        ("draft_model", "draft_model"),
        ("extract_hidden_states", "extract_hidden_states"),
        ("DRAFT", "draft_model"),
        ("eagle_parallel", "eagle_parallel"),
    ],
)
async def test_detect_spec_decoding_accepts_canonical_and_variant_hints(
    hint: str, expected: str
) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(404))
    async with MeasurementTestClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1", backend_hint=hint)
    assert info.active is True
    assert info.method == expected


@pytest.mark.asyncio
async def test_detect_spec_decoding_ignores_unknown_hint_and_errors() -> None:
    for transport in (
        httpx.MockTransport(lambda r: httpx.Response(404)),
        httpx.MockTransport(lambda r: (_ for _ in ()).throw(ConnectionError("down"))),
    ):
        async with MeasurementTestClient(transport=transport) as client:
            info = await detect_spec_decoding(
                client, "http://host:8000/v1", backend_hint="unknown_method"
            )
        assert info.active is False


@pytest.mark.asyncio
async def test_speculative_measurement_prometheus_and_llamacpp_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import speculative

    counters = iter(
        [
            speculative.SpecDecodeCounters(accepted_tokens=10, draft_tokens=20, num_drafts=5),
            speculative.SpecDecodeCounters(accepted_tokens=24, draft_tokens=40, num_drafts=9),
        ]
    )

    async def scrape(*args: object, **kwargs: object):
        return next(counters)

    async def stream(*args: object, **kwargs: object):
        return _cov_sample()

    monkeypatch.setattr(speculative, "scrape_spec_metrics", scrape)
    monkeypatch.setattr(speculative, "_stream_one", stream)
    sample = await speculative.measure_spec_single(
        MagicAsyncClient(),
        "url",
        "model",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True, method="mtp"),
    )
    assert sample.acceptance_source == "prometheus"
    assert sample.acceptance_rate == 0.7
    assert sample.acceptance_length == 4.5

    seen_messages: list[list[dict[str, object]]] = []

    async def stream_with_timings(
        client, base_url, model, messages, *args: object, **kwargs: object
    ):
        seen_messages.append(messages)
        return _cov_sample(draft_n=10, draft_n_accepted=6)

    monkeypatch.setattr(speculative, "_stream_one", stream_with_timings)
    monkeypatch.setattr(speculative, "scrape_spec_metrics", lambda *args, **kwargs: None)
    cfg = TokenizerConfig(chars_per_token=2.0)
    fallback = await speculative.measure_spec_single(
        MagicAsyncClient(),
        "url",
        "model",
        depth=512,
        prompt_type="structured",
        tok_cfg=cfg,
        spec_info=SpecDecodeInfo(),
    )
    assert fallback.acceptance_source == "timings"
    assert fallback.acceptance_rate == 0.6
    assert len(str(seen_messages[0][0]["content"])) >= 512


@pytest.mark.asyncio
async def test_speculative_full_sweep_invokes_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.runner import speculative

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object):
            return None

    async def calibrate_stub(*args: object, **kwargs: object) -> TokenizerConfig:
        return TokenizerConfig()

    async def detect_stub(*args: object, **kwargs: object) -> SpecDecodeInfo:
        return SpecDecodeInfo(has_per_request_timings=True)

    async def measure(*args: object, depth=0, prompt_type="filler", **kwargs: object):
        return SpecDecodeSample(depth=depth, prompt_type=prompt_type)

    seen: list[tuple[int, int]] = []

    async def callback(sample: SpecDecodeSample, idx: int, total: int) -> None:
        seen.append((idx, total))

    monkeypatch.setattr(speculative, "calibrate", calibrate_stub)
    monkeypatch.setattr(speculative, "detect_spec_decoding", detect_stub)
    monkeypatch.setattr(speculative, "measure_spec_single", measure)
    samples = await speculative.run_spec_bench(
        "url",
        "model",
        client_factory=lambda **kwargs: Client(),
        depths=[0, 1024],
        prompt_types=["filler", "code"],
        on_sample=callback,
    )

    assert len(samples) == 4
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]


@pytest.mark.asyncio
async def test_speculative_prometheus_reset_and_metric_parsers() -> None:
    empty_pooled = pool_spec_samples([])
    assert empty_pooled.error == "no samples"
    assert empty_pooled.runs == 0

    scrapes = [
        (
            "vllm:spec_decode_num_draft_tokens_total 100\n"
            "vllm:spec_decode_num_accepted_tokens_total 80\n"
            "vllm:spec_decode_num_drafts_total 25\n"
        ),
        (
            "vllm:spec_decode_num_draft_tokens_total 10\n"
            "vllm:spec_decode_num_accepted_tokens_total 8\n"
            "vllm:spec_decode_num_drafts_total 2\n"
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/metrics"):
            return httpx.Response(200, text=scrapes.pop(0))
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 16})
        sse = (
            'data: {"choices": [{"delta": {"content": "hello"}}], '
            '"usage": {"prompt_tokens": 16, "completion_tokens": 4}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse.encode(),
        )

    from tool_eval_bench.runner.speculative import measure_spec_single

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        spec_sample: SpecDecodeSample = await measure_spec_single(
            client,
            "http://localhost:8000",
            "m",
            pp=32,
            tg=4,
            tok_cfg=TokenizerConfig(
                has_tokenize_endpoint=True,
                supports_return_token_ids=False,
            ),
            spec_info=SpecDecodeInfo(method="eagle", has_prometheus=True),
        )
        assert spec_sample.error is None
        assert spec_sample.draft_tokens_delta is None
        assert spec_sample.accepted_tokens_delta is None

    prom_text = (
        "vllm:spec_decode_num_draft_tokens_total 1e400\n"
        "vllm:spec_decode_num_accepted_tokens_total 12\n"
        "vllm:spec_decode_num_drafts_total 1e400\n"
    )
    counters = parse_prometheus_spec_metrics(prom_text)
    assert counters.accepted_tokens == 12.0
    assert counters.draft_tokens == 0.0
    assert counters.has_num_drafts is False


def test_pool_spec_samples_keeps_non_null_acceptance_metadata_from_later_runs() -> None:
    s1 = SpecDecodeSample(
        pp_tokens=100,
        tg_tokens=20,
        ttft_ms=50.0,
        total_ms=250.0,
        pp_tps=2000.0,
        tg_tps=100.0,
        acceptance_source=None,
        num_spec_tokens=None,
    )
    s2 = SpecDecodeSample(
        pp_tokens=100,
        tg_tokens=20,
        ttft_ms=50.0,
        total_ms=250.0,
        pp_tps=2000.0,
        tg_tps=100.0,
        draft_tokens_delta=30,
        accepted_tokens_delta=24,
        acceptance_rate=0.8,
        acceptance_source="prometheus",
        num_spec_tokens=4,
    )
    pooled = pool_spec_samples([s1, s2])
    assert pooled.acceptance_rate == pytest.approx(0.8)
    assert pooled.acceptance_source == "prometheus"
    assert pooled.num_spec_tokens == 4
