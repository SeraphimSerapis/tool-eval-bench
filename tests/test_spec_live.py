"""Tests for the live speculative decoding monitor.

Covers:
- runner/spec_live.py: Prometheus parsing, snapshot, delta computation
- cli/spec_live_display.py: dashboard rendering, gauge helpers, sparklines
"""

from __future__ import annotations

import time
from collections import deque

import pytest

from tool_eval_bench.runner.spec_live import (
    MetricsSnapshot,
    SpecLiveDelta,
    _parse_labels,
    _parse_snapshot,
    compute_delta,
    metrics_url_from_base,
)

# ---------------------------------------------------------------------------
# MetricsSnapshot parsing
# ---------------------------------------------------------------------------


class TestParseSnapshot:
    """Test Prometheus text → MetricsSnapshot parsing."""

    FULL_VLLM_METRICS = """\
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Qwen3.6-35B"} 1500.0
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Qwen3.6-35B"} 5000.0
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="Qwen3.6-35B"} 300.0
# HELP vllm:avg_prompt_throughput_toks_per_s Avg prompt throughput.
# TYPE vllm:avg_prompt_throughput_toks_per_s gauge
vllm:avg_prompt_throughput_toks_per_s{engine="0",model_name="Qwen3.6-35B"} 2580.9
# HELP vllm:avg_generation_throughput_toks_per_s Avg generation throughput.
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s{engine="0",model_name="Qwen3.6-35B"} 10.5
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{engine="0",model_name="Qwen3.6-35B"} 0.034
# HELP vllm:num_requests_running Number of running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen3.6-35B"} 1.0
# HELP vllm:num_requests_waiting Number of waiting requests.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen3.6-35B"} 0.0
# HELP vllm:prefix_cache_hit_rate Prefix cache hit rate.
# TYPE vllm:prefix_cache_hit_rate gauge
vllm:prefix_cache_hit_rate{engine="0",model_name="Qwen3.6-35B"} 0.45
"""

    def test_full_vllm_parse(self):
        snap = _parse_snapshot(self.FULL_VLLM_METRICS)
        assert snap.accepted_tokens == pytest.approx(1500.0)
        assert snap.draft_tokens == pytest.approx(5000.0)
        assert snap.num_drafts == pytest.approx(300.0)
        assert snap.prompt_tps == pytest.approx(2580.9)
        assert snap.generation_tps == pytest.approx(10.5)
        assert snap.gpu_cache_usage == pytest.approx(0.034)
        assert snap.running_reqs == pytest.approx(1.0)
        assert snap.waiting_reqs == pytest.approx(0.0)
        assert snap.prefix_cache_hit == pytest.approx(0.45)

    def test_has_spec_decode_true(self):
        snap = _parse_snapshot(self.FULL_VLLM_METRICS)
        assert snap.has_spec_decode is True

    def test_has_spec_decode_false(self):
        snap = _parse_snapshot("vllm:num_requests_running 0\n")
        assert snap.has_spec_decode is False

    def test_without_vllm_prefix(self):
        text = """\
spec_decode_num_accepted_tokens 800
spec_decode_num_draft_tokens 1200
spec_decode_num_drafts 300
"""
        snap = _parse_snapshot(text)
        assert snap.accepted_tokens == pytest.approx(800)
        assert snap.draft_tokens == pytest.approx(1200)
        assert snap.num_drafts == pytest.approx(300)
        assert snap.has_spec_decode is True

    def test_with_total_suffix(self):
        text = """\
spec_decode_num_accepted_tokens_total 500.0
spec_decode_num_draft_tokens_total 750.0
spec_decode_num_drafts_total 200.0
"""
        snap = _parse_snapshot(text)
        assert snap.accepted_tokens == pytest.approx(500.0)
        assert snap.draft_tokens == pytest.approx(750.0)
        assert snap.num_drafts == pytest.approx(200.0)

    def test_per_position_rates(self):
        text = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0"} 500.0
vllm:spec_decode_per_position_acceptance_rate{engine="0",position="0"} 0.658
vllm:spec_decode_per_position_acceptance_rate{engine="0",position="1"} 0.447
vllm:spec_decode_per_position_acceptance_rate{engine="0",position="2"} 0.263
vllm:spec_decode_per_position_acceptance_rate{engine="0",position="3"} 0.184
"""
        snap = _parse_snapshot(text)
        assert len(snap.per_position_rates) == 4
        assert snap.per_position_rates[0] == pytest.approx(0.658)
        assert snap.per_position_rates[1] == pytest.approx(0.447)
        assert snap.per_position_rates[2] == pytest.approx(0.263)
        assert snap.per_position_rates[3] == pytest.approx(0.184)

    def test_no_per_position_for_mtp(self):
        """MTP servers typically don't expose per-position rates."""
        text = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0"} 500.0
"""
        snap = _parse_snapshot(text)
        assert snap.per_position_rates == {}
        assert snap.has_spec_decode is True

    def test_empty_text(self):
        snap = _parse_snapshot("")
        assert snap.accepted_tokens == 0.0
        assert snap.draft_tokens == 0.0
        assert snap.has_spec_decode is False
        assert snap.per_position_rates == {}

    def test_scientific_notation_values(self):
        r"""vLLM reports large counters in scientific notation (e.g. 1.378e+06).

        Regression: the old regex (\d+(?:\.\d+)?) only captured the mantissa,
        dropping the exponent and causing wildly wrong calculations.
        """
        text = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0"} 500.0
vllm:prefix_cache_queries_total{engine="0"} 1.378852e+06
vllm:prefix_cache_hits_total{engine="0"} 41920.0
vllm:prompt_tokens_total{engine="0"} 2.5e+05
vllm:generation_tokens_total{engine="0"} 3.79e+04
"""
        snap = _parse_snapshot(text)
        assert snap.prefix_cache_queries == pytest.approx(1_378_852.0)
        assert snap.prefix_cache_hits == pytest.approx(41920.0)
        assert snap.prompt_tokens_total == pytest.approx(250_000.0)
        assert snap.generation_tokens_total == pytest.approx(37_900.0)

    def test_scientific_notation_negative_exponent(self):
        """Small values with negative exponents (e.g. cache fractions)."""
        text = """\
vllm:kv_cache_usage_perc{engine="0"} 7.643e-03
"""
        snap = _parse_snapshot(text)
        assert snap.kv_cache_usage == pytest.approx(0.007643, rel=1e-3)

    def test_kv_cache_none_when_not_present(self):
        """KV cache fields default to None when metric is absent."""
        snap = _parse_snapshot("vllm:num_requests_running 0\n")
        assert snap.kv_cache_usage is None
        assert snap.gpu_cache_usage is None

    def test_kv_cache_set_when_present(self):
        """KV cache fields are float when metric is found."""
        text = """\
vllm:kv_cache_usage_perc{engine="0"} 0.0
vllm:gpu_cache_usage_perc{engine="0"} 0.034
"""
        snap = _parse_snapshot(text)
        assert snap.kv_cache_usage == pytest.approx(0.0)
        assert snap.gpu_cache_usage == pytest.approx(0.034)

    def test_timestamp_set(self):
        before = time.time()
        snap = _parse_snapshot("spec_decode_num_accepted_tokens 100\n")
        after = time.time()
        assert before <= snap.timestamp <= after

    def test_vllm_counter_series_are_aggregated_across_engines(self):
        """Cumulative counters are additive, including per-position totals."""
        text = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 100
vllm:spec_decode_num_accepted_tokens_total{engine="1"} 50
vllm:spec_decode_num_draft_tokens_total{engine="0"} 200
vllm:spec_decode_num_draft_tokens_total{engine="1"} 100
vllm:spec_decode_num_drafts_total{engine="0"} 20
vllm:spec_decode_num_drafts_total{engine="1"} 10
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 80
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 40
"""
        snap = _parse_snapshot(text)
        assert snap.accepted_tokens == pytest.approx(150)
        assert snap.draft_tokens == pytest.approx(300)
        assert snap.num_drafts == pytest.approx(30)
        assert snap.per_position_counters == {0: pytest.approx(120)}

    def test_generic_counters_do_not_claim_a_method(self):
        snap = _parse_snapshot("vllm:spec_decode_num_draft_tokens_total 100\n")
        assert snap.spec_method == "unknown"

    def test_empty_or_unrelated_metrics_remain_graceful(self):
        snap = _parse_snapshot('python_gc_collections_total{generation="0"} 3\n')
        assert snap.has_spec_decode is False
        assert snap.has_llamacpp_metrics is False
        assert snap.has_sglang_metrics is False


# ---------------------------------------------------------------------------
# compute_delta
# ---------------------------------------------------------------------------


class TestComputeDelta:
    """Test delta computation between snapshots."""

    def _make_snap(self, **kwargs) -> MetricsSnapshot:
        defaults = dict(
            timestamp=time.time(),
            accepted_tokens=0.0,
            draft_tokens=0.0,
            num_drafts=0.0,
            prompt_tps=0.0,
            generation_tps=0.0,
            running_reqs=0.0,
            waiting_reqs=0.0,
            prefix_cache_hit=0.0,
        )
        defaults.update(kwargs)
        return MetricsSnapshot(**defaults)

    def test_basic_delta(self):
        prev = self._make_snap(
            timestamp=100.0, accepted_tokens=100, draft_tokens=400, num_drafts=50
        )
        curr = self._make_snap(
            timestamp=110.0,
            accepted_tokens=200,
            draft_tokens=800,
            num_drafts=100,
            generation_tps=10.5,
            prompt_tps=2580.9,
            gpu_cache_usage=0.034,
            running_reqs=1,
            waiting_reqs=0,
            prefix_cache_hit=0.45,
        )
        delta = compute_delta(prev, curr)

        assert delta.elapsed_s == pytest.approx(10.0)
        assert delta.had_activity is True

        # Interval rates
        assert delta.acceptance_rate == pytest.approx(100 / 400)
        assert delta.waste_ratio == pytest.approx(1.0 - 100 / 400)
        assert delta.acceptance_length == pytest.approx(1 + 100 / 50)
        assert delta.draft_window == pytest.approx(400 / 50)
        assert delta.accepted_tps == pytest.approx(100 / 10.0)
        assert delta.drafted_tps == pytest.approx(400 / 10.0)

        # Cumulative rates
        assert delta.cumulative_acceptance_rate == pytest.approx(200 / 800)
        assert delta.cumulative_acceptance_length == pytest.approx(1 + 200 / 100)
        assert delta.cumulative_draft_window == pytest.approx(800 / 100)

        # Gauges from current snapshot
        assert delta.generation_tps == pytest.approx(10.5)
        assert delta.prompt_tps == pytest.approx(2580.9)
        assert delta.gpu_cache_pct == pytest.approx(3.4)
        assert delta.running_reqs == 1
        assert delta.waiting_reqs == 0
        assert delta.prefix_cache_hit_pct == pytest.approx(45.0)

        # Totals
        assert delta.total_accepted == 200
        assert delta.total_drafted == 800

    def test_no_activity_delta(self):
        """When counters don't change (vLLM 10s interval), interval rates are None."""
        prev = self._make_snap(
            timestamp=100.0, accepted_tokens=500, draft_tokens=2000, num_drafts=250
        )
        curr = self._make_snap(
            timestamp=101.0, accepted_tokens=500, draft_tokens=2000, num_drafts=250
        )
        delta = compute_delta(prev, curr)

        assert delta.had_activity is False
        assert delta.acceptance_rate is None
        assert delta.waste_ratio is None
        assert delta.acceptance_length is None
        assert delta.draft_window is None
        assert delta.accepted_tps == pytest.approx(0.0)
        assert delta.drafted_tps == pytest.approx(0.0)

        # But cumulative rates are still valid
        assert delta.cumulative_acceptance_rate == pytest.approx(500 / 2000)
        assert delta.cumulative_acceptance_length == pytest.approx(1 + 500 / 250)

    def test_counter_reset_never_produces_negative_rates(self):
        """Engine restarts reset counters and must not create negative rates."""
        prev = self._make_snap(
            timestamp=100.0,
            accepted_tokens=1_000,
            draft_tokens=2_000,
            num_drafts=500,
            generation_tokens_total=10_000,
            prompt_tokens_total=20_000,
        )
        curr = self._make_snap(
            timestamp=110.0,
            accepted_tokens=20,
            draft_tokens=40,
            num_drafts=10,
            generation_tokens_total=100,
            prompt_tokens_total=200,
        )

        delta = compute_delta(prev, curr)

        assert delta.had_activity is True
        assert delta.acceptance_rate == pytest.approx(0.5)
        assert delta.accepted_tps == pytest.approx(2.0)
        assert delta.drafted_tps == pytest.approx(4.0)
        assert delta.generation_tps == pytest.approx(10.0)
        assert delta.prompt_tps == pytest.approx(20.0)
        assert delta.accepted_tps >= 0
        assert delta.drafted_tps >= 0

    def test_cumulative_rates_with_zero_totals(self):
        """Before any spec decode activity, cumulative rates are None."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0)
        delta = compute_delta(prev, curr)

        assert delta.cumulative_acceptance_rate is None
        assert delta.cumulative_acceptance_length is None
        assert delta.cumulative_draft_window is None

    def test_zero_elapsed_time(self):
        """Zero elapsed time should not cause division by zero."""
        prev = self._make_snap(timestamp=100.0, accepted_tokens=100, draft_tokens=400)
        curr = self._make_snap(timestamp=100.0, accepted_tokens=200, draft_tokens=800)
        delta = compute_delta(prev, curr)
        # Should use dt=1.0 fallback
        assert delta.elapsed_s == pytest.approx(1.0)
        assert delta.accepted_tps == pytest.approx(100.0)

    def test_per_position_rates_forwarded(self):
        """Per-position rates from snapshot are forwarded to delta."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0, accepted_tokens=100, draft_tokens=500)
        curr.per_position_rates = {0: 0.658, 1: 0.447, 2: 0.263}
        delta = compute_delta(prev, curr)
        assert delta.per_position_rates == {0: 0.658, 1: 0.447, 2: 0.263}

    def test_kv_cache_prefers_new_metric_even_if_zero(self):
        """kv_cache_usage=0.0 (present) should NOT fall back to gpu_cache_usage."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0, accepted_tokens=100, draft_tokens=500)
        curr.kv_cache_usage = 0.0  # genuinely zero (idle)
        curr.gpu_cache_usage = None  # not present (old metric)
        delta = compute_delta(prev, curr)
        assert delta.gpu_cache_pct == pytest.approx(0.0)

    def test_kv_cache_falls_back_to_gpu_cache(self):
        """When kv_cache_usage is None, fall back to gpu_cache_usage."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0, accepted_tokens=100, draft_tokens=500)
        curr.kv_cache_usage = None  # new metric not present
        curr.gpu_cache_usage = 0.05  # old metric present
        delta = compute_delta(prev, curr)
        assert delta.gpu_cache_pct == pytest.approx(5.0)

    def test_kv_cache_both_none(self):
        """When neither cache metric is present, cache_pct defaults to 0."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0, accepted_tokens=100, draft_tokens=500)
        curr.kv_cache_usage = None
        curr.gpu_cache_usage = None
        delta = compute_delta(prev, curr)
        assert delta.gpu_cache_pct == pytest.approx(0.0)

    def test_counter_derived_gen_tps(self):
        """When generation_tps gauge is 0, derive from token counter deltas."""
        prev = self._make_snap(
            timestamp=100.0,
            accepted_tokens=0,
            draft_tokens=0,
        )
        prev.generation_tokens_total = 1000.0
        curr = self._make_snap(
            timestamp=110.0,
            accepted_tokens=100,
            draft_tokens=500,
            generation_tps=0.0,  # gauge removed in vLLM ≥0.8
        )
        curr.generation_tokens_total = 1500.0  # 500 tokens in 10s
        delta = compute_delta(prev, curr)
        assert delta.generation_tps == pytest.approx(50.0)  # 500/10

    def test_counter_derived_prompt_tps(self):
        """When prompt_tps gauge is 0, derive from prompt_tokens_total deltas."""
        prev = self._make_snap(
            timestamp=100.0,
            accepted_tokens=0,
            draft_tokens=0,
        )
        prev.prompt_tokens_total = 10000.0
        curr = self._make_snap(
            timestamp=110.0,
            accepted_tokens=100,
            draft_tokens=500,
            prompt_tps=0.0,  # gauge removed
        )
        curr.prompt_tokens_total = 30000.0  # 20000 tokens in 10s
        delta = compute_delta(prev, curr)
        assert delta.prompt_tps == pytest.approx(2000.0)  # 20000/10

    def test_prefix_cache_counter_derived_rate(self):
        """When prefix_cache_hit gauge is 0, derive from hit/query counters."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(
            timestamp=101.0,
            accepted_tokens=100,
            draft_tokens=500,
        )
        curr.prefix_cache_hit = 0.0  # old gauge not present
        curr.prefix_cache_queries = 10000  # counter
        curr.prefix_cache_hits = 800  # counter
        delta = compute_delta(prev, curr)
        # 800/10000 = 0.08 → 8%
        assert delta.prefix_cache_hit_pct == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# metrics_url_from_base
# ---------------------------------------------------------------------------


class TestMetricsUrlFromBase:
    def test_plain_url(self):
        assert metrics_url_from_base("http://host:8000") == "http://host:8000/metrics"

    def test_trailing_slash(self):
        assert metrics_url_from_base("http://host:8000/") == "http://host:8000/metrics"

    def test_with_v1_suffix(self):
        assert metrics_url_from_base("http://host:8000/v1") == "http://host:8000/metrics"

    def test_with_v1_trailing_slash(self):
        assert metrics_url_from_base("http://host:8000/v1/") == "http://host:8000/metrics"


# ---------------------------------------------------------------------------
# Dashboard rendering helpers
# ---------------------------------------------------------------------------


class TestDashboardHelpers:
    """Test display helper functions from spec_live_display."""

    def test_ar_color_gradient(self):
        from tool_eval_bench.cli.spec_live_rendering import _ar_color

        assert _ar_color(0.0) == "bright_red"
        assert _ar_color(0.1) == "bright_red"
        assert _ar_color(0.25) == "red"
        assert _ar_color(0.4) == "dark_orange"
        assert _ar_color(0.55) == "yellow"
        assert _ar_color(0.7) == "green_yellow"
        assert _ar_color(0.85) == "bright_green"

    def test_gauge_bar_length(self):
        from tool_eval_bench.cli.spec_live_rendering import _gauge_bar

        bar = _gauge_bar(0.5, width=20)
        text_str = str(bar)
        # Should contain the percentage
        assert "50.0%" in text_str

    def test_gauge_bar_clamped(self):
        from tool_eval_bench.cli.spec_live_rendering import _gauge_bar

        # Values outside 0-1 should be clamped
        bar_high = _gauge_bar(1.5, width=10)
        assert "150.0%" in str(bar_high)  # label shows actual
        bar_low = _gauge_bar(-0.1, width=10)
        assert str(bar_low)  # should not crash

    def test_sparkline_empty(self):
        from tool_eval_bench.cli.spec_live_rendering import _sparkline

        spark = _sparkline([], width=10)
        assert len(str(spark)) == 10  # all dashes

    def test_sparkline_single_value(self):
        from tool_eval_bench.cli.spec_live_rendering import _sparkline

        spark = _sparkline([5.0], width=10)
        assert str(spark)  # should not crash

    def test_sparkline_constant_values(self):
        from tool_eval_bench.cli.spec_live_rendering import _sparkline

        spark = _sparkline([5.0, 5.0, 5.0], width=10)
        assert str(spark)  # should not crash, all same → range=0

    def test_sparkline_varying(self):
        from tool_eval_bench.cli.spec_live_rendering import _sparkline

        spark = _sparkline([1.0, 5.0, 3.0, 7.0, 2.0], width=10)
        text = str(spark)
        assert len(text) == 10  # padded to width

    def test_format_uptime(self):
        from tool_eval_bench.cli.spec_live_rendering import _format_uptime

        assert _format_uptime(0) == "00:00"
        assert _format_uptime(65) == "01:05"
        assert _format_uptime(3661) == "1:01:01"

    def test_position_bars_empty(self):
        from tool_eval_bench.cli.spec_live_rendering import _position_bars

        table = _position_bars({})
        # Should render without error — empty table (panel hidden when no data)
        from io import StringIO

        from rich.console import Console

        out = StringIO()
        Console(file=out, width=60, no_color=True).print(table)
        text = out.getvalue().strip()
        # Empty table should produce no content rows
        assert text == ""

    def test_position_bars_with_data(self):
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.spec_live_rendering import _position_bars

        rates = {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.2}
        table = _position_bars(rates)
        out = StringIO()
        Console(file=out, width=60, no_color=True).print(table)
        text = out.getvalue()
        assert "p0" in text
        assert "80.0%" in text
        assert "p3" in text


# ---------------------------------------------------------------------------
# Dashboard build (smoke tests)
# ---------------------------------------------------------------------------


class TestBuildDashboard:
    """Smoke tests for _build_dashboard — ensures it renders without errors."""

    def _make_delta(self, **kwargs) -> SpecLiveDelta:
        defaults = dict(
            elapsed_s=1.0,
            had_activity=True,
            cumulative_acceptance_rate=0.22,
            cumulative_acceptance_length=2.75,
            cumulative_draft_window=8.0,
            acceptance_rate=0.22,
            acceptance_length=2.75,
            draft_window=8.0,
            waste_ratio=0.78,
            accepted_tps=8.4,
            drafted_tps=14.0,
            prompt_tps=2500.0,
            generation_tps=10.5,
            gpu_cache_pct=3.4,
            running_reqs=1,
            waiting_reqs=0,
            prefix_cache_hit_pct=0.0,
            per_position_rates={0: 0.65, 1: 0.45, 2: 0.26},
            total_accepted=1500,
            total_drafted=5000,
            total_drafts=300,
        )
        defaults.update(kwargs)
        return SpecLiveDelta(**defaults)

    def _render(self, panel) -> str:
        from io import StringIO

        from rich.console import Console

        out = StringIO()
        Console(file=out, width=100, no_color=True).print(panel)
        return out.getvalue()

    def test_renders_with_data(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 60, "TestModel", "http://localhost:8000/metrics", 60
        )
        text = self._render(panel)
        assert "SPECULATIVE DECODING MONITOR" in text
        assert "TestModel" in text
        assert "ACCEPTANCE RATE" in text

    def test_renders_waiting_state(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        panel = _build_dashboard(
            None, deque(maxlen=60), time.time() - 5, "TestModel", "http://localhost:8000/metrics", 5
        )
        text = self._render(panel)
        assert "Connecting to" in text
        assert "spec decode enabled" in text

    def test_renders_without_per_position(self):
        """MTP models don't have per-position rates — panel is hidden."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(per_position_rates={})
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 30, "MTPModel", "http://localhost:8000/metrics", 30
        )
        text = self._render(panel)
        assert "Per-Position" not in text
        # Engine panel should still be present
        assert "Engine" in text

    def test_renders_with_rolling_averages(self):
        """Rolling averages panel appears after 5+ data points."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 30, "TestModel", "http://localhost:8000/metrics", 30
        )
        text = self._render(panel)
        assert "Rolling Averages" in text

    def test_renders_rolling_averages_early(self):
        """Rolling averages panel visible immediately with zero values."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 3, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 3, "TestModel", "http://localhost:8000/metrics", 3
        )
        text = self._render(panel)
        assert "Rolling Averages" in text

    def test_renders_high_acceptance(self):
        """Dashboard with excellent acceptance rate."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(
            cumulative_acceptance_rate=0.75,
            cumulative_acceptance_length=6.0,
            cumulative_draft_window=8.0,
        )
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 60, "TestModel", "http://localhost:8000/metrics", 60
        )
        text = self._render(panel)
        assert "Excellent" in text

    def test_renders_poor_acceptance(self):
        """Dashboard with poor acceptance rate shows warning."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(
            cumulative_acceptance_rate=0.10,
            cumulative_acceptance_length=0.8,
            cumulative_draft_window=8.0,
        )
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 60, "TestModel", "http://localhost:8000/metrics", 60
        )
        text = self._render(panel)
        assert "Poor" in text

    def test_efficiency_insight_tuning_hint(self):
        """Low utilization triggers num_speculative_tokens suggestion."""
        from tool_eval_bench.cli.spec_live_rendering import _efficiency_insight

        delta = self._make_delta(
            cumulative_acceptance_rate=0.15,
            cumulative_acceptance_length=1.5,
            cumulative_draft_window=8.0,
        )
        text = str(_efficiency_insight(delta))
        assert "num_speculative_tokens" in text

    def test_efficiency_insight_no_hint_when_good(self):
        """Good utilization does not trigger tuning hint."""
        from tool_eval_bench.cli.spec_live_rendering import _efficiency_insight

        delta = self._make_delta(
            cumulative_acceptance_rate=0.75,
            cumulative_acceptance_length=6.0,
            cumulative_draft_window=8.0,
        )
        text = str(_efficiency_insight(delta))
        assert "num_speculative_tokens" not in text

    def test_efficiency_insight_awaiting(self):
        """No cumulative rate → awaiting message."""
        from tool_eval_bench.cli.spec_live_rendering import _efficiency_insight

        delta = self._make_delta(cumulative_acceptance_rate=None)
        text = str(_efficiency_insight(delta))
        assert "awaiting" in text


# ---------------------------------------------------------------------------
# llama.cpp metrics support
# ---------------------------------------------------------------------------


class TestLlamaCppSnapshot:
    """Test parsing of llama.cpp Prometheus metrics."""

    LLAMACPP_METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 19345
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 1157
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 28.3926
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 1234.5
# HELP llamacpp:requests_processing Number of requests currently being processed.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
# HELP llamacpp:requests_deferred Number of requests waiting.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 1
"""

    def test_parse_llamacpp_metrics(self):
        snap = _parse_snapshot(self.LLAMACPP_METRICS)
        assert snap.llamacpp_prompt_tokens_total == pytest.approx(19345.0)
        assert snap.llamacpp_predicted_tokens_total == pytest.approx(1157.0)
        assert snap.llamacpp_predicted_tokens_seconds == pytest.approx(28.3926)
        assert snap.llamacpp_prompt_tokens_seconds == pytest.approx(1234.5)
        assert snap.llamacpp_requests_processing == pytest.approx(2.0)
        assert snap.llamacpp_requests_deferred == pytest.approx(1.0)

    def test_has_spec_decode_false_for_llamacpp(self):
        """llama.cpp metrics don't have spec_decode counters."""
        snap = _parse_snapshot(self.LLAMACPP_METRICS)
        assert snap.has_spec_decode is False

    def test_has_llamacpp_metrics_true(self):
        """has_llamacpp_metrics should be True when llama.cpp counters are present."""
        snap = _parse_snapshot(self.LLAMACPP_METRICS)
        assert snap.has_llamacpp_metrics is True

    def test_has_llamacpp_metrics_false_for_vllm(self):
        """has_llamacpp_metrics should be False for vLLM metrics."""
        snap = _parse_snapshot(TestParseSnapshot.FULL_VLLM_METRICS)
        assert snap.has_llamacpp_metrics is False

    def test_has_llamacpp_metrics_false_for_empty(self):
        snap = _parse_snapshot("")
        assert snap.has_llamacpp_metrics is False

    def test_llamacpp_kv_cache_usage_ratio(self):
        """Parse llama.cpp KV cache usage ratio."""
        text = """\
llamacpp:kv_cache_usage_ratio 0.42
llamacpp:tokens_predicted_total 100
"""
        snap = _parse_snapshot(text)
        assert snap.llamacpp_kv_cache_usage_ratio == pytest.approx(0.42)


class TestComputeDeltaLlamaCpp:
    """Test compute_delta with llama.cpp metrics."""

    def _make_snap(self, **kwargs) -> MetricsSnapshot:
        defaults = dict(
            timestamp=time.time(),
            accepted_tokens=0.0,
            draft_tokens=0.0,
            num_drafts=0.0,
            prompt_tps=0.0,
            generation_tps=0.0,
            running_reqs=0.0,
            waiting_reqs=0.0,
            prefix_cache_hit=0.0,
        )
        defaults.update(kwargs)
        return MetricsSnapshot(**defaults)

    def test_llamacpp_generation_tps_fallback(self):
        """When vLLM gauges are zero, use llamacpp:predicted_tokens_seconds."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(
            timestamp=110.0,
            generation_tps=0.0,  # vLLM gauge not present
        )
        curr.llamacpp_predicted_tokens_seconds = 28.39
        delta = compute_delta(prev, curr)
        assert delta.generation_tps == pytest.approx(28.39)

    def test_llamacpp_prompt_tps_fallback(self):
        """When vLLM prompt gauge is zero, use llamacpp:prompt_tokens_seconds."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(
            timestamp=110.0,
            prompt_tps=0.0,
        )
        curr.llamacpp_prompt_tokens_seconds = 1234.5
        delta = compute_delta(prev, curr)
        assert delta.prompt_tps == pytest.approx(1234.5)

    def test_llamacpp_counter_derived_throughput(self):
        """Derive throughput from llama.cpp cumulative token counters."""
        prev = self._make_snap(timestamp=100.0)
        prev.llamacpp_predicted_tokens_total = 1000.0
        curr = self._make_snap(
            timestamp=110.0,
            generation_tps=0.0,
        )
        curr.llamacpp_predicted_tokens_total = 1500.0  # 500 in 10s
        curr.llamacpp_predicted_tokens_seconds = 0.0  # gauge also zero (edge case)
        delta = compute_delta(prev, curr)
        assert delta.generation_tps == pytest.approx(50.0)

    def test_llamacpp_running_requests_fallback(self):
        """Use llamacpp:requests_processing when vLLM running_reqs is 0."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0, running_reqs=0.0)
        curr.llamacpp_requests_processing = 3.0
        delta = compute_delta(prev, curr)
        assert delta.running_reqs == 3

    def test_llamacpp_waiting_requests_fallback(self):
        """Use llamacpp:requests_deferred when vLLM waiting_reqs is 0."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0, waiting_reqs=0.0)
        curr.llamacpp_requests_deferred = 2.0
        delta = compute_delta(prev, curr)
        assert delta.waiting_reqs == 2

    def test_llamacpp_kv_cache_fallback(self):
        """Use llamacpp:kv_cache_usage_ratio when vLLM cache metrics are absent."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0)
        curr.kv_cache_usage = None
        curr.gpu_cache_usage = None
        curr.llamacpp_kv_cache_usage_ratio = 0.35
        delta = compute_delta(prev, curr)
        assert delta.gpu_cache_pct == pytest.approx(35.0)

    def test_vllm_takes_precedence_over_llamacpp(self):
        """When vLLM gauges are non-zero, they should win over llama.cpp values."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(
            timestamp=110.0,
            generation_tps=55.0,  # vLLM gauge present
            running_reqs=2.0,
        )
        curr.llamacpp_predicted_tokens_seconds = 28.39  # should be ignored
        curr.llamacpp_requests_processing = 5.0  # should be ignored
        delta = compute_delta(prev, curr)
        assert delta.generation_tps == pytest.approx(55.0)
        assert delta.running_reqs == 2


# ---------------------------------------------------------------------------
# SGLang direct gauges and current llama.cpp spec counters
# ---------------------------------------------------------------------------


class TestSGLangSpecGauges:
    """Test the current SGLang gauge contract without inventing counters."""

    METRICS = """\
sglang:spec_accept_rate{tp_rank="1",pp_rank="0"} 0.80
sglang:spec_accept_rate{tp_rank="0",pp_rank="0"} 0.40
sglang:spec_accept_length{tp_rank="1",pp_rank="0"} 2.80
sglang:spec_accept_length{tp_rank="0",pp_rank="0"} 1.40
sglang:spec_cap_length{tp_rank="0",pp_rank="0"} 4.0
sglang:spec_block_accept_length{tp_rank="0",pp_rank="0"} 3.2
sglang:spec_num_steps{tp_rank="0",pp_rank="0"} 2
sglang:spec_num_draft_tokens{tp_rank="0",pp_rank="0"} 6
sglang:gen_throughput{tp_rank="0",pp_rank="0"} 42.0
sglang:token_usage{tp_rank="0",pp_rank="0"} 0.25
sglang:num_running_reqs{tp_rank="0",pp_rank="0"} 1
sglang:num_queue_reqs{tp_rank="0",pp_rank="0"} 2
"""

    def test_rank_zero_is_selected_for_replicated_gauges(self):
        snap = _parse_snapshot(self.METRICS)
        assert snap.sglang_acceptance_rate == pytest.approx(0.40)
        assert snap.sglang_acceptance_length == pytest.approx(1.40)
        assert snap.sglang_spec_metrics_present is True
        assert snap.spec_backend == "sglang"
        assert snap.generation_tps == pytest.approx(42.0)
        assert snap.kv_cache_usage == pytest.approx(0.25)
        assert snap.running_reqs == pytest.approx(1.0)
        assert snap.waiting_reqs == pytest.approx(2.0)

    def test_gauges_feed_delta_without_fake_token_totals(self):
        snap = _parse_snapshot(self.METRICS)
        delta = compute_delta(
            MetricsSnapshot(timestamp=100.0),
            MetricsSnapshot(**{**snap.__dict__, "timestamp": 101.0}),
        )
        assert delta.counter_metrics_available is False
        assert delta.spec_metrics_source == "sglang"
        assert delta.cumulative_acceptance_rate == pytest.approx(0.40)
        assert delta.cumulative_acceptance_length == pytest.approx(1.40)
        assert delta.cumulative_draft_window == pytest.approx(6.0)
        assert delta.spec_num_steps == 2
        assert delta.spec_cap_length == pytest.approx(4.0)
        assert delta.spec_block_accept_length == pytest.approx(3.2)
        assert delta.total_accepted == 0
        assert delta.total_drafted == 0

    def test_zero_gauges_are_present_but_not_summed(self):
        text = (
            'sglang:spec_accept_rate{tp_rank="0"} 0\n'
            'sglang:spec_accept_length{tp_rank="0"} 0\n'
            'sglang:spec_num_steps{tp_rank="0"} 0\n'
            'sglang:spec_num_draft_tokens{tp_rank="0"} 0\n'
        )
        snap = _parse_snapshot(text)
        assert snap.has_spec_decode is True
        assert snap.sglang_acceptance_rate == pytest.approx(0.0)
        assert snap.sglang_acceptance_length == pytest.approx(0.0)


class TestLlamaCppSpecCounters:
    """Test llama.cpp's current Prometheus spec counters and positions."""

    METRICS = """\
llamacpp:spec_decode_num_draft_tokens_total 300
llamacpp:spec_decode_num_accepted_tokens_total 90
llamacpp:spec_decode_num_drafts_total 30
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 30
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 20
"""

    def test_current_spec_metrics_are_parsed(self):
        snap = _parse_snapshot(self.METRICS)
        assert snap.spec_backend == "llamacpp"
        assert snap.has_spec_decode is True
        assert snap.accepted_tokens == pytest.approx(90)
        assert snap.draft_tokens == pytest.approx(300)
        assert snap.num_drafts == pytest.approx(30)
        assert snap.per_position_counters == {0: 30.0, 1: 20.0}
        assert snap.per_position_rates[0] == pytest.approx(1.0)
        assert snap.per_position_rates[1] == pytest.approx(2 / 3)

    def test_current_spec_counters_use_bonus_token_length(self):
        snap = _parse_snapshot(self.METRICS)
        delta = compute_delta(MetricsSnapshot(timestamp=100.0), snap)
        assert delta.spec_metrics_source == "llamacpp"
        assert delta.cumulative_acceptance_rate == pytest.approx(0.3)
        assert delta.cumulative_acceptance_length == pytest.approx(4.0)
        assert delta.cumulative_draft_window == pytest.approx(10.0)

    def test_spec_counter_reset_is_non_negative(self):
        prev = _parse_snapshot(
            """\
llamacpp:spec_decode_num_draft_tokens_total 300
llamacpp:spec_decode_num_accepted_tokens_total 90
llamacpp:spec_decode_num_drafts_total 30
"""
        )
        curr = _parse_snapshot(
            """\
llamacpp:spec_decode_num_draft_tokens_total 20
llamacpp:spec_decode_num_accepted_tokens_total 8
llamacpp:spec_decode_num_drafts_total 2
"""
        )
        prev.timestamp = 100.0
        curr.timestamp = 101.0
        delta = compute_delta(prev, curr)
        assert delta.acceptance_rate == pytest.approx(0.4)
        assert delta.accepted_tps == pytest.approx(8.0)
        assert delta.drafted_tps == pytest.approx(20.0)
        assert delta.accepted_tps >= 0
        assert delta.drafted_tps >= 0


# ---------------------------------------------------------------------------
# Spec method detection
# ---------------------------------------------------------------------------


class TestSpecMethodDetection:
    """Test _detect_spec_method inference from Prometheus text."""

    def test_dflash_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = """\
# HELP vllm:spec_decode_num_draft_tokens_total Total draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Qwen3.6-35B",spec_method="dflash"} 5000.0
"""
        assert _detect_spec_method(text) == "dflash"

    def test_mtp_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = """\
# HELP vllm:spec_decode_num_draft_tokens_total MTP draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",spec_method="mtp"} 3000.0
"""
        assert _detect_spec_method(text) == "mtp"

    def test_multi_token_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'spec_decode_num_draft_tokens{spec_method="mtp"} 100\n'
        assert _detect_spec_method(text) == "mtp"

    def test_eagle3_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'spec_decode_num_draft_tokens{spec_method="eagle3"} 100\n'
        assert _detect_spec_method(text) == "eagle3"

    def test_eagle_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'spec_decode_num_draft_tokens{spec_method="eagle"} 100\n'
        assert _detect_spec_method(text) == "eagle"

    def test_ngram_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'spec_decode_num_draft_tokens{spec_method="ngram"} 100\n'
        assert _detect_spec_method(text) == "ngram"

    def test_prompt_lookup_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'spec_decode_num_draft_tokens{spec_method="ngram_gpu"} 100\n'
        assert _detect_spec_method(text) == "ngram_gpu"

    @pytest.mark.parametrize(
        ("reported", "canonical"),
        [
            ("NEXTN", "mtp"),
            ("STANDALONE", "draft_model"),
            ("draft_flash", "dflash"),
            ("prompt_lookup", "ngram"),
        ],
    )
    def test_engine_specific_method_aliases(self, reported, canonical):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = f'spec_decode_num_draft_tokens{{spec_method="{reported}"}} 100\n'
        assert _detect_spec_method(text) == canonical

    def test_generic_draft_model_detection(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        # Generic counters prove activity but not the configured method.
        text = "spec_decode_num_draft_tokens 100\n"
        assert _detect_spec_method(text) == "unknown"

    def test_unknown_when_no_spec_decode(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = "vllm:num_requests_running 0\n"
        assert _detect_spec_method(text) == "unknown"

    def test_dflash_takes_precedence_over_draft(self):
        """dflash should be detected before generic draft_model."""
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'spec_decode_num_draft_tokens{method="dflash"} 100\n'
        assert _detect_spec_method(text) == "dflash"

    def test_eagle3_before_eagle(self):
        """eagle3 is more specific than eagle and should match first."""
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'spec_decode_num_draft_tokens{method="eagle3"} 100\n'
        assert _detect_spec_method(text) == "eagle3"

    def test_method_in_parse_snapshot(self):
        """spec_method is populated by _parse_snapshot."""
        text = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0",spec_method="dflash"} 1500.0
vllm:spec_decode_num_draft_tokens_total{engine="0",spec_method="dflash"} 5000.0
"""
        snap = _parse_snapshot(text)
        assert snap.spec_method == "dflash"

    def test_method_forwarded_to_delta(self):
        """spec_method propagates from snapshot to delta."""
        prev = MetricsSnapshot(timestamp=100.0)
        curr = MetricsSnapshot(
            timestamp=110.0,
            accepted_tokens=100,
            draft_tokens=500,
            num_drafts=50,
            spec_method="mtp",
        )
        delta = compute_delta(prev, curr)
        assert delta.spec_method == "mtp"


# ---------------------------------------------------------------------------
# num_spec_tokens inference
# ---------------------------------------------------------------------------


class TestNumSpecTokens:
    """Test inference of num_speculative_tokens from draft window."""

    def test_inferred_from_draft_window(self):
        """num_spec_tokens = round(draft_tokens / num_drafts)."""
        prev = MetricsSnapshot(timestamp=100.0)
        curr = MetricsSnapshot(
            timestamp=110.0,
            accepted_tokens=150,
            draft_tokens=500,
            num_drafts=100,
        )
        delta = compute_delta(prev, curr)
        # 500 / 100 = 5.0 → 5
        assert delta.num_spec_tokens == 5

    def test_rounded_correctly(self):
        """Rounding for non-integer draft windows."""
        prev = MetricsSnapshot(timestamp=100.0)
        curr = MetricsSnapshot(
            timestamp=110.0,
            accepted_tokens=150,
            draft_tokens=330,
            num_drafts=100,
        )
        delta = compute_delta(prev, curr)
        # 330 / 100 = 3.3 → 3
        assert delta.num_spec_tokens == 3

    def test_none_when_no_drafts(self):
        """num_spec_tokens is None when there are no drafts."""
        prev = MetricsSnapshot(timestamp=100.0)
        curr = MetricsSnapshot(timestamp=110.0)
        delta = compute_delta(prev, curr)
        assert delta.num_spec_tokens is None

    def test_single_spec_token_mtp(self):
        """MTP with num_speculative_tokens=1."""
        prev = MetricsSnapshot(timestamp=100.0)
        curr = MetricsSnapshot(
            timestamp=110.0,
            accepted_tokens=450,
            draft_tokens=500,
            num_drafts=500,
            spec_method="mtp",
        )
        delta = compute_delta(prev, curr)
        # 500 / 500 = 1.0 → 1
        assert delta.num_spec_tokens == 1
        assert delta.spec_method == "mtp"


# ---------------------------------------------------------------------------
# Per-position counter → rate computation
# ---------------------------------------------------------------------------


class TestPerPositionCounters:
    """Test per-position acceptance rate from counter metrics (vLLM v1)."""

    COUNTER_METRICS = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 1500.0
vllm:spec_decode_num_draft_tokens_total{engine="0"} 3000.0
vllm:spec_decode_num_drafts_total{engine="0"} 500.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 454.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 383.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="2"} 280.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="3"} 220.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="4"} 163.0
"""

    def test_counters_parsed(self):
        """Per-position counters are captured in per_position_counters."""
        snap = _parse_snapshot(self.COUNTER_METRICS)
        assert len(snap.per_position_counters) == 5
        assert snap.per_position_counters[0] == 454.0
        assert snap.per_position_counters[4] == 163.0

    def test_rates_computed_from_counters(self):
        """Rates are derived as counter[pos] / num_drafts."""
        snap = _parse_snapshot(self.COUNTER_METRICS)
        assert len(snap.per_position_rates) == 5
        assert snap.per_position_rates[0] == pytest.approx(454.0 / 500.0)
        assert snap.per_position_rates[4] == pytest.approx(163.0 / 500.0)

    def test_rates_decrease_monotonically(self):
        """Rates should decrease across positions (natural decay)."""
        snap = _parse_snapshot(self.COUNTER_METRICS)
        rates = [snap.per_position_rates[i] for i in sorted(snap.per_position_rates)]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]

    def test_gauge_rates_take_priority(self):
        """If gauge rates are present, counters don't override them."""
        text = self.COUNTER_METRICS + (
            'vllm:spec_decode_per_position_acceptance_rate{position="0"} 0.99\n'
            'vllm:spec_decode_per_position_acceptance_rate{position="1"} 0.88\n'
        )
        snap = _parse_snapshot(text)
        # Gauge rates present → they take priority
        assert snap.per_position_rates[0] == pytest.approx(0.99)
        assert snap.per_position_rates[1] == pytest.approx(0.88)
        # Counters are still parsed independently
        assert len(snap.per_position_counters) == 5

    def test_no_rates_when_zero_drafts(self):
        """With zero num_drafts, counters can't produce rates."""
        text = """\
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 0.0
vllm:spec_decode_num_drafts_total 0.0
"""
        snap = _parse_snapshot(text)
        assert snap.per_position_counters == {0: 0.0}
        assert snap.per_position_rates == {}  # can't divide by zero

    def test_underscore_prefix_variant(self):
        """Also match vllm_ prefix (underscore instead of colon)."""
        text = """\
vllm_spec_decode_num_accepted_tokens_per_pos_total{position="0"} 100.0
vllm_spec_decode_num_accepted_tokens_per_pos_total{position="1"} 80.0
vllm_spec_decode_num_drafts_total 200.0
"""
        snap = _parse_snapshot(text)
        assert len(snap.per_position_rates) == 2
        assert snap.per_position_rates[0] == pytest.approx(0.5)
        assert snap.per_position_rates[1] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Per-position decay summary
# ---------------------------------------------------------------------------


class TestPerPositionDecaySummary:
    """Test _per_position_decay_summary analysis."""

    def test_no_rates_returns_none(self):
        from tool_eval_bench.cli.spec_live_rendering import _per_position_decay_summary

        assert _per_position_decay_summary({}) is None

    def test_single_position_returns_none(self):
        from tool_eval_bench.cli.spec_live_rendering import _per_position_decay_summary

        assert _per_position_decay_summary({0: 0.8}) is None

    def test_effective_positions_count(self):
        from tool_eval_bench.cli.spec_live_rendering import _per_position_decay_summary

        rates = {0: 0.80, 1: 0.60, 2: 0.40, 3: 0.15, 4: 0.05}
        result = _per_position_decay_summary(rates)
        text = str(result)
        # 3 positions above 20%: p0=80%, p1=60%, p2=40%
        assert "3/5" in text
        assert "effective" in text

    def test_half_point_reported(self):
        from tool_eval_bench.cli.spec_live_rendering import _per_position_decay_summary

        rates = {0: 0.80, 1: 0.60, 2: 0.35, 3: 0.15}
        result = _per_position_decay_summary(rates)
        text = str(result)
        # 50% of 0.80 = 0.40 → p2 (0.35) is first below
        assert "p2" in text

    def test_no_half_point_when_all_high(self):
        from tool_eval_bench.cli.spec_live_rendering import _per_position_decay_summary

        rates = {0: 0.90, 1: 0.85, 2: 0.80}
        result = _per_position_decay_summary(rates)
        text = str(result)
        assert "3/3" in text
        # No 50% drop
        assert "50% drop" not in text

    def test_decay_rate_gamma(self):
        from tool_eval_bench.cli.spec_live_rendering import _per_position_decay_summary

        rates = {0: 0.80, 1: 0.60, 2: 0.40, 3: 0.25}
        result = _per_position_decay_summary(rates)
        text = str(result)
        # γ should be present
        assert "γ=" in text


# ---------------------------------------------------------------------------
# Spec method label and dashboard badge
# ---------------------------------------------------------------------------


class TestSpecMethodLabel:
    """Test _spec_method_label formatting."""

    def test_dflash_label(self):
        from tool_eval_bench.cli.spec_live_rendering import _spec_method_label

        label, style = _spec_method_label("dflash")
        assert label == "Draft Flash"
        assert "cyan" in style

    def test_mtp_label(self):
        from tool_eval_bench.cli.spec_live_rendering import _spec_method_label

        label, style = _spec_method_label("mtp")
        assert label == "Multi-Token Prediction"
        assert "yellow" in style

    def test_eagle_label(self):
        from tool_eval_bench.cli.spec_live_rendering import _spec_method_label

        label, _ = _spec_method_label("eagle")
        assert label == "EAGLE"

    def test_eagle3_label(self):
        from tool_eval_bench.cli.spec_live_rendering import _spec_method_label

        label, _ = _spec_method_label("eagle3")
        assert label == "EAGLE-3"

    def test_ngram_label(self):
        from tool_eval_bench.cli.spec_live_rendering import _spec_method_label

        label, _ = _spec_method_label("ngram")
        assert label == "N-Gram"

    def test_unknown_label(self):
        from tool_eval_bench.cli.spec_live_rendering import _spec_method_label

        label, style = _spec_method_label("unknown")
        assert label == "Speculative Decoding"
        assert style == "bold dim"

    def test_mlp_speculator_label(self):
        from tool_eval_bench.cli.spec_live_rendering import _spec_method_label

        label, _ = _spec_method_label("mlp_speculator")
        assert label == "MLP Speculator"


class TestDraftModelDetection:
    """Test draft model name extraction and method detection improvements."""

    def test_draft_flash_detected(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'vllm:spec_decode_num_draft_tokens_total{spec_method="dflash"} 100\n'
        assert _detect_spec_method(text) == "dflash"

    def test_dflash_detected(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'vllm:spec_decode_num_draft_tokens_total{method="dflash"} 100\n'
        assert _detect_spec_method(text) == "dflash"

    def test_mlp_speculator_detected(self):
        from tool_eval_bench.runner.spec_live import _detect_spec_method

        text = 'vllm:spec_decode_num_draft_tokens_total{spec_method="mlp_speculator"} 100\n'
        assert _detect_spec_method(text) == "mlp_speculator"

    def test_model_names_extracted(self):
        from tool_eval_bench.runner.spec_live import _extract_model_names

        text = (
            'vllm:spec_decode_num_accepted_tokens_total{model_name="Qwen/Qwen3-8B"} 100\n'
            'vllm:generation_tokens_total{model_name="Qwen/Qwen3-8B"} 500\n'
            'vllm:generation_tokens_total{model_name="Qwen/Qwen3-0.6B"} 200\n'
        )
        names = _extract_model_names(text)
        assert names == {"Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B"}

    def test_model_names_empty(self):
        from tool_eval_bench.runner.spec_live import _extract_model_names

        text = "vllm:spec_decode_num_accepted_tokens_total 100\n"
        names = _extract_model_names(text)
        assert names == set()

    def test_model_names_in_snapshot(self):
        from tool_eval_bench.runner.spec_live import _parse_snapshot

        text = (
            'vllm:spec_decode_num_accepted_tokens_total{model_name="MainModel"} 100\n'
            'vllm:spec_decode_num_draft_tokens_total{model_name="DraftModel"} 200\n'
        )
        snap = _parse_snapshot(text)
        assert "MainModel" in snap.model_names
        assert "DraftModel" in snap.model_names


class TestHorizontalBarsScaling:
    """Test horizontal per-position bars scale to many positions."""

    def test_six_positions_single_row(self):
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.spec_live_rendering import _position_bars_horizontal

        rates = {i: max(0.1, 0.9 - i * 0.15) for i in range(6)}
        table = _position_bars_horizontal(rates, inner_w=120)
        out = StringIO()
        Console(file=out, width=120, no_color=True).print(table)
        text = out.getvalue()
        assert "p0" in text
        assert "p5" in text

    def test_twelve_positions_multi_row(self):
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.spec_live_rendering import _position_bars_horizontal

        rates = {i: max(0.05, 0.9 - i * 0.07) for i in range(12)}
        table = _position_bars_horizontal(rates, inner_w=100)
        out = StringIO()
        Console(file=out, width=100, no_color=True).print(table)
        text = out.getvalue()
        # All 12 positions should be rendered
        for i in range(12):
            assert f"p{i}" in text
        # Should have multiple lines (wrapping)
        lines = [line for line in text.strip().split("\n") if line.strip()]
        assert len(lines) >= 2

    def test_narrow_terminal_still_works(self):
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.spec_live_rendering import _position_bars_horizontal

        rates = {i: max(0.1, 0.9 - i * 0.15) for i in range(6)}
        table = _position_bars_horizontal(rates, inner_w=50)
        out = StringIO()
        Console(file=out, width=50, no_color=True).print(table)
        text = out.getvalue()
        assert "p0" in text
        assert "p5" in text


class TestDashboardSpecBadge:
    """Test dashboard renders spec method badge and num_spec_tokens."""

    def _make_delta(self, **kwargs) -> SpecLiveDelta:
        defaults = dict(
            elapsed_s=1.0,
            had_activity=True,
            cumulative_acceptance_rate=0.30,
            cumulative_acceptance_length=2.5,
            cumulative_draft_window=5.0,
            acceptance_rate=0.30,
            accepted_tps=8.0,
            drafted_tps=14.0,
            prompt_tps=2500.0,
            generation_tps=10.5,
            gpu_cache_pct=3.4,
            running_reqs=1,
            waiting_reqs=0,
            prefix_cache_hit_pct=0.0,
            per_position_rates={},
            total_accepted=1500,
            total_drafted=5000,
            total_drafts=1000,
            spec_method="dflash",
            num_spec_tokens=5,
        )
        defaults.update(kwargs)
        return SpecLiveDelta(**defaults)

    def _render(self, panel) -> str:
        from io import StringIO

        from rich.console import Console

        out = StringIO()
        Console(file=out, width=120, no_color=True).print(panel)
        return out.getvalue()

    def test_dflash_badge_in_header(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(spec_method="dflash")
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 30, "TestModel", "http://localhost:8000/metrics", 30
        )
        text = self._render(panel)
        assert "Draft Flash" in text

    def test_mtp_badge_in_header(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(spec_method="mtp")
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 30, "TestModel", "http://localhost:8000/metrics", 30
        )
        text = self._render(panel)
        assert "Multi-Token Prediction" in text

    def test_unknown_method_no_badge(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(spec_method="unknown")
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 30, "TestModel", "http://localhost:8000/metrics", 30
        )
        text = self._render(panel)
        assert "Draft Flash" not in text
        assert "Multi-Token" not in text

    def test_num_spec_tokens_shown(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(num_spec_tokens=5)
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 30, "TestModel", "http://localhost:8000/metrics", 30
        )
        text = self._render(panel)
        assert "k=5" in text
        assert "Spec Tokens" in text

    def test_per_position_rendered_in_dashboard(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(
            per_position_rates={0: 0.80, 1: 0.60, 2: 0.30, 3: 0.08, 4: 0.02},
        )
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta, history, time.time() - 30, "TestModel", "http://localhost:8000/metrics", 30
        )
        text = self._render(panel)
        # Per-position panel should be rendered with position labels
        assert "p0" in text
        assert "p1" in text
        assert "Per-Position" in text
        # Should show percentage values
        assert "80%" in text
        assert "60%" in text

    def test_dflash_efficiency_insight_with_nst(self):
        """Dflash with high draft tokens and low utilization shows hint."""
        from tool_eval_bench.cli.spec_live_rendering import _efficiency_insight

        delta = self._make_delta(
            spec_method="dflash",
            cumulative_acceptance_rate=0.20,
            cumulative_acceptance_length=1.5,
            cumulative_draft_window=8.0,
            num_spec_tokens=8,
        )
        text = str(_efficiency_insight(delta))
        assert "num_speculative_tokens" in text
        assert "current: 8" in text

    def test_mtp_efficiency_insight_guidance(self):
        """MTP with good utilization shows MTP-specific guidance."""
        from tool_eval_bench.cli.spec_live_rendering import _efficiency_insight

        delta = self._make_delta(
            spec_method="mtp",
            cumulative_acceptance_rate=0.65,
            cumulative_acceptance_length=0.65,
            cumulative_draft_window=1.0,
            num_spec_tokens=1,
        )
        text = str(_efficiency_insight(delta))
        assert "MTP" in text


# ---------------------------------------------------------------------------
# ServerSpecInfo and probe_server_spec_info
# ---------------------------------------------------------------------------


class TestServerSpecInfo:
    """Test the ServerSpecInfo dataclass and probe function."""

    def test_default_values(self):
        from tool_eval_bench.runner.spec_live import ServerSpecInfo

        info = ServerSpecInfo()
        assert info.spec_method is None
        assert info.draft_model_name is None
        assert info.target_model_name is None
        assert info.num_speculative_tokens is None

    def test_fields_populated(self):
        from tool_eval_bench.runner.spec_live import ServerSpecInfo

        info = ServerSpecInfo(
            spec_method="dflash",
            draft_model_name="Qwen/Qwen3-0.6B",
            target_model_name="Qwen/Qwen3-35B",
            num_speculative_tokens=5,
        )
        assert info.spec_method == "dflash"
        assert info.draft_model_name == "Qwen/Qwen3-0.6B"
        assert info.target_model_name == "Qwen/Qwen3-35B"
        assert info.num_speculative_tokens == 5


@pytest.mark.asyncio
class TestProbeServerSpecInfo:
    """Test probe_server_spec_info against mock HTTP responses."""

    async def test_models_do_not_claim_a_draft_model(self):
        """A second /v1/models item is not proof of a drafter relationship."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from tool_eval_bench.runner.spec_live import probe_server_spec_info

        # Mock httpx.AsyncClient to return a models response with 2 models
        mock_response_models = MagicMock()
        mock_response_models.status_code = 200
        mock_response_models.json.return_value = {
            "object": "list",
            "data": [
                {"id": "Qwen/Qwen3-35B", "object": "model"},
                {"id": "Qwen/Qwen3-0.6B", "object": "model"},
            ],
        }

        mock_response_version = MagicMock()
        mock_response_version.status_code = 404  # no /version

        async def mock_get(url, **kwargs):
            if "/models" in url:
                return mock_response_models
            return mock_response_version

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tool_eval_bench.runner.spec_live.httpx.AsyncClient", return_value=mock_client):
            info = await probe_server_spec_info(
                "http://localhost:8000/v1",
                primary_model="Qwen/Qwen3-35B",
            )

        assert info.draft_model_name is None
        assert info.target_model_name == "Qwen/Qwen3-35B"
        assert info.spec_method is None

    async def test_probe_handles_connection_failure(self):
        """probe_server_spec_info should not raise on connection errors."""
        from tool_eval_bench.runner.spec_live import probe_server_spec_info

        # Point to a non-existent server
        info = await probe_server_spec_info(
            "http://localhost:1",  # unreachable
            primary_model="test",
        )
        # Should return default ServerSpecInfo, not raise
        assert info.spec_method is None
        assert info.draft_model_name is None


# ---------------------------------------------------------------------------
# Dashboard with ServerSpecInfo
# ---------------------------------------------------------------------------


class TestDashboardWithServerSpecInfo:
    """Test dashboard rendering with ServerSpecInfo for draft model display."""

    def _make_delta(self, **kwargs) -> SpecLiveDelta:
        defaults = dict(
            elapsed_s=1.0,
            had_activity=True,
            cumulative_acceptance_rate=0.30,
            cumulative_acceptance_length=2.5,
            cumulative_draft_window=5.0,
            acceptance_rate=0.30,
            accepted_tps=8.0,
            drafted_tps=14.0,
            prompt_tps=2500.0,
            generation_tps=10.5,
            gpu_cache_pct=3.4,
            running_reqs=1,
            waiting_reqs=0,
            prefix_cache_hit_pct=0.0,
            per_position_rates={},
            total_accepted=1500,
            total_drafted=5000,
            total_drafts=1000,
            spec_method="dflash",
            num_spec_tokens=5,
        )
        defaults.update(kwargs)
        return SpecLiveDelta(**defaults)

    def _render(self, panel) -> str:
        from io import StringIO

        from rich.console import Console

        out = StringIO()
        Console(file=out, width=120, no_color=True).print(panel)
        return out.getvalue()

    def test_draft_model_from_server_spec_info(self):
        """ServerSpecInfo draft model name shown in header."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard
        from tool_eval_bench.runner.spec_live import ServerSpecInfo

        info = ServerSpecInfo(
            draft_model_name="Qwen/Qwen3-0.6B",
            spec_method="dflash",
        )
        delta = self._make_delta()
        history = deque([delta] * 5, maxlen=60)
        panel = _build_dashboard(
            delta,
            history,
            time.time() - 30,
            "Qwen/Qwen3-35B",
            "http://localhost:8000/metrics",
            30,
            server_spec_info=info,
        )
        text = self._render(panel)
        assert "Qwen/Qwen3-0.6B" in text

    def test_draft_model_from_server_spec_info_preferred_over_prometheus(self):
        """ServerSpecInfo takes priority over Prometheus label heuristic."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard
        from tool_eval_bench.runner.spec_live import ServerSpecInfo

        info = ServerSpecInfo(draft_model_name="real-draft-model")
        delta = self._make_delta()
        delta.model_names = {"Qwen/Qwen3-35B", "prom-draft-model"}
        history = deque([delta] * 5, maxlen=60)
        panel = _build_dashboard(
            delta,
            history,
            time.time() - 30,
            "Qwen/Qwen3-35B",
            "http://localhost:8000/metrics",
            30,
            server_spec_info=info,
        )
        text = self._render(panel)
        assert "real-draft-model" in text
        # Prometheus heuristic draft should NOT appear when ServerSpecInfo has one
        assert "prom-draft-model" not in text

    def test_reset_flash_shown(self):
        """Reset flash banner appears when reset_flash=True."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 5, maxlen=60)
        panel = _build_dashboard(
            delta,
            history,
            time.time() - 30,
            "TestModel",
            "http://localhost:8000/metrics",
            30,
            reset_flash=True,
        )
        text = self._render(panel)
        assert "Session reset" in text

    def test_reset_flash_hidden_by_default(self):
        """Reset flash not shown when reset_flash=False (default)."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 5, maxlen=60)
        panel = _build_dashboard(
            delta,
            history,
            time.time() - 30,
            "TestModel",
            "http://localhost:8000/metrics",
            30,
        )
        text = self._render(panel)
        assert "Session reset" not in text

    def test_subtitle_shows_ctrl_r_hint(self):
        """Subtitle includes Ctrl+R reset hint."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 5, maxlen=60)
        panel = _build_dashboard(
            delta,
            history,
            time.time() - 30,
            "TestModel",
            "http://localhost:8000/metrics",
            30,
        )
        text = self._render(panel)
        assert "Ctrl+R" in text

    def test_waiting_state_shows_ctrl_r_hint(self):
        """Waiting/connecting state also shows Ctrl+R hint."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        panel = _build_dashboard(
            None,
            deque(maxlen=60),
            time.time() - 5,
            "TestModel",
            "http://localhost:8000/metrics",
            5,
        )
        text = self._render(panel)
        assert "Ctrl+R" in text


# ---------------------------------------------------------------------------
# High-k per-position scaling
# ---------------------------------------------------------------------------


class TestHighKPositionScaling:
    """Test per-position bars with k > 16 (high speculative token counts)."""

    def test_twenty_positions(self):
        """20 positions should render without errors."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.spec_live_rendering import _position_bars_horizontal

        rates = {i: max(0.02, 0.95 - i * 0.04) for i in range(20)}
        table = _position_bars_horizontal(rates, inner_w=120)
        out = StringIO()
        Console(file=out, width=120, no_color=True).print(table)
        text = out.getvalue()
        for i in range(20):
            assert f"p{i}" in text

    def test_thirty_two_positions(self):
        """32 positions (extreme k) should wrap to multiple rows."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.spec_live_rendering import _position_bars_horizontal

        rates = {i: max(0.01, 0.90 - i * 0.025) for i in range(32)}
        table = _position_bars_horizontal(rates, inner_w=120)
        out = StringIO()
        Console(file=out, width=120, no_color=True).print(table)
        text = out.getvalue()
        # All 32 positions visible
        for i in range(32):
            assert f"p{i}" in text
        # Must wrap (at 120 cols, ~8 positions per row → 4 rows)
        lines = [line for line in text.strip().split("\n") if line.strip()]
        assert len(lines) >= 4

    def test_dashboard_with_high_k_positions(self):
        """Full dashboard renders correctly with 20 per-position rates."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = SpecLiveDelta(
            elapsed_s=1.0,
            had_activity=True,
            cumulative_acceptance_rate=0.40,
            cumulative_acceptance_length=8.0,
            cumulative_draft_window=20.0,
            accepted_tps=12.0,
            drafted_tps=30.0,
            prompt_tps=2000.0,
            generation_tps=15.0,
            gpu_cache_pct=5.0,
            running_reqs=1,
            waiting_reqs=0,
            prefix_cache_hit_pct=0.0,
            per_position_rates={i: max(0.03, 0.85 - i * 0.04) for i in range(20)},
            total_accepted=3000,
            total_drafted=7500,
            total_drafts=375,
            num_spec_tokens=20,
        )
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(
            delta,
            history,
            time.time() - 30,
            "TestModel",
            "http://localhost:8000/metrics",
            30,
        )
        out = StringIO()
        Console(file=out, width=120, no_color=True).print(panel)
        text = out.getvalue()
        assert "k=20" in text
        assert "p0" in text
        assert "p19" in text
        assert "Per-Position" in text


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('model_name="gpt-4", method="eagle"', {"model_name": "gpt-4", "method": "eagle"}),
        (r'a="x\"y"', {"a": 'x"y'}),  # the escape is unwrapped
        ('a=""', {"a": ""}),
        ('a="unclosed', {}),
        ("", {}),
    ],
)
def test_parse_labels_reads_prometheus_label_sets(raw, expected):
    assert _parse_labels(raw) == expected


def test_parse_labels_does_not_backtrack_exponentially_on_an_unclosed_label():
    """A label that opens a quote and never closes it must not hang the monitor.

    `[^"]` also matched a backslash, so every escape had two possible parses and
    a non-matching tail took exponential time. Metrics text arrives from
    whatever server the run points at, so this input is reachable. The old
    pattern needed roughly half a second at 22 repetitions and doubled with
    each one after that; the ceiling here is loose enough to be stable on a
    loaded CI runner while still failing outright on exponential behaviour.
    """
    hostile = 'A="' + "\\!" * 64

    start = time.perf_counter()
    assert _parse_labels(hostile) == {}
    assert time.perf_counter() - start < 1.0
