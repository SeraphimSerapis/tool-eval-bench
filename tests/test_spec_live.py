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
            timestamp=110.0, accepted_tokens=200, draft_tokens=800, num_drafts=100,
            generation_tps=10.5, prompt_tps=2580.9, gpu_cache_usage=0.034,
            running_reqs=1, waiting_reqs=0, prefix_cache_hit=0.45,
        )
        delta = compute_delta(prev, curr)

        assert delta.elapsed_s == pytest.approx(10.0)
        assert delta.had_activity is True

        # Interval rates
        assert delta.acceptance_rate == pytest.approx(100 / 400)
        assert delta.waste_ratio == pytest.approx(1.0 - 100 / 400)
        assert delta.acceptance_length == pytest.approx(100 / 50)
        assert delta.draft_window == pytest.approx(400 / 50)
        assert delta.accepted_tps == pytest.approx(100 / 10.0)
        assert delta.drafted_tps == pytest.approx(400 / 10.0)

        # Cumulative rates
        assert delta.cumulative_acceptance_rate == pytest.approx(200 / 800)
        assert delta.cumulative_acceptance_length == pytest.approx(200 / 100)
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
        assert delta.cumulative_acceptance_length == pytest.approx(500 / 250)

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
        curr.kv_cache_usage = 0.0   # genuinely zero (idle)
        curr.gpu_cache_usage = None  # not present (old metric)
        delta = compute_delta(prev, curr)
        assert delta.gpu_cache_pct == pytest.approx(0.0)

    def test_kv_cache_falls_back_to_gpu_cache(self):
        """When kv_cache_usage is None, fall back to gpu_cache_usage."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(timestamp=101.0, accepted_tokens=100, draft_tokens=500)
        curr.kv_cache_usage = None   # new metric not present
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
            timestamp=100.0, accepted_tokens=0, draft_tokens=0,
        )
        prev.generation_tokens_total = 1000.0
        curr = self._make_snap(
            timestamp=110.0, accepted_tokens=100, draft_tokens=500,
            generation_tps=0.0,  # gauge removed in vLLM ≥0.8
        )
        curr.generation_tokens_total = 1500.0  # 500 tokens in 10s
        delta = compute_delta(prev, curr)
        assert delta.generation_tps == pytest.approx(50.0)  # 500/10

    def test_counter_derived_prompt_tps(self):
        """When prompt_tps gauge is 0, derive from prompt_tokens_total deltas."""
        prev = self._make_snap(
            timestamp=100.0, accepted_tokens=0, draft_tokens=0,
        )
        prev.prompt_tokens_total = 10000.0
        curr = self._make_snap(
            timestamp=110.0, accepted_tokens=100, draft_tokens=500,
            prompt_tps=0.0,  # gauge removed
        )
        curr.prompt_tokens_total = 30000.0  # 20000 tokens in 10s
        delta = compute_delta(prev, curr)
        assert delta.prompt_tps == pytest.approx(2000.0)  # 20000/10

    def test_prefix_cache_counter_derived_rate(self):
        """When prefix_cache_hit gauge is 0, derive from hit/query counters."""
        prev = self._make_snap(timestamp=100.0)
        curr = self._make_snap(
            timestamp=101.0, accepted_tokens=100, draft_tokens=500,
        )
        curr.prefix_cache_hit = 0.0       # old gauge not present
        curr.prefix_cache_queries = 10000  # counter
        curr.prefix_cache_hits = 800       # counter
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
        from tool_eval_bench.cli.spec_live_display import _ar_color

        assert _ar_color(0.0) == "bright_red"
        assert _ar_color(0.1) == "bright_red"
        assert _ar_color(0.25) == "red"
        assert _ar_color(0.4) == "dark_orange"
        assert _ar_color(0.55) == "yellow"
        assert _ar_color(0.7) == "green_yellow"
        assert _ar_color(0.85) == "bright_green"

    def test_gauge_bar_length(self):
        from tool_eval_bench.cli.spec_live_display import _gauge_bar

        bar = _gauge_bar(0.5, width=20)
        text_str = str(bar)
        # Should contain the percentage
        assert "50.0%" in text_str

    def test_gauge_bar_clamped(self):
        from tool_eval_bench.cli.spec_live_display import _gauge_bar

        # Values outside 0-1 should be clamped
        bar_high = _gauge_bar(1.5, width=10)
        assert "150.0%" in str(bar_high)  # label shows actual
        bar_low = _gauge_bar(-0.1, width=10)
        assert str(bar_low)  # should not crash

    def test_sparkline_empty(self):
        from tool_eval_bench.cli.spec_live_display import _sparkline

        spark = _sparkline([], width=10)
        assert len(str(spark)) == 10  # all dashes

    def test_sparkline_single_value(self):
        from tool_eval_bench.cli.spec_live_display import _sparkline

        spark = _sparkline([5.0], width=10)
        assert str(spark)  # should not crash

    def test_sparkline_constant_values(self):
        from tool_eval_bench.cli.spec_live_display import _sparkline

        spark = _sparkline([5.0, 5.0, 5.0], width=10)
        assert str(spark)  # should not crash, all same → range=0

    def test_sparkline_varying(self):
        from tool_eval_bench.cli.spec_live_display import _sparkline

        spark = _sparkline([1.0, 5.0, 3.0, 7.0, 2.0], width=10)
        text = str(spark)
        assert len(text) == 10  # padded to width

    def test_format_uptime(self):
        from tool_eval_bench.cli.spec_live_display import _format_uptime

        assert _format_uptime(0) == "00:00"
        assert _format_uptime(65) == "01:05"
        assert _format_uptime(3661) == "1:01:01"

    def test_position_bars_empty(self):
        from tool_eval_bench.cli.spec_live_display import _position_bars

        table = _position_bars({})
        # Should render without error and contain the MTP fallback text
        from rich.console import Console
        from io import StringIO

        out = StringIO()
        Console(file=out, width=60, no_color=True).print(table)
        text = out.getvalue()
        assert "not exposed" in text

    def test_position_bars_with_data(self):
        from tool_eval_bench.cli.spec_live_display import _position_bars
        from rich.console import Console
        from io import StringIO

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
        from rich.console import Console
        from io import StringIO

        out = StringIO()
        Console(file=out, width=100, no_color=True).print(panel)
        return out.getvalue()

    def test_renders_with_data(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(delta, history, time.time() - 60,
                                 "TestModel", "http://localhost:8000/metrics", 60)
        text = self._render(panel)
        assert "SPECULATIVE DECODING MONITOR" in text
        assert "TestModel" in text
        assert "ACCEPTANCE RATE" in text

    def test_renders_waiting_state(self):
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        panel = _build_dashboard(None, deque(maxlen=60), time.time() - 5,
                                 "TestModel", "http://localhost:8000/metrics", 5)
        text = self._render(panel)
        assert "Connecting to" in text
        assert "spec decode enabled" in text

    def test_renders_without_per_position(self):
        """MTP models don't have per-position rates — panel is hidden."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(per_position_rates={})
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(delta, history, time.time() - 30,
                                 "MTPModel", "http://localhost:8000/metrics", 30)
        text = self._render(panel)
        assert "Per-Position" not in text
        # Engine panel should still be present
        assert "Engine" in text

    def test_renders_with_rolling_averages(self):
        """Rolling averages panel appears after 5+ data points."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(delta, history, time.time() - 30,
                                 "TestModel", "http://localhost:8000/metrics", 30)
        text = self._render(panel)
        assert "Rolling Averages" in text

    def test_renders_without_rolling_averages(self):
        """Rolling averages panel hidden with < 5 data points."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta()
        history = deque([delta] * 3, maxlen=60)
        panel = _build_dashboard(delta, history, time.time() - 3,
                                 "TestModel", "http://localhost:8000/metrics", 3)
        text = self._render(panel)
        assert "Rolling Averages" not in text

    def test_renders_high_acceptance(self):
        """Dashboard with excellent acceptance rate."""
        from tool_eval_bench.cli.spec_live_display import _build_dashboard

        delta = self._make_delta(
            cumulative_acceptance_rate=0.75,
            cumulative_acceptance_length=6.0,
            cumulative_draft_window=8.0,
        )
        history = deque([delta] * 10, maxlen=60)
        panel = _build_dashboard(delta, history, time.time() - 60,
                                 "TestModel", "http://localhost:8000/metrics", 60)
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
        panel = _build_dashboard(delta, history, time.time() - 60,
                                 "TestModel", "http://localhost:8000/metrics", 60)
        text = self._render(panel)
        assert "Poor" in text

    def test_efficiency_insight_tuning_hint(self):
        """Low utilization triggers num_speculative_tokens suggestion."""
        from tool_eval_bench.cli.spec_live_display import _efficiency_insight

        delta = self._make_delta(
            cumulative_acceptance_rate=0.15,
            cumulative_acceptance_length=1.5,
            cumulative_draft_window=8.0,
        )
        text = str(_efficiency_insight(delta))
        assert "num_speculative_tokens" in text

    def test_efficiency_insight_no_hint_when_good(self):
        """Good utilization does not trigger tuning hint."""
        from tool_eval_bench.cli.spec_live_display import _efficiency_insight

        delta = self._make_delta(
            cumulative_acceptance_rate=0.75,
            cumulative_acceptance_length=6.0,
            cumulative_draft_window=8.0,
        )
        text = str(_efficiency_insight(delta))
        assert "num_speculative_tokens" not in text

    def test_efficiency_insight_awaiting(self):
        """No cumulative rate → awaiting message."""
        from tool_eval_bench.cli.spec_live_display import _efficiency_insight

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
        curr.llamacpp_predicted_tokens_seconds = 0.0   # gauge also zero (edge case)
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
        curr.llamacpp_requests_processing = 5.0         # should be ignored
        delta = compute_delta(prev, curr)
        assert delta.generation_tps == pytest.approx(55.0)
        assert delta.running_reqs == 2
