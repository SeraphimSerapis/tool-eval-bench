"""Rolling-window rates, scrape health, and exact per-position denominators.

The dashboard used to plot the session running average everywhere, which
converges and then hides workload changes, and a dead server kept a green
spinner. These tests pin the replacements.
"""

from __future__ import annotations

from collections import deque
from io import StringIO

import pytest
from rich.console import Console

from tool_eval_bench.cli.spec_live_display import _build_dashboard
from tool_eval_bench.runner.spec_live import (
    SpecLiveDelta,
    _parse_snapshot,
    apply_rolling_rates,
    per_position_rates_from_counters,
)

# ---------------------------------------------------------------------------
# apply_rolling_rates
# ---------------------------------------------------------------------------


def _delta(accepted: float, drafted: float, drafts: float, elapsed: float = 10.0) -> SpecLiveDelta:
    return SpecLiveDelta(
        elapsed_s=elapsed,
        delta_accepted=accepted,
        delta_drafted=drafted,
        delta_drafts=drafts,
        had_activity=drafted > 0,
    )


def test_rolling_window_pools_only_recent_intervals() -> None:
    history = deque(
        [
            _delta(80, 100, 25),  # old, high acceptance: outside the window
            _delta(80, 100, 25),
            _delta(10, 100, 25),
            _delta(10, 100, 25),
            _delta(10, 100, 25),
        ]
    )
    apply_rolling_rates(history, window_s=30.0)
    latest = history[-1]
    assert latest.rolling_acceptance_rate == pytest.approx(0.1)
    assert latest.rolling_acceptance_length == pytest.approx(1.0 + 30 / 75)
    assert latest.rolling_draft_window == pytest.approx(4.0)
    # Only the newest delta is annotated.
    assert history[0].rolling_acceptance_rate is None


def test_rolling_window_spans_idle_polls_and_stays_none_without_drafts() -> None:
    history = deque([_delta(50, 100, 25, elapsed=1.0)] + [_delta(0, 0, 0, elapsed=1.0)] * 5)
    apply_rolling_rates(history, window_s=30.0)
    assert history[-1].rolling_acceptance_rate == pytest.approx(0.5)

    idle = deque([_delta(0, 0, 0)] * 3)
    apply_rolling_rates(idle, window_s=30.0)
    assert idle[-1].rolling_acceptance_rate is None
    assert idle[-1].rolling_acceptance_length is None

    apply_rolling_rates(deque(), window_s=30.0)  # no history is not an error


def test_rolling_window_leaves_gauge_backends_alone() -> None:
    sglang = _delta(50, 100, 25)
    sglang.counter_metrics_available = False
    history = deque([sglang])
    apply_rolling_rates(history, window_s=30.0)
    assert sglang.rolling_acceptance_rate is None


# ---------------------------------------------------------------------------
# per-position denominators
# ---------------------------------------------------------------------------


def test_drafted_per_pos_counter_is_the_denominator_when_present() -> None:
    accepted = {0: 90.0, 1: 40.0, 2: 5.0}
    drafted = {0: 100.0, 1: 50.0, 2: 10.0}  # ngram-style: fewer drafts reach later positions
    assert per_position_rates_from_counters(accepted, drafted, 100.0) == pytest.approx(
        {0: 0.9, 1: 0.8, 2: 0.5}
    )
    # Without drafted counters every draft is assumed full length.
    assert per_position_rates_from_counters(accepted, {}, 100.0) == pytest.approx(
        {0: 0.9, 1: 0.4, 2: 0.05}
    )
    # A position with no drafts yields no rate rather than a division error.
    assert per_position_rates_from_counters({0: 1.0}, {0: 0.0}, 5.0) == {}


def test_parse_snapshot_reads_drafted_per_pos_counters() -> None:
    text = """\
vllm:spec_decode_num_drafts_total{engine="0"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0"} 150.0
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 95.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 90.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 5.0
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",position="0"} 1.7e+09
vllm:spec_decode_num_draft_tokens_per_pos_total{engine="0",position="0"} 100.0
vllm:spec_decode_num_draft_tokens_per_pos_total{engine="0",position="1"} 50.0
vllm:spec_decode_num_draft_tokens_per_pos_created{engine="0",position="0"} 1.7e+09
"""
    snap = _parse_snapshot(text)
    assert snap.per_position_drafted_counters == {0: 100.0, 1: 50.0}
    # The per-pos series must not leak into the plain draft-token counter.
    assert snap.draft_tokens == 150.0
    assert snap.per_position_rates == pytest.approx({0: 0.9, 1: 0.1})


# ---------------------------------------------------------------------------
# dashboard rendering
# ---------------------------------------------------------------------------


def _render(panel) -> str:
    out = StringIO()
    Console(file=out, width=120, no_color=True).print(panel)
    return out.getvalue()


def test_gauge_and_grid_show_rolling_and_session_separately() -> None:
    delta = SpecLiveDelta(
        elapsed_s=1.0,
        had_activity=True,
        cumulative_acceptance_rate=0.4,
        cumulative_acceptance_length=2.6,
        cumulative_draft_window=4.0,
        rolling_acceptance_rate=0.2,
        rolling_acceptance_length=1.8,
        generation_tps=50.0,
    )
    text = _render(
        _build_dashboard(delta, deque([delta]), 0.0, "m", "http://x/metrics", 1, poll_interval=5.0)
    )
    assert "ACCEPTANCE RATE (30s)" in text
    assert "20.0%" in text
    assert "Session α" in text and "40.0%" in text
    assert "τ (30s)" in text and "1.80" in text
    assert "Session Averages" in text
    assert "Refreshing every 5s" in text
    assert "60 polls · 05:00" in text  # 60 × 5 s of history


def test_gauge_falls_back_to_session_without_a_rolling_value() -> None:
    delta = SpecLiveDelta(cumulative_acceptance_rate=0.4, generation_tps=50.0)
    text = _render(_build_dashboard(delta, deque([delta]), 0.0, "m", "http://x/metrics", 1))
    assert "ACCEPTANCE RATE  " in text and "(30s)" not in text.split("Session α")[0]
    assert "40.0%" in text


def test_failed_polls_turn_the_header_red_and_date_the_data() -> None:
    delta = SpecLiveDelta(cumulative_acceptance_rate=0.4, generation_tps=50.0)
    text = _render(
        _build_dashboard(
            delta,
            deque([delta]),
            0.0,
            "m",
            "http://x/metrics",
            7,
            failed_polls=3,
            last_scrape_age_s=42.0,
        )
    )
    assert "✗" in text
    assert "3 failed polls, data 00:42 old" in text
    # Before any data at all, the connecting screen still reports failures.
    waiting = _render(
        _build_dashboard(None, deque(), 0.0, "m", "http://x/metrics", 2, failed_polls=2)
    )
    assert "2 failed polls" in waiting and "Connecting to" in waiting
