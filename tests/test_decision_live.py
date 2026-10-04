"""Live decision-model monitor: rolling stats, server load, dashboard, and run loop."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Mapping
from io import StringIO
from typing import Any

import httpx
import pytest
from rich.console import Console

from tests.test_decision import FakeDecisionBackend
from tool_eval_bench.cli import decision_live_display as display
from tool_eval_bench.domain.decision import (
    DecisionBackend,
    DecisionQuestion,
    DecisionResult,
    DecisionUnsupportedError,
)
from tool_eval_bench.plugins.decision.dataset import build_items
from tool_eval_bench.plugins.decision.live import (
    HISTOGRAM_BINS,
    LiveStats,
    LoadTrend,
    ProbeEvent,
    ServerLoad,
    canary_items,
    parse_server_load,
    probe_once,
    scrape_server_load,
)


def _event(**overrides: Any) -> ProbeEvent:
    defaults: dict[str, Any] = {
        "item_id": "route-01",
        "category": "routing",
        "state": "charged twice",
        "gold": "billing",
        "predicted": "billing",
        "correct": True,
        "confidence": 0.9,
        "brier": 0.02,
        "distribution": {"billing": 0.9, "shipping": 0.1},
        "is_score_scale": False,
        "latency_ms": 20.0,
        "input_tokens": 40,
        "at": 1000.0,
    }
    return ProbeEvent(**{**defaults, **overrides})


LLAMA_METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
llamacpp:prompt_tokens_total 16279
llamacpp:n_decode_total 250
llamacpp:requests_processing 1
llamacpp:requests_deferred 2
llamacpp:n_busy_slots_per_decode 1
"""


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


class TestCanary:
    def test_cycles_every_base_item_and_repeats(self) -> None:
        base = build_items(variants=False)
        gen = canary_items()
        first_cycle = [next(gen).id for _ in range(len(base))]
        second_cycle = [next(gen).id for _ in range(len(base))]
        assert sorted(first_cycle) == sorted(i.id for i in base)
        assert first_cycle == second_cycle

    def test_order_is_mixed_and_stable(self) -> None:
        first_ten = [i for _, i in zip(range(10), canary_items(), strict=False)]
        again = [i for _, i in zip(range(10), canary_items(), strict=False)]
        # A fixed shuffle: the same every run, and not one category in a row.
        assert [i.id for i in first_ten] == [i.id for i in again]
        assert len({i.category for i in first_ten}) > 1


class TestProbeOnce:
    @pytest.mark.asyncio
    async def test_scores_the_answer_against_the_gold_label(self) -> None:
        items = build_items(variants=False)
        item = items[0]
        event = await probe_once(
            FakeDecisionBackend(items, confidence=0.8),  # type: ignore[arg-type]
            item,
            model="m",
            base_url="u",
            api_key=None,
            timeout_seconds=1.0,
        )
        assert event.item_id == item.id and event.correct
        assert event.confidence == pytest.approx(0.8)
        assert event.latency_ms == 5.0 and event.input_tokens == 10
        assert not event.confident_mistake

    @pytest.mark.asyncio
    async def test_a_wrong_answer_at_high_confidence_is_a_confident_mistake(self) -> None:
        items = build_items(variants=False)
        wrong = next(i for i in items if i.category == "routing" and i.gold != "billing")
        event = await probe_once(
            FakeDecisionBackend(items, confidence=0.95, always_first=True),  # type: ignore[arg-type]
            wrong,
            model="m",
            base_url="u",
            api_key=None,
            timeout_seconds=1.0,
        )
        assert not event.correct and event.confident_mistake

    @pytest.mark.asyncio
    async def test_score_questions_are_flagged_for_scale_ordering(self) -> None:
        items = build_items(variants=False)
        urgency = next(i for i in items if i.category == "urgency")
        event = await probe_once(
            FakeDecisionBackend(items),  # type: ignore[arg-type]
            urgency,
            model="m",
            base_url="u",
            api_key=None,
            timeout_seconds=1.0,
        )
        assert event.is_score_scale


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


class TestLiveStats:
    def test_empty_stats_are_zero_not_an_error(self) -> None:
        stats = LiveStats()
        assert stats.rolling_accuracy == 0.0 and stats.rolling_ece == 0.0
        assert stats.session_accuracy == 0.0 and stats.category_accuracy() == {}
        assert stats.latency_percentiles() == (0.0, 0.0)

    def test_records_accuracy_by_category_and_remembers_the_last_miss(self) -> None:
        stats = LiveStats()
        stats.record(_event(item_id="a"))
        miss = _event(item_id="b", category="urgency", correct=False, confidence=0.6)
        stats.record(miss)
        assert stats.total == 2 and stats.correct == 1
        assert stats.session_accuracy == 0.5
        assert stats.category_accuracy() == {"routing": 1.0, "urgency": 0.0}
        assert stats.last_miss is miss and stats.last_event is miss

    def test_only_a_confident_wrong_answer_counts_as_a_confident_mistake(self) -> None:
        stats = LiveStats()
        stats.record(_event(correct=False, confidence=0.89))
        stats.record(_event(correct=True, confidence=0.99))
        assert stats.confident_mistakes == 0
        stats.record(_event(correct=False, confidence=0.9))
        assert stats.confident_mistakes == 1

    def test_the_rolling_window_forgets_old_probes_but_the_session_does_not(self) -> None:
        stats = LiveStats(window=3)
        for _ in range(3):
            stats.record(_event(correct=False, confidence=0.5))
        for _ in range(3):
            stats.record(_event(correct=True, confidence=0.5))
        assert stats.rolling_accuracy == 1.0
        assert stats.session_accuracy == 0.5 and stats.total == 6

    def test_overconfidence_shows_up_in_rolling_ece(self) -> None:
        stats = LiveStats()
        for ok in (True, True, False, False):
            stats.record(_event(correct=ok, confidence=0.95))
        assert stats.rolling_ece == pytest.approx(0.45)
        assert stats.rolling_confidence == pytest.approx(0.95)

    def test_histogram_puts_certainty_in_the_last_bin(self) -> None:
        stats = LiveStats()
        stats.record(_event(confidence=1.0))
        stats.record(_event(confidence=0.05))
        counts = stats.confidence_histogram()
        assert len(counts) == HISTOGRAM_BINS
        assert counts[0] == 1 and counts[-1] == 1 and sum(counts) == 2

    def test_trends_are_bounded(self) -> None:
        stats = LiveStats()
        for _ in range(200):
            stats.record(_event())
        assert len(stats.accuracy_trend) == len(stats.ece_trend) == 60

    def test_errors_do_not_count_as_probes(self) -> None:
        stats = LiveStats()
        stats.record_error("TimeoutError: slow")
        assert stats.errors == 1 and stats.total == 0
        assert stats.last_error == "TimeoutError: slow"

    def test_reset_clears_the_session_but_keeps_the_window_size(self) -> None:
        stats = LiveStats(window=7)
        stats.record(_event(correct=False, confidence=0.99))
        stats.record_error("x")
        stats.reset()
        assert (stats.total, stats.errors, stats.confident_mistakes) == (0, 0, 0)
        assert stats.last_miss is None and stats.window == 7 and not stats.events
        assert stats.events.maxlen == 7


# ---------------------------------------------------------------------------
# Server load
# ---------------------------------------------------------------------------


class TestServerLoad:
    def test_parses_llamacpp_counters(self) -> None:
        load = parse_server_load(LLAMA_METRICS, at=5.0)
        assert load == ServerLoad(
            at=5.0, prompt_tokens=16279, decodes=250, processing=1, deferred=2, busy_slots=1
        )

    def test_a_scrape_without_llamacpp_counters_is_not_a_load(self) -> None:
        assert parse_server_load("vllm:num_requests_running 3\n") is None
        assert parse_server_load("") is None

    def test_missing_optional_gauges_default_to_zero(self) -> None:
        load = parse_server_load("llamacpp:n_decode_total 9\n", at=1.0)
        assert load is not None and load.processing == 0 and load.prompt_tokens == 0

    def test_rates_come_from_successive_scrapes(self) -> None:
        trend = LoadTrend()
        trend.update(ServerLoad(10.0, 1000, 100, 0, 0, 0))
        assert not trend.requests_per_s
        trend.update(ServerLoad(12.0, 1500, 120, 1, 0, 1))
        assert list(trend.requests_per_s) == [10.0]
        assert list(trend.input_tokens_per_s) == [250.0]
        assert trend.latest is not None and trend.latest.processing == 1

    def test_a_server_restart_is_not_a_negative_rate(self) -> None:
        trend = LoadTrend()
        trend.update(ServerLoad(10.0, 9000, 900, 0, 0, 0))
        trend.update(ServerLoad(11.0, 50, 5, 0, 0, 0))
        assert list(trend.requests_per_s) == [0.0]
        assert list(trend.input_tokens_per_s) == [0.0]

    def test_a_scrape_with_no_elapsed_time_adds_no_rate(self) -> None:
        trend = LoadTrend()
        trend.update(ServerLoad(10.0, 0, 1, 0, 0, 0))
        trend.update(ServerLoad(10.0, 0, 2, 0, 0, 0))
        assert not trend.requests_per_s

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("handler", "expected"),
        [
            (lambda r: httpx.Response(200, text=LLAMA_METRICS), True),
            (lambda r: httpx.Response(500, text="no"), False),
            (lambda r: httpx.Response(200, text="unrelated 1\n"), False),
        ],
    )
    async def test_scrape_returns_a_load_or_nothing(
        self, handler: Callable[[httpx.Request], httpx.Response], expected: bool
    ) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            load = await scrape_server_load(client, "http://h/metrics", {})
        assert (load is not None) is expected

    @pytest.mark.asyncio
    async def test_scrape_survives_a_connection_error_and_sends_headers(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            raise httpx.ConnectError("down")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            load = await scrape_server_load(
                client, "http://h/metrics", {"Authorization": "Bearer t"}
            )
        assert load is None and seen["authorization"] == "Bearer t"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def _render(**kwargs: Any) -> str:
    buffer = StringIO()
    console = Console(file=buffer, width=120, record=False, force_terminal=False)
    console.print(display.build_dashboard(**kwargs))
    return buffer.getvalue()


def _dashboard_args(stats: LiveStats | None = None, trend: LoadTrend | None = None, **kw: Any):
    return {
        "stats": stats or LiveStats(),
        "trend": trend or LoadTrend(),
        "model_name": "laya",
        "url": "http://host:8084",
        "tick": 0,
        **kw,
    }


class TestDashboard:
    def test_an_empty_session_renders_without_error(self) -> None:
        text = _render(**_dashboard_args())
        assert "Decision Live" in text and "laya" in text
        assert "waiting for the first answer" in text
        assert "no data yet" in text
        assert "no /metrics" in text
        assert "Ctrl+R reset" in text

    def test_shows_the_current_distribution_with_gold_and_prediction_marked(self) -> None:
        stats = LiveStats()
        stats.record(
            _event(
                correct=False,
                predicted="shipping",
                confidence=0.7,
                distribution={"billing": 0.3, "shipping": 0.7},
            )
        )
        text = _render(**_dashboard_args(stats))
        assert "◀ gold" in text and "◀ predicted" in text
        assert "0.700" in text and "20 ms" in text
        assert "last miss" in text and "route-01" in text

    def test_score_levels_are_drawn_in_scale_order(self) -> None:
        stats = LiveStats()
        stats.record(
            _event(
                item_id="urg-01",
                category="urgency",
                gold="0",
                predicted="3",
                correct=False,
                is_score_scale=True,
                distribution={"3": 0.7, "2": 0.2, "1": 0.07, "0": 0.03},
            )
        )
        text = _render(**_dashboard_args(stats))
        rows = [ln.split()[1] for ln in text.splitlines() if ln.startswith("│   ")]
        levels = [r for r in rows if r in {"0", "1", "2", "3"}]
        assert levels == ["0", "1", "2", "3"]

    def test_shows_server_load_when_it_is_available(self) -> None:
        trend = LoadTrend()
        trend.update(ServerLoad(10.0, 1000, 100, 0, 0, 0))
        trend.update(ServerLoad(11.0, 1250, 110, 1, 2, 1))
        text = _render(**_dashboard_args(trend=trend))
        assert "requests/s" in text and "10.0" in text
        assert "input tokens/s" in text and "250" in text
        assert "1 busy" in text and "2 queued" in text
        assert "no /metrics" not in text

    def test_flash_and_unreachable_banners(self) -> None:
        stats = LiveStats()
        stats.record_error("ConnectError: refused")
        text = _render(**_dashboard_args(stats, flash="⚠ CONFIDENT MISTAKE x", unreachable=True))
        assert "CONFIDENT MISTAKE" in text
        assert "model unreachable: ConnectError: refused" in text

    def test_a_long_message_in_the_last_miss_is_clipped(self) -> None:
        stats = LiveStats()
        stats.record(_event(correct=False, state="word " * 60))
        text = _render(**_dashboard_args(stats))
        assert "…" in text
        assert all(len(line) <= 120 for line in text.splitlines())


class TestTrend:
    def test_no_data_is_a_dashed_line_of_full_width(self) -> None:
        assert display._trend([], "cyan").plain == "─" * 30

    def test_is_anchored_at_zero_so_a_flat_series_stays_flat(self) -> None:
        plain = display._trend([5.0] * 8, "cyan").plain
        assert plain.endswith("█" * 8) and plain.startswith("─")

    def test_a_fixed_ceiling_scales_proportions(self) -> None:
        plain = display._trend([0.5], "cyan", ceiling=1.0).plain
        assert plain.endswith("▄")

    def test_values_above_the_ceiling_do_not_overflow(self) -> None:
        assert display._trend([9.0], "cyan", ceiling=1.0).plain.endswith("█")


class TestWait:
    @pytest.mark.asyncio
    async def test_without_a_terminal_it_just_sleeps(self) -> None:
        stop = asyncio.Event()
        assert await display._wait(stop, 0.01) is False

    @pytest.mark.asyncio
    async def test_a_stop_signal_ends_the_wait_early(self) -> None:
        stop = asyncio.Event()
        stop.set()
        loop = asyncio.get_running_loop()
        started = loop.time()
        assert await display._wait(stop, 5.0) is False
        assert loop.time() - started < 1.0


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


class _CapturingLive:
    """The subset of Rich Live the monitor uses, keeping every frame."""

    frames: list[Any] = []

    def __init__(self, renderable: Any, *args: Any, **kwargs: Any) -> None:
        type(self).frames = [renderable]

    def __enter__(self) -> _CapturingLive:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update(self, renderable: Any) -> None:
        type(self).frames.append(renderable)


class _FailingBackend(DecisionBackend):
    def __init__(self) -> None:
        self.calls = 0

    async def decide(
        self,
        *,
        model: str,
        state: str,
        questions: Mapping[str, DecisionQuestion],
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        base_url: str = "",
    ) -> DecisionResult:
        self.calls += 1
        raise TimeoutError("synthetic")

    async def aclose(self) -> None:
        return None


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    """Run the monitor for a scripted number of cycles, then stop it with SIGINT."""
    handlers: dict[signal.Signals, Callable[[], None]] = {}
    removed: list[signal.Signals] = []
    buffer = StringIO()
    monkeypatch.setattr(display, "Console", lambda *a, **k: Console(file=buffer, width=120))
    monkeypatch.setattr(display, "Live", _CapturingLive)

    async def run(backend: Any, *, cycles: int, reset_on: int | None = None, **kwargs: Any):
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(
            loop, "add_signal_handler", lambda sig, fn, *a: handlers.__setitem__(sig, fn)
        )
        monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: removed.append(sig) or True)
        monkeypatch.setattr(display, "build_decision_adapter", lambda **_kw: backend)

        waits = 0
        stop_holder: list[asyncio.Event] = []

        async def fake_wait(stop: asyncio.Event, timeout: float) -> bool:
            nonlocal waits
            waits += 1
            stop_holder.append(stop)
            if waits >= cycles:
                handlers[signal.SIGINT]()
            return reset_on is not None and waits == reset_on

        async def no_load(*args: Any, **kw: Any) -> None:
            return None

        monkeypatch.setattr(display, "_wait", fake_wait)
        monkeypatch.setattr(display, "scrape_server_load", no_load)
        await asyncio.wait_for(
            display.run_decision_live("http://h:1", model="laya", interval=0.0, **kwargs),
            timeout=10.0,
        )
        return waits

    return run, handlers, removed, buffer


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_probes_until_interrupted_and_detaches_its_handlers(self, harness: Any) -> None:
        run, handlers, removed, _ = harness
        backend = FakeDecisionBackend(build_items(variants=False))
        waits = await run(backend, cycles=4)
        # One probe before the screen opens, then one per cycle.
        assert backend.calls == 1 + waits == 5
        assert set(removed) == {signal.SIGINT, signal.SIGTERM}
        last_frame = _CapturingLive.frames[-1]
        buffer = StringIO()
        Console(file=buffer, width=120).print(last_frame)
        assert "5 probes" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_ctrl_r_resets_the_session(self, harness: Any) -> None:
        run, *_ = harness
        backend = FakeDecisionBackend(build_items(variants=False))
        await run(backend, cycles=3, reset_on=2)
        buffer = StringIO()
        Console(file=buffer, width=120).print(_CapturingLive.frames[-1])
        # Reset after cycle 2 leaves only the third cycle's probe in the session.
        assert " 1 probes" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_an_unsupported_endpoint_stops_before_the_screen_opens(
        self, harness: Any
    ) -> None:
        run, handlers, removed, _ = harness
        backend = FakeDecisionBackend(build_items(variants=False), unsupported=True)
        with pytest.raises(DecisionUnsupportedError):
            await run(backend, cycles=3)
        assert not handlers and not removed

    @pytest.mark.asyncio
    async def test_a_dead_model_shows_a_banner_instead_of_crashing(self, harness: Any) -> None:
        run, *_ = harness
        backend = _FailingBackend()
        await run(backend, cycles=5)
        buffer = StringIO()
        Console(file=buffer, width=120).print(_CapturingLive.frames[-1])
        text = buffer.getvalue()
        assert "model unreachable" in text and "TimeoutError" in text

    @pytest.mark.asyncio
    async def test_a_confident_mistake_raises_the_banner(self, harness: Any) -> None:
        run, *_ = harness
        items = build_items(variants=False)
        backend = FakeDecisionBackend(items, confidence=0.97, always_first=True)
        await run(backend, cycles=len(items))
        texts = []
        for frame in _CapturingLive.frames:
            buffer = StringIO()
            Console(file=buffer, width=120).print(frame)
            texts.append(buffer.getvalue())
        assert any("CONFIDENT MISTAKE" in t for t in texts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> Any:
    from tool_eval_bench.cli.legacy_parser import make_parser
    from tool_eval_bench.cli.parser import parse_cli_args

    _, args = parse_cli_args(make_parser, argv)
    return args


class TestCli:
    def test_flat_flag_and_interval(self) -> None:
        args = _parse(["--decision-live", "--decision-live-interval", "0.25"])
        assert args.decision_live is True and args.decision_live_interval == 0.25

    def test_defaults_off(self) -> None:
        args = _parse(["--short"])
        assert args.decision_live is False and args.decision_live_interval == 0.5

    def test_subcommand_spelling(self) -> None:
        assert _parse(["decision-live"]).decision_live is True

    def test_the_command_is_in_the_public_schema(self) -> None:
        from tool_eval_bench.cli.command_registry import commands_schema

        assert "decision-live" in commands_schema()
