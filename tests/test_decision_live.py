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

from tests.test_decision_typed import (
    FakeTypedBackend,
    _hand_case,
    _right_on_yes_no_only,
)
from tool_eval_bench.cli import decision_live_display as display
from tool_eval_bench.domain.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionBackend,
    DecisionQuestion,
    DecisionResult,
    DecisionState,
    DecisionUnsupportedError,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)
from tool_eval_bench.plugins.decision import live
from tool_eval_bench.plugins.decision.live import (
    HISTOGRAM_BINS,
    LATENCY_WINDOW,
    CaseProbe,
    LiveStats,
    LoadTrend,
    ProbeEvent,
    ServerLoad,
    canary_cases,
    parse_server_load,
    probe_once,
    scrape_server_load,
)
from tool_eval_bench.plugins.decision.typed_decisions import (
    DatasetIntegrityError,
    GoldAnswer,
    load_cases,
)


def _event(**overrides: Any) -> ProbeEvent:
    defaults: dict[str, Any] = {
        "decision_id": "customer_service_000001/category",
        "question": "category",
        "question_type": "choice",
        "gold": "billing",
        "predicted": "billing",
        "correct": True,
        "confidence": 0.9,
        "brier": 0.02,
    }
    return ProbeEvent(**{**defaults, **overrides})


def _probe(*events: ProbeEvent, **overrides: Any) -> CaseProbe:
    defaults: dict[str, Any] = {
        "case_id": "customer_service_000001",
        "state": '{"thread": ["charged twice"]}',
        "decisions": events or (_event(),),
        "latency_ms": 20.0,
        "input_tokens": 40,
    }
    return CaseProbe(**{**defaults, **overrides})


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
    def test_cycles_every_test_case_and_repeats(self) -> None:
        cases = load_cases()
        gen = canary_cases()
        first_cycle = [next(gen).id for _ in range(len(cases))]
        second_cycle = [next(gen).id for _ in range(len(cases))]
        assert sorted(first_cycle) == sorted(c.id for c in cases)
        assert first_cycle == second_cycle

    def test_order_is_mixed_and_stable(self) -> None:
        first_ten = [c.id for _, c in zip(range(10), canary_cases(), strict=False)]
        again = [c.id for _, c in zip(range(10), canary_cases(), strict=False)]
        # A fixed shuffle: the same every run, and not one workflow in a row.
        assert first_ten == again
        assert len({i.rsplit("_", 1)[0] for i in first_ten}) > 1

    def test_takes_an_explicit_case_list(self) -> None:
        cases = [_hand_case("a1", "alpha"), _hand_case("b1", "beta")]
        gen = canary_cases(cases)
        assert sorted(next(gen).id for _ in range(2)) == ["a1", "b1"]


class TestProbeOnce:
    @pytest.mark.asyncio
    async def test_one_request_scores_every_question_of_the_case(self) -> None:
        case = _hand_case("a1", "alpha")
        backend = FakeTypedBackend([case], _right_on_yes_no_only)
        probe = await probe_once(
            backend, case, model="m", base_url="u", api_key=None, timeout_seconds=1.0
        )
        assert backend.calls == 1
        assert probe.case_id == "a1"
        assert [d.decision_id for d in probe.decisions] == ["a1/flag", "a1/pick", "a1/level"]
        assert [d.question for d in probe.decisions] == ["flag", "pick", "level"]
        assert [d.question_type for d in probe.decisions] == ["noul", "choice", "score"]
        assert [d.correct for d in probe.decisions] == [True, False, False]
        assert probe.correct == 1
        assert probe.latency_ms == 40.0 and probe.input_tokens == 100
        assert probe.state == '{"id": "a1"}'

    @pytest.mark.asyncio
    async def test_scores_against_the_gold_distribution(self) -> None:
        case = _hand_case("a1", "alpha")
        probe = await probe_once(
            FakeTypedBackend([case], _right_on_yes_no_only),
            case,
            model="m",
            base_url="u",
            api_key=None,
            timeout_seconds=1.0,
        )
        flag, pick, level = probe.decisions
        # yes/no: (0.9 - 0.8)^2 + (0.1 - 0.2)^2
        assert flag.brier == pytest.approx(0.02)
        assert pick.gold == "b" and pick.predicted == "a"
        # 90% on the wrong answer is a confident mistake.
        assert pick.confident_mistake and level.confident_mistake
        assert not flag.confident_mistake


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


class TestLiveStats:
    def test_empty_stats_are_zero_not_an_error(self) -> None:
        stats = LiveStats()
        assert stats.rolling_accuracy == 0.0 and stats.rolling_ece == 0.0
        assert stats.rolling_brier == 0.0
        assert stats.session_accuracy == 0.0 and stats.type_accuracy() == {}
        assert stats.latency_percentiles() == (0.0, 0.0)

    def test_folds_every_decision_of_a_case_and_remembers_the_last_miss(self) -> None:
        stats = LiveStats()
        miss = _event(
            decision_id="c/urgency", question="urgency", question_type="score", correct=False
        )
        probe = _probe(_event(), _event(question_type="noul"), miss)
        stats.record(probe)
        assert (stats.probes, stats.total, stats.correct) == (1, 3, 2)
        assert stats.session_accuracy == pytest.approx(2 / 3)
        assert list(stats.type_accuracy().items()) == [
            ("noul", 1.0),
            ("choice", 1.0),
            ("score", 0.0),
        ]
        assert stats.last_miss is miss and stats.last_probe is probe

    def test_tokens_and_latency_count_once_per_request(self) -> None:
        stats = LiveStats()
        stats.record(_probe(_event(), _event(), latency_ms=10.0, input_tokens=900))
        stats.record(_probe(_event(), _event(), latency_ms=30.0, input_tokens=800))
        assert stats.input_tokens == 1700
        assert stats.latency_percentiles() == (20.0, pytest.approx(29.0))
        assert list(stats.latency_trend) == [10.0, 30.0]
        assert len(stats.accuracy_trend) == 2

    def test_rolling_brier_is_the_mean_over_decisions(self) -> None:
        stats = LiveStats()
        stats.record(_probe(_event(brier=0.1), _event(brier=0.3)))
        assert stats.rolling_brier == pytest.approx(0.2)

    def test_only_a_confident_wrong_answer_counts_as_a_confident_mistake(self) -> None:
        stats = LiveStats()
        stats.record(
            _probe(_event(correct=False, confidence=0.89), _event(correct=True, confidence=0.99))
        )
        assert stats.confident_mistakes == 0
        stats.record(_probe(_event(correct=False, confidence=0.9)))
        assert stats.confident_mistakes == 1

    def test_the_rolling_window_forgets_old_decisions_but_the_session_does_not(self) -> None:
        stats = LiveStats(window=3)
        stats.record(_probe(*[_event(correct=False, confidence=0.5)] * 3))
        stats.record(_probe(*[_event(correct=True, confidence=0.5)] * 3))
        assert stats.rolling_accuracy == 1.0
        assert stats.session_accuracy == 0.5 and stats.total == 6

    def test_overconfidence_shows_up_in_rolling_ece(self) -> None:
        stats = LiveStats()
        stats.record(
            _probe(*(_event(correct=ok, confidence=0.95) for ok in (True, True, False, False)))
        )
        assert stats.rolling_ece == pytest.approx(0.45)
        assert stats.rolling_confidence == pytest.approx(0.95)

    def test_histogram_puts_certainty_in_the_last_bin(self) -> None:
        stats = LiveStats()
        stats.record(_probe(_event(confidence=1.0), _event(confidence=0.05)))
        counts = stats.confidence_histogram()
        assert len(counts) == HISTOGRAM_BINS
        assert counts[0] == 1 and counts[-1] == 1 and sum(counts) == 2

    def test_trends_are_bounded(self) -> None:
        stats = LiveStats()
        for _ in range(200):
            stats.record(_probe())
        assert len(stats.accuracy_trend) == len(stats.ece_trend) == 60

    def test_errors_do_not_count_as_probes(self) -> None:
        stats = LiveStats()
        stats.record_error("TimeoutError: slow")
        assert stats.errors == 1 and stats.total == 0 and stats.probes == 0
        assert stats.last_error == "TimeoutError: slow"

    def test_reset_clears_the_session_but_keeps_the_window_size(self) -> None:
        stats = LiveStats(window=7)
        stats.record(_probe(_event(correct=False, confidence=0.99)))
        stats.record_error("x")
        stats.reset()
        assert (stats.total, stats.probes, stats.errors, stats.confident_mistakes) == (0, 0, 0, 0)
        assert stats.input_tokens == 0 and stats.last_probe is None
        assert stats.last_miss is None and stats.window == 7 and not stats.events
        assert stats.events.maxlen == 7 and stats.latencies.maxlen == LATENCY_WINDOW


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

    def test_shows_every_decision_of_the_current_case(self) -> None:
        stats = LiveStats()
        stats.record(
            _probe(
                _event(
                    decision_id="c1/needs_human",
                    question="needs_human",
                    question_type="noul",
                    predicted="true",
                ),
                _event(
                    decision_id="c1/category",
                    correct=False,
                    predicted="shipping",
                    gold="billing",
                    confidence=0.7,
                ),
                case_id="c1",
                input_tokens=1674,
            )
        )
        text = _render(**_dashboard_args(stats))
        assert "1/2" in text and "c1" in text
        assert "needs_human" in text and "pred true" in text
        assert "pred shipping" in text and "gold billing" in text and "0.700" in text
        assert "20 ms" in text and "1,674 input tokens" in text
        assert "last miss" in text and "c1/category" in text

    def test_shows_brier_tokens_and_accuracy_by_question_type(self) -> None:
        stats = LiveStats()
        stats.record(_probe(_event(brier=0.25), _event(question_type="score", correct=False)))
        text = _render(**_dashboard_args(stats))
        assert "Brier vs gold" in text and "0.135" in text
        assert "by question type" in text and "choice" in text and "score" in text
        assert "1 cases · 2 decisions · 0 failed cases" in text

    def test_credits_the_dataset(self) -> None:
        text = _render(**_dashboard_args(attribution="Typed Decisions (LocalLLaMA/x), Apache-2.0"))
        assert "Cases: Typed Decisions (LocalLLaMA/x), Apache-2.0" in text

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

    def test_a_long_state_is_clipped(self) -> None:
        stats = LiveStats()
        stats.record(_probe(state="word " * 60))
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
        state: DecisionState,
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


def _confidently_first(_name: str, question: DecisionQuestion, _gold: GoldAnswer) -> DecisionAnswer:
    """97% on the first answer of every question, whatever the case says."""
    if isinstance(question, YesNoQuestion):
        return YesNoAnswer(0.97)
    if isinstance(question, ChoiceQuestion):
        names = list(question.options)
        rest = 0.03 / (len(names) - 1)
        return ChoiceAnswer(names[0], {n: 0.97 if i == 0 else rest for i, n in enumerate(names)})
    assert isinstance(question, ScoreQuestion)
    levels = len(question.levels)
    rest = 0.03 / (levels - 1)
    return ScoreAnswer(0.0, {str(i): 0.97 if i == 0 else rest for i in range(levels)})


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_probes_until_interrupted_and_detaches_its_handlers(self, harness: Any) -> None:
        run, handlers, removed, _ = harness
        backend = FakeTypedBackend(load_cases())
        waits = await run(backend, cycles=4)
        # One probe before the screen opens, then one per cycle.
        assert backend.calls == 1 + waits == 5
        assert set(removed) == {signal.SIGINT, signal.SIGTERM}
        text = _frame_text(_CapturingLive.frames[-1])
        assert "5 cases · 25 decisions" in text
        assert "Cases: Typed Decisions (LocalLLaMA/typed-decisions" in text

    @pytest.mark.asyncio
    async def test_every_probe_sends_a_whole_case(self, harness: Any) -> None:
        run, *_ = harness
        cases = load_cases()
        seen: list[int] = []

        class Counting(FakeTypedBackend):
            async def decide(self, **kwargs: Any) -> DecisionResult:
                seen.append(len(kwargs["questions"]))
                return await super().decide(**kwargs)

        await run(Counting(cases), cycles=3)
        assert seen == [5, 5, 5, 5]

    @pytest.mark.asyncio
    async def test_ctrl_r_resets_the_session(self, harness: Any) -> None:
        run, *_ = harness
        await run(FakeTypedBackend(load_cases()), cycles=3, reset_on=2)
        # Reset after cycle 2 leaves only the third cycle's case in the session.
        assert " 1 cases · 5 decisions" in _frame_text(_CapturingLive.frames[-1])

    @pytest.mark.asyncio
    async def test_an_unsupported_endpoint_stops_before_the_screen_opens(
        self, harness: Any
    ) -> None:
        run, handlers, removed, _ = harness
        with pytest.raises(DecisionUnsupportedError):
            await run(FakeTypedBackend([], unsupported=True), cycles=3)
        assert not handlers and not removed

    @pytest.mark.asyncio
    async def test_damaged_data_stops_before_an_adapter_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[object] = []

        def broken(*_a: Any) -> Any:
            raise DatasetIntegrityError("typed-decisions data has sha256 abc")

        monkeypatch.setattr(live, "load_cases", broken)
        monkeypatch.setattr(display, "build_decision_adapter", lambda **_kw: built.append(1))
        with pytest.raises(DatasetIntegrityError, match="sha256 abc"):
            await display.run_decision_live("http://h:1", model="laya")
        # No adapter, so no HTTP client left open behind the error.
        assert not built

    @pytest.mark.asyncio
    async def test_a_dead_model_shows_a_banner_instead_of_crashing(self, harness: Any) -> None:
        run, *_ = harness
        await run(_FailingBackend(), cycles=5)
        text = _frame_text(_CapturingLive.frames[-1])
        assert "model unreachable" in text and "TimeoutError" in text

    @pytest.mark.asyncio
    async def test_a_confident_mistake_raises_the_banner(self, harness: Any) -> None:
        run, *_ = harness
        await run(FakeTypedBackend(load_cases(), _confidently_first), cycles=5)
        assert any("CONFIDENT MISTAKE" in _frame_text(f) for f in _CapturingLive.frames)


def _frame_text(frame: Any) -> str:
    buffer = StringIO()
    Console(file=buffer, width=120).print(frame)
    return buffer.getvalue()


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


@pytest.mark.usefixtures("no_dns")
def test_the_cli_reports_damaged_data_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch

    def broken(*_a: Any) -> Any:
        raise DatasetIntegrityError("typed-decisions data has sha256 abc")

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(live, "load_cases", broken)
    # --no-probe-engine skips backend detection, which would GET http://h:1/metrics.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "http://h:1",
            "--no-probe-engine",
            "--decision-live",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        dispatch.main()
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "Decision monitor error:" in out and "sha256 abc" in out
    assert "Traceback" not in out
