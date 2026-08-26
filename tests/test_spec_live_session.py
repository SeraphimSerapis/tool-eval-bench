"""The live monitor's actual work: deltas, sticky gauges, and the exit summary.

Every existing `run_spec_live` test scrapes `None`, so the loop body that
computes and renders a session was never exercised. That body is most of the
module, and it is what a user watching the dashboard is looking at.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from tool_eval_bench.cli import spec_live_display as display
from tool_eval_bench.runner.spec_live import MetricsSnapshot


class _CapturingLive:
    """The subset of Rich Live that `run_spec_live` uses, keeping every frame."""

    frames: list[Any] = []

    def __init__(self, renderable: Any, *args: Any, **kwargs: Any) -> None:
        type(self).frames = [renderable]

    def __enter__(self) -> _CapturingLive:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update(self, renderable: Any) -> None:
        type(self).frames.append(renderable)


@pytest.fixture
def monitor(monkeypatch: pytest.MonkeyPatch):
    """Run the monitor over a scripted series of scrapes and return what it drew.

    Unlike the shutdown tests, `_build_dashboard` is left real: the point here
    is to exercise the rendering, not to stub past it.
    """
    handlers: dict[signal.Signals, Callable[[], None]] = {}
    buffer = StringIO()

    monkeypatch.setattr(display, "Console", lambda *a, **k: Console(file=buffer, width=120))
    monkeypatch.setattr(display, "Live", _CapturingLive)

    async def no_probe(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(display, "probe_server_spec_info", no_probe)

    async def run(snapshots: list[MetricsSnapshot | None], **kwargs: Any):
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(
            loop,
            "add_signal_handler",
            lambda sig, fn, *a: handlers.__setitem__(sig, lambda: fn(*a)),
        )
        monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: True)
        remaining = list(snapshots)

        async def scrape(*args: Any, **kw: Any) -> MetricsSnapshot | None:
            if not remaining:
                handlers[signal.SIGINT]()
                return None
            return remaining.pop(0)

        monkeypatch.setattr(display, "scrape_snapshot", scrape)
        await asyncio.wait_for(
            display.run_spec_live("http://127.0.0.1:8000/v1", poll_interval=0.001, **kwargs),
            timeout=5.0,
        )
        return _CapturingLive.frames, buffer.getvalue()

    return run


def _vllm(timestamp: float, *, accepted: float, drafted: float, gen_tps: float) -> MetricsSnapshot:
    return MetricsSnapshot(
        timestamp=timestamp,
        accepted_tokens=accepted,
        draft_tokens=drafted,
        num_drafts=drafted / 4,
        generation_tps=gen_tps,
        prompt_tps=gen_tps * 3,
        gpu_cache_usage=0.42,
        running_reqs=2,
        spec_method="eagle",
        vllm_spec_metrics_present=True,
        spec_backend="vllm",
        per_position_rates={0: 0.9, 1: 0.6, 2: 0.3},
    )


@pytest.mark.asyncio
async def test_a_session_renders_a_frame_for_every_scrape(monitor) -> None:
    frames, _ = await monitor(
        [
            _vllm(1000.0, accepted=100, drafted=200, gen_tps=40.0),
            _vllm(1001.0, accepted=180, drafted=300, gen_tps=45.0),
            _vllm(1002.0, accepted=260, drafted=400, gen_tps=50.0),
        ]
    )

    # One frame built before the loop, then one per scrape.
    assert len(frames) >= 4


@pytest.mark.asyncio
async def test_the_exit_summary_reports_the_session(monitor) -> None:
    _, output = await monitor(
        [
            _vllm(1000.0, accepted=100, drafted=200, gen_tps=40.0),
            _vllm(1001.0, accepted=180, drafted=300, gen_tps=45.0),
        ]
    )

    assert "spec-live" in output and "stopped" in output
    assert "Duration" in output


@pytest.mark.asyncio
async def test_a_single_scrape_leaves_no_delta_to_summarise(monitor) -> None:
    """One snapshot cannot produce a rate, and must not divide by zero."""
    _, output = await monitor([_vllm(1000.0, accepted=100, drafted=200, gen_tps=40.0)])

    assert "stopped" in output


@pytest.mark.asyncio
async def test_a_server_without_spec_metrics_still_renders(monitor) -> None:
    plain = MetricsSnapshot(timestamp=1000.0, generation_tps=30.0)

    frames, output = await monitor([plain, plain])

    assert frames
    assert "stopped" in output


@pytest.mark.asyncio
async def test_a_llamacpp_session_is_rendered(monitor) -> None:
    def snap(timestamp: float, accepted: float, drafted: float) -> MetricsSnapshot:
        return MetricsSnapshot(
            timestamp=timestamp,
            llamacpp_accepted_tokens=accepted,
            llamacpp_draft_tokens=drafted,
            llamacpp_num_drafts=drafted / 5,
            llamacpp_predicted_tokens_seconds=55.0,
            llamacpp_prompt_tokens_seconds=210.0,
            llamacpp_predicted_tokens_total=drafted,
            llamacpp_kv_cache_usage_ratio=0.3,
            llamacpp_metrics_present=True,
            llamacpp_spec_metrics_present=True,
            spec_backend="llamacpp",
        )

    frames, output = await monitor([snap(1000.0, 50, 100), snap(1001.0, 130, 200)])

    assert len(frames) >= 3
    assert "stopped" in output


@pytest.mark.asyncio
async def test_an_sglang_session_uses_its_gauges(monitor) -> None:
    def snap(timestamp: float, rate: float) -> MetricsSnapshot:
        return MetricsSnapshot(
            timestamp=timestamp,
            sglang_acceptance_rate=rate,
            sglang_acceptance_length=2.4,
            sglang_num_steps=1000,
            sglang_num_draft_tokens=4000,
            sglang_spec_metrics_present=True,
            generation_tps=60.0,
            spec_backend="sglang",
        )

    frames, output = await monitor([snap(1000.0, 0.7), snap(1001.0, 0.75)])

    assert len(frames) >= 3
    assert "stopped" in output


@pytest.mark.asyncio
async def test_a_zero_gauge_does_not_wipe_the_last_real_reading(monitor) -> None:
    """vLLM zeroes its gauges between its internal updates; the panel must not flicker.

    The sticky value is only established once a delta has carried a non-zero
    reading, so this needs three scrapes: two to produce that delta, and a
    third reporting zero.
    """
    zeroed = _vllm(1002.0, accepted=260, drafted=400, gen_tps=0.0)
    zeroed.gpu_cache_usage = 0.0

    frames, _ = await monitor(
        [
            _vllm(1000.0, accepted=100, drafted=200, gen_tps=40.0),
            _vllm(1001.0, accepted=180, drafted=300, gen_tps=45.0),
            zeroed,
        ]
    )

    final = _render(frames[-1])
    gen_line = next(line for line in final.split("\n") if "Gen t/s" in line)
    assert "45" in gen_line, f"a zero reading wiped the last real one: {gen_line.strip()}"


def _render(frame: Any) -> str:
    buffer = StringIO()
    Console(file=buffer, width=140, no_color=True).print(frame)
    return buffer.getvalue()


class _FakeStdin:
    """A stdin with a real-looking descriptor; pytest's capture has none."""

    def __init__(self, char: str = "\x12") -> None:
        self._char = char

    def fileno(self) -> int:
        return 0

    def read(self, count: int) -> str:
        return self._char


class TestReadKeypress:
    """Ctrl+R resets the session counters, but only where a tty exists."""

    @pytest.fixture(autouse=True)
    def _stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setattr(sys, "stdin", _FakeStdin())

    @pytest.mark.asyncio
    async def test_piped_stdin_is_not_a_tty_and_reads_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import termios

        def not_a_tty(fd: int) -> None:
            raise termios.error("not a typewriter")

        monkeypatch.setattr(termios, "tcgetattr", not_a_tty)

        assert await display._read_keypress(asyncio.Event()) is None

    @pytest.mark.asyncio
    async def test_a_stop_event_ends_the_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ctrl+C while waiting on a key must not leave the reader hanging."""
        import termios
        import tty

        monkeypatch.setattr(termios, "tcgetattr", lambda fd: [])
        monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, old: None)
        monkeypatch.setattr(tty, "setraw", lambda fd: None)

        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_reader", lambda fd, fn: None)
        monkeypatch.setattr(loop, "remove_reader", lambda fd: True)

        stop = asyncio.Event()
        stop.set()

        assert await display._read_keypress(stop) is None

    @pytest.mark.asyncio
    async def test_a_keypress_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import termios
        import tty

        monkeypatch.setattr(termios, "tcgetattr", lambda fd: [])
        monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, old: None)
        monkeypatch.setattr(tty, "setraw", lambda fd: None)

        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_reader", lambda fd, fn: loop.call_soon(fn))
        monkeypatch.setattr(loop, "remove_reader", lambda fd: True)

        assert await display._read_keypress(asyncio.Event()) == "\x12"
