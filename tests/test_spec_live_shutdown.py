"""Regression tests for graceful shutdown of ``tool-eval-bench --spec-live``."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from typing import Any

import pytest

from tests.conftest import requires_termios
from tool_eval_bench.cli import spec_live_display as display


class DummyConsole:
    """Small stand-in for Rich Console used by the live monitor."""

    width = 100

    def print(self, *args: Any, **kwargs: Any) -> None:
        return None


class DummyLive:
    """Context manager matching the subset of Rich Live used by run_spec_live."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.updates: list[Any] = []

    def __enter__(self) -> "DummyLive":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, renderable: Any) -> None:
        self.updates.append(renderable)


def install_spec_live_harness(monkeypatch: pytest.MonkeyPatch):
    """Patch terminal/Rich dependencies and capture production signal handlers."""

    loop = asyncio.get_running_loop()
    handlers: dict[signal.Signals, Callable[[], None]] = {}
    removed: list[signal.Signals] = []

    def add_signal_handler(sig: signal.Signals, handler: Callable[..., None], *args: Any) -> None:
        handlers[sig] = lambda: handler(*args)

    def remove_signal_handler(sig: signal.Signals) -> bool:
        removed.append(sig)
        return True

    async def probe_server_spec_info(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
    monkeypatch.setattr(loop, "remove_signal_handler", remove_signal_handler)
    monkeypatch.setattr(display, "Console", DummyConsole)
    monkeypatch.setattr(display, "Live", DummyLive)
    monkeypatch.setattr(display, "_build_dashboard", lambda *args, **kwargs: "dashboard")
    monkeypatch.setattr(display, "probe_server_spec_info", probe_server_spec_info)

    return handlers, removed


@pytest.mark.asyncio
async def test_signal_handlers_include_sighup_and_are_removed(monkeypatch):
    """Production run_spec_live must wire SIGHUP and detach handlers on exit."""
    handlers, removed = install_spec_live_harness(monkeypatch)
    scrape_started = asyncio.Event()

    async def scrape_snapshot(*args: Any, **kwargs: Any) -> None:
        scrape_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(display, "scrape_snapshot", scrape_snapshot)

    task = asyncio.create_task(
        display.run_spec_live("http://127.0.0.1:8000/v1", poll_interval=0.01)
    )
    await asyncio.wait_for(scrape_started.wait(), timeout=1.0)

    assert signal.SIGINT in handlers
    assert signal.SIGTERM in handlers
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in handlers
        handlers[signal.SIGHUP]()
    else:
        handlers[signal.SIGTERM]()

    await asyncio.wait_for(task, timeout=1.0)
    assert set(removed) == set(handlers)


@pytest.mark.asyncio
async def test_first_signal_cancels_hung_scrape_and_exits(monkeypatch):
    """A single Ctrl+C must cancel the in-flight scrape and exit gracefully."""
    handlers, _removed = install_spec_live_harness(monkeypatch)
    scrape_started = asyncio.Event()
    scrape_cancelled = asyncio.Event()

    async def scrape_snapshot(*args: Any, **kwargs: Any) -> None:
        scrape_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            scrape_cancelled.set()
            raise

    monkeypatch.setattr(display, "scrape_snapshot", scrape_snapshot)

    task = asyncio.create_task(
        display.run_spec_live("http://127.0.0.1:8000/v1", poll_interval=30.0)
    )
    await asyncio.wait_for(scrape_started.wait(), timeout=1.0)

    handlers[signal.SIGINT]()

    await asyncio.wait_for(scrape_cancelled.wait(), timeout=1.0)
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_second_signal_forces_nonzero_exit(monkeypatch):
    """A second termination signal must use the production os._exit path."""
    handlers, _removed = install_spec_live_harness(monkeypatch)
    scrape_started = asyncio.Event()
    exit_codes: list[int] = []

    async def scrape_snapshot(*args: Any, **kwargs: Any) -> None:
        scrape_started.set()
        await asyncio.Event().wait()

    def fake_exit(code: int) -> None:
        exit_codes.append(code)

    monkeypatch.setattr(display, "scrape_snapshot", scrape_snapshot)
    monkeypatch.setattr(display.os, "_exit", fake_exit)

    task = asyncio.create_task(
        display.run_spec_live("http://127.0.0.1:8000/v1", poll_interval=30.0)
    )
    await asyncio.wait_for(scrape_started.wait(), timeout=1.0)

    handlers[signal.SIGINT]()
    handlers[signal.SIGINT]()

    assert exit_codes == [130]
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
@requires_termios
async def test_signal_during_idle_wait_exits_promptly(monkeypatch):
    """A signal arriving during the idle poll wait (not during a scrape) must
    unblock ``_check_stdin`` via the ``stop_event`` watcher within the loop
    tick, not after ``poll_interval``."""
    handlers, _removed = install_spec_live_harness(monkeypatch)

    # Force _check_stdin past its termios/fileno guards so the asyncio.wait
    # set (the code under test) actually executes instead of falling back.
    import sys as _sys
    import termios as termios_mod
    import tty as tty_mod

    monkeypatch.setattr(_sys.stdin, "fileno", lambda: 99)
    monkeypatch.setattr(termios_mod, "tcgetattr", lambda fd: [0] * 7)
    monkeypatch.setattr(termios_mod, "tcsetattr", lambda *a, **k: None)
    monkeypatch.setattr(tty_mod, "setcbreak", lambda fd: None)

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_reader", lambda fd, cb: None)
    monkeypatch.setattr(loop, "remove_reader", lambda fd: None)

    scrape_calls = 0
    idle_waiting = asyncio.Event()

    async def scrape_snapshot(*args: Any, **kwargs: Any) -> None:
        nonlocal scrape_calls
        scrape_calls += 1
        # After the first scrape the loop renders and enters _check_stdin.
        # Signal that we have reached the idle wait so the test can fire.
        if scrape_calls >= 1:
            idle_waiting.set()
        return None

    monkeypatch.setattr(display, "scrape_snapshot", scrape_snapshot)

    task = asyncio.create_task(
        display.run_spec_live("http://127.0.0.1:8000/v1", poll_interval=30.0)
    )
    await asyncio.wait_for(idle_waiting.wait(), timeout=1.0)
    # Let the loop actually reach the asyncio.wait inside _check_stdin.
    await asyncio.sleep(0.05)

    loop.call_soon(handlers[signal.SIGINT])

    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_sighup_suppresses_terminal_summary(monkeypatch):
    """On SIGHUP the controlling terminal is dead — run_spec_live must skip
    the post-shutdown summary instead of writing to a closed fd."""
    if not hasattr(signal, "SIGHUP"):
        pytest.skip("SIGHUP not available on this platform")

    handlers, _removed = install_spec_live_harness(monkeypatch)
    prints: list[Any] = []

    class CountingConsole(DummyConsole):
        def print(self, *args: Any, **kwargs: Any) -> None:
            prints.append(args)

    monkeypatch.setattr(display, "Console", CountingConsole)
    scrape_started = asyncio.Event()

    async def scrape_snapshot(*args: Any, **kwargs: Any) -> None:
        scrape_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(display, "scrape_snapshot", scrape_snapshot)

    task = asyncio.create_task(
        display.run_spec_live("http://127.0.0.1:8000/v1", poll_interval=30.0)
    )
    await asyncio.wait_for(scrape_started.wait(), timeout=1.0)

    handlers[signal.SIGHUP]()

    await asyncio.wait_for(task, timeout=1.0)
    assert prints == [], "SIGHUP path must not print a summary to a dead terminal"
