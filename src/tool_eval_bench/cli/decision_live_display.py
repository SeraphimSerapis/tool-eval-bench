"""Rich Live terminal dashboard for decision models.

Powers ``--decision-live``.  Sends one canary item at a time to the model and
draws the probability distribution it returns, rolling accuracy and calibration,
a confidence histogram, latency, and the server's load counters.  Ctrl+R resets
the session and Ctrl+C exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
import time
from typing import Any

import httpx
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tool_eval_bench.adapters.factory import build_decision_adapter
from tool_eval_bench.cli.spec_live_rendering import _ACTIVITY_FRAMES, _format_uptime
from tool_eval_bench.domain.decision import DecisionUnsupportedError
from tool_eval_bench.plugins.decision.live import (
    HISTOGRAM_BINS,
    ROLLING_WINDOW,
    LiveStats,
    LoadTrend,
    ProbeEvent,
    canary_items,
    probe_once,
    scrape_server_load,
)
from tool_eval_bench.plugins.decision.render import bar
from tool_eval_bench.utils.urls import metrics_request_target, redact_url

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 0.5
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"
_TREND_WIDTH = 30
_BAR_WIDTH = 20
# Ticks a banner stays up after a confident mistake or a reset.
_FLASH_TICKS = 6
# Probes in a row that may fail before the dashboard says the model is unreachable.
_UNREACHABLE_AFTER = 3


def _trend(
    values: list[float], style: str, *, ceiling: float | None = None, width: int = _TREND_WIDTH
) -> Text:
    """A one-colour sparkline anchored at zero.

    The speculative-decoding sparkline colours low values red, which is wrong
    for latency and calibration error, where low is good.  Anchoring at zero
    keeps a flat series flat instead of stretching its noise to full height.
    """
    data = values[-width:]
    if not data:
        return Text("─" * width, style="dim")
    top = ceiling if ceiling is not None else (max(data) or 1.0)
    line = "".join(
        _SPARK_CHARS[max(0, min(len(_SPARK_CHARS) - 1, round(v / top * (len(_SPARK_CHARS) - 1))))]
        for v in data
    )
    return Text.assemble(Text("─" * (width - len(data)), style="dim"), Text(line, style=style))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _quality_style(accuracy: float) -> str:
    return "bright_green" if accuracy >= 0.85 else "yellow" if accuracy >= 0.6 else "bright_red"


def _now_panel(event: ProbeEvent | None, spinner: str) -> Panel:
    if event is None:
        body: RenderableType = Text("  waiting for the first answer…", style="dim italic")
        return Panel(body, title=f"{spinner} now", border_style="dim")

    mark = Text("✓ ", style="bold green") if event.correct else Text("✗ ", style="bold red")
    lines: list[RenderableType] = [
        Text.assemble(
            mark,
            Text(f"{event.item_id}  ", style="bold"),
            Text(event.state, style="italic"),
        )
    ]
    if event.is_score_scale:
        ordered = sorted(event.distribution.items(), key=lambda kv: int(kv[0]))
    else:
        ordered = sorted(event.distribution.items(), key=lambda kv: -kv[1])
    width = max(len(name) for name, _ in ordered)
    for name, p in ordered:
        is_gold, is_pred = name == event.gold, name == event.predicted
        style = "bright_green" if is_gold and is_pred else "bright_red" if is_pred else "cyan"
        row = Text(f"  {name:<{width}}  ")
        row.append(bar(p, _BAR_WIDTH), style=style if (is_gold or is_pred) else "dim")
        row.append(f" {p:5.3f}", style="bold" if is_pred else "")
        if is_gold:
            row.append("  ◀ gold", style="green")
        if is_pred and not is_gold:
            row.append("  ◀ predicted", style="red")
        lines.append(row)
    lines.append(
        Text(f"  {event.latency_ms:.0f} ms · {event.input_tokens} input tokens", style="dim")
    )
    border = "green" if event.correct else "red"
    return Panel(Group(*lines), title=f"{spinner} now", border_style=border)


def _quality_panel(stats: LiveStats) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)

    acc = stats.rolling_accuracy
    style = _quality_style(acc)
    grid.add_row(
        f"accuracy ({min(len(stats.events), ROLLING_WINDOW)} probes)",
        Text.assemble(Text(bar(acc, _BAR_WIDTH), style=style), Text(f" {acc:6.1%}", style=style)),
        _trend(list(stats.accuracy_trend), style, ceiling=1.0),
    )
    ece = stats.rolling_ece
    grid.add_row(
        "calibration error (ECE)",
        Text(f"{ece:.3f}", style="bold"),
        _trend(list(stats.ece_trend), "cyan"),
    )
    grid.add_row(
        "mean confidence",
        Text(f"{stats.rolling_confidence:.1%}", style="bold"),
        Text(f"session accuracy {stats.session_accuracy:.1%}", style="dim"),
    )
    cm = stats.confident_mistakes
    grid.add_row(
        "confident mistakes",
        Text(str(cm), style="bold bright_red" if cm else "bold bright_green"),
        Text("wrong at 90% confidence or more", style="dim"),
    )
    return Panel(grid, title="quality", border_style="bright_cyan")


def _categories_panel(stats: LiveStats) -> Panel:
    accuracy = stats.category_accuracy()
    if not accuracy:
        return Panel(Text("  no data yet", style="dim"), title="by category", border_style="dim")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", style="dim", no_wrap=True)
    for category, value in accuracy.items():
        seen = stats.categories[category][1]
        grid.add_row(
            category,
            Text.assemble(
                Text(bar(value, _BAR_WIDTH), style=_quality_style(value)),
                Text(f" {value:5.1%}", style="bold"),
            ),
            f"n={seen}",
        )
    return Panel(grid, title="by category", border_style="bright_cyan")


def _histogram_panel(stats: LiveStats) -> Panel:
    counts = stats.confidence_histogram()
    peak = max(counts) or 1
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(style="dim", justify="right", no_wrap=True)
    for i, count in enumerate(counts):
        low = i / HISTOGRAM_BINS
        grid.add_row(
            f"{low:.1f}–{low + 1 / HISTOGRAM_BINS:.1f}",
            Text(bar(count / peak, _BAR_WIDTH), style="bright_cyan" if count else "dim"),
            str(count),
        )
    return Panel(grid, title="confidence histogram", border_style="bright_cyan")


def _server_panel(stats: LiveStats, trend: LoadTrend) -> Panel:
    p50, p95 = stats.latency_percentiles()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(
        "latency",
        Text(f"p50 {p50:.0f} ms · p95 {p95:.0f} ms", style="bold"),
        _trend(list(stats.latency_trend), "yellow"),
    )
    load = trend.latest
    if load is None:
        grid.add_row("server", Text("no /metrics (load panels off)", style="dim italic"), "")
    else:
        rps = list(trend.requests_per_s)
        tps = list(trend.input_tokens_per_s)
        grid.add_row(
            "requests/s",
            Text(f"{rps[-1]:.1f}" if rps else "–", style="bold"),
            _trend(rps, "bright_cyan"),
        )
        grid.add_row(
            "input tokens/s",
            Text(f"{tps[-1]:,.0f}" if tps else "–", style="bold"),
            _trend(tps, "bright_cyan"),
        )
        grid.add_row(
            "slots",
            Text(
                f"{load.busy_slots:.0f} busy · {load.processing:.0f} processing · "
                f"{load.deferred:.0f} queued",
                style="bold",
            ),
            "",
        )
    return Panel(grid, title="server", border_style="bright_cyan")


def _alerts(stats: LiveStats, flash: str | None, unreachable: bool) -> list[RenderableType]:
    lines: list[RenderableType] = []
    if unreachable:
        lines.append(Text(f"  ⚠ model unreachable: {stats.last_error}", style="bold bright_red"))
    if flash:
        lines.append(Text(f"  {flash}", style="bold black on yellow"))
    miss = stats.last_miss
    if miss is not None:
        lines.append(
            Text.assemble(
                Text("  last miss  ", style="dim"),
                Text(f"{miss.item_id} ", style="bold"),
                Text(f"gold {miss.gold}, predicted {miss.predicted} at {miss.confidence:.2f}  "),
                Text(_clip(miss.state, 40), style="italic dim"),
            )
        )
    return lines


def build_dashboard(
    stats: LiveStats,
    trend: LoadTrend,
    *,
    model_name: str,
    url: str,
    tick: int,
    flash: str | None = None,
    unreachable: bool = False,
) -> RenderableType:
    """Compose the whole screen.  Pure, so it can be rendered in a test."""
    spinner = _ACTIVITY_FRAMES[tick % len(_ACTIVITY_FRAMES)]
    header = Text.assemble(
        Text(f"{spinner} ⚖️  Decision Live  ", style="bold bright_cyan"),
        Text(model_name, style="bold"),
        Text(f"  {url}", style="dim"),
        Text(
            f"  up {_format_uptime(time.time() - stats.started)} · "
            f"{stats.total} probes · {stats.errors} errors",
            style="dim",
        ),
    )
    middle = Table.grid(expand=True, padding=(0, 1))
    middle.add_column(ratio=1)
    middle.add_column(ratio=1)
    middle.add_row(_categories_panel(stats), _histogram_panel(stats))
    footer = Text("  Ctrl+R reset session · Ctrl+C quit", style="dim")
    return Group(
        header,
        _now_panel(stats.last_event, spinner),
        _quality_panel(stats),
        middle,
        _server_panel(stats, trend),
        *_alerts(stats, flash, unreachable),
        footer,
    )


async def _wait(stop: asyncio.Event, timeout: float) -> bool:
    """Wait up to ``timeout`` seconds.  Returns True when Ctrl+R was pressed.

    Reads one key in cbreak mode so Ctrl+C still raises its signal.  Without a
    terminal (piped output, Windows) it just sleeps, so the loop never spins.
    """
    try:
        import termios
        import tty
    except ImportError:
        termios = tty = None  # type: ignore[assignment]

    fd = None
    old = None
    if termios is not None and tty is not None and sys.stdin.isatty():
        try:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
        except (termios.error, OSError, ValueError):
            fd = None

    reset = asyncio.Event()
    loop = asyncio.get_running_loop()
    reader_added = False
    try:
        if fd is not None and tty is not None:
            tty.setcbreak(fd)

            def on_key() -> None:
                if sys.stdin.read(1) == "\x12":  # Ctrl+R
                    reset.set()

            loop.add_reader(fd, on_key)
            reader_added = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=timeout)
    finally:
        if reader_added and fd is not None:
            loop.remove_reader(fd)
        if fd is not None and old is not None and termios is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return reset.is_set()


async def run_decision_live(
    base_url: str,
    *,
    model: str,
    api_key: str | None = None,
    metrics_url: str | None = None,
    display_url: str | None = None,
    interval: float = DEFAULT_INTERVAL,
    timeout_seconds: float = 10.0,
    adapter_options: dict[str, Any] | None = None,
) -> None:
    """Run the live decision monitor until Ctrl+C.

    Raises ``DecisionUnsupportedError`` before the screen is taken over when the
    server has no decision endpoint, so the message lands in the normal terminal.
    """
    adapter = build_decision_adapter(**(adapter_options or {}))
    items = canary_items()
    scrape_url, scrape_headers = metrics_request_target(base_url, metrics_url, api_key)

    stats = LiveStats()
    trend = LoadTrend()
    # A first probe outside the alternate screen surfaces an unsupported
    # endpoint as a normal error instead of an empty dashboard.
    first = next(items)
    try:
        stats.record(
            await probe_once(
                adapter,
                first,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        )
    except DecisionUnsupportedError:
        await adapter.aclose()
        raise
    except Exception as exc:
        logger.debug("First probe failed: %s", exc)
        stats.record_error(f"{type(exc).__name__}: {exc}")

    # Handlers go in only after the first probe, so a Ctrl+C during it takes the
    # normal KeyboardInterrupt path and the unsupported-endpoint exit has nothing
    # to detach.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    attached: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
            attached.append(sig)

    console = Console()
    shown_url = display_url or redact_url(base_url)
    tick = 0
    flash: str | None = None
    flash_left = 0
    consecutive_failures = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            with Live(
                build_dashboard(stats, trend, model_name=model, url=shown_url, tick=tick),
                console=console,
                refresh_per_second=4,
                screen=True,
            ) as live:
                while not stop.is_set():
                    item = next(items)
                    probe = asyncio.create_task(
                        probe_once(
                            adapter,
                            item,
                            model=model,
                            base_url=base_url,
                            api_key=api_key,
                            timeout_seconds=timeout_seconds,
                        )
                    )
                    stopper = asyncio.create_task(stop.wait())
                    done, _ = await asyncio.wait(
                        {probe, stopper}, return_when=asyncio.FIRST_COMPLETED
                    )
                    stopper.cancel()
                    if probe not in done:
                        probe.cancel()
                        break
                    try:
                        event = probe.result()
                    except DecisionUnsupportedError:
                        raise
                    except Exception as exc:
                        consecutive_failures += 1
                        stats.record_error(f"{type(exc).__name__}: {exc}")
                    else:
                        consecutive_failures = 0
                        stats.record(event)
                        if event.confident_mistake:
                            flash = (
                                f"⚠ CONFIDENT MISTAKE  {event.item_id}: predicted "
                                f"{event.predicted} at {event.confidence:.2f}, gold {event.gold}"
                            )
                            flash_left = _FLASH_TICKS

                    load = await scrape_server_load(client, scrape_url, scrape_headers)
                    if load is not None:
                        trend.update(load)

                    tick += 1
                    if flash_left > 0:
                        flash_left -= 1
                        if flash_left == 0:
                            flash = None
                    live.update(
                        build_dashboard(
                            stats,
                            trend,
                            model_name=model,
                            url=shown_url,
                            tick=tick,
                            flash=flash,
                            unreachable=consecutive_failures >= _UNREACHABLE_AFTER,
                        )
                    )

                    if await _wait(stop, interval):
                        stats.reset()
                        trend = LoadTrend()
                        flash, flash_left = "↺ session reset", _FLASH_TICKS
    finally:
        for sig in attached:
            loop.remove_signal_handler(sig)
        await adapter.aclose()
