"""Rich Live terminal dashboard for speculative decoding stats.

Powers the ``--spec-live`` monitor.  Polls Prometheus ``/metrics`` on a
configurable interval and renders a full-terminal Rich Live dashboard with
acceptance rate gauges, throughput sparklines, per-position bars, and engine
status.  Press Ctrl+R to reset session counters; Ctrl+C to exit.

Layout builders live in ``spec_live_dashboard`` and rendering primitives live
in ``spec_live_rendering``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, replace

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from tool_eval_bench.cli.spec_live_dashboard import _build_dashboard
from tool_eval_bench.cli.spec_live_rendering import (
    _HISTORY_LEN,
    _POLL_INTERVAL,
    _ROLLING_WINDOW_S,
    _ar_color,
    _format_uptime,
)
from tool_eval_bench.runner.spec_live import (
    MetricsSnapshot,
    ServerSpecInfo,
    SpecLiveDelta,
    apply_rolling_rates,
    compute_delta,
    counter_delta,
    per_position_rates_from_counters,
    probe_server_spec_info,
    scrape_snapshot,
)
from tool_eval_bench.utils.urls import metrics_request_target, redact_url

logger = logging.getLogger(__name__)

# vLLM's deprecated throughput gauges read 0 between its internal updates,
# roughly every 10 s. A zero is bridged for at most this long; after that the
# server is idle, and the dashboard must say so.
_STICKY_GAUGE_MAX_AGE_S = 10.0


@dataclass
class _StickyGauge:
    """The last non-zero reading of a gauge that can flicker to zero."""

    value: float = 0.0
    seen_at: float | None = None

    def apply(self, reading: float, now: float, *, may_hold: bool) -> float:
        """Return *reading*, or the held value when *reading* is a recent flicker.

        *may_hold* is False for counter-derived readings: a zero counter delta
        means nothing happened, so it is never papered over.
        """
        if reading > 0:
            self.value, self.seen_at = reading, now
            return reading
        if may_hold and self.seen_at is not None and now - self.seen_at <= _STICKY_GAUGE_MAX_AGE_S:
            return self.value
        return reading


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def _read_keypress(stop_event: asyncio.Event) -> str | None:
    """Non-blocking stdin key reader for the async loop.

    Returns a single character when a key is pressed, or None if
    the stop event fires first.  Only works on Unix (uses termios
    raw mode); returns None immediately on unsupported platforms.
    """
    import sys

    try:
        import termios
        import tty
    except ImportError:
        return None  # Windows — Ctrl+R not supported

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        return None  # not a real tty (e.g. piped input)

    try:
        tty.setraw(fd)
        loop = asyncio.get_event_loop()
        # Wait for stdin to become readable or stop_event
        future: asyncio.Future[str | None] = loop.create_future()

        def _on_readable() -> None:
            if not future.done():
                ch = sys.stdin.read(1)
                future.set_result(ch)

        loop.add_reader(fd, _on_readable)
        try:
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait(
                [future, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )  # type: ignore[type-var]
            if future.done():
                return future.result()
            return None
        finally:
            loop.remove_reader(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def run_spec_live(
    base_url: str,
    *,
    api_key: str | None = None,
    metrics_url: str | None = None,
    model_name: str = "unknown",
    poll_interval: float = _POLL_INTERVAL,
    spec_method: str | None = None,
    redact_endpoint: bool = False,
) -> None:
    """Run the live speculative decoding monitor (blocking).

    Polls /metrics at ``poll_interval`` and renders a Rich Live dashboard.
    Press Ctrl+C to exit gracefully, Ctrl+R to reset session counters.
    ``redact_endpoint`` (``--redact-url``) masks the host of the metrics
    endpoint the dashboard header shows.

    Uses the terminal alternate screen buffer so the dashboard occupies
    the entire terminal without disturbing previous output.
    """
    import sys

    # An explicit --metrics-url may live on a different host than the inference
    # API, so the bearer token only travels when the origins match.
    url, metrics_headers = metrics_request_target(base_url, metrics_url, api_key)
    shown_endpoint = redact_url(url) if redact_endpoint else url

    # ── Probe server for spec decode config (draft model, method, k) ──
    server_spec_info: ServerSpecInfo | None = None
    try:
        server_spec_info = await probe_server_spec_info(
            base_url,
            api_key=api_key,
            primary_model=model_name,
        )
    except Exception:
        logger.debug("Server spec info probe failed — using Prometheus heuristics")

    history: deque[SpecLiveDelta] = deque(maxlen=_HISTORY_LEN)
    prev_snap: MetricsSnapshot | None = None
    baseline_snap: MetricsSnapshot | None = None  # first snapshot — for session-relative counters
    start_time = time.time()
    poll_count = 0
    last_delta: SpecLiveDelta | None = None
    reset_flash_remaining = 0  # show reset banner for N poll cycles
    failed_polls = 0  # consecutive scrapes that returned nothing
    last_scrape_ok: float | None = None
    # Generation throughput over the polls that generated tokens, across the
    # whole session rather than the history window.  Accumulated from raw
    # readings, before any sticky value stands in for a zero.
    gen_tps_sum = 0.0
    gen_tps_count = 0
    gen_tps_peak = 0.0

    # Sticky gauges: see _StickyGauge.  KV cache is not held, because 0% is a
    # valid idle reading (compute_delta treats it as one).
    sticky_gen_tps = _StickyGauge()
    sticky_prompt_tps = _StickyGauge()
    sticky_prefix_cache_pct = _StickyGauge()

    stop_event = asyncio.Event()
    signal_count = 0
    received_hup = False

    poll_task: asyncio.Task[MetricsSnapshot | None] | None = None
    tty_restore = None

    def _restore_terminal_for_exit() -> None:
        if tty_restore is not None:
            try:
                tty_restore()
            except (OSError, ValueError):
                logger.debug("Failed to restore terminal settings before force exit")
        try:
            sys.stdout.write("\033[?1049l")  # rmcup — leave alt screen
            sys.stdout.flush()
        except (OSError, ValueError):
            # Last-resort cleanup before os._exit: a closed/dead stdout can
            # raise ValueError or OSError — never block the force exit.
            pass

    def _handle_signal(sig: signal.Signals) -> None:
        nonlocal received_hup, signal_count
        if hasattr(signal, "SIGHUP") and sig == signal.SIGHUP:
            received_hup = True
        signal_count += 1
        if signal_count >= 2:
            # Restore the screen FIRST so the warning is visible on the real
            # terminal, then log, then bail out unconditionally.
            _restore_terminal_for_exit()
            logger.warning("Second termination signal received — forcing exit")
            os._exit(130)
        stop_event.set()
        if poll_task is not None:
            poll_task.cancel()

    loop = asyncio.get_event_loop()
    signals: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM)
    if hasattr(signal, "SIGHUP"):
        signals = (*signals, signal.SIGHUP)
    for sig in signals:
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except NotImplementedError:
            pass  # Windows — fall back to default handler

    # ── Enter alternate screen buffer ──
    # This gives us a clean, full-terminal canvas (like htop/vim).
    # The original terminal content is restored when we leave.
    sys.stdout.write("\033[?1049h")  # smcup — enter alt screen
    sys.stdout.flush()

    try:
        console = Console()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        ) as client:
            with Live(
                _build_dashboard(
                    None,
                    history,
                    start_time,
                    model_name,
                    shown_endpoint,
                    0,
                    baseline_snap,
                    term_width=console.width,
                    server_spec_info=server_spec_info,
                ),
                console=console,
                refresh_per_second=2,
                transient=False,
                screen=False,  # we manage the screen ourselves
            ) as live:
                while not stop_event.is_set():
                    poll_task = asyncio.create_task(scrape_snapshot(client, url, metrics_headers))
                    try:
                        snap = await poll_task
                    except asyncio.CancelledError:
                        if stop_event.is_set():
                            break
                        raise
                    finally:
                        poll_task = None
                    poll_count += 1
                    if snap is None:
                        failed_polls += 1
                    else:
                        failed_polls = 0
                        last_scrape_ok = time.time()

                    if snap is not None and (snap.has_spec_decode or snap.has_llamacpp_metrics):
                        # Capture baseline on first successful scrape (session-relative counters)
                        if baseline_snap is None:
                            baseline_snap = snap

                        if prev_snap is not None:
                            delta = compute_delta(prev_snap, snap)

                            # ── Make everything session-relative ──
                            # All cumulative metrics should reflect only what
                            # happened while the dashboard is open.
                            if (
                                baseline_snap is not None
                                and delta.counter_metrics_available
                                and snap.has_counter_spec_decode
                            ):
                                sess_accepted = counter_delta(
                                    baseline_snap.accepted_tokens,
                                    snap.accepted_tokens,
                                )
                                sess_drafted = counter_delta(
                                    baseline_snap.draft_tokens,
                                    snap.draft_tokens,
                                )
                                sess_drafts = counter_delta(
                                    baseline_snap.num_drafts,
                                    snap.num_drafts,
                                )

                                # Session acceptance rate
                                if sess_drafted > 0:
                                    delta.cumulative_acceptance_rate = sess_accepted / sess_drafted
                                else:
                                    delta.cumulative_acceptance_rate = None

                                # Session acceptance length (τ), draft window,
                                # and the k inferred from the window.
                                if sess_drafts > 0:
                                    delta.cumulative_acceptance_length = (
                                        1.0 + sess_accepted / sess_drafts
                                    )
                                    delta.cumulative_draft_window = sess_drafted / sess_drafts
                                    delta.num_spec_tokens = round(sess_drafted / sess_drafts)
                                else:
                                    delta.cumulative_acceptance_length = None
                                    delta.cumulative_draft_window = None
                                    delta.num_spec_tokens = None

                                # Session per-position rates from counters
                                if snap.per_position_counters:
                                    if sess_drafts > 0:
                                        sess_accepted_pos = {
                                            pos: counter_delta(
                                                baseline_snap.per_position_counters.get(pos, 0.0),
                                                count,
                                            )
                                            for pos, count in snap.per_position_counters.items()
                                        }
                                        sess_drafted_pos = {
                                            pos: counter_delta(
                                                baseline_snap.per_position_drafted_counters.get(
                                                    pos, 0.0
                                                ),
                                                count,
                                            )
                                            for pos, count in (
                                                snap.per_position_drafted_counters.items()
                                            )
                                        }
                                        delta.per_position_rates = per_position_rates_from_counters(
                                            sess_accepted_pos, sess_drafted_pos, sess_drafts
                                        )
                                    else:
                                        # No new drafts yet — don't show stale all-time rates
                                        delta.per_position_rates = {}
                                elif sess_drafts == 0:
                                    delta.per_position_rates = {}

                            # Override spec method from --spec-method CLI flag
                            if spec_method is not None:
                                delta.spec_method = spec_method
                            # Override spec method from ServerSpecInfo (API probe)
                            elif server_spec_info and server_spec_info.spec_method:
                                delta.spec_method = server_spec_info.spec_method

                            # History and the session average keep the raw
                            # readings, so an idle poll counts as idle.
                            history.append(delta)
                            apply_rolling_rates(history, _ROLLING_WINDOW_S)
                            if delta.generation_tps > 0:
                                gen_tps_sum += delta.generation_tps
                                gen_tps_count += 1
                                gen_tps_peak = max(gen_tps_peak, delta.generation_tps)

                            # Only the copy on screen bridges a gauge flicker.
                            now = snap.timestamp
                            last_delta = replace(
                                delta,
                                generation_tps=sticky_gen_tps.apply(
                                    delta.generation_tps,
                                    now,
                                    may_hold=delta.generation_tps_is_gauge,
                                ),
                                prompt_tps=sticky_prompt_tps.apply(
                                    delta.prompt_tps, now, may_hold=delta.prompt_tps_is_gauge
                                ),
                                prefix_cache_hit_pct=sticky_prefix_cache_pct.apply(
                                    delta.prefix_cache_hit_pct, now, may_hold=True
                                ),
                            )
                        prev_snap = snap
                    elif snap is not None and prev_snap is None:
                        # First scrape, no spec decode counters yet — store for next
                        prev_snap = snap

                    # Decrement reset flash counter
                    if reset_flash_remaining > 0:
                        reset_flash_remaining -= 1

                    live.update(
                        _build_dashboard(
                            last_delta,
                            history,
                            start_time,
                            model_name,
                            shown_endpoint,
                            poll_count,
                            baseline_snap,
                            term_width=console.width,
                            server_spec_info=server_spec_info,
                            reset_flash=reset_flash_remaining > 0,
                            poll_interval=poll_interval,
                            failed_polls=failed_polls,
                            last_scrape_age_s=(
                                time.time() - last_scrape_ok if last_scrape_ok else None
                            ),
                        )
                    )

                    # ── Wait for poll interval OR Ctrl+R keypress ──
                    reset_event = asyncio.Event()

                    async def _check_stdin() -> bool:
                        """Wait one poll interval while watching stdin for Ctrl+R.

                        Returns False when stdin cannot be watched (no termios,
                        or not a TTY) so the caller sleeps instead; otherwise
                        the loop would spin at full speed whenever output is
                        piped.
                        """
                        nonlocal tty_restore

                        try:
                            import termios  # noqa: F811
                            import tty  # noqa: F811
                        except ImportError:
                            return False
                        import sys as _sys  # avoid shadowing outer

                        fd = _sys.stdin.fileno()
                        try:
                            old = termios.tcgetattr(fd)
                        except termios.error:
                            return False

                        def _restore_tty() -> None:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old)

                        try:
                            tty.setcbreak(fd)  # cbreak: signals still work
                            tty_restore = _restore_tty
                            _loop = asyncio.get_event_loop()
                            fut: asyncio.Future[None] = _loop.create_future()

                            def _readable(_evt=reset_event) -> None:  # noqa: B023
                                if not fut.done():
                                    ch = _sys.stdin.read(1)
                                    if ch == "\x12":  # Ctrl+R
                                        _evt.set()
                                    fut.set_result(None)

                            _loop.add_reader(fd, _readable)
                            # Create the wait tasks only AFTER add_reader succeeds;
                            # otherwise a failing add_reader would orphan them.
                            stop_task = _loop.create_task(stop_event.wait())
                            timeout_task = _loop.create_task(asyncio.sleep(poll_interval))
                            try:
                                await asyncio.wait(
                                    {fut, stop_task, timeout_task},
                                    return_when=asyncio.FIRST_COMPLETED,
                                )  # type: ignore[type-var]
                            finally:
                                stop_task.cancel()
                                timeout_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await stop_task
                                with contextlib.suppress(asyncio.CancelledError):
                                    await timeout_task
                                try:
                                    _loop.remove_reader(fd)
                                except (OSError, ValueError):
                                    logger.debug("Failed to remove stdin reader")
                        finally:
                            try:
                                _restore_tty()
                            except (OSError, ValueError):
                                logger.debug("Failed to restore terminal settings")
                            tty_restore = None
                        return True

                    # Run stdin check with poll timeout
                    try:
                        waited = await _check_stdin()
                    except Exception:
                        waited = False
                    if not waited:
                        # Plain wait: no stdin support or stdin is not a TTY.
                        try:
                            await asyncio.wait_for(
                                stop_event.wait(),
                                timeout=poll_interval,
                            )
                            break
                        except asyncio.TimeoutError:
                            pass

                    if stop_event.is_set():
                        break

                    # ── Handle Ctrl+R session reset ──
                    if reset_event.is_set():
                        history.clear()
                        prev_snap = None
                        baseline_snap = None
                        last_delta = None
                        start_time = time.time()
                        poll_count = 0
                        gen_tps_sum = 0.0
                        gen_tps_count = 0
                        gen_tps_peak = 0.0
                        sticky_gen_tps = _StickyGauge()
                        sticky_prompt_tps = _StickyGauge()
                        sticky_prefix_cache_pct = _StickyGauge()
                        reset_flash_remaining = 3  # show banner for 3 poll cycles
    except OSError:
        # On SIGHUP the controlling terminal (PTY) is already gone, so Rich's
        # final ``Live`` refresh writes to a dead fd and raises.  Suppress it
        # ONLY on that path — never mask a real I/O error in normal operation.
        if not received_hup:
            raise

    finally:
        # ── Leave alternate screen buffer ──
        try:
            sys.stdout.write("\033[?1049l")  # rmcup — leave alt screen
            sys.stdout.flush()
        except OSError:
            pass
        for sig in signals:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

    if received_hup:
        return

    # Print session summary to the restored normal terminal
    console = Console()
    console.print()
    console.print("  [bold bright_magenta]◆ spec-live[/] stopped.")

    # Print session summary.  Session α is the pooled counter ratio for the
    # whole session, not an average of the running average; the spread is
    # the range of the rolling window, which is what actually varied.
    if last_delta is not None and last_delta.cumulative_acceptance_rate is not None:
        session_ar = last_delta.cumulative_acceptance_rate
        session_tau = last_delta.cumulative_acceptance_length
        rolling_vals = [
            d.rolling_acceptance_rate for d in history if d.rolling_acceptance_rate is not None
        ]
        avg_gen = gen_tps_sum / gen_tps_count if gen_tps_count else 0.0

        # Session-relative totals for exit summary
        if last_delta.counter_metrics_available:
            if baseline_snap:
                sess_accepted_text = f"{int(counter_delta(baseline_snap.accepted_tokens, last_delta.total_accepted)):,}"
                sess_drafted_text = (
                    f"{int(counter_delta(baseline_snap.draft_tokens, last_delta.total_drafted)):,}"
                )
            else:
                sess_accepted_text = f"{last_delta.total_accepted:,}"
                sess_drafted_text = f"{last_delta.total_drafted:,}"
        else:
            sess_accepted_text = sess_drafted_text = "—"

        alpha_line: list[tuple[str, str]] = [
            ("  Session α:       ", "dim"),
            (f"{session_ar * 100:.1f}%", f"bold {_ar_color(session_ar)}"),
        ]
        if len(rolling_vals) > 1:
            alpha_line.append(
                (
                    f"  ({_ROLLING_WINDOW_S:g}s window ranged "
                    f"{min(rolling_vals) * 100:.0f}–{max(rolling_vals) * 100:.0f}%)",
                    "dim",
                )
            )
        if session_tau is not None:
            alpha_line.extend([("  │  τ ", "dim"), (f"{session_tau:.2f}", "bold cyan")])

        console.print()
        console.print(
            Panel(
                Text.assemble(
                    ("  Duration:        ", "dim"),
                    (_format_uptime(time.time() - start_time), "bold"),
                    ("  │  ", "dim"),
                    (f"{poll_count} polls", "dim"),
                    ("\n", ""),
                    *alpha_line,
                    ("\n", ""),
                    ("  Avg Gen t/s:     ", "dim"),
                    (f"{avg_gen:.1f}", "bold bright_green"),
                    ("  ", ""),
                    ("peak ", "dim"),
                    (f"{gen_tps_peak:.1f}", "bold"),
                    ("\n", ""),
                    ("  Session tokens:  ", "dim"),
                    (sess_accepted_text, "bold"),
                    (" accepted  / ", "dim"),
                    (sess_drafted_text, "bold"),
                    (" drafted", "dim"),
                ),
                title="[bold]Session Summary[/]",
                border_style="bright_magenta",
                padding=(0, 1),
            )
        )
    console.print()


__all__ = ["_build_dashboard", "run_spec_live"]
