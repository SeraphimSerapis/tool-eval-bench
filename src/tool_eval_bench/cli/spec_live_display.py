"""Rich Live terminal dashboard for speculative decoding stats.

Powers the ``--spec-live`` monitor.  Polls Prometheus ``/metrics`` on a
configurable interval and renders a full-terminal Rich Live dashboard with
acceptance rate gauges, throughput sparklines, per-position bars, and engine
status.  Press Ctrl+R to reset session counters; Ctrl+C to exit.

Rendering helpers (gauge bars, sparklines, etc.) live in the shared
``spec_live_rendering`` module.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from collections import deque

import httpx
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tool_eval_bench.cli.spec_live_rendering import (
    _ACTIVITY_FRAMES,
    _HISTORY_LEN,
    _POLL_INTERVAL,
    _ar_color,
    _efficiency_insight,
    _format_uptime,
    _gauge_bar,
    _per_position_decay_summary,
    _position_bars_horizontal,
    _sparkline,
    _spec_method_label,
)
from tool_eval_bench.runner.spec_live import (
    MetricsSnapshot,
    ServerSpecInfo,
    SpecLiveDelta,
    compute_delta,
    metrics_url_from_base,
    probe_server_spec_info,
    scrape_snapshot,
)

logger = logging.getLogger(__name__)


def _build_dashboard(
    delta: SpecLiveDelta | None,
    history: deque[SpecLiveDelta],
    start_time: float,
    model_name: str,
    metrics_endpoint: str,
    poll_count: int,
    baseline_snap: MetricsSnapshot | None = None,
    term_width: int = 120,
    server_spec_info: ServerSpecInfo | None = None,
    reset_flash: bool = False,
) -> Panel:
    """Build the full dashboard layout.

    Parameters
    ----------
    term_width : int
        Terminal width in columns. Used to scale all widget widths
        so the dashboard fills but never overflows the terminal.
    """
    now = time.time()
    uptime = now - start_time

    # ── Responsive width calculations ──
    # Panel border + padding eats 4 chars (│ + padding each side)
    inner_w = max(60, term_width - 4)
    # Gauge bar scales: ~40 chars at 120 cols, wider/narrower proportionally
    gauge_w = max(20, min(60, int(inner_w * 0.35)))
    # Sparkline width: use available right-column space minus labels
    spark_w = max(16, min(50, int(inner_w * 0.25)))
    # Divider
    divider_w = max(40, inner_w - 4)
    # Bottom column split: left gets ~38% for per-position + engine
    left_col_w = max(30, int(inner_w * 0.38))

    # Activity indicator
    activity = _ACTIVITY_FRAMES[poll_count % len(_ACTIVITY_FRAMES)]
    activity_color = "bright_green" if delta is not None else "yellow"

    # ── Header ──
    header = Table.grid(padding=0, expand=True)
    header.add_column("left", no_wrap=True, ratio=1)
    header.add_column("right", no_wrap=True, justify="right")

    left_text = Text()
    left_text.append(f" {activity} ", style=f"bold {activity_color}")
    left_text.append("SPECULATIVE DECODING MONITOR", style="bold bright_magenta")

    # Show spec method badge when spec decoding is detected
    if delta is not None:
        method_label, method_style = _spec_method_label(delta.spec_method)
        if method_label:
            left_text.append("  ╌╌ ", style="dim")
            left_text.append(f"⟨ {method_label} ⟩", style=method_style)

    left_text.append("\n ", style="")
    left_text.append(" ▸ ", style="dim cyan")
    left_text.append(model_name, style="bold cyan")

    # Show draft model name — prefer ServerSpecInfo (probed from /v1/models at
    # startup) over Prometheus label heuristic (rarely contains draft model)
    draft_name: str | None = None
    if server_spec_info and server_spec_info.draft_model_name:
        draft_name = server_spec_info.draft_model_name
    elif delta is not None and delta.model_names:
        # Fallback: look for a model_name label different from primary
        other_models = {m for m in delta.model_names if m != model_name}
        if other_models:
            draft_name = sorted(other_models)[0]

    if draft_name:
        left_text.append("  ← ", style="dim")
        left_text.append(draft_name, style="dim italic cyan")

    right_text = Text()
    right_text.append(f"⏱  {_format_uptime(uptime)}", style="dim")
    right_text.append("  │  ", style="dim")
    right_text.append(f"📡 {poll_count}", style="dim")
    if delta is not None and delta.num_spec_tokens is not None:
        right_text.append("  │  ", style="dim")
        right_text.append(f"k={delta.num_spec_tokens}", style="bold cyan")

    header.add_row(left_text, right_text)

    if delta is None:
        # No data yet — show waiting state
        waiting = Text.assemble(
            ("\n\n  ", ""),
            ("⏳ ", "bold yellow"),
            ("Connecting to ", ""),
            (metrics_endpoint, "bold cyan"),
            (" …\n", ""),
            ("  Waiting for speculative decoding metrics.\n", "dim"),
            ("  Make sure the server has spec decode enabled and is serving requests.\n\n", "dim"),
        )
        return Panel(
            Group(header, waiting),
            border_style="bright_magenta",
            title="[bold bright_magenta]─── ◆ spec-live ◆ ───[/]",
            subtitle="[dim italic]Ctrl+R reset  ·  Ctrl+C exit  ·  Refreshing every 1s[/]",
        )

    # ── Use CUMULATIVE rates for gauges (always meaningful) ──
    # vLLM updates Prometheus counters every ~10s, so per-interval
    # rates are zero most of the time.  Cumulative α is always valid.
    ar = delta.cumulative_acceptance_rate if delta.cumulative_acceptance_rate is not None else 0.0

    gauge_line = Text()
    gauge_line.append("\n ")
    gauge_line.append(" ◈ ACCEPTANCE RATE  ", style="bold bright_magenta")
    gauge_line.append_text(_gauge_bar(ar, width=gauge_w))

    # Annotate with τ/window utilization and inferred num_speculative_tokens
    tau = delta.cumulative_acceptance_length
    win = delta.cumulative_draft_window
    nst = delta.num_spec_tokens
    if tau and win and win > 0:
        gauge_line.append(f"  τ={tau:.1f}/{win:.0f}", style="dim")
        if nst is not None:
            gauge_line.append(f"  [k={nst}]", style="dim cyan")

    # ── Insight line ──
    insight = _efficiency_insight(delta)

    # ── Key Metrics Grid ──
    metrics = Table.grid(padding=(0, 1), expand=True)
    metrics.add_column("l1", no_wrap=True, width=15)
    metrics.add_column("v1", no_wrap=True, width=10)
    metrics.add_column("sep1", no_wrap=True, width=1)
    metrics.add_column("l2", no_wrap=True, width=15)
    metrics.add_column("v2", no_wrap=True, width=10)
    metrics.add_column("sep2", no_wrap=True, width=1)
    metrics.add_column("l3", no_wrap=True, width=15)
    metrics.add_column("v3", no_wrap=True, width=10)

    tau_str = f"{tau:.2f}" if tau is not None else "—"
    win_str = f"{win:.1f}" if win is not None else "—"
    nst_str = str(nst) if nst is not None else "—"
    # Cumulative waste
    waste = (1.0 - ar) if ar > 0 else None
    waste_str = f"{waste * 100:.1f}%" if waste is not None else "—"
    waste_color = (
        "bright_green"
        if waste and waste < 0.3
        else "yellow"
        if waste and waste < 0.6
        else "bright_red"
    )

    metrics.add_row(
        Text("  τ Acc Length", style="dim"),
        Text(tau_str, style="bold cyan"),
        Text("│", style="dim"),
        Text("  Draft Window", style="dim"),
        Text(win_str, style="bold"),
        Text("│", style="dim"),
        Text("  Spec Tokens", style="dim"),
        Text(nst_str, style="bold cyan"),
    )
    metrics.add_row(
        Text("  Accepted t/s", style="dim"),
        Text(f"{delta.accepted_tps:.1f}", style="bold green"),
        Text("│", style="dim"),
        Text("  Drafted t/s", style="dim"),
        Text(f"{delta.drafted_tps:.1f}", style="bold"),
        Text("│", style="dim"),
        Text("  Waste Ratio", style="dim"),
        Text(waste_str, style=f"bold {waste_color}" if waste is not None else "dim"),
    )
    metrics.add_row(
        Text("  Gen t/s", style="dim"),
        Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
        Text("│", style="dim"),
        Text("", style="dim"),
        Text("", style="dim"),
        Text("", style="dim"),
        Text("", style="dim"),
        Text("", style="dim"),
    )

    # ── Divider ──
    divider = Text("  " + "╶" + "─" * divider_w + "╴", style="dim bright_magenta")

    # ── Bottom Layout: two-column table ──
    bottom = Table.grid(padding=(0, 1), expand=True)
    bottom.add_column("left", width=left_col_w, no_wrap=False)
    bottom.add_column("right", ratio=1, no_wrap=False)

    # ── Left Column: Per-Position (if available) + Engine ──

    # Engine status block — bar + pct must fit in ~20 chars at narrow widths
    cache_pct = delta.gpu_cache_pct
    cache_color = "bright_green" if cache_pct < 50 else "yellow" if cache_pct < 80 else "bright_red"
    cache_bar_w = max(6, min(10, left_col_w - 20))
    cache_filled = int(cache_pct / 100 * cache_bar_w)
    cache_bar = Text()
    cache_bar.append("▰" * cache_filled, style=f"bold {cache_color}")
    cache_bar.append("▱" * (cache_bar_w - cache_filled), style="dim")
    cache_bar.append(f" {cache_pct:.0f}%", style=f"bold {cache_color}")

    engine_table = Table.grid(padding=(0, 1))
    engine_table.add_column("label", no_wrap=True, width=13)
    engine_table.add_column("value", no_wrap=True)

    engine_table.add_row(Text("KV Cache", style="dim"), cache_bar)

    prefix_pct = delta.prefix_cache_hit_pct
    prefix_color = "bright_green" if prefix_pct > 50 else "cyan" if prefix_pct > 0 else "dim"
    engine_table.add_row(
        Text("Prefix Cache", style="dim"),
        Text(f"{prefix_pct:.1f}%", style=f"bold {prefix_color}"),
    )

    run_style = "bold yellow" if delta.running_reqs > 0 else "dim"
    wait_style = "bold red" if delta.waiting_reqs > 0 else "dim"
    engine_table.add_row(
        Text("Running", style="dim"),
        Text(f"{delta.running_reqs}", style=run_style),
    )
    engine_table.add_row(
        Text("Waiting", style="dim"),
        Text(f"{delta.waiting_reqs}", style=wait_style),
    )
    engine_table.add_row(
        Text("Prompt t/s", style="dim"),
        Text(f"{delta.prompt_tps:,.0f}", style="bold cyan"),
    )

    # Session totals — show tokens since monitor launch, not server lifetime
    session_table = Table.grid(padding=(0, 2))
    session_table.add_column("label", no_wrap=True, width=15)
    session_table.add_column("value", no_wrap=True)

    if baseline_snap is not None:
        session_accepted = delta.total_accepted - int(baseline_snap.accepted_tokens)
        session_drafted = delta.total_drafted - int(baseline_snap.draft_tokens)
    else:
        session_accepted = delta.total_accepted
        session_drafted = delta.total_drafted

    session_table.add_row(
        Text("Accepted", style="dim"),
        Text(f"{session_accepted:,}", style="bold"),
    )
    session_table.add_row(
        Text("Drafted", style="dim"),
        Text(f"{session_drafted:,}", style="bold"),
    )
    session_ar = session_accepted / session_drafted if session_drafted > 0 else 0.0
    session_table.add_row(
        Text("Session α", style="dim"),
        Text(f"{session_ar * 100:.1f}%", style=f"bold {_ar_color(session_ar)}"),
    )

    engine_panel = Panel(
        Group(engine_table, Text(""), session_table),
        title="[bold]⚙ Engine & Session[/]",
        border_style="dim cyan",
        padding=(0, 1),
    )

    left_col = engine_panel

    # ── Right Column: Sparklines + Throughput History ──
    # Use cumulative α for sparklines (always available)
    ar_hist = [
        d.cumulative_acceptance_rate for d in history if d.cumulative_acceptance_rate is not None
    ]
    # For throughput, use gen_tps gauge (always updated) and filter accepted to active intervals
    gen_hist = [d.generation_tps for d in history]
    acc_hist = [d.accepted_tps for d in history if d.had_activity]
    waste_hist = [
        1.0 - d.cumulative_acceptance_rate
        for d in history
        if d.cumulative_acceptance_rate is not None
    ]

    spark_table = Table.grid(padding=(0, 1))
    spark_table.add_column("label", width=13, no_wrap=True)
    spark_table.add_column("spark", no_wrap=True)
    spark_table.add_column("val", width=7, justify="right", no_wrap=True)
    spark_table.add_column("range", width=16, justify="right", no_wrap=True)

    # Accept Rate sparkline
    ar_current = f"{ar * 100:.1f}%" if ar else "—"
    ar_range = ""
    if len(ar_hist) > 1:
        ar_range = f"↕{min(ar_hist) * 100:.0f}–{max(ar_hist) * 100:.0f}%"
    spark_table.add_row(
        Text("Accept Rate", style="bold"),
        _sparkline(ar_hist, width=spark_w),
        Text(ar_current, style=f"bold {_ar_color(ar)}"),
        Text(ar_range, style="dim"),
    )

    # Gen t/s sparkline
    gen_range = ""
    if len(gen_hist) > 1:
        gen_range = f"↕{min(gen_hist):.0f}–{max(gen_hist):.0f}"
    spark_table.add_row(
        Text("Gen t/s", style="bold"),
        _sparkline(gen_hist, width=spark_w),
        Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
        Text(gen_range, style="dim"),
    )

    # Accepted t/s sparkline
    acc_range = ""
    if len(acc_hist) > 1:
        acc_range = f"↕{min(acc_hist):.0f}–{max(acc_hist):.0f}"
    spark_table.add_row(
        Text("Accepted t/s", style="bold"),
        _sparkline(acc_hist, width=spark_w),
        Text(f"{delta.accepted_tps:.1f}", style="bold green"),
        Text(acc_range, style="dim"),
    )

    # Waste ratio sparkline
    waste_current = f"{waste * 100:.0f}%" if waste is not None else "—"
    waste_range = ""
    if len(waste_hist) > 1:
        waste_range = f"↕{min(waste_hist) * 100:.0f}–{max(waste_hist) * 100:.0f}%"
    waste_style = f"bold {_ar_color(1.0 - waste)}" if waste is not None else "dim"
    spark_table.add_row(
        Text("Waste", style="bold"),
        _sparkline(
            [1.0 - w for w in waste_hist] if waste_hist else [],
            width=spark_w,
        ),
        Text(waste_current, style=waste_style),
        Text(waste_range, style="dim"),
    )

    sparkline_panel = Panel(
        spark_table,
        title=f"[bold]📊 History ({len(history)}/{_HISTORY_LEN}s)[/]",
        border_style="bright_cyan",
        padding=(0, 1),
    )

    # Averages panel — always visible; shows 0.0 until enough data accumulates
    from statistics import mean

    avg_table = Table.grid(padding=(0, 2))
    avg_table.add_column("label", no_wrap=True, width=14)
    avg_table.add_column("value", no_wrap=True)

    avg_ar = mean(ar_hist) if ar_hist else 0.0
    avg_gen = mean(gen_hist) if gen_hist else 0.0
    avg_acc = mean(acc_hist) if acc_hist else 0.0

    avg_table.add_row(
        Text("Avg α", style="dim"),
        Text(f"{avg_ar * 100:.1f}%", style=f"bold {_ar_color(avg_ar)}"),
    )
    avg_table.add_row(
        Text("Avg Gen t/s", style="dim"),
        Text(f"{avg_gen:.1f}", style="bold bright_green"),
    )
    avg_table.add_row(
        Text("Avg Acc t/s", style="dim"),
        Text(f"{avg_acc:.1f}", style="bold green"),
    )

    avg_panel = Panel(
        avg_table,
        title="[bold]⌀ Rolling Averages[/]",
        border_style="dim cyan",
        padding=(0, 1),
    )

    right_col = Group(sparkline_panel, avg_panel)

    bottom.add_row(left_col, right_col)

    # ── Per-Position Acceptance (full-width, below the two-column layout) ──
    per_pos_panel: Panel | None = None
    if delta.per_position_rates:
        pos_bars_hz = _position_bars_horizontal(
            delta.per_position_rates,
            inner_w=inner_w,
        )
        # Add compact decay summary on same line
        decay_summary = _per_position_decay_summary(delta.per_position_rates)
        pos_content: RenderableType
        if decay_summary:
            pos_content = Group(pos_bars_hz, decay_summary)
        else:
            pos_content = pos_bars_hz
        per_pos_panel = Panel(
            pos_content,
            title="[bold]▊ Per-Position Acceptance[/]",
            border_style="bright_cyan",
            padding=(0, 1),
            expand=True,
        )

    # ── Final Assembly ──
    parts: list[Text | Table | Panel] = [
        header,
        gauge_line,
        insight,
        Text(""),
        metrics,
        divider,
        Text(""),
        bottom,
    ]
    if per_pos_panel is not None:
        parts.append(Text(""))
        parts.append(per_pos_panel)

    # Reset flash banner
    if reset_flash:
        flash_text = Text()
        flash_text.append("\n  ⟳ Session reset  ", style="bold bright_yellow")
        flash_text.append("— counters & history cleared", style="dim yellow")
        parts.append(flash_text)

    return Panel(
        Group(*parts),
        border_style="bright_magenta",
        title="[bold bright_magenta]─── ◆ spec-live ◆ ───[/]",
        subtitle="[dim italic]Ctrl+R reset  ·  Ctrl+C exit  ·  Refreshing every 1s[/]",
        padding=(0, 1),
    )


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
) -> None:
    """Run the live speculative decoding monitor (blocking).

    Polls /metrics at ``poll_interval`` and renders a Rich Live dashboard.
    Press Ctrl+C to exit gracefully, Ctrl+R to reset session counters.

    Uses the terminal alternate screen buffer so the dashboard occupies
    the entire terminal without disturbing previous output.
    """
    import sys

    url = metrics_url or metrics_url_from_base(base_url)

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

    # Sticky gauges — vLLM resets gauge metrics to 0 between its ~10s
    # internal update intervals.  We keep the last non-zero value so the
    # dashboard doesn't flicker between real values and zero.
    _sticky_gen_tps: float = 0.0
    _sticky_prompt_tps: float = 0.0
    _sticky_gpu_cache_pct: float = 0.0
    _sticky_prefix_cache_pct: float = 0.0

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
                    url,
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
                    poll_task = asyncio.create_task(scrape_snapshot(client, url, api_key))
                    try:
                        snap = await poll_task
                    except asyncio.CancelledError:
                        if stop_event.is_set():
                            break
                        raise
                    finally:
                        poll_task = None
                    poll_count += 1

                    if snap is not None and (snap.has_spec_decode or snap.has_llamacpp_metrics):
                        # Capture baseline on first successful scrape (session-relative counters)
                        if baseline_snap is None:
                            baseline_snap = snap

                        if prev_snap is not None:
                            delta = compute_delta(prev_snap, snap)

                            # ── Make everything session-relative ──
                            # All cumulative metrics should reflect only what
                            # happened while the dashboard is open.
                            if baseline_snap is not None:
                                sess_accepted = snap.accepted_tokens - baseline_snap.accepted_tokens
                                sess_drafted = snap.draft_tokens - baseline_snap.draft_tokens
                                sess_drafts = snap.num_drafts - baseline_snap.num_drafts

                                # Session acceptance rate
                                if sess_drafted > 0:
                                    delta.cumulative_acceptance_rate = sess_accepted / sess_drafted
                                else:
                                    delta.cumulative_acceptance_rate = None

                                # Session acceptance length (τ)
                                if sess_drafts > 0:
                                    delta.cumulative_acceptance_length = sess_accepted / sess_drafts
                                else:
                                    delta.cumulative_acceptance_length = None

                                # Session per-position rates from counters
                                if (
                                    snap.per_position_counters
                                    and baseline_snap.per_position_counters
                                ):
                                    if sess_drafts > 0:
                                        sess_rates: dict[int, float] = {}
                                        for pos, count in snap.per_position_counters.items():
                                            base_count = baseline_snap.per_position_counters.get(
                                                pos, 0.0
                                            )
                                            sess_rates[pos] = (count - base_count) / sess_drafts
                                        delta.per_position_rates = sess_rates
                                    else:
                                        # No new drafts yet — don't show stale all-time rates
                                        delta.per_position_rates = {}
                                elif sess_drafts == 0:
                                    delta.per_position_rates = {}

                            # Update sticky gauges — keep last non-zero value
                            if delta.generation_tps > 0:
                                _sticky_gen_tps = delta.generation_tps
                            if delta.prompt_tps > 0:
                                _sticky_prompt_tps = delta.prompt_tps
                            if delta.gpu_cache_pct > 0:
                                _sticky_gpu_cache_pct = delta.gpu_cache_pct
                            if delta.prefix_cache_hit_pct > 0:
                                _sticky_prefix_cache_pct = delta.prefix_cache_hit_pct

                            # Apply sticky values when current reading is zero
                            if delta.generation_tps == 0:
                                delta.generation_tps = _sticky_gen_tps
                            if delta.prompt_tps == 0:
                                delta.prompt_tps = _sticky_prompt_tps
                            if delta.gpu_cache_pct == 0:
                                delta.gpu_cache_pct = _sticky_gpu_cache_pct
                            if delta.prefix_cache_hit_pct == 0:
                                delta.prefix_cache_hit_pct = _sticky_prefix_cache_pct

                            history.append(delta)
                            last_delta = delta

                            # Override spec method from --spec-method CLI flag
                            if spec_method is not None:
                                delta.spec_method = spec_method
                            # Override spec method from ServerSpecInfo (API probe)
                            elif server_spec_info and server_spec_info.spec_method:
                                delta.spec_method = server_spec_info.spec_method
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
                            url,
                            poll_count,
                            baseline_snap,
                            term_width=console.width,
                            server_spec_info=server_spec_info,
                            reset_flash=reset_flash_remaining > 0,
                        )
                    )

                    # ── Wait for poll interval OR Ctrl+R keypress ──
                    reset_event = asyncio.Event()

                    async def _check_stdin() -> None:
                        """Check stdin for Ctrl+R (\x12) keypresses."""
                        nonlocal tty_restore

                        try:
                            import termios  # noqa: F811
                            import tty  # noqa: F811
                        except ImportError:
                            return
                        import sys as _sys  # avoid shadowing outer

                        fd = _sys.stdin.fileno()
                        try:
                            old = termios.tcgetattr(fd)
                        except termios.error:
                            return

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

                    # Run stdin check with poll timeout
                    try:
                        await _check_stdin()
                    except Exception:
                        # Fallback: plain wait (no stdin support)
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
                        _sticky_gen_tps = 0.0
                        _sticky_prompt_tps = 0.0
                        _sticky_gpu_cache_pct = 0.0
                        _sticky_prefix_cache_pct = 0.0
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

    # Print session summary
    if history:
        ar_vals = [
            d.cumulative_acceptance_rate
            for d in history
            if d.cumulative_acceptance_rate is not None
        ]
        gen_vals = [d.generation_tps for d in history]
        if ar_vals:
            from statistics import mean, stdev

            avg_ar = mean(ar_vals)
            std_ar = stdev(ar_vals) if len(ar_vals) > 1 else 0.0
            avg_gen = mean(gen_vals) if gen_vals else 0.0
            max_gen = max(gen_vals) if gen_vals else 0.0

            console.print()

            # Session-relative totals for exit summary
            if last_delta and baseline_snap:
                sess_accepted = last_delta.total_accepted - int(baseline_snap.accepted_tokens)
                sess_drafted = last_delta.total_drafted - int(baseline_snap.draft_tokens)
            elif last_delta:
                sess_accepted = last_delta.total_accepted
                sess_drafted = last_delta.total_drafted
            else:
                sess_accepted = 0
                sess_drafted = 0

            console.print(
                Panel(
                    Text.assemble(
                        ("  Duration:        ", "dim"),
                        (_format_uptime(time.time() - start_time), "bold"),
                        ("  │  ", "dim"),
                        (f"{poll_count} polls", "dim"),
                        ("\n", ""),
                        ("  Avg α:           ", "dim"),
                        (f"{avg_ar * 100:.1f}%", f"bold {_ar_color(avg_ar)}"),
                        (" ± ", "dim"),
                        (f"{std_ar * 100:.1f}%", "dim"),
                        ("\n", ""),
                        ("  Avg Gen t/s:     ", "dim"),
                        (f"{avg_gen:.1f}", "bold bright_green"),
                        ("  ", ""),
                        ("peak ", "dim"),
                        (f"{max_gen:.1f}", "bold"),
                        ("\n", ""),
                        ("  Session tokens:  ", "dim"),
                        (f"{sess_accepted:,}", "bold"),
                        (" accepted  / ", "dim"),
                        (f"{sess_drafted:,}", "bold"),
                        (" drafted", "dim"),
                    ),
                    title="[bold]Session Summary[/]",
                    border_style="bright_magenta",
                    padding=(0, 1),
                )
            )
    console.print()
