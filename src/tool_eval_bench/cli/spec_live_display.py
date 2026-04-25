"""Rich Live terminal dashboard for speculative decoding stats.

Renders a continuously-updating dashboard with:
- Acceptance rate gauge with color gradient
- Per-position acceptance waterfall bar chart
- Throughput sparklines (rolling 60s history)
- Draft efficiency analysis & utilization gauge
- Engine status (KV cache, requests, prefix cache)
- Cumulative session stats with session α
"""

from __future__ import annotations

import asyncio
import signal
import time
from collections import deque

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tool_eval_bench.runner.spec_live import (
    MetricsSnapshot,
    SpecLiveDelta,
    compute_delta,
    metrics_url_from_base,
    scrape_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HISTORY_LEN = 60  # keep 60 samples ≈ 60 seconds at 1 Hz
_POLL_INTERVAL = 1.0  # seconds between scrapes

# Sparkline block characters (⅛ blocks, bottom-up)
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

# Color thresholds for acceptance rate
_AR_COLORS = [
    (0.0, "bright_red"),
    (0.2, "red"),
    (0.35, "dark_orange"),
    (0.5, "yellow"),
    (0.65, "green_yellow"),
    (0.8, "bright_green"),
]

# Activity indicator cycle
_ACTIVITY_FRAMES = ["◉", "◎"]


def _ar_color(rate: float) -> str:
    """Return a Rich color for an acceptance rate value."""
    color = "bright_green"
    for threshold, c in _AR_COLORS:
        if rate >= threshold:
            color = c
    return color


def _gauge_bar(value: float, width: int = 40, fill: str = "━", empty: str = "╌") -> Text:
    """Render a horizontal gauge bar with color gradient."""
    filled = int(value * width)
    filled = max(0, min(width, filled))
    color = _ar_color(value)
    bar = Text()
    bar.append(fill * filled, style=f"bold {color}")
    bar.append(empty * (width - filled), style="dim")
    bar.append(f" {value * 100:5.1f}%", style=f"bold {color}")
    return bar


def _mini_gauge(value: float, width: int = 12) -> Text:
    """Render a small gauge for inline use."""
    filled = int(value * width)
    filled = max(0, min(width, filled))
    color = _ar_color(value) if value <= 1.0 else "bright_red"
    bar = Text()
    bar.append("▓" * filled, style=color)
    bar.append("░" * (width - filled), style="dim")
    return bar


def _sparkline(values: list[float], width: int = 40) -> Text:
    """Render a sparkline from a list of values."""
    if not values:
        return Text("─" * width, style="dim")

    # Take last `width` values
    data = values[-width:]
    if not data:
        return Text("─" * width, style="dim")

    mn = min(data)
    mx = max(data)
    rng = mx - mn if mx > mn else 1.0

    spark = Text()
    for v in data:
        idx = int((v - mn) / rng * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        # Color based on relative position
        if idx >= 6:
            style = "bright_green"
        elif idx >= 4:
            style = "green"
        elif idx >= 2:
            style = "yellow"
        else:
            style = "bright_red"
        spark.append(_SPARK_CHARS[idx], style=style)

    # Pad if shorter than width
    if len(data) < width:
        padding = Text("─" * (width - len(data)), style="dim")
        return Text.assemble(padding, spark)

    return spark


def _position_bars(rates: dict[int, float], max_positions: int = 8) -> Table:
    """Render per-position acceptance rates as a waterfall bar chart."""
    table = Table.grid(padding=(0, 1))
    table.add_column("pos", justify="right", width=3, no_wrap=True)
    table.add_column("bar", width=22, no_wrap=True)
    table.add_column("pct", justify="right", width=6, no_wrap=True)

    positions = sorted(rates.keys())[:max_positions]
    if not positions:
        table.add_row(
            Text("", style="dim"),
            Text("not exposed by server", style="dim italic"),
            Text("", style="dim"),
        )
        table.add_row(
            Text("", style="dim"),
            Text("(MTP may not report", style="dim italic"),
            Text("", style="dim"),
        )
        table.add_row(
            Text("", style="dim"),
            Text(" per-position rates)", style="dim italic"),
            Text("", style="dim"),
        )
        return table

    for pos in positions:
        rate = rates[pos]
        color = _ar_color(rate)
        bar_width = 20
        filled = int(rate * bar_width)
        filled = max(0, min(bar_width, filled))

        bar = Text()
        bar.append("█" * filled, style=f"{color}")
        bar.append("░" * (bar_width - filled), style="dim")

        table.add_row(
            Text(f"p{pos}", style="bold"),
            bar,
            Text(f"{rate * 100:5.1f}%", style=f"bold {color}"),
        )

    return table


def _format_uptime(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _efficiency_insight(delta: SpecLiveDelta) -> Text:
    """Generate a one-line efficiency insight based on current metrics."""
    text = Text()

    ar = delta.cumulative_acceptance_rate
    if ar is None:
        text.append("  ℹ ", style="dim")
        text.append("awaiting acceptance data from server", style="dim italic")
        return text

    tau = delta.cumulative_acceptance_length
    win = delta.cumulative_draft_window

    if ar >= 0.6:
        text.append("  ✦ ", style="bright_green")
        text.append("Excellent", style="bold bright_green")
        text.append(" — draft model is well-aligned", style="dim")
    elif ar >= 0.4:
        text.append("  ✦ ", style="yellow")
        text.append("Good", style="bold yellow")
        text.append(" — moderate acceptance, decent speedup", style="dim")
    elif ar >= 0.2:
        text.append("  ⚡ ", style="dark_orange")
        text.append("Fair", style="bold dark_orange")
        text.append(" — high waste ratio", style="dim")
    else:
        text.append("  ⚠ ", style="bright_red")
        text.append("Poor", style="bold bright_red")
        text.append(" — draft tokens mostly rejected", style="dim")

    if win is not None and tau is not None and win > 0:
        utilization = tau / win
        if utilization < 0.3:
            optimal = max(int(tau * 1.5), 2)
            text.append(
                f"\n  💡 Consider reducing num_speculative_tokens to ~{optimal} "
                f"(current window ≈{win:.0f})",
                style="dim yellow",
            )

    return text


def _build_dashboard(
    delta: SpecLiveDelta | None,
    history: deque[SpecLiveDelta],
    start_time: float,
    model_name: str,
    metrics_endpoint: str,
    poll_count: int,
    baseline_snap: MetricsSnapshot | None = None,
) -> Panel:
    """Build the full dashboard layout."""
    now = time.time()
    uptime = now - start_time

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
    left_text.append("  ", style="")
    left_text.append(model_name, style="bold cyan")

    right_text = Text()
    right_text.append(f"⏱ {_format_uptime(uptime)}", style="dim")
    right_text.append("  │  ", style="dim")
    right_text.append(f"📡 {poll_count}", style="dim")

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
            title="[bold bright_magenta]◆ spec-live ◆[/]",
            subtitle="[dim]Ctrl+C to exit[/]",
        )

    # ── Use CUMULATIVE rates for gauges (always meaningful) ──
    # vLLM updates Prometheus counters every ~10s, so per-interval
    # rates are zero most of the time.  Cumulative α is always valid.
    ar = delta.cumulative_acceptance_rate if delta.cumulative_acceptance_rate is not None else 0.0

    gauge_line = Text()
    gauge_line.append("\n  ACCEPTANCE RATE  ", style="bold")
    gauge_line.append_text(_gauge_bar(ar, width=40))

    # Annotate with τ/window utilization when draft window data is available
    tau = delta.cumulative_acceptance_length
    win = delta.cumulative_draft_window
    if tau and win and win > 0:
        gauge_line.append(f"  τ={tau:.1f}/{win:.0f}", style="dim")

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
    # Cumulative waste
    waste = (1.0 - ar) if ar > 0 else None
    waste_str = f"{waste * 100:.1f}%" if waste is not None else "—"
    waste_color = "bright_green" if waste and waste < 0.3 else "yellow" if waste and waste < 0.6 else "bright_red"

    metrics.add_row(
        Text("  τ Acc Length", style="dim"), Text(tau_str, style="bold cyan"),
        Text("│", style="dim"),
        Text("  Draft Window", style="dim"), Text(win_str, style="bold"),
        Text("│", style="dim"),
        Text("  Waste Ratio", style="dim"),
        Text(waste_str, style=f"bold {waste_color}" if waste is not None else "dim"),
    )
    metrics.add_row(
        Text("  Accepted t/s", style="dim"), Text(f"{delta.accepted_tps:.1f}", style="bold green"),
        Text("│", style="dim"),
        Text("  Drafted t/s", style="dim"), Text(f"{delta.drafted_tps:.1f}", style="bold"),
        Text("│", style="dim"),
        Text("  Gen t/s", style="dim"),
        Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
    )

    # ── Divider ──
    divider = Text("  " + "─" * 76, style="dim")

    # ── Bottom Layout: two-column table ──
    bottom = Table.grid(padding=(0, 1), expand=True)
    bottom.add_column("left", width=40, no_wrap=False)
    bottom.add_column("right", ratio=1, no_wrap=False)

    # ── Left Column: Per-Position (if available) + Engine ──

    # Engine status block
    cache_pct = delta.gpu_cache_pct
    cache_color = "bright_green" if cache_pct < 50 else "yellow" if cache_pct < 80 else "bright_red"
    cache_bar_w = 10
    cache_filled = int(cache_pct / 100 * cache_bar_w)
    cache_bar = Text()
    cache_bar.append("▓" * cache_filled, style=cache_color)
    cache_bar.append("░" * (cache_bar_w - cache_filled), style="dim")
    cache_bar.append(f" {cache_pct:.1f}%", style=f"bold {cache_color}")

    engine_table = Table.grid(padding=(0, 2))
    engine_table.add_column("label", no_wrap=True, width=15)
    engine_table.add_column("value", no_wrap=True)

    engine_table.add_row(Text("KV Cache Fill", style="dim"), cache_bar)

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
    if session_drafted > 0:
        session_ar = session_accepted / session_drafted
        session_table.add_row(
            Text("Session α", style="dim"),
            Text(f"{session_ar * 100:.1f}%", style=f"bold {_ar_color(session_ar)}"),
        )

    engine_panel = Panel(
        Group(engine_table, Text(""), session_table),
        title="[bold]Engine & Session[/]",
        border_style="bright_cyan",
        padding=(0, 1),
    )

    # Only show Per-Position panel if the server exposes per-position rates
    if delta.per_position_rates:
        pos_panel = Panel(
            _position_bars(delta.per_position_rates),
            title="[bold]Per-Position Acceptance[/]",
            border_style="bright_cyan",
            padding=(0, 1),
        )
        left_col = Group(pos_panel, engine_panel)
    else:
        left_col = engine_panel

    # ── Right Column: Sparklines + Throughput History ──
    # Use cumulative α for sparklines (always available)
    ar_hist = [d.cumulative_acceptance_rate for d in history if d.cumulative_acceptance_rate is not None]
    # For throughput, use gen_tps gauge (always updated) and filter accepted to active intervals
    gen_hist = [d.generation_tps for d in history]
    acc_hist = [d.accepted_tps for d in history if d.had_activity]
    waste_hist = [1.0 - d.cumulative_acceptance_rate for d in history if d.cumulative_acceptance_rate is not None]

    spark_table = Table.grid(padding=(0, 1))
    spark_table.add_column("label", width=13, no_wrap=True)
    spark_table.add_column("spark", no_wrap=True)
    spark_table.add_column("val", width=7, justify="right", no_wrap=True)
    spark_table.add_column("range", width=16, justify="right", no_wrap=True)

    spark_width = 28

    # Accept Rate sparkline
    ar_current = f"{ar * 100:.1f}%" if ar else "—"
    ar_range = ""
    if len(ar_hist) > 1:
        ar_range = f"↕{min(ar_hist) * 100:.0f}–{max(ar_hist) * 100:.0f}%"
    spark_table.add_row(
        Text("Accept Rate", style="bold"),
        _sparkline(ar_hist, width=spark_width),
        Text(ar_current, style=f"bold {_ar_color(ar)}"),
        Text(ar_range, style="dim"),
    )

    # Gen t/s sparkline
    gen_range = ""
    if len(gen_hist) > 1:
        gen_range = f"↕{min(gen_hist):.0f}–{max(gen_hist):.0f}"
    spark_table.add_row(
        Text("Gen t/s", style="bold"),
        _sparkline(gen_hist, width=spark_width),
        Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
        Text(gen_range, style="dim"),
    )

    # Accepted t/s sparkline
    acc_range = ""
    if len(acc_hist) > 1:
        acc_range = f"↕{min(acc_hist):.0f}–{max(acc_hist):.0f}"
    spark_table.add_row(
        Text("Accepted t/s", style="bold"),
        _sparkline(acc_hist, width=spark_width),
        Text(f"{delta.accepted_tps:.1f}", style="bold green"),
        Text(acc_range, style="dim"),
    )

    # Waste ratio sparkline
    waste_current = f"{waste * 100:.0f}%" if waste is not None else "—"
    waste_range = ""
    if len(waste_hist) > 1:
        waste_range = f"↕{min(waste_hist) * 100:.0f}–{max(waste_hist) * 100:.0f}%"
    waste_style = (
        f"bold {_ar_color(1.0 - waste)}" if waste is not None else "dim"
    )
    spark_table.add_row(
        Text("Waste", style="bold"),
        _sparkline(
            [1.0 - w for w in waste_hist] if waste_hist else [],
            width=spark_width,
        ),
        Text(waste_current, style=waste_style),
        Text(waste_range, style="dim"),
    )

    sparkline_panel = Panel(
        spark_table,
        title=f"[bold]History ({len(history)}/{_HISTORY_LEN}s)[/]",
        border_style="bright_cyan",
        padding=(0, 1),
    )

    # Averages panel (if enough history)
    avg_panel: Panel | None = None
    if len(ar_hist) >= 5:
        from statistics import mean

        avg_table = Table.grid(padding=(0, 2))
        avg_table.add_column("label", no_wrap=True, width=14)
        avg_table.add_column("value", no_wrap=True)

        avg_ar = mean(ar_hist)
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
            title="[bold]Rolling Averages[/]",
            border_style="bright_cyan",
            padding=(0, 1),
        )

    right_col = Group(sparkline_panel, avg_panel) if avg_panel else sparkline_panel

    bottom.add_row(left_col, right_col)

    # ── Final Assembly ──
    return Panel(
        Group(
            header,
            gauge_line,
            insight,
            Text(""),
            metrics,
            divider,
            Text(""),
            bottom,
        ),
        border_style="bright_magenta",
        title="[bold bright_magenta]◆ spec-live ◆[/]",
        subtitle="[dim]Ctrl+C to exit[/]",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def run_spec_live(
    base_url: str,
    *,
    api_key: str | None = None,
    metrics_url: str | None = None,
    model_name: str = "unknown",
    poll_interval: float = _POLL_INTERVAL,
) -> None:
    """Run the live speculative decoding monitor (blocking).

    Polls /metrics at ``poll_interval`` and renders a Rich Live dashboard.
    Press Ctrl+C to exit gracefully.
    """
    url = metrics_url or metrics_url_from_base(base_url)
    console = Console()

    history: deque[SpecLiveDelta] = deque(maxlen=_HISTORY_LEN)
    prev_snap: MetricsSnapshot | None = None
    baseline_snap: MetricsSnapshot | None = None  # first snapshot — for session-relative counters
    start_time = time.time()
    poll_count = 0
    last_delta: SpecLiveDelta | None = None

    # Sticky gauges — vLLM resets gauge metrics to 0 between its ~10s
    # internal update intervals.  We keep the last non-zero value so the
    # dashboard doesn't flicker between real values and zero.
    _sticky_gen_tps: float = 0.0
    _sticky_prompt_tps: float = 0.0
    _sticky_gpu_cache_pct: float = 0.0
    _sticky_prefix_cache_pct: float = 0.0

    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # Windows

    console.print()
    console.print(
        f"  [bold bright_magenta]◆ spec-live[/] starting…  "
        f"[dim]polling {url} every {poll_interval}s[/]"
    )
    console.print("  [dim]Press Ctrl+C to stop.[/]")
    console.print()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
    ) as client:
        with Live(
            _build_dashboard(None, history, start_time, model_name, url, 0, baseline_snap),
            console=console,
            refresh_per_second=2,
            transient=False,
        ) as live:
            while not stop_event.is_set():
                snap = await scrape_snapshot(client, url, api_key)
                poll_count += 1

                if snap is not None and snap.has_spec_decode:
                    # Capture baseline on first successful scrape (session-relative counters)
                    if baseline_snap is None:
                        baseline_snap = snap

                    if prev_snap is not None:
                        delta = compute_delta(prev_snap, snap)

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
                    prev_snap = snap
                elif snap is not None and prev_snap is None:
                    # First scrape, no spec decode counters yet — store for next
                    prev_snap = snap

                live.update(
                    _build_dashboard(
                        last_delta, history, start_time,
                        model_name, url, poll_count,
                        baseline_snap,
                    )
                )

                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=poll_interval,
                    )
                    break  # stop_event was set
                except asyncio.TimeoutError:
                    pass  # normal: poll again

    console.print()
    console.print("  [bold bright_magenta]◆ spec-live[/] stopped.")

    # Print session summary
    if history:
        ar_vals = [d.cumulative_acceptance_rate for d in history if d.cumulative_acceptance_rate is not None]
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
