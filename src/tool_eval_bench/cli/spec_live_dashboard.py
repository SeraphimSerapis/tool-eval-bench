"""Pure Rich layout builders for the ``--spec-live`` speculative decoding dashboard.

Separated from ``spec_live_display`` so layout composition has no async loop,
signal, or ``termios`` dependencies.
"""

from __future__ import annotations

import time
from collections import deque
from statistics import mean

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tool_eval_bench.cli.spec_live_rendering import (
    _ACTIVITY_FRAMES,
    _HISTORY_LEN,
    _POLL_INTERVAL,
    _ROLLING_WINDOW_S,
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
    counter_delta,
)


def _build_header(
    delta: SpecLiveDelta | None,
    *,
    uptime: float,
    model_name: str,
    poll_count: int,
    server_spec_info: ServerSpecInfo | None,
    failed_polls: int,
    last_scrape_age_s: float | None,
) -> Table:
    """Build the top status header with activity indicator, model/draft names, and poll counters."""
    stale = failed_polls > 0
    activity = _ACTIVITY_FRAMES[poll_count % len(_ACTIVITY_FRAMES)]
    if stale:
        activity = "✗"
        activity_color = "bright_red"
    else:
        activity_color = "bright_green" if delta is not None else "yellow"

    header = Table.grid(padding=0, expand=True)
    header.add_column("left", no_wrap=True, ratio=1)
    header.add_column("right", no_wrap=True, justify="right")

    left_text = Text()
    left_text.append(f" {activity} ", style=f"bold {activity_color}")
    left_text.append("SPECULATIVE DECODING MONITOR", style="bold bright_magenta")

    if delta is not None:
        method_label, method_style = _spec_method_label(delta.spec_method)
        if method_label:
            left_text.append("  ╌╌ ", style="dim")
            left_text.append(f"⟨ {method_label} ⟩", style=method_style)

    left_text.append("\n ", style="")
    left_text.append(" ▸ ", style="dim cyan")
    left_text.append(model_name, style="bold cyan")

    # Show a draft model only when the server explicitly reports it in its
    # speculative configuration.  A second model_name label or /v1/models
    # entry is not enough to establish a drafter relationship.
    if server_spec_info and server_spec_info.draft_model_name:
        left_text.append("  ← ", style="dim")
        left_text.append(server_spec_info.draft_model_name, style="dim italic cyan")

    right_text = Text()
    if stale:
        age = f", data {_format_uptime(last_scrape_age_s)} old" if last_scrape_age_s else ""
        right_text.append(f"⚠ {failed_polls} failed polls{age}", style="bold bright_red")
        right_text.append("  │  ", style="dim")
    right_text.append(f"⏱  {_format_uptime(uptime)}", style="dim")
    right_text.append("  │  ", style="dim")
    right_text.append(f"📡 {poll_count}", style="dim")
    if delta is not None and delta.num_spec_tokens is not None:
        right_text.append("  │  ", style="dim")
        right_text.append(f"k={delta.num_spec_tokens}", style="bold cyan")

    header.add_row(left_text, right_text)
    return header


def _build_acceptance_panel(
    delta: SpecLiveDelta,
    *,
    gauge_w: int,
) -> tuple[Text, Text, Table, float | None, float | None]:
    """Build the rolling acceptance gauge, efficiency insight, and 3-column metrics grid."""
    # Session α is a running average: it converges and then hides workload
    # changes.  The gauge and the sparklines follow the rolling window so a
    # switch from code to prose is visible; the session figure lives in the
    # grid and the exit summary.
    session_ar = delta.cumulative_acceptance_rate
    rolling = delta.rolling_acceptance_rate
    # None means no acceptance data yet; 0.0 is a drafter whose every token
    # was rejected, which is exactly what this monitor exists to show.
    ar = rolling if rolling is not None else session_ar
    ar_label = (
        f" ◈ ACCEPTANCE RATE ({_ROLLING_WINDOW_S:g}s)  "
        if rolling is not None
        else " ◈ ACCEPTANCE RATE  "
    )

    gauge_line = Text()
    gauge_line.append("\n ")
    gauge_line.append(ar_label, style="bold bright_magenta")
    gauge_line.append_text(_gauge_bar(ar if ar is not None else 0.0, width=gauge_w))

    # Annotate with τ, the draft window, and inferred num_speculative_tokens.
    # τ counts the bonus token, so it is not a fraction of the window.
    tau = delta.cumulative_acceptance_length
    win = delta.cumulative_draft_window
    nst = delta.num_spec_tokens
    if tau and win and win > 0:
        gauge_line.append(f"  τ={tau:.1f} win={win:.0f}", style="dim")
        if nst is not None:
            gauge_line.append(f"  [k={nst}]", style="dim cyan")

    insight = _efficiency_insight(delta)

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
    waste = (1.0 - ar) if ar is not None else None
    waste_str = f"{waste * 100:.1f}%" if waste is not None else "—"
    waste_color = (
        "bright_green"
        if waste is not None and waste < 0.3
        else "yellow"
        if waste is not None and waste < 0.6
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
    # Gauge-only backends (SGLang) have no token counters, so these rates are
    # unknown rather than zero, like the session totals below.
    has_counters = delta.counter_metrics_available
    metrics.add_row(
        Text("  Accepted t/s", style="dim"),
        Text(f"{delta.accepted_tps:.1f}", style="bold green") if has_counters else Text("—", "dim"),
        Text("│", style="dim"),
        Text("  Drafted t/s", style="dim"),
        Text(f"{delta.drafted_tps:.1f}", style="bold") if has_counters else Text("—", "dim"),
        Text("│", style="dim"),
        Text("  Waste Ratio", style="dim"),
        Text(waste_str, style=f"bold {waste_color}" if waste is not None else "dim"),
    )
    session_ar_str = f"{session_ar * 100:.1f}%" if session_ar is not None else "—"
    rolling_tau = delta.rolling_acceptance_length
    rolling_tau_str = f"{rolling_tau:.2f}" if rolling_tau is not None else "—"
    metrics.add_row(
        Text("  Gen t/s", style="dim"),
        Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
        Text("│", style="dim"),
        Text("  Session α", style="dim"),
        Text(
            session_ar_str,
            style=f"bold {_ar_color(session_ar)}" if session_ar is not None else "dim",
        ),
        Text("│", style="dim"),
        Text(f"  τ ({_ROLLING_WINDOW_S:g}s)", style="dim"),
        Text(rolling_tau_str, style="bold cyan"),
    )
    return gauge_line, insight, metrics, ar, waste


def _build_engine_panel(
    delta: SpecLiveDelta,
    baseline_snap: MetricsSnapshot | None,
    *,
    left_col_w: int,
    fallback_ar: float | None,
) -> tuple[Panel, float | None]:
    """Build the Engine & Session panel and return ``(panel, session_ar)``."""
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

    if delta.counter_metrics_available:
        if baseline_snap is not None:
            session_accepted = counter_delta(
                baseline_snap.accepted_tokens,
                float(delta.total_accepted),
            )
            session_drafted = counter_delta(
                baseline_snap.draft_tokens,
                float(delta.total_drafted),
            )
        else:
            session_accepted = float(delta.total_accepted)
            session_drafted = float(delta.total_drafted)
        accepted_text = f"{int(session_accepted):,}"
        drafted_text = f"{int(session_drafted):,}"
        session_ar: float | None = (
            session_accepted / session_drafted if session_drafted > 0 else 0.0
        )
    else:
        # SGLang's current Prometheus contract exposes gauges, not cumulative
        # accepted/drafted totals.  Do not display unknown totals as zero.
        accepted_text = drafted_text = "—"
        session_ar = fallback_ar

    session_table.add_row(Text("Accepted", style="dim"), Text(accepted_text, style="bold"))
    session_table.add_row(Text("Drafted", style="dim"), Text(drafted_text, style="bold"))
    session_table.add_row(
        Text("Session α", style="dim"),
        Text(f"{session_ar * 100:.1f}%", style=f"bold {_ar_color(session_ar)}")
        if session_ar is not None
        else Text("—", style="dim"),
    )

    panel = Panel(
        Group(engine_table, Text(""), session_table),
        title="[bold]⚙ Engine & Session[/]",
        border_style="dim cyan",
        padding=(0, 1),
    )
    return panel, session_ar


def _build_history_column(
    delta: SpecLiveDelta,
    history: deque[SpecLiveDelta],
    *,
    ar: float | None,
    waste: float | None,
    session_ar: float | None,
    spark_w: int,
    poll_interval: float,
) -> Group:
    """Build the sparklines + session averages right-hand column."""
    has_counters = delta.counter_metrics_available
    # Sparklines follow the rolling window; direct-gauge backends have no
    # counter deltas, so they fall back to the gauge value itself.
    ar_hist = [
        rate
        for d in history
        if (
            rate := (
                d.rolling_acceptance_rate
                if d.rolling_acceptance_rate is not None
                else d.cumulative_acceptance_rate
            )
        )
        is not None
    ]
    # History holds raw readings, so an idle poll is a zero here even while the
    # grid holds a flickering gauge's last value. Accepted t/s needs counters.
    gen_hist = [d.generation_tps for d in history]
    acc_hist = [d.accepted_tps for d in history if d.had_activity and d.counter_metrics_available]
    waste_hist = [1.0 - rate for rate in ar_hist]

    spark_table = Table.grid(padding=(0, 1))
    spark_table.add_column("label", width=13, no_wrap=True)
    spark_table.add_column("spark", no_wrap=True)
    spark_table.add_column("val", width=7, justify="right", no_wrap=True)
    spark_table.add_column("range", width=16, justify="right", no_wrap=True)

    ar_current = f"{ar * 100:.1f}%" if ar is not None else "—"
    ar_range = f"↕{min(ar_hist) * 100:.0f}–{max(ar_hist) * 100:.0f}%" if len(ar_hist) > 1 else ""
    spark_table.add_row(
        Text("Accept Rate", style="bold"),
        _sparkline(ar_hist, width=spark_w),
        Text(ar_current, style=f"bold {_ar_color(ar)}" if ar is not None else "dim"),
        Text(ar_range, style="dim"),
    )

    gen_range = f"↕{min(gen_hist):.0f}–{max(gen_hist):.0f}" if len(gen_hist) > 1 else ""
    spark_table.add_row(
        Text("Gen t/s", style="bold"),
        _sparkline(gen_hist, width=spark_w),
        Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
        Text(gen_range, style="dim"),
    )

    acc_range = f"↕{min(acc_hist):.0f}–{max(acc_hist):.0f}" if len(acc_hist) > 1 else ""
    spark_table.add_row(
        Text("Accepted t/s", style="bold"),
        _sparkline(acc_hist, width=spark_w),
        Text(f"{delta.accepted_tps:.1f}", style="bold green") if has_counters else Text("—", "dim"),
        Text(acc_range, style="dim"),
    )

    waste_current = f"{waste * 100:.0f}%" if waste is not None else "—"
    waste_range = (
        f"↕{min(waste_hist) * 100:.0f}–{max(waste_hist) * 100:.0f}%" if len(waste_hist) > 1 else ""
    )
    waste_style = f"bold {_ar_color(1.0 - waste)}" if waste is not None else "dim"
    spark_table.add_row(
        Text("Waste", style="bold"),
        _sparkline([1.0 - w for w in waste_hist] if waste_hist else [], width=spark_w),
        Text(waste_current, style=waste_style),
        Text(waste_range, style="dim"),
    )

    sparkline_panel = Panel(
        spark_table,
        title=(
            f"[bold]📊 History ({len(history)}/{_HISTORY_LEN} polls"
            f" · {_format_uptime(_HISTORY_LEN * poll_interval)})[/]"
        ),
        border_style="bright_cyan",
        padding=(0, 1),
    )

    avg_table = Table.grid(padding=(0, 2))
    avg_table.add_column("label", no_wrap=True, width=14)
    avg_table.add_column("value", no_wrap=True)

    avg_ar = session_ar if session_ar is not None else 0.0
    # Both throughput averages cover the polls that did the work, the same
    # definition the exit summary uses for gen t/s.
    generating = [value for value in gen_hist if value > 0]
    avg_gen = mean(generating) if generating else 0.0
    avg_acc = mean(acc_hist) if acc_hist else 0.0

    avg_table.add_row(
        Text("Session α", style="dim"),
        Text(f"{avg_ar * 100:.1f}%", style=f"bold {_ar_color(avg_ar)}"),
    )
    avg_table.add_row(
        Text("Avg Gen t/s", style="dim"),
        Text(f"{avg_gen:.1f}", style="bold bright_green"),
    )
    avg_table.add_row(
        Text("Avg Acc t/s", style="dim"),
        Text(f"{avg_acc:.1f}", style="bold green") if has_counters else Text("—", "dim"),
    )

    avg_panel = Panel(
        avg_table,
        title="[bold]⌀ Session Averages[/]",
        border_style="dim cyan",
        padding=(0, 1),
    )
    return Group(sparkline_panel, avg_panel)


def _build_position_panel(delta: SpecLiveDelta, *, inner_w: int) -> Panel | None:
    """Build the full-width per-position acceptance panel when rates are present."""
    if not delta.per_position_rates:
        return None
    pos_bars_hz = _position_bars_horizontal(delta.per_position_rates, inner_w=inner_w)
    decay_summary = _per_position_decay_summary(delta.per_position_rates)
    pos_content: RenderableType = (
        Group(pos_bars_hz, decay_summary) if decay_summary else pos_bars_hz
    )
    return Panel(
        pos_content,
        title="[bold]▊ Per-Position Acceptance[/]",
        border_style="bright_cyan",
        padding=(0, 1),
        expand=True,
    )


def _build_footer_subtitle(poll_interval: float) -> str:
    return f"[dim italic]Ctrl+R reset  ·  Ctrl+C exit  ·  Refreshing every {poll_interval:g}s[/]"


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
    poll_interval: float = _POLL_INTERVAL,
    failed_polls: int = 0,
    last_scrape_age_s: float | None = None,
) -> Panel:
    """Build the full dashboard layout.

    Parameters
    ----------
    term_width : int
        Terminal width in columns. Used to scale all widget widths
        so the dashboard fills but never overflows the terminal.
    failed_polls : int
        Consecutive scrapes that returned nothing. Non-zero turns the header
        red and shows how old the numbers on screen are, so a dead server
        never looks like a quiet one.
    last_scrape_age_s : float | None
        Seconds since the last successful scrape, when there has been one.
    """
    uptime = time.time() - start_time

    # ── Responsive width calculations ──
    # Panel border + padding eats 4 chars (│ + padding each side)
    inner_w = max(60, term_width - 4)
    gauge_w = max(20, min(60, int(inner_w * 0.35)))
    spark_w = max(16, min(50, int(inner_w * 0.25)))
    divider_w = max(40, inner_w - 4)
    left_col_w = max(30, int(inner_w * 0.38))

    header = _build_header(
        delta,
        uptime=uptime,
        model_name=model_name,
        poll_count=poll_count,
        server_spec_info=server_spec_info,
        failed_polls=failed_polls,
        last_scrape_age_s=last_scrape_age_s,
    )

    if delta is None:
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
            subtitle=_build_footer_subtitle(poll_interval),
        )

    gauge_line, insight, metrics, ar, waste = _build_acceptance_panel(delta, gauge_w=gauge_w)
    divider = Text("  " + "╶" + "─" * divider_w + "╴", style="dim bright_magenta")

    bottom = Table.grid(padding=(0, 1), expand=True)
    bottom.add_column("left", width=left_col_w, no_wrap=False)
    bottom.add_column("right", ratio=1, no_wrap=False)

    engine_panel, session_ar = _build_engine_panel(
        delta, baseline_snap, left_col_w=left_col_w, fallback_ar=ar
    )
    right_col = _build_history_column(
        delta,
        history,
        ar=ar,
        waste=waste,
        session_ar=session_ar,
        spark_w=spark_w,
        poll_interval=poll_interval,
    )
    bottom.add_row(engine_panel, right_col)

    per_pos_panel = _build_position_panel(delta, inner_w=inner_w)

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

    if reset_flash:
        flash_text = Text()
        flash_text.append("\n  ⟳ Session reset  ", style="bold bright_yellow")
        flash_text.append("— counters & history cleared", style="dim yellow")
        parts.append(flash_text)

    return Panel(
        Group(*parts),
        border_style="bright_magenta",
        title="[bold bright_magenta]─── ◆ spec-live ◆ ───[/]",
        subtitle=_build_footer_subtitle(poll_interval),
        padding=(0, 1),
    )
