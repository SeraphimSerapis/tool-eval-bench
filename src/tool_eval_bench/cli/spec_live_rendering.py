"""Shared rendering helpers for the speculative decoding dashboard.

These are pure functions that produce ``rich.text.Text`` or ``rich.table.Table``
objects.  They have **no** display-layer dependencies (no ``Live``, no Textual,
no ``termios``).  Both the Rich Live fallback path
(``cli/spec_live_display.py``) and the Textual TUI widgets import from here.
"""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from tool_eval_bench.runner.spec_live import SpecLiveDelta

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
_ACTIVITY_FRAMES = ["◆", "◇", "◈", "◇"]


# ---------------------------------------------------------------------------
# Color / format helpers
# ---------------------------------------------------------------------------


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


def _utilization_bar(value: float, width: int = 12, label: str = "") -> Text:
    """Render a compact utilization bar (0.0–1.0) with a label."""
    clamped = max(0.0, min(1.0, value))
    filled = int(clamped * width)
    color = "bright_green" if clamped < 0.5 else "yellow" if clamped < 0.8 else "bright_red"
    bar = Text()
    if label:
        bar.append(f"{label} ", style="dim")
    bar.append("▰" * filled, style=f"bold {color}")
    bar.append("▱" * (width - filled), style="dim")
    bar.append(f" {clamped * 100:.0f}%", style=f"bold {color}")
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


def _position_bars(rates: dict[int, float], max_positions: int = 8, bar_width: int = 20) -> Table:
    """Render per-position acceptance rates as a waterfall bar chart."""
    table = Table.grid(padding=(0, 1))
    table.add_column("pos", justify="right", width=3, no_wrap=True)
    table.add_column("bar", width=bar_width + 2, no_wrap=True)
    table.add_column("pct", justify="right", width=6, no_wrap=True)
    table.add_column("note", width=6, no_wrap=True)

    positions = sorted(rates.keys())[:max_positions]
    if not positions:
        return table

    # Find the tail-off point: first position where acceptance drops below 10%
    tail_off_pos: int | None = None
    for pos in positions:
        if rates[pos] < 0.10:
            tail_off_pos = pos
            break

    for pos in positions:
        rate = rates[pos]
        color = _ar_color(rate)
        filled = int(rate * bar_width)
        filled = max(0, min(bar_width, filled))

        bar = Text()
        bar.append("█" * filled, style=f"{color}")
        bar.append("░" * (bar_width - filled), style="dim")

        # Annotate significant positions
        note = Text()
        if pos == 0:
            note.append("best", style="dim")
        elif tail_off_pos is not None and pos == tail_off_pos:
            note.append("↓tail", style="dim yellow")

        table.add_row(
            Text(f"p{pos}", style="bold"),
            bar,
            Text(f"{rate * 100:5.1f}%", style=f"bold {color}"),
            note,
        )

    return table


def _position_bars_horizontal(
    rates: dict[int, float],
    max_positions: int = 64,
    inner_w: int = 80,
) -> Table:
    """Render per-position acceptance rates as horizontal inline bars.

    Automatically wraps to multiple rows when there are too many positions
    to fit in a single line (e.g., k=12 at 80 columns gets 2 rows of 6).
    Supports up to 64 positions by default (enough for any current setup).
    """
    positions = sorted(rates.keys())[:max_positions]
    if not positions:
        return Table.grid()

    # Determine how many positions fit per row (min cell width: 14 chars)
    min_cell_w = 14  # "p0 ████ 83%" needs ~14 chars minimum
    per_row = max(1, (inner_w - 4) // min_cell_w)
    per_row = min(per_row, len(positions))  # don't exceed actual count

    # Compute bar width from actual cells-per-row
    cell_w = max(min_cell_w, (inner_w - 4) // per_row - 2)
    bar_w = max(4, cell_w - 10)  # reserve space for "p0 " + " 83%"

    table = Table.grid(padding=(0, 1), expand=True)
    for _ in range(per_row):
        table.add_column(no_wrap=True)

    # Build cells and add rows
    for row_start in range(0, len(positions), per_row):
        row_positions = positions[row_start : row_start + per_row]
        cells = []
        for pos in row_positions:
            rate = rates[pos]
            color = _ar_color(rate)
            filled = int(rate * bar_w)
            filled = max(0, min(bar_w, filled))

            cell = Text()
            cell.append(f"p{pos} ", style="bold dim")
            cell.append("\u2588" * filled, style=f"{color}")
            cell.append("\u2591" * (bar_w - filled), style="dim")
            cell.append(f" {rate * 100:.0f}%", style=f"bold {color}")
            cells.append(cell)

        # Pad with empty cells if the last row is short
        while len(cells) < per_row:
            cells.append(Text(""))

        table.add_row(*cells)
    return table


def _format_uptime(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _spec_method_label(method: str) -> tuple[str, str]:
    """Return a human-readable label and Rich style for a spec decode method."""
    labels: dict[str, tuple[str, str]] = {
        "dflash": ("Draft Flash", "bold bright_cyan"),
        "dspark": ("DSpark", "bold bright_cyan"),
        "mtp": ("Multi-Token Prediction", "bold bright_yellow"),
        "eagle": ("EAGLE", "bold bright_green"),
        "eagle3": ("EAGLE-3", "bold bright_green"),
        "extract_hidden_states": ("Extract Hidden States", "bold bright_green"),
        "mlp_speculator": ("MLP Speculator", "bold bright_cyan"),
        "ngram": ("N-Gram", "bold bright_magenta"),
        "ngram_gpu": ("N-Gram (GPU)", "bold bright_magenta"),
        "medusa": ("Medusa", "bold bright_yellow"),
        "suffix": ("Suffix Decoding", "bold bright_magenta"),
        "custom_class": ("Custom Speculator", "bold bright_cyan"),
        "draft_model": ("Draft Model", "bold bright_cyan"),
        "draft_model_parallel": ("Draft Model (parallel)", "bold bright_cyan"),
        "draft_model_dynamic": ("Draft Model (dynamic)", "bold bright_cyan"),
        "eagle_parallel": ("EAGLE (parallel)", "bold bright_green"),
        "eagle_dynamic": ("EAGLE (dynamic)", "bold bright_green"),
        "unknown": ("Speculative Decoding", "bold dim"),
    }
    label, style = labels.get(method, (method.upper(), "dim"))
    return label, style


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
    method = delta.spec_method

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
            nst = delta.num_spec_tokens
            current_label = f"(current: {nst})" if nst else f"(current window ≈{win:.0f})"
            text.append(
                f"\n  💡 Consider reducing num_speculative_tokens to ~{optimal} {current_label}",
                style="dim yellow",
            )
        # Method-specific guidance
        if method == "mtp" and utilization > 0.5:
            text.append(
                f"\n  ℹ MTP benefits from native multi-token heads — "
                f"acceptance at {ar * 100:.0f}% is typical for MTP",
                style="dim cyan",
            )
        elif (
            method == "dflash" and utilization < 0.5 and (nst := delta.num_spec_tokens) and nst > 3
        ):
            text.append(
                f"\n  ℹ dflash with {nst} draft tokens — "
                f"try reducing for better latency/throughput balance",
                style="dim cyan",
            )

    return text


def _per_position_decay_summary(rates: dict[int, float]) -> Text | None:
    """Generate a summary of per-position acceptance decay.

    Shows: effective positions (>20% acceptance), geometric decay rate,
    and the position where most value is exhausted.
    """
    if not rates:
        return None

    positions = sorted(rates.keys())
    if len(positions) < 2:
        return None

    text = Text()

    # Effective positions: count positions with >20% acceptance
    effective = sum(1 for p in positions if rates[p] > 0.20)
    total = len(positions)

    # Find the 50% drop point (position where rate drops below half of p0)
    p0_rate = rates.get(0, rates[positions[0]])
    half_point: int | None = None
    if p0_rate > 0:
        for pos in positions[1:]:
            if rates[pos] < p0_rate * 0.5:
                half_point = pos
                break

    text.append("  ", style="")
    eff_color = (
        "bright_green"
        if effective >= total * 0.6
        else "yellow"
        if effective >= total * 0.3
        else "bright_red"
    )
    text.append(f"{effective}/{total}", style=f"bold {eff_color}")
    text.append(" effective positions (>20%)", style="dim")

    if half_point is not None:
        text.append(f"  │  50% drop at p{half_point}", style="dim")

    # Geometric decay rate between p0 and last position
    last_pos = positions[-1]
    last_rate = rates[last_pos]
    if p0_rate > 0 and last_rate > 0 and last_pos > 0:
        import math

        decay_per_pos = math.exp(math.log(last_rate / p0_rate) / last_pos)
        text.append(f"  │  γ={decay_per_pos:.2f}/pos", style="dim")

    return text
