"""Shared card drawing primitives, empty states, and axis/scale helpers."""

from __future__ import annotations

import math
from typing import Literal

from tool_eval_bench.storage.reports.card._fonts import FontKey, text_width
from tool_eval_bench.storage.reports.card._svg import (
    Run,
    attrs,
    first_fit,
    fit,
    line,
    polyline,
    rect,
    runs_text,
    runs_width,
    text,
)
from tool_eval_bench.storage.reports.card._theme import (
    FAIL,
    GOOD,
    MUTED,
    PAD,
    PARTIAL,
    PASS,
    RADIUS,
    STROKE,
    TEXT,
    TEXT_2,
    WARN,
    Box,
)

MID = " \u00b7 "
HEADER_H = 32
DOT_FILL = "#0B1020"

# Heights a panel needs in its compact (summary-only) state and as a hint strip.
COMPACT_H = 128.0
HINT_H = 88.0
# Small labels (axes, legends) stay at least this size so a card shared at
# phone width is still readable.
LABEL = 13.0
FLAG_SIZE = 13.0

GhostKind = Literal["line", "bars", "heatmap", "rows"]

# Time ticks in milliseconds. Past 10 s they follow minutes and hours rather
# than decades, so labels read "1 min" and "1 h" instead of "100 s" or "1e+06 s".
_TIME_TICKS = (0.1, 1, 10, 100, 1e3, 1e4, 6e4, 6e5, 3.6e6, 3.6e7, 3.6e8)


# --------------------------------------------------------------------------- frames


def frame(box: Box, *, dashed: bool = False, radius: float = RADIUS) -> str:
    geometry = {"x": float(box.x), "y": float(box.y), "width": float(box.w), "height": float(box.h)}
    edge = {"x": box.x + 0.5, "y": box.y + 0.5, "width": box.w - 1.0, "height": box.h - 1.0}
    if dashed:
        return (
            f"<rect {attrs(**geometry, rx=float(radius), fill='url(#panel)', opacity=0.5)}/>"
            f"<rect {attrs(**edge, rx=float(radius), fill='none', stroke=STROKE, stroke_opacity=0.16, stroke_dasharray='6 6')}/>"
        )
    return (
        f"<rect {attrs(**geometry, rx=float(radius), fill='url(#panel)')}/>"
        f"<rect {attrs(**edge, rx=float(radius), fill='none', stroke='url(#panelEdge)')}/>"
    )


def pill(
    x: float,
    y: float,
    label: str,
    *,
    size: float = 13,
    font: FontKey = "mono",
    weight: int = 500,
    color: str = TEXT_2,
    fill: str = STROKE,
    fill_opacity: float = 0.05,
    stroke_opacity: float = 0.12,
    height: float = 28,
    pad: float = 12,
    dashed: bool = False,
    tracking: float = 0.0,
    max_width: float | None = None,
) -> tuple[str, float]:
    """A rounded label. Returns markup and its width."""
    if max_width is not None:
        label = fit(label, max_width - 2 * pad, size, font=font, weight=weight, tracking=tracking)
    w = text_width(label, size, font=font, weight=weight, tracking=tracking) + 2 * pad
    body = _pill_shape(x, y, w, height, fill, fill_opacity, stroke_opacity, dashed) + text(
        x + pad,
        y + height / 2 + size * 0.36,
        label,
        size=size,
        fill=color,
        font=font,
        weight=weight,
        tracking=tracking,
    )
    return body, w


def _pill_shape(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    fill_opacity: float,
    stroke_opacity: float,
    dashed: bool = False,
) -> str:
    shape = attrs(
        x=float(x),
        y=float(y),
        width=float(w),
        height=float(h),
        rx=h / 2,
        fill=fill,
        fill_opacity=fill_opacity,
        stroke=fill,
        stroke_opacity=stroke_opacity,
        stroke_dasharray="4 4" if dashed else None,
    )
    return f"<rect {shape}/>"


def warn_pill(x: float, y: float, label: str, *, max_width: float) -> tuple[str, float]:
    return pill(
        x,
        y,
        label,
        size=13,
        font="sans",
        weight=600,
        color=WARN,
        fill=WARN,
        fill_opacity=0.10,
        stroke_opacity=0.45,
        height=28,
        max_width=max_width,
    )


def flag_pill(
    x: float, y: float, flag: str, color: str, *, max_width: float | None = None
) -> tuple[str, float]:
    return pill(
        x,
        y,
        flag,
        size=FLAG_SIZE,
        color=color,
        fill=color,
        fill_opacity=0.08,
        stroke_opacity=0.35,
        height=28,
        pad=10,
        max_width=max_width,
    )


def pills_width(flags: tuple[str, ...]) -> float:
    return sum(text_width(f, FLAG_SIZE, font="mono", weight=500) + 20 for f in flags) + 8 * max(
        len(flags) - 1, 0
    )


def _flag_row(x: float, y: float, room: float, flags: tuple[str, ...], color: str) -> str:
    """Flags left to right; a flag that would not fit is shortened, never dropped."""
    out = []
    px = x
    for i, flag in enumerate(flags):
        left = len(flags) - i - 1
        # The small allowance absorbs float rounding when the row is sized exactly.
        budget = x + room - px - left * 48 + 0.01
        chip, w = flag_pill(px, y, flag, color, max_width=max(budget, 40.0))
        out.append(chip)
        px += w + 8
    return "".join(out)


def panel_header(
    box: Box,
    title: str,
    color: str,
    *,
    subtitle: str | list[str] | None = None,
    subtitle_color: str = MUTED,
    callout: str | None = None,
    callout_label: str | None = None,
    callout_color: str = TEXT,
    right_reserved: float = 0.0,
) -> str:
    """Title on the left and an optional callout on the right, sharing a baseline.

    ``subtitle`` may list wordings, longest first; the first that fits is used.
    """
    base = box.y + PAD + 24
    x = box.x + PAD
    out = [rect(x, base - 10, 8, 8, fill=color, rx=2)]
    title_x = x + 16
    room = box.w - 2 * PAD - 16
    title = fit(title, room, 13, weight=600, tracking=1.6)
    out.append(text(title_x, base, title, size=13, fill=TEXT_2, weight=600, tracking=1.6))
    title_w = text_width(title, 13, weight=600, tracking=1.6)
    right = box.right - PAD
    used = right_reserved
    if callout:
        runs = [Run(callout, 26, callout_color, "display", 700)]
        if callout_label:
            runs.append(Run(callout_label, 13, MUTED, "sans", 500, gap=8))
        used = runs_width(runs)
        out.append(runs_text(right, base, runs, anchor="end"))
    if subtitle:
        space = right - used - 16 - (title_x + title_w + 12)
        options = [subtitle] if isinstance(subtitle, str) else subtitle
        if space > 60:
            out.append(
                text(
                    title_x + title_w + 12,
                    base,
                    first_fit(options, space, 13),
                    size=13,
                    fill=subtitle_color,
                )
            )
    return "".join(out)


def content_box(box: Box, *, extra: float = 0) -> Box:
    top = box.y + PAD + HEADER_H + 16 + extra
    return Box(box.x + PAD, top, box.w - 2 * PAD, box.bottom - PAD - top)


def ghost(kind: GhostKind, body: Box) -> str:
    """A faint skeleton of the chart a panel would show, behind its empty state."""
    out = ['<g opacity="0.06">']
    if kind == "line":
        pts = []
        for i in range(9):
            t = i / 8
            pts.append((body.x + t * body.w, body.y + body.h * (0.25 + 0.55 * t * t)))
        out.append(
            f'<polyline points="{polyline(pts)}" fill="none" stroke="{TEXT}" stroke-width="3" stroke-linejoin="round"/>'
        )
        for i in range(4):
            y = body.y + body.h * i / 3
            out.append(line(body.x, y, body.right, y, stroke=TEXT, width=1))
    elif kind == "bars":
        count = 6
        slot = body.w / count
        for i in range(count):
            h = body.h * (0.18 + 0.8 * (i + 1) / count)
            out.append(
                rect(
                    body.x + slot * i + slot * 0.2, body.bottom - h, slot * 0.6, h, fill=TEXT, rx=4
                )
            )
    elif kind == "heatmap":
        cols, rows, gap = 8, 5, 4.0
        cw = (body.w - gap * (cols - 1)) / cols
        ch = (body.h - gap * (rows - 1)) / rows
        for r in range(rows):
            for c in range(cols):
                out.append(
                    rect(body.x + c * (cw + gap), body.y + r * (ch + gap), cw, ch, fill=TEXT, rx=4)
                )
    else:
        rows = max(int(body.h // 28), 1)
        for r in range(rows):
            y = body.y + r * 28 + 10
            out.append(rect(body.x, y, 22, 8, fill=TEXT, rx=4))
            out.append(
                rect(
                    body.x + body.w * 0.55,
                    y,
                    body.w * 0.4 * (1 - r / (rows + 2)),
                    8,
                    fill=TEXT,
                    rx=4,
                )
            )
    out.append("</g>")
    return "".join(out)


def missing_panel(
    box: Box,
    title: str,
    color: str,
    message: str,
    flags: tuple[str, ...],
    *,
    note: str | None = None,
    ghost_kind: GhostKind | None = None,
) -> str:
    """The intentional empty state: what is missing and the flag that fills it.

    A strip-height box is a one- or two-line hint; a short box puts the flags
    in the header row; a tall one centers the message over a faint skeleton of
    the missing chart.
    """
    if box.h < 120:
        return hint_strip(box, title, color, message, flags)
    out = [frame(box, dashed=True)]
    body = content_box(box)
    if box.h < 200:
        flags_w = min(pills_width(flags), body.w)
        title_w = 16 + text_width(title, 13, weight=600, tracking=1.6)
        y = body.y + 6
        message_line = text(
            body.x,
            y + 14,
            fit(message, body.w, 17, weight=500),
            size=17,
            fill=TEXT_2,
            weight=500,
        )
        if title_w + 16 + flags_w > body.w:
            # Too narrow for title and flags on one row: the flags go under the message.
            out.append(panel_header(box, title, color))
            out.append(message_line)
            out.append(_flag_row(body.x, y + 26, body.w, flags, color))
            return "".join(out)
        out.append(panel_header(box, title, color, right_reserved=flags_w))
        out.append(_flag_row(box.right - PAD - flags_w, box.y + PAD + 5, flags_w, flags, color))
        out.append(message_line)
        if note and body.h > 44:
            out.append(text(body.x, y + 40, fit(note, body.w, 13), size=13, fill=MUTED))
        return "".join(out)
    out.append(panel_header(box, title, color))
    if ghost_kind:
        out.append(ghost(ghost_kind, body))
    block_h = 24 + 16 + 28 + (30 if note else 0)
    y = body.y + max((body.h - block_h) / 2, 0)
    cx = box.x + box.w / 2
    out.append(
        text(
            cx,
            y + 20,
            fit(message, body.w, 19, weight=500),
            size=19,
            fill=TEXT_2,
            weight=500,
            anchor="middle",
        )
    )
    y += 40
    flags_w = min(pills_width(flags), body.w)
    out.append(_flag_row(cx - flags_w / 2, y, flags_w, flags, color))
    if note:
        out.append(text(cx, y + 56, fit(note, body.w, 13), size=13, fill=MUTED, anchor="middle"))
    return "".join(out)


def hint_strip(box: Box, title: str, color: str, message: str, flags: tuple[str, ...]) -> str:
    """A compact hint: what was not measured and the flag that measures it.

    Wide strips use one row; narrow ones put the flags on a second row. The
    title already names the section, so a message too long for its room gives
    way to a short "Not measured" rather than an ellipsis.
    """
    out = [frame(box, dashed=True, radius=min(RADIUS, box.h / 3))]
    short = "Not scored" if message.lower().endswith("not scored") else "Not measured"
    wordings = [message, short]
    x = box.x + PAD
    room = box.w - 2 * PAD
    title_text = fit(title, room - 16, 13, weight=600, tracking=1.6)
    title_w = 16 + text_width(title_text, 13, weight=600, tracking=1.6)
    flags_w = pills_width(flags)
    message_w = text_width(message, 15, weight=500)
    one_row = title_w + 16 + min(message_w, 160) + 16 + flags_w <= room
    base = box.y + box.h / 2 + 5 if one_row else box.y + box.h / 2 - 9
    out.append(rect(x, base - 10, 8, 8, fill=color, rx=2))
    out.append(text(x + 16, base, title_text, size=13, fill=TEXT_2, weight=600, tracking=1.6))
    if one_row:
        msg_room = room - title_w - 16 - flags_w - 16
        out.append(
            text(
                x + title_w + 16,
                base,
                first_fit(wordings, msg_room, 15, weight=500),
                size=15,
                fill=TEXT_2,
                weight=500,
            )
        )
        out.append(
            _flag_row(box.right - PAD - flags_w, box.y + box.h / 2 - 14, flags_w, flags, color)
        )
        return "".join(out)
    msg_room = room - title_w - 12
    if msg_room >= 80:
        out.append(
            text(
                x + title_w + 12,
                base,
                first_fit(wordings, msg_room, 14, weight=500),
                size=14,
                fill=TEXT_2,
                weight=500,
            )
        )
    out.append(_flag_row(x, base + 10, room, flags, color))
    return "".join(out)


# --------------------------------------------------------------------------- scales


def nice_ticks(max_value: float, target: int = 4) -> list[float]:
    """Round ticks from zero that cover ``max_value`` with a little headroom."""
    if not math.isfinite(max_value) or max_value <= 0:
        return [0.0, 1.0]
    raw = max_value * 1.08 / target
    exp = 10 ** math.floor(math.log10(raw))
    step = next(m * exp for m in (1, 2, 2.5, 5, 10) if m * exp >= raw)
    top = math.ceil(max_value * 1.04 / step) * step
    count = int(round(top / step))
    return [i * step for i in range(count + 1)]


def log_decades(values: list[float]) -> tuple[int, int]:
    positive = [v for v in values if v > 0 and math.isfinite(v)]
    if not positive:
        return 0, 1
    lo = math.floor(math.log10(min(positive)))
    hi = math.ceil(math.log10(max(positive)))
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def log_step(decades: int, height: float, min_gap: float = 22.0) -> int:
    """Label every Nth decade so labels on a log axis stay at least ``min_gap`` apart."""
    step = 1
    while step < decades and height * step / max(decades, 1) < min_gap:
        step += 1
    return step


def time_ticks(lo: int, hi: int, height: float, min_gap: float = 22.0) -> list[float]:
    """Readable ticks between 10**lo and 10**hi ms, at least ``min_gap`` px apart."""
    span = (hi - lo) or 1
    candidates = [t for t in _TIME_TICKS if 10.0**lo <= t <= 10.0**hi]
    kept: list[float] = []
    for t in candidates:
        y = math.log10(t)
        if not kept or (y - math.log10(kept[-1])) / span * height >= min_gap:
            kept.append(t)
    return kept


def depth_position(depth: int) -> float:
    """Log-ish depth axis: 0 sits one step left of 1K and spacing doubles per octave."""
    return math.log2(1 + max(depth, 0) / 1024)


def spaced_labels(positions: list[float], widths: list[float], gap: float = 10) -> list[bool]:
    """Which labels to draw so neighbours do not collide. First and last always win."""
    count = len(positions)
    keep = [False] * count
    if not count:
        return keep
    keep[0] = True
    if count == 1:
        return keep
    keep[-1] = True
    last_right = positions[0] + widths[0] / 2
    final_left = positions[-1] - widths[-1] / 2
    for i in range(1, count - 1):
        left = positions[i] - widths[i] / 2
        right = positions[i] + widths[i] / 2
        if left >= last_right + gap and right <= final_left - gap:
            keep[i] = True
            last_right = right
    return keep


def grid_line(x1: float, x2: float, y: float) -> str:
    return line(x1, y, x2, y, stroke=STROKE, width=1, opacity=0.07, dash="2 6")


def axis_label(x: float, y: float, label: str, *, anchor: str = "end") -> str:
    return text(x, y, label, size=LABEL, fill=MUTED, font="mono", anchor=anchor)


def mono_width(label: str, size: float = LABEL) -> float:
    return text_width(label, size, font="mono")


def bar_color(percent: float) -> str:
    """One colour scale for every score bar and accuracy value on the card."""
    if percent >= 85:
        return PASS
    if percent >= 60:
        return GOOD
    if percent >= 40:
        return PARTIAL
    return FAIL
