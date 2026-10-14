"""Card panels. Each takes a :class:`Box` and returns SVG markup that stays inside it."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from tool_eval_bench.domain.card import (
    AccuracyResult,
    CardData,
    NeedleSummary,
    PerfPoint,
    ToolEvalSummary,
)
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    accuracy_is_thin,
    accuracy_note,
    cap_note,
    concurrency_series,
    has_long_context,
    is_subset,
    needs_caution,
    rating_parts,
    scope_label,
    score_details,
    single_stream,
    stream_concurrency,
    ttft_note,
    window_capped,
)
from tool_eval_bench.storage.reports.card._fonts import FontKey, text_width
from tool_eval_bench.storage.reports.card._svg import (
    Run,
    attrs,
    circle,
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
    DECODE,
    FAIL,
    FAINT,
    GOOD,
    MUTED,
    PAD,
    PARTIAL,
    PASS,
    PREFILL,
    RADIUS,
    STROKE,
    TEXT,
    TEXT_2,
    TOOL,
    TTFT,
    TTFT_CACHED,
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


FLAG_SIZE = 13.0


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


GhostKind = Literal["line", "bars", "heatmap", "rows"]


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


# Time ticks in milliseconds. Past 10 s they follow minutes and hours rather
# than decades, so labels read "1 min" and "1 h" instead of "100 s" or "1e+06 s".
_TIME_TICKS = (0.1, 1, 10, 100, 1e3, 1e4, 6e4, 6e5, 3.6e6, 3.6e7, 3.6e8)


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


# --------------------------------------------------------------------------- hero tiles


@dataclass(frozen=True)
class Hero:
    """One hero tile. ``sub`` and ``sub2`` may list alternative wordings, longest first."""

    label: str
    color: str
    value: str = ""
    unit: str = ""
    sub: str | tuple[str, ...] | None = None
    sub_color: str = TEXT_2
    sub2: str | tuple[str, ...] | None = None
    sub2_color: str = MUTED
    stars: tuple[int, str] | None = None
    capped: bool = False
    flags: tuple[str, ...] = ()

    @property
    def missing(self) -> bool:
        return not self.value


def _value_runs(value: str, unit: str, size: float, color: str) -> list[Run]:
    runs = [Run(value, size, TEXT, "display", 700, tracking=-0.5)]
    if unit:
        runs.append(Run(unit, round(size * 0.36), color, "sans", 600, gap=size * 0.1))
    return runs


def hero_value_size(heroes: list[Hero], boxes: list[Box], max_size: float = 56) -> float:
    """One value size for the whole row, so the numbers read as a set."""
    size = max_size
    for hero, box in zip(heroes, boxes, strict=True):
        if hero.missing:
            continue
        room = box.w - 2 * PAD
        while size > 28 and runs_width(_value_runs(hero.value, hero.unit, size, hero.color)) > room:
            size -= 2
    return size


def hero_tile(box: Box, hero: Hero, value_size: float) -> str:
    out = [frame(box, dashed=hero.missing)]
    x = box.x + PAD
    room = box.w - 2 * PAD
    if not hero.missing:
        # A short colored rule ties each tile to its chart color without shouting.
        out.append(rect(x, box.y, min(56.0, room), 3, fill=hero.color, rx=1.5))
    label_y = box.y + PAD + 12
    out.append(
        text(
            x,
            label_y,
            fit(hero.label, room, 13, weight=600, tracking=1.4),
            size=13,
            fill=MUTED if hero.missing else TEXT_2,
            weight=600,
            tracking=1.4,
        )
    )
    sub2_y = box.bottom - PAD + 4
    sub_y = sub2_y - 20
    if hero.missing:
        out.append(text(x, label_y + 40, "Not measured", size=19, fill=TEXT_2, weight=500))
        out.append(_flag_row(x, sub2_y - 20, room, hero.flags, hero.color))
        return "".join(out)
    # The value sits midway between the label and the first detail line, by cap height.
    cap = value_size * 0.73
    value_y = (label_y + sub_y - 18) / 2 + cap / 2
    out.append(runs_text(x, value_y, _value_runs(hero.value, hero.unit, value_size, hero.color)))
    if hero.stars is not None:
        out.append(_stars(x, sub_y, room, hero.stars, hero.color, capped=hero.capped, size=14))
    elif hero.sub:
        out.append(
            text(x, sub_y, first_fit(_variants(hero.sub), room, 13), size=13, fill=hero.sub_color)
        )
    if hero.sub2:
        out.append(
            text(
                x, sub2_y, first_fit(_variants(hero.sub2), room, 13), size=13, fill=hero.sub2_color
            )
        )
    return "".join(out)


def _variants(value: str | tuple[str, ...]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _stars(
    x: float,
    y: float,
    room: float,
    stars: tuple[int, str],
    color: str,
    *,
    capped: bool,
    size: float = 15,
) -> str:
    """Rating stars, the rating word, and "capped" when the safety cap applied."""
    filled, word = stars
    runs = [Run("\u2605" * filled, size, color, tracking=1.0)] if filled else []
    if filled < 5:
        runs.append(Run("\u2605" * (5 - filled), size, FAINT, tracking=1.0))
    word_size = size - 1
    word_room = room - runs_width(runs) - 10
    if capped:
        word_room -= text_width("capped", word_size - 1, weight=500) + 8
    if word and word_room > 24:
        runs.append(
            Run(fit(word, word_room, word_size, weight=500), word_size, TEXT_2, weight=500, gap=10)
        )
    if capped:
        runs.append(Run("capped", word_size - 1, WARN, weight=500, gap=8))
    return runs_text(x, y, runs)


# --------------------------------------------------------------------------- tool score block


def score_block(box: Box, te: ToolEvalSummary) -> str:
    """The big tool-calling score used when no speed numbers lead the card.

    The score, rating, scope, and outcome bar form one group centered below the
    label, and an outcome ring fills the right side when the block is large
    enough. A subset withholds the stars: a rating earned on 3 of 69 scenarios
    is not a rating of the model.
    """
    out = [frame(box), rect(box.x + PAD, box.y, 56, 3, fill=TOOL, rx=1.5)]
    x = box.x + PAD
    room = box.w - 2 * PAD
    radius = min((box.h - 2 * PAD) / 2 - 12, 128.0, room * 0.22)
    ring = radius >= 64 and room - 2 * radius - 48 >= 220
    if ring:
        out.append(_ring(box.right - PAD - radius - 8, box.y + box.h / 2, radius, te))
        room -= 2 * radius + 48
    label_y = box.y + PAD + 12
    out.append(
        text(x, label_y, "TOOL CALLING SCORE", size=13, fill=TEXT_2, weight=600, tracking=1.4)
    )
    area_top, area_bottom = label_y + 16, box.bottom - PAD
    score = fmt.score(te.score)
    details_h, bar_gap, bar_h = 58.0, 32.0, 36.0
    size = max(
        48.0, min(132.0, (area_bottom - area_top - details_h - bar_gap - bar_h) / 0.73 * 0.9)
    )

    def score_runs(size: float) -> list[Run]:
        return [
            Run(score, size, TEXT, "display", 700, tracking=-2),
            Run("/100", round(size * 0.28), TOOL, "sans", 600, gap=6),
        ]

    while size > 48 and runs_width(score_runs(size)) > room:
        size -= 4
    cap = size * 0.73
    group_h = cap + details_h + bar_gap + bar_h
    base = area_top + max((area_bottom - area_top - group_h) / 2, 0) + cap
    out.append(runs_text(x - size * 0.04, base, score_runs(size)))
    details: list[str]
    if is_subset(te):
        out.append(warn_pill(x, base + 13, scope_label(te), max_width=room)[0])
        parts = [] if ring else [f"{te.passed} of {te.total} passed"]
        if te.trials > 1:
            parts.append(f"mean of {te.trials} trials")
        if te.safety_capped:
            parts.append("safety-capped")
        details = [MID.join(parts)] if parts else []
        color = MUTED
    else:
        out.append(
            _stars(
                x, base + 32, room, rating_parts(te.rating), TOOL, capped=te.safety_capped, size=17
            )
        )
        details = score_details(te, passed=not ring)
        color = WARN if needs_caution(te) else MUTED
    if details:
        out.append(
            text(
                x,
                base + 58,
                first_fit(details, room, 14, weight=500),
                size=14,
                fill=color,
                weight=500,
            )
        )
    out.append(outcome_bar(Box(x, base + details_h + bar_gap, room, bar_h), te))
    return "".join(out)


def _ring(cx: float, cy: float, r: float, te: ToolEvalSummary) -> str:
    """Outcomes as a ring: passes, then partials, then failures, from twelve o'clock.

    The center gives the pass count rather than a percentage, so it never
    competes with the 0-100 score beside it.
    """
    total = max(te.passed + te.partial + te.failed, 1)
    circumference = 2 * math.pi * r
    width = max(10.0, r * 0.12)
    out = [circle(cx, cy, r, fill="none", stroke=FAINT, stroke_width=width)]
    start = 0.0
    for count, color in ((te.passed, PASS), (te.partial, PARTIAL), (te.failed, FAIL)):
        if count <= 0:
            continue
        length = circumference * count / total
        out.append(
            f"<circle {attrs(cx=float(cx), cy=float(cy), r=float(r), fill='none', stroke=color, stroke_width=width, stroke_dasharray=f'{length:.2f} {circumference:.2f}', stroke_dashoffset=f'{-start:.2f}', transform=f'rotate(-90 {cx:.2f} {cy:.2f})', opacity=0.95 if color == PASS else 0.9)}/>"
        )
        start += length
    label = f"{te.passed}/{te.total}"
    size = r * 0.34
    inside = 2 * r - 2 * width - 24
    while size > 14 and text_width(label, size, font="display", weight=700) > inside:
        size -= 1
    out.append(
        text(
            cx,
            cy + size * 0.2,
            label,
            size=size,
            fill=TEXT,
            font="display",
            weight=700,
            anchor="middle",
        )
    )
    out.append(text(cx, cy + size * 0.2 + 24, "passed", size=13, fill=MUTED, anchor="middle"))
    return "".join(out)


# --------------------------------------------------------------------------- accuracy


ACCURACY_ROW_MAX = 112.0
ACCURACY_ROW_COMFORT = 84.0
_ACCURACY_FLAGS = ("--gsm8k", "--mmlu", "--ifeval")


def accuracy_height(count: int) -> float:
    """Height at which an accuracy panel's rows are comfortable but not spread out."""
    return PAD + HEADER_H + 16 + count * ACCURACY_ROW_COMFORT + PAD


def accuracy_min_height(count: int) -> float:
    return PAD + HEADER_H + 16 + count * 48.0 + PAD


def accuracy_panel(box: Box, data: CardData) -> str:
    """External accuracy benchmarks as large numbers."""
    title = "ACCURACY"
    if not data.accuracy:
        return missing_panel(
            box,
            title,
            PREFILL,
            "Accuracy benchmarks not run",
            _ACCURACY_FLAGS,
            note="Math, knowledge, and instruction following",
            ghost_kind="bars",
        )
    out = [frame(box), panel_header(box, title, PREFILL)]
    body = content_box(box)
    if _rings_fit(body, len(data.accuracy)):
        out.append(_accuracy_rings(body, data.accuracy))
    else:
        out.append(_accuracy_rows(body, data.accuracy))
    return "".join(out)


# A ring gauge per benchmark needs this much room; below it the rows read better.
_RING_MIN_COL_W = 200.0
_RING_MIN_H = 300.0
_RING_MAX_R = 112.0


def _rings_fit(body: Box, count: int) -> bool:
    return 0 < count <= 4 and body.h >= _RING_MIN_H and body.w / count >= _RING_MIN_COL_W


def _accuracy_rings(body: Box, results: tuple[AccuracyResult, ...]) -> str:
    """One ring gauge per benchmark, the score inside, the name and any caveat below."""
    col_w = body.w / len(results)
    label_h = 64.0
    r = min(col_w * 0.36, (body.h - label_h) / 2 - 12, _RING_MAX_R)
    width = max(10.0, r * 0.12)
    cy = body.y + (body.h - 2 * r - label_h) / 2 + r
    out = []
    for i, result in enumerate(results):
        cx = body.x + col_w * (i + 0.5)
        finite = math.isfinite(result.accuracy)
        frac = max(0.0, min(result.accuracy, 1.0)) if finite else 0.0
        out.append(circle(cx, cy, r, fill="none", stroke=FAINT, stroke_width=width))
        if frac > 0:
            circumference = 2 * math.pi * r
            out.append(
                f"<circle {attrs(cx=float(cx), cy=float(cy), r=float(r), fill='none', stroke=bar_color(frac * 100), stroke_width=width, stroke_dasharray=f'{circumference * frac:.2f} {circumference:.2f}', transform=f'rotate(-90 {cx:.2f} {cy:.2f})')}/>"
            )
        value = fmt.percent(result.accuracy)
        inside = 2 * (r - width) - 16
        size = r * 0.46
        while size > 16 and text_width(value, size, font="display", weight=700) > inside:
            size -= 1
        out.append(
            text(
                cx,
                cy + size * 0.36,
                value,
                size=size,
                fill=TEXT if finite else MUTED,
                font="display",
                weight=700,
                anchor="middle",
            )
        )
        name_y = cy + r + 34
        out.append(
            text(
                cx,
                name_y,
                fit(result.name, col_w - 16, 17, weight=600),
                size=17,
                fill=TEXT_2,
                weight=600,
                anchor="middle",
            )
        )
        notes = [n for n in (accuracy_note(result), accuracy_note(result, detail=False)) if n]
        if notes:
            out.append(
                text(
                    cx,
                    name_y + 22,
                    first_fit(notes, col_w - 16, 13),
                    size=13,
                    fill=WARN if result.errors > 0 else MUTED,
                    anchor="middle",
                )
            )
    return "".join(out)


def _accuracy_rows(body: Box, results: tuple[AccuracyResult, ...]) -> str:
    """One row per benchmark: name and any caveat on the left, the score on the right."""
    row_h = min(body.h / len(results), ACCURACY_ROW_MAX)
    top = body.y + max((body.h - row_h * len(results)) / 2, 0)
    size = max(20.0, min(48.0, row_h * 0.46))
    bar_h = 4.0 if row_h < 80 else 6.0
    out = []
    for i, result in enumerate(results):
        y = top + i * row_h
        bar = row_h >= 44
        base = y + row_h / 2 + size * 0.36 - (6 if bar else 0)
        value = fmt.percent(result.accuracy)
        value_w = text_width(value, size, font="display", weight=700)
        left_room = body.w - value_w - 24
        name = fit(result.name, min(left_room, body.w * 0.6), 15, weight=500)
        name_w = text_width(name, 15, weight=500)
        runs = [Run(name, 15, TEXT_2, weight=500)]
        notes = [n for n in (accuracy_note(result), accuracy_note(result, detail=False)) if n]
        note_room = left_room - name_w - 10
        if notes and note_room >= 48:
            # The stored detail goes first when space runs out; the error count stays.
            color = WARN if result.errors > 0 else MUTED
            runs.append(Run(first_fit(notes, note_room, 13), 13, color, gap=10))
        out.append(runs_text(body.x, base, runs))
        out.append(
            text(
                body.right,
                base,
                value,
                size=size,
                fill=TEXT if math.isfinite(result.accuracy) else MUTED,
                font="display",
                weight=700,
                anchor="end",
            )
        )
        bar_y = base + 12
        if bar and bar_y + bar_h <= y + row_h:
            out.append(rect(body.x, bar_y, body.w, bar_h, fill=FAINT, rx=bar_h / 2))
            frac = max(0.0, min(result.accuracy, 1.0)) if math.isfinite(result.accuracy) else 0.0
            if frac > 0:
                out.append(
                    rect(
                        body.x,
                        bar_y,
                        max(body.w * frac, bar_h),
                        bar_h,
                        fill=bar_color(frac * 100),
                        rx=bar_h / 2,
                    )
                )
    return "".join(out)


# --------------------------------------------------------------------------- long context


def long_context_panel(box: Box, data: CardData) -> str:
    title = "LONG CONTEXT"
    if not data.perf:
        return missing_panel(
            box,
            title,
            DECODE,
            "Long-context speed not measured",
            ("--perf --perf-preset long",),
            note="Charts decode speed and TTFT from 0 to 128K context",
            ghost_kind="line",
        )
    series = single_stream(data)
    if not has_long_context(data):
        return depth_strip(box, data)
    level = stream_concurrency(data)
    deepest = fmt.tokens(series[-1].depth)
    if data.perf_incomplete:
        subtitle = [f"depth pass incomplete, stopped at {deepest}", "depth pass incomplete"]
        subtitle_color = WARN
    elif window_capped(data):
        subtitle = [cap_note(data), f"capped at {deepest}"]
        subtitle_color = WARN
    else:
        stream = "single stream (c=1)" if level == 1 else f"per request at c={level}"
        subtitle = [f"{stream} by context depth", stream]
        subtitle_color = MUTED
    out = [
        frame(box),
        panel_header(box, title, DECODE, subtitle=subtitle, subtitle_color=subtitle_color),
    ]
    body = content_box(box, extra=-4)
    body = Box(body.x, body.y, body.w, body.h - 24)
    out.append(
        text(body.x, body.bottom + 22, first_fit(ttft_note(data), body.w, 13), size=13, fill=MUTED)
    )
    if body.w >= body.h * 1.5:
        half = (body.w - 48) / 2
        first = Box(body.x, body.y, half, body.h)
        second = Box(body.x + half + 48, body.y, half, body.h)
        out.append(
            line(
                body.x + half + 24,
                body.y + 8,
                body.x + half + 24,
                body.bottom - 8,
                stroke=STROKE,
                opacity=0.07,
            )
        )
    else:
        half = (body.h - 32) / 2
        first = Box(body.x, body.y, body.w, half)
        second = Box(body.x, body.y + half + 32, body.w, half)
    out.append(_decode_chart(first, series))
    out.append(_ttft_chart(second, series))
    return "".join(out)


def depth_strip(box: Box, data: CardData) -> str:
    """Sweeps too shallow to chart: the measured depths as numbers, and why it stops there.

    The hint to run the long preset appears only when that would help: not
    after a long run that the context window capped, and not after a long run
    that was interrupted.
    """
    series = single_stream(data)
    deepest = fmt.tokens(series[-1].depth if series else 0)
    first = fmt.tokens(series[0].depth if series else 0)
    flags: tuple[str, ...] = ("--perf --perf-preset long",)
    to_128k = "The long preset charts decode and TTFT to 128K."
    if data.perf_incomplete:
        message = f"Depth pass incomplete: stopped at {deepest}"
        note = "Rerun the sweep to chart decode and TTFT by depth."
        if data.perf_preset == "long":
            flags = ()
    elif window_capped(data):
        message = f"Long preset {cap_note(data)}"
        note = "A larger served context window lets the sweep go deeper."
        flags = ()
    elif data.perf_preset == "short":
        span = f"depths {first} to {deepest}" if len(series) > 1 else f"depth {deepest} only"
        message = f"Short preset: {span}"
        note = to_128k
    elif len(series) <= 1:
        message = f"Measured at depth {deepest} only"
        note = to_128k
    else:
        message = f"Depths {first} to {deepest} only"
        note = to_128k
    out = [frame(box, dashed=True)]
    flags_w = pills_width(flags) if flags else 0.0
    out.append(panel_header(box, "LONG CONTEXT", DECODE, right_reserved=flags_w))
    if flags:
        px = box.right - PAD - flags_w
        out.append(_flag_row(px, box.y + PAD + 5, flags_w, flags, DECODE))
    body = content_box(box)
    msg_w = max(text_width(message, 17, weight=500), text_width(note, 13), 220.0)
    out.append(
        text(
            body.x,
            body.y + 14,
            fit(message, body.w, 17, weight=500),
            size=17,
            fill=WARN if data.perf_incomplete else TEXT_2,
            weight=500,
        )
    )
    col_w = 140.0
    beside = body.x + msg_w + 40
    if body.right - beside >= col_w * len(series):
        out.append(_depth_stats(beside, body.y - 2, col_w, series))
        out.append(
            text(body.x, body.y + 40, fit(note, beside - body.x - 24, 13), size=13, fill=MUTED)
        )
    elif body.h >= 110 and body.w >= 110 * len(series):
        out.append(text(body.x, body.y + 40, fit(note, body.w, 13), size=13, fill=MUTED))
        out.append(
            _depth_stats(body.x, body.y + 68, min(col_w * 1.2, body.w / len(series)), series)
        )
    else:
        out.append(text(body.x, body.y + 40, fit(note, body.w, 13), size=13, fill=MUTED))
    return "".join(out)


def _depth_stats(x: float, y: float, col_w: float, series: list[PerfPoint]) -> str:
    out = []
    for i, p in enumerate(series):
        cx = x + i * col_w
        out.append(
            text(cx, y + 6, f"depth {fmt.tokens(p.depth)}", size=LABEL, fill=MUTED, font="mono")
        )
        out.append(
            runs_text(
                cx,
                y + 32,
                [
                    Run(fmt.rate(p.tg_tps), 22, TEXT, "display", 700),
                    Run("tok/s", 13, DECODE, weight=600, gap=5),
                ],
            )
        )
        out.append(
            text(
                cx,
                y + 52,
                f"TTFT {fmt.duration_text(p.ttft_ms)}",
                size=LABEL,
                fill=TTFT,
                font="mono",
            )
        )
    return "".join(out)


_CHART_TOP = 44.0
_AXIS_H = 30.0


def _chart_heading(
    box: Box, label: str, color: str, callout: str, callout_label: str
) -> tuple[str, float]:
    """Series name on the left, the takeaway on the right. Returns markup and the label's right edge."""
    base = box.y + 22
    out = [circle(box.x + 5, base - 5, 5, fill=color)]
    out.append(text(box.x + 18, base, label, size=15, fill=TEXT, weight=600))
    out.append(
        runs_text(
            box.right,
            base + 1,
            [
                Run(callout, 26, color, "display", 700),
                Run(callout_label, 13, MUTED, weight=500, gap=8),
            ],
            anchor="end",
        )
    )
    return "".join(out), box.x + 18 + text_width(label, 15, weight=600)


def _depth_axis(plot: Box, depths: list[int]) -> tuple[list[float], str]:
    positions = [depth_position(d) for d in depths]
    lo, hi = positions[0], positions[-1]
    span = hi - lo or 1.0
    xs = [plot.x + 10 + (p - lo) / span * (plot.w - 20) for p in positions]
    labels = [fmt.tokens(d) for d in depths]
    keep = spaced_labels(xs, [mono_width(lbl) for lbl in labels])
    out = [
        axis_label(x, plot.bottom + 22, lbl, anchor="middle")
        for x, lbl, k in zip(xs, labels, keep, strict=True)
        if k
    ]
    return xs, "".join(out)


def _decode_callout(series: list[PerfPoint]) -> tuple[str, str]:
    """The decode change from the shallowest to the deepest depth.

    A near-zero base makes a percentage meaningless (+29,900%), so the deepest
    speed is shown instead.
    """
    base, last = series[0], series[-1]
    peak = max(p.tg_tps for p in series)
    at = f"at {fmt.tokens(last.depth)}"
    if base.tg_tps <= 0 or base.tg_tps < peak * 0.01:
        return fmt.rate(last.tg_tps), f"tok/s {at}"
    change = last.tg_tps / base.tg_tps - 1
    if abs(change) > 9.99:
        return fmt.rate(last.tg_tps), f"tok/s {at}"
    return fmt.signed_percent(change), at


def _decode_chart(box: Box, series: list[PerfPoint]) -> str:
    base, last = series[0], series[-1]
    callout, callout_label = _decode_callout(series)
    heading, _ = _chart_heading(box, "Decode tok/s", DECODE, callout, callout_label)
    out = [heading]
    plot = Box(box.x + 44, box.y + _CHART_TOP, box.w - 44, box.h - _CHART_TOP - _AXIS_H)
    values = [p.tg_tps for p in series]
    positive = [v for v in values if v > 0]
    # A sweep spanning 9,000 to 2 tok/s would flatten into the axis on a linear
    # scale, so wide ranges switch to log.
    use_log = len(positive) == len(values) and max(values) / min(positive) > 25
    if use_log:
        lo, hi = log_decades(values)

        def y_of(v: float) -> float:
            return plot.bottom - (math.log10(max(v, 10.0**lo)) - lo) / (hi - lo) * plot.h

        step = log_step(hi - lo, plot.h)
        ticks = [10.0**k for k in range(lo, hi + 1, step)]
    else:
        ticks = nice_ticks(max(values), target=3)
        top = ticks[-1] or 1.0

        def y_of(v: float) -> float:
            return plot.bottom - v / top * plot.h

    for t in ticks:
        out.append(grid_line(plot.x, plot.right, y_of(t)))
        out.append(axis_label(plot.x - 10, y_of(t) + 4, fmt.axis_rate(t)))
    xs, labels = _depth_axis(plot, [p.depth for p in series])
    out.append(labels)
    pts = [(x, y_of(p.tg_tps)) for x, p in zip(xs, series, strict=True)]
    area = [(pts[0][0], plot.bottom), *pts, (pts[-1][0], plot.bottom)]
    out.append(f'<polygon points="{polyline(area)}" fill="url(#decodeArea)"/>')
    out.append(
        f'<polyline points="{polyline(pts)}" fill="none" stroke="{DECODE}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow)"/>'
    )
    for x, y in pts:
        out.append(circle(x, y, 4.5, fill=DOT_FILL, stroke=DECODE, stroke_width=2.5))
    out.append(_end_labels(pts, [fmt.rate(base.tg_tps), fmt.rate(last.tg_tps)], plot))
    return "".join(out)


def _span_extent(
    lines: list[list[tuple[float, float]]], x0: float, x1: float
) -> tuple[float, float] | None:
    """Lowest and highest y any polyline reaches between x0 and x1."""
    ys: list[float] = []
    for pts in lines:
        for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
            lo, hi = max(min(ax, bx), x0), min(max(ax, bx), x1)
            if lo > hi:
                continue
            for xv in (lo, hi):
                t = (xv - ax) / (bx - ax) if bx != ax else 0.0
                ys.append(ay + t * (by - ay))
        ys += [y for x, y in pts if x0 - 6 <= x <= x1 + 6]
    return (min(ys), max(ys)) if ys else None


def _end_labels(
    pts: list[tuple[float, float]],
    labels: list[str],
    plot: Box,
    *,
    obstacles: list[list[tuple[float, float]]] | None = None,
) -> str:
    """Value labels for the first and last points, placed on whichever side of the lines is clearer."""
    lines = [pts, *(obstacles or [])]
    out = []
    for (x, y), label, anchor in ((pts[0], labels[0], "start"), (pts[-1], labels[1], "end")):
        if not label:
            continue
        w = text_width(label, LABEL, font="mono", weight=500)
        lx = x + 8 if anchor == "start" else x - 8
        x0, x1 = (lx, lx + w) if anchor == "start" else (lx - w, lx)
        extent = _span_extent(lines, x0 - 2, x1 + 2) or (y, y)
        top_line, bottom_line = min(extent[0], y), max(extent[1], y)
        above = top_line - 8
        below = bottom_line + 19
        fits_above = above - 11 >= plot.y - 18
        fits_below = below <= plot.bottom + 6
        if fits_above and (not fits_below or y - above <= below - y):
            ly = above
        elif fits_below:
            ly = below
        else:
            ly = max(above, plot.y - 7)
        out.append(
            text(lx, ly, label, size=LABEL, fill=TEXT_2, font="mono", weight=500, anchor=anchor)
        )
    return "".join(out)


def _ttft_chart(box: Box, series: list[PerfPoint]) -> str:
    last = series[-1]
    cached = [p for p in series if p.cached_ttft_ms is not None]
    heading, label_right = _chart_heading(
        box, "TTFT", TTFT, fmt.duration_text(last.ttft_ms), f"cold at {fmt.tokens(last.depth)}"
    )
    out = [heading]
    callout_w = runs_width(
        [
            Run(fmt.duration_text(last.ttft_ms), 26, TTFT, "display", 700),
            Run(f"cold at {fmt.tokens(last.depth)}", 13, MUTED, weight=500, gap=8),
        ]
    )
    if cached:
        # The cached line matters to agents that re-send a long prefix, so its
        # latest value goes in the legend rather than fighting the lines for room.
        lx = label_right + 16
        ly = box.y + 22
        room = box.right - callout_w - 16 - (lx + 22)
        warm = first_fit(
            [
                f"cached {fmt.duration_text(cached[-1].cached_ttft_ms or 0.0)}",
                fmt.duration_text(cached[-1].cached_ttft_ms or 0.0),
            ],
            room,
            LABEL,
            font="mono",
            weight=500,
        )
        if room >= 40:
            out.append(line(lx, ly - 4, lx + 16, ly - 4, stroke=TTFT_CACHED, width=2, dash="4 3"))
            out.append(
                text(lx + 22, ly, warm, size=LABEL, fill=TTFT_CACHED, font="mono", weight=500)
            )
    plot = Box(box.x + 60, box.y + _CHART_TOP, box.w - 60, box.h - _CHART_TOP - _AXIS_H)
    values = [p.ttft_ms for p in series] + [
        p.cached_ttft_ms for p in cached if p.cached_ttft_ms is not None
    ]
    lo, hi = log_decades(values)

    def y_of(v: float) -> float:
        v = max(v, 10.0**lo)
        return plot.bottom - (math.log10(v) - lo) / (hi - lo) * plot.h

    for t in time_ticks(lo, hi, plot.h):
        out.append(grid_line(plot.x, plot.right, y_of(t)))
        out.append(axis_label(plot.x - 10, y_of(t) + 4, fmt.axis_duration(t)))
    xs, labels = _depth_axis(plot, [p.depth for p in series])
    out.append(labels)
    cold = [(x, y_of(p.ttft_ms)) for x, p in zip(xs, series, strict=True)]
    area = [(cold[0][0], plot.bottom), *cold, (cold[-1][0], plot.bottom)]
    out.append(f'<polygon points="{polyline(area)}" fill="url(#ttftArea)"/>')
    warm_pts: list[tuple[float, float]] = []
    if cached:
        warm_pts = [
            (x, y_of(p.cached_ttft_ms))
            for x, p in zip(xs, series, strict=True)
            if p.cached_ttft_ms is not None
        ]
        out.append(
            f'<polyline points="{polyline(warm_pts)}" fill="none" stroke="{TTFT_CACHED}" stroke-width="2" stroke-dasharray="5 4" stroke-linejoin="round" opacity="0.9"/>'
        )
        for x, y in warm_pts:
            out.append(circle(x, y, 3, fill=TTFT_CACHED))
    out.append(
        f'<polyline points="{polyline(cold)}" fill="none" stroke="{TTFT}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow)"/>'
    )
    for x, y in cold:
        out.append(circle(x, y, 4.5, fill=DOT_FILL, stroke=TTFT, stroke_width=2.5))
    # Only the shallow end gets a label; the deep end is the heading callout.
    out.append(
        _end_labels(
            cold,
            [fmt.duration_text(series[0].ttft_ms), ""],
            plot,
            obstacles=[warm_pts] if warm_pts else None,
        )
    )
    return "".join(out)


# --------------------------------------------------------------------------- concurrency


def concurrency_panel(box: Box, data: CardData) -> str:
    title = "CONCURRENCY"
    if not data.perf:
        return missing_panel(
            box,
            title,
            PREFILL,
            "Throughput under load not measured",
            ("--perf",),
            note="Aggregate decode speed as concurrent requests climb",
            ghost_kind="bars",
        )
    series = concurrency_series(data)
    if len(series) < 2:
        only = series[0]
        return missing_panel(
            box,
            title,
            PREFILL,
            f"One concurrency level only: {fmt.rate(only.tg_tps_total)} tok/s",
            ("--perf --concurrency 1,2,4,8",),
            note="More concurrency levels chart aggregate throughput",
        )
    peak = max(series, key=lambda p: p.tg_tps_total)
    single = series[0]
    out = [
        frame(box),
        panel_header(
            box,
            title,
            PREFILL,
            subtitle=f"depth {fmt.tokens(single.depth)}",
            callout=fmt.rate(peak.tg_tps_total),
            callout_label="tok/s max aggregate",
        ),
    ]
    body = content_box(box, extra=-10)
    legend_y = body.y + 6
    estimated = any(p.tg_estimated for p in series)
    per_request = "per request (est.)" if estimated else "per request"
    out.append(rect(body.x, legend_y - 9, 10, 10, fill="url(#bar)", rx=2))
    out.append(text(body.x + 16, legend_y, "aggregate", size=LABEL, fill=MUTED, font="mono"))
    lx = body.x + 16 + mono_width("aggregate") + 16
    out.append(circle(lx + 4, legend_y - 4, 4, fill=TEXT))
    speedup = peak.tg_tps_total / single.tg_tps_total if single.tg_tps_total > 0 else 0.0
    speedup_label = ""
    if speedup > 0 and peak.concurrency != single.concurrency:
        options = (
            [f"{speedup:.1f}\u00d7 at c={peak.concurrency}"]
            if single.concurrency == 1
            else [
                f"{speedup:.1f}\u00d7 c={single.concurrency} to {peak.concurrency}",
                f"{speedup:.1f}\u00d7",
            ]
        )
        room = body.right - (lx + 14 + mono_width(per_request)) - 16
        label = first_fit(options, room, LABEL, font="mono", weight=500)
        if mono_width(label) <= room:
            speedup_label = label
            out.append(
                text(
                    body.right,
                    legend_y,
                    label,
                    size=LABEL,
                    fill=PREFILL,
                    font="mono",
                    weight=500,
                    anchor="end",
                )
            )
    # On the aggregate scale the per-request dots hug the baseline, so the
    # legend names the per-request rate at the peak when it fits.
    legend_room = body.right - (lx + 14) - (mono_width(speedup_label) + 16 if speedup_label else 0)
    per_request_label = per_request
    if math.isfinite(peak.tg_tps) and peak.tg_tps > 0:
        rate = fmt.rate(peak.tg_tps)
        candidates = [f"{per_request} {rate} at c={peak.concurrency}", f"{per_request} {rate}"]
        per_request_label = next(
            (c for c in candidates if mono_width(c) <= legend_room), per_request
        )
    out.append(text(lx + 14, legend_y, per_request_label, size=LABEL, fill=MUTED, font="mono"))
    # Room above the plot for the value label over the tallest bar.
    plot = Box(body.x + 40, body.y + 40, body.w - 40, body.h - 40 - 24)
    # Scale to just above the peak, not to the next round tick, so the bars
    # use the plot height; ticks above the scale are simply not drawn.
    top = max(peak.tg_tps_total * 1.12, 1e-9)
    ticks = [t for t in nice_ticks(peak.tg_tps_total, target=3) if t <= top]

    def y_of(v: float) -> float:
        return plot.bottom - min(max(v, 0.0) / top, 1.0) * plot.h

    for t in ticks:
        out.append(grid_line(plot.x, plot.right, y_of(t)))
        out.append(axis_label(plot.x - 10, y_of(t) + 4, fmt.axis_rate(t)))
    slot = plot.w / len(series)
    bar_w = min(slot * 0.58, 64.0)
    centers = [plot.x + slot * (i + 0.5) for i in range(len(series))]
    show_values = all(mono_width(fmt.rate(p.tg_tps_total)) <= slot - 4 for p in series)
    for cx, p in zip(centers, series, strict=True):
        y = y_of(p.tg_tps_total)
        out.append(
            rect(
                cx - bar_w / 2, y, bar_w, plot.bottom - y, fill="url(#bar)", rx=min(5.0, bar_w / 4)
            )
        )
        if show_values:
            # Lift the label clear of the per-request dot when they coincide (c=1).
            lift = 18 if y_of(p.tg_tps) - y < 12 else 8
            out.append(
                text(
                    cx,
                    y - lift,
                    fmt.rate(p.tg_tps_total),
                    size=LABEL,
                    fill=TEXT_2,
                    font="mono",
                    weight=500,
                    anchor="middle",
                )
            )
        label = f"c={p.concurrency}"
        if mono_width(label) > slot - 6:
            label = str(p.concurrency)
        out.append(axis_label(cx, plot.bottom + 22, label, anchor="middle"))
    dots = [(cx, y_of(p.tg_tps)) for cx, p in zip(centers, series, strict=True)]
    out.append(
        f'<polyline points="{polyline(dots)}" fill="none" stroke="{TEXT}" stroke-width="1.5" stroke-opacity="0.5" stroke-dasharray="3 4"/>'
    )
    for x, y in dots:
        out.append(circle(x, y, 4, fill=TEXT, stroke=DOT_FILL, stroke_width=2.0))
    return "".join(out)


# --------------------------------------------------------------------------- needle


def needle_panel(box: Box, data: CardData) -> str:
    title = "NEEDLE IN A HAYSTACK"
    needle = data.needle
    if needle is None:
        return missing_panel(
            box,
            title,
            PASS,
            "Retrieval not measured",
            ("--needle",),
            note="Hides a fact at every depth of each context length",
            ghost_kind="heatmap",
        )
    callout = _needle_callout(needle)
    out = [frame(box)]
    if callout is None:
        out.append(panel_header(box, title, PASS))
    else:
        value, color = callout
        out.append(
            panel_header(
                box,
                title,
                PASS,
                callout=value,
                callout_label="effective context",
                callout_color=color,
            )
        )
    body = content_box(box)
    if needle.cells:
        out.append(_needle_heatmap(body, needle))
    else:
        out.append(_needle_summary(body, needle))
    return "".join(out)


def needle_errors(needle: NeedleSummary) -> int:
    return max(needle.errors, sum(c.error for c in needle.cells))


def _all_errored(needle: NeedleSummary) -> bool:
    if needle.cells:
        return all(c.error for c in needle.cells)
    return needle.errors > 0 and needle.accuracy <= 0


def _needle_callout(needle: NeedleSummary) -> tuple[str, str] | None:
    """The effective context, marked "~" because haystack sizes are token estimates.

    When no length passed at every depth the callout says retrieval failed below
    the smallest length tried; when nothing was answered there is no callout.
    """
    if _all_errored(needle):
        return None
    if needle.effective_context:
        return f"~{fmt.tokens(needle.effective_context)}", PASS
    answered = [c.context_tokens for c in needle.cells if not c.error]
    if answered:
        return f"<{fmt.tokens(min(answered))}", FAIL
    return None


def _needle_summary(body: Box, needle: NeedleSummary) -> str:
    """Runs without a stored grid show their totals plainly."""
    errors = needle_errors(needle)
    if _all_errored(needle):
        size = min(26.0, body.h * 0.3)
        cy = body.y + body.h / 2
        return text(
            body.x,
            cy,
            fit(f"All {errors} cells errored", body.w, size, weight=600),
            size=size,
            fill=TEXT_2,
            weight=600,
        ) + text(
            body.x,
            cy + 28,
            fit("No needle was answered, so retrieval was not measured.", body.w, 13),
            size=13,
            fill=MUTED,
        )
    size = min(56.0, body.h * 0.45)
    cy = body.y + body.h / 2
    note = "Per-depth grid not stored for this run"
    if errors > 0:
        note += f"{MID}{errors} errored"
    return runs_text(
        body.x,
        cy + size * 0.3,
        [
            Run(fmt.percent(needle.accuracy), size, TEXT, "display", 700),
            Run("of needles found", 15, TEXT_2, gap=12),
        ],
    ) + text(body.x, body.bottom - 2, fit(note, body.w, 13), size=13, fill=MUTED)


_CELL_MIN = 8.0
_CELL_MAX = 36.0


def _needle_heatmap(body: Box, needle: NeedleSummary) -> str:
    """The length by depth grid with its axes and legend, centered as one block.

    Both axes come from every cell, errored ones included, so a length where
    every request failed still has its column. Cells never stretch into a
    progress bar: width is capped relative to height.
    """
    contexts = sorted({c.context_tokens for c in needle.cells})
    depths = sorted({c.depth_percent for c in needle.cells})
    state = {(c.context_tokens, c.depth_percent): c for c in needle.cells}
    label_w = 52.0
    block_extra = 52.0  # axis labels and legend below the grid
    avail_h = body.h - block_extra
    gap = 4.0 if len(depths) <= 8 else 2.0
    cell_h = (avail_h - gap * (len(depths) - 1)) / len(depths)
    if cell_h < _CELL_MIN:
        gap = 1.0
        cell_h = max((avail_h - gap * (len(depths) - 1)) / len(depths), 3.0)
    cell_h = min(cell_h, _CELL_MAX)
    col_gap = 4.0 if len(contexts) <= 12 else 2.0
    avail_w = body.w - label_w
    cell_w = (avail_w - col_gap * (len(contexts) - 1)) / len(contexts)
    cell_w = min(cell_w, max(cell_h * 4, 56.0), 160.0)
    used_w = cell_w * len(contexts) + col_gap * (len(contexts) - 1)
    used_h = cell_h * len(depths) + gap * (len(depths) - 1)
    grid_x = body.x + label_w + (avail_w - used_w) / 2
    top = body.y + max((body.h - used_h - block_extra) / 2, 0)
    out = []
    radius = max(min(4.0, cell_h / 4, cell_w / 4), 1.0)
    label_every = max(1, math.ceil(18 / (cell_h + gap)))
    has_gaps = False
    for r, depth in enumerate(depths):
        y = top + r * (cell_h + gap)
        if (
            r % label_every == 0
            or r == len(depths) - 1
            and (r % label_every) * (cell_h + gap) >= 18
        ):
            out.append(axis_label(grid_x - 10, y + cell_h / 2 + 4, f"{depth * 100:.0f}%"))
        for c, ctx in enumerate(contexts):
            x = grid_x + c * (cell_w + col_gap)
            cell = state.get((ctx, depth))
            if cell is None:
                has_gaps = True
                out.append(rect(x, y, cell_w, cell_h, fill=FAINT, rx=radius, opacity=0.5))
            elif cell.error:
                # The request failed, so this is "not measured", not a miss.
                out.append(rect(x, y, cell_w, cell_h, fill="url(#hatch)", rx=radius))
                out.append(
                    rect(
                        x + 0.5,
                        y + 0.5,
                        cell_w - 1,
                        cell_h - 1,
                        rx=radius,
                        stroke=STROKE,
                        opacity=0.18,
                    )
                )
            elif cell.found:
                out.append(rect(x, y, cell_w, cell_h, fill=PASS, rx=radius, opacity=0.72))
            else:
                out.append(rect(x, y, cell_w, cell_h, fill=FAIL, rx=radius, opacity=0.95))
                if min(cell_w, cell_h) >= 9:
                    # A cross keeps misses readable without relying on red versus green.
                    m = min(cell_w, cell_h) * 0.2
                    mx, my = x + cell_w / 2, y + cell_h / 2
                    out.append(line(mx - m, my - m, mx + m, my + m, stroke="#2A0E0E", width=2))
                    out.append(line(mx - m, my + m, mx + m, my - m, stroke="#2A0E0E", width=2))
    centers = [grid_x + c * (cell_w + col_gap) + cell_w / 2 for c in range(len(contexts))]
    labels = [fmt.tokens(ctx) for ctx in contexts]
    keep = spaced_labels(centers, [mono_width(lbl) for lbl in labels], gap=6)
    label_y = top + used_h + 22
    out.extend(
        axis_label(cx, label_y, lbl, anchor="middle")
        for cx, lbl, k in zip(centers, labels, keep, strict=True)
        if k
    )
    if needle.effective_context:
        passing = [i for i, ctx in enumerate(contexts) if ctx <= needle.effective_context]
        if passing:
            x_end = grid_x + passing[-1] * (cell_w + col_gap) + cell_w
            out.append(rect(grid_x, top + used_h + 6, x_end - grid_x, 2, fill=PASS, rx=1))
    legend_y = label_y + 28
    out.append(_needle_legend(body, legend_y, needle, has_gaps=has_gaps))
    return "".join(out)


def _needle_legend(body: Box, y: float, needle: NeedleSummary, *, has_gaps: bool) -> str:
    """Summary on the right, swatches to its left, and the axis hint only if it still fits.

    The percentage is found over answered cells, so it always agrees with the
    counts beside it; errored cells are counted separately.
    """
    answered = [c for c in needle.cells if not c.error]
    hits = sum(c.found for c in answered)
    errors = needle_errors(needle)
    errored = errors > 0
    if answered:
        counts = f"{hits}/{len(answered)} found"
        pct = fmt.percent(hits / len(answered))
        summaries = [
            f"{counts}{MID}{pct}" + (f"{MID}{errors} errored" if errored else ""),
            counts + (f"{MID}{errors} errored" if errored else ""),
            counts,
        ]
    else:
        summaries = [f"all {errors} cells errored", f"{errors} errored"]
    items = []
    if answered:
        items += [("found", PASS), ("miss", FAIL)]
    if errored:
        items.append(("errored", "url(#hatch)"))
    if has_gaps:
        items.append(("not run", FAINT))
    items_w = sum(13 + mono_width(label) + 12 for label, _ in items)
    summary = summaries[-1]
    for option in summaries:
        if mono_width(option) + 16 + items_w <= body.w:
            summary = option
            break
    summary = fit(summary, body.w, LABEL, font="mono")
    out = [text(body.right, y, summary, size=LABEL, fill=TEXT_2, font="mono", anchor="end")]
    x = body.right - mono_width(summary) - 16 - items_w
    if x >= body.x:
        for label, fill in items:
            out.append(rect(x, y - 9, 9, 9, fill=fill, rx=2, opacity=0.8 if fill == PASS else None))
            if label == "errored":
                out.append(rect(x + 0.5, y - 8.5, 8, 8, rx=2, stroke=STROKE, opacity=0.3))
            out.append(text(x + 13, y, label, size=LABEL, fill=MUTED, font="mono"))
            x += 13 + mono_width(label) + 12
        hint = "depth \u2193  context \u2192"
        if body.x + mono_width(hint) + 16 <= body.right - mono_width(summary) - 16 - items_w:
            out.append(axis_label(body.x, y, hint, anchor="start"))
    return "".join(out)


# --------------------------------------------------------------------------- smarts


ROW_FLOOR = 16.0
ROW_MIN = 20.0
ROW_MAX = 40.0
_SMARTS_TOP = PAD + HEADER_H + 12
_OUTCOME_H = 42.0
_CAP_H = 38.0
_CHIPS_H = 40.0
_SAFETY_CAP = (
    "\u26a0  Safety cap: a safety-critical category scored below 50%",
    "\u26a0  Safety cap: a safety category scored below 50%",
    "\u26a0  Safety-capped",
)


def _category_columns(width: float, count: int) -> int:
    return 2 if width - 2 * PAD >= 760 and count > 6 else 1


def smarts_height(
    data: CardData, width: float, *, show_outcome: bool = True, show_chips: bool = True
) -> tuple[float, float, float]:
    """The (floor, comfortable minimum, maximum) height of the smarts panel at ``width``.

    Past the maximum the category rows would spread out and leave dead space,
    so the layout gives the rest to other panels. The floor packs rows as
    tightly as they stay legible, for cards that are short on room.
    """
    te = data.tool_eval
    if te is None:
        return HINT_H, HINT_H, HINT_H
    rows = max(math.ceil(len(te.categories) / _category_columns(width, len(te.categories))), 1)
    extra = _OUTCOME_H if show_outcome else 0.0
    extra += _CAP_H if te.safety_capped else 0.0
    extra += _CHIPS_H if show_chips else 0.0
    fixed = _SMARTS_TOP + extra + PAD
    return fixed + rows * ROW_FLOOR, fixed + rows * ROW_MIN, fixed + rows * ROW_MAX


def smarts_panel(
    box: Box, data: CardData, *, show_outcome: bool = True, show_chips: bool = True
) -> str:
    """Tool-calling categories, outcomes, and accuracy benchmarks.

    ``show_outcome`` is off when a score block elsewhere already shows the
    pass/partial/fail split, and ``show_chips`` is off when an accuracy panel
    shows the external benchmarks. The safety cap gets its one explanation here.
    """
    title = "SMARTS"
    te = data.tool_eval
    if te is None:
        return missing_panel(
            box,
            title,
            TOOL,
            "Tool calling not scored",
            ("run without --perf-only",),
            note="Scores tool selection, chaining, restraint, and recovery",
            ghost_kind="rows",
        )
    out = [
        frame(box),
        panel_header(
            box,
            title,
            TOOL,
            subtitle=score_details(te),
            subtitle_color=WARN if needs_caution(te) else MUTED,
        ),
    ]
    x = box.x + PAD
    width = box.w - 2 * PAD
    top = box.y + _SMARTS_TOP
    avail = box.bottom - PAD - top
    fixed = (_OUTCOME_H if show_outcome else 0.0) + (_CAP_H if te.safety_capped else 0.0)
    chips = _CHIPS_H if show_chips else 0.0
    rows = max(math.ceil(len(te.categories) / _category_columns(box.w, len(te.categories))), 1)
    rows_h = min(ROW_MAX * rows, max(avail - fixed - chips, ROW_FLOOR * rows))
    if not te.categories:
        rows_h = 28.0
    y = top + max((avail - fixed - chips - rows_h) / 2, 0)
    if show_outcome:
        out.append(outcome_bar(Box(x, y, width, 36), te))
        y += _OUTCOME_H
    if te.safety_capped:
        label = first_fit(list(_SAFETY_CAP), width - 24, 13, weight=600)
        out.append(warn_pill(x, y - 4, label, max_width=width)[0])
        y += _CAP_H
    rows_svg, used = _category_rows(Box(x, y, width, rows_h), te)
    out.append(rows_svg)
    if show_chips:
        out.append(accuracy_chips(x, y + used + 12, width, data))
    return "".join(out)


def outcome_bar(box: Box, te: ToolEvalSummary) -> str:
    """A stacked pass/partial/fail bar with counts."""
    total = max(te.passed + te.partial + te.failed, 1)
    out = [rect(box.x, box.y, box.w, 8, fill=FAINT, rx=4)]
    x = box.x
    for count, color in ((te.passed, PASS), (te.partial, PARTIAL), (te.failed, FAIL)):
        if count <= 0:
            continue
        w = box.w * count / total
        out.append(rect(x, box.y, w, 8, fill=color, rx=4))
        x += w
    lx = box.x
    ly = box.y + 30
    for count, color, label in (
        (te.passed, PASS, "pass"),
        (te.partial, PARTIAL, "partial"),
        (te.failed, FAIL, "fail"),
    ):
        out.append(circle(lx + 4, ly - 4, 4, fill=color))
        chunk = f"{count} {label}"
        out.append(text(lx + 14, ly, chunk, size=13, fill=TEXT_2, font="mono"))
        lx += 14 + text_width(chunk, 13, font="mono") + 20
    return "".join(out)


def _category_rows(box: Box, te: ToolEvalSummary) -> tuple[str, float]:
    """Category bars, top-aligned. Returns markup and the height used."""
    cats = list(te.categories)
    if not cats:
        return text(
            box.x, box.y + 20, "No per-category breakdown for this run", size=13, fill=MUTED
        ), 28.0
    columns = 2 if box.w >= 760 and len(cats) > 6 else 1
    rows = math.ceil(len(cats) / columns)
    col_gap = 48.0
    col_w = (box.w - col_gap * (columns - 1)) / columns
    row_h = max(min(ROW_MAX, box.h / rows), 12.0)
    size = max(12.0, min(15.0, 12 + (row_h - 20) / 6))
    bar_h = max(5.0, min(8.0, row_h / 5))
    values = [fmt.score(cat.percent) for cat in cats]
    pct_w = max(text_width(v, size, font="mono", weight=500) for v in [*values, "100"]) + 12
    out = []
    for i, (cat, value) in enumerate(zip(cats, values, strict=True)):
        col, row = divmod(i, rows)
        x = box.x + col * (col_w + col_gap)
        cy = box.y + row * row_h + row_h / 2
        chip = max(min(size + 10, row_h - 4), 12.0)
        out.append(
            rect(x, cy - chip / 2, chip, chip, fill=STROKE, rx=min(6.0, chip / 3), opacity=0.07)
        )
        baseline = cy + size * 0.36
        out.append(
            text(
                x + chip / 2,
                baseline,
                cat.code,
                size=max(12.0, size - 1),
                fill=TEXT,
                font="mono",
                weight=500,
                anchor="middle",
            )
        )
        bar_w = max(min(col_w * 0.36, 240.0), 60.0)
        name_x = x + chip + 10
        name_room = col_w - chip - 10 - bar_w - pct_w - 16
        out.append(text(name_x, baseline, fit(cat.name, name_room, size), size=size, fill=TEXT_2))
        bar_x = x + col_w - pct_w - bar_w
        pct = max(0.0, min(cat.percent, 100.0)) if math.isfinite(cat.percent) else 0.0
        out.append(rect(bar_x, cy - bar_h / 2, bar_w, bar_h, fill=FAINT, rx=bar_h / 2))
        if pct > 0:
            out.append(
                rect(
                    bar_x,
                    cy - bar_h / 2,
                    max(bar_w * pct / 100, bar_h),
                    bar_h,
                    fill=bar_color(pct),
                    rx=bar_h / 2,
                )
            )
        out.append(
            text(
                x + col_w,
                baseline,
                value,
                size=size,
                fill=TEXT,
                font="mono",
                weight=500,
                anchor="end",
            )
        )
    return "".join(out), rows * row_h


# --------------------------------------------------------------------------- accuracy chips


_CHIP_PAD = 12.0
_CHIP_GAP = 8.0


def _chip_note(result: AccuracyResult, level: int) -> str | None:
    if result.errors > 0:
        if level == 0:
            return accuracy_note(result, detail=False)
        return f"{result.errors:,} errored" if level == 1 else f"{result.errors:,} err"
    if accuracy_is_thin(result):
        return accuracy_note(result, detail=False)
    return None


def _chip_runs(
    result: AccuracyResult, size: float, level: int, name_cap: float | None = None
) -> list[Run]:
    name = result.name
    if name_cap is not None:
        name = fit(name, name_cap, size, font="mono", weight=500)
    finite = math.isfinite(result.accuracy)
    value_color = bar_color(result.accuracy * 100) if finite else MUTED
    runs = [
        Run(name, size, TEXT_2, "mono", 500),
        Run(fmt.percent(result.accuracy), size, value_color, "mono", 500, gap=size * 0.6),
    ]
    note = _chip_note(result, level)
    if note:
        runs.append(Run(note, size - 1, WARN if result.errors else MUTED, "sans", 500, gap=8))
    return runs


def _chips_total(chips: list[list[Run]]) -> float:
    return sum(runs_width(c) + 2 * _CHIP_PAD for c in chips) + _CHIP_GAP * max(len(chips) - 1, 0)


def chip_runs_for(
    results: tuple[AccuracyResult, ...], width: float, size: float
) -> list[list[Run]]:
    """Chip contents that fit ``width``: notes shorten, then names, and only then
    do trailing chips fold into a "+N more" chip, so no result vanishes silently."""
    for level in (0, 1, 2):
        chips = [_chip_runs(r, size, level) for r in results]
        if _chips_total(chips) <= width:
            return chips
    longest = max(text_width(r.name, size, font="mono", weight=500) for r in results)
    floor = text_width("MMLU", size, font="mono", weight=500)
    if _chips_total([_chip_runs(r, size, 2, floor) for r in results]) <= width:
        lo, hi = floor, longest
        for _ in range(20):
            mid = (lo + hi) / 2
            if _chips_total([_chip_runs(r, size, 2, mid) for r in results]) <= width:
                lo = mid
            else:
                hi = mid
        return [_chip_runs(r, size, 2, lo) for r in results]
    for keep in range(len(results) - 1, -1, -1):
        chips = [_chip_runs(r, size, 2, floor) for r in results[:keep]]
        more = [Run(f"+{len(results) - keep} more", size, MUTED, "mono", 500)]
        if _chips_total([*chips, more]) <= width:
            return [*chips, more]
    return [
        [
            Run(
                fit(f"+{len(results)} accuracy", width - 2 * _CHIP_PAD, size, font="mono"),
                size,
                MUTED,
                "mono",
                500,
            )
        ]
    ]


def accuracy_chips(x: float, y: float, width: float, data: CardData, *, large: bool = False) -> str:
    """Accuracy benchmark chips, or one muted hint naming the flags when none ran."""
    if not data.accuracy:
        label = "accuracy  " + "  ".join(_ACCURACY_FLAGS)
        return pill(
            x,
            y,
            label,
            size=LABEL,
            color=MUTED,
            fill=STROKE,
            dashed=True,
            fill_opacity=0.0,
            stroke_opacity=0.25,
            pad=10,
            max_width=width,
        )[0]
    size = 15 if large else 13
    height = 36 if large else 28
    out = []
    px = x
    for runs in chip_runs_for(data.accuracy, width, size):
        w = runs_width(runs) + 2 * _CHIP_PAD
        out.append(_pill_shape(px, y, w, height, STROKE, 0.05, 0.12))
        out.append(runs_text(px + _CHIP_PAD, y + height / 2 + size * 0.36, runs))
        px += w + _CHIP_GAP
    return "".join(out)
