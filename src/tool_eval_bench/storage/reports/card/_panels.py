"""Card panels. Each takes a :class:`Box` and returns SVG markup that stays inside it."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from tool_eval_bench.domain.card import CardData, NeedleSummary, PerfPoint, ToolEvalSummary
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    LONG_CONTEXT_MIN_DEPTH,
    concurrency_series,
    has_long_context,
    is_subset,
    rating_parts,
    scenario_label,
    score_details,
    single_stream,
    ttft_note,
)
from tool_eval_bench.storage.reports.card._fonts import FAMILIES, FontKey, text_width
from tool_eval_bench.storage.reports.card._svg import (
    attrs,
    circle,
    esc,
    first_fit,
    fit,
    line,
    polyline,
    rect,
    text,
    tspan,
)
from tool_eval_bench.storage.reports.card._theme import (
    DECODE,
    FAIL,
    FAINT,
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

# Heights a panel needs in its compact (not measured or summary-only) state.
COMPACT_H = 128.0


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
    shape = attrs(
        x=float(x),
        y=float(y),
        width=float(w),
        height=float(height),
        rx=height / 2,
        fill=fill,
        fill_opacity=fill_opacity,
        stroke=fill,
        stroke_opacity=stroke_opacity,
        stroke_dasharray="4 4" if dashed else None,
    )
    label_y = y + height / 2 + size * 0.36
    body = f"<rect {shape}/>" + text(
        x + pad, label_y, label, size=size, fill=color, font=font, weight=weight, tracking=tracking
    )
    return body, w


def flag_pill(
    x: float, y: float, flag: str, color: str, *, max_width: float | None = None
) -> tuple[str, float]:
    return pill(
        x,
        y,
        flag,
        size=12,
        color=color,
        fill=color,
        fill_opacity=0.08,
        stroke_opacity=0.35,
        height=26,
        pad=10,
        max_width=max_width,
    )


def pills_width(flags: tuple[str, ...]) -> float:
    return sum(text_width(f, 12, font="mono", weight=500) + 20 for f in flags) + 8 * max(
        len(flags) - 1, 0
    )


def panel_header(
    box: Box,
    title: str,
    color: str,
    *,
    subtitle: str | None = None,
    callout: str | None = None,
    callout_label: str | None = None,
    callout_color: str = TEXT,
    right_reserved: float = 0.0,
) -> str:
    """Title on the left and an optional callout on the right, sharing a baseline."""
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
        label_w = text_width(callout_label, 13, weight=500) + 8 if callout_label else 0.0
        number_w = text_width(callout, 26, font="display", weight=700)
        used = number_w + label_w
        out.append(
            text(
                right - label_w,
                base,
                callout,
                size=26,
                fill=callout_color,
                font="display",
                weight=700,
                anchor="end",
            )
        )
        if callout_label:
            out.append(
                text(right, base, callout_label, size=13, fill=MUTED, weight=500, anchor="end")
            )
    if subtitle:
        space = right - used - 16 - (title_x + title_w + 12)
        if space > 60:
            out.append(
                text(title_x + title_w + 12, base, fit(subtitle, space, 13), size=13, fill=MUTED)
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

    Tall boxes center the message over a faint skeleton of the missing chart;
    short ones put the flags in the header row so the panel still reads at a
    glance.
    """
    out = [frame(box, dashed=True)]
    body = content_box(box)
    if box.h < 200:
        flags_w = pills_width(flags)
        out.append(panel_header(box, title, color, right_reserved=flags_w))
        px = box.right - PAD - min(flags_w, body.w)
        for flag in flags:
            chip, w = flag_pill(px, box.y + PAD + 6, flag, color, max_width=body.w)
            out.append(chip)
            px += w + 8
        y = body.y + 6
        out.append(
            text(
                body.x,
                y + 14,
                fit(message, body.w, 17, weight=500),
                size=17,
                fill=TEXT_2,
                weight=500,
            )
        )
        if note and body.h > 44:
            out.append(text(body.x, y + 40, fit(note, body.w, 13), size=13, fill=MUTED))
        return "".join(out)
    out.append(panel_header(box, title, color))
    if ghost_kind:
        out.append(ghost(ghost_kind, body))
    block_h = 24 + 16 + 26 + (30 if note else 0)
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
    px = cx - min(pills_width(flags), body.w) / 2
    for flag in flags:
        chip, w = flag_pill(px, y, flag, color, max_width=body.w)
        out.append(chip)
        px += w + 8
    if note:
        out.append(text(cx, y + 54, fit(note, body.w, 13), size=13, fill=MUTED, anchor="middle"))
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


def tick_duration(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:g} s"
    return f"{ms:g} ms"


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
    return text(x, y, label, size=12, fill=MUTED, font="mono", anchor=anchor)


# --------------------------------------------------------------------------- hero tiles


@dataclass(frozen=True)
class Hero:
    """One hero tile. ``sub`` and ``sub2`` may list alternative wordings, longest first."""

    label: str
    color: str
    value: str = ""
    unit: str = ""
    sub: str | tuple[str, ...] | None = None
    sub2: str | tuple[str, ...] | None = None
    sub2_color: str = MUTED
    stars: tuple[int, str] | None = None
    capped: bool = False
    flags: tuple[str, ...] = ()

    @property
    def missing(self) -> bool:
        return not self.value


def hero_value_size(heroes: list[Hero], boxes: list[Box], max_size: float = 56) -> float:
    """One value size for the whole row, so the numbers read as a set."""
    size = max_size
    for hero, box in zip(heroes, boxes, strict=True):
        if hero.missing:
            continue
        room = box.w - 2 * PAD
        while size > 28:
            w = text_width(hero.value, size, font="display", weight=700, tracking=-0.5)
            if hero.unit:
                w += text_width(hero.unit, round(size * 0.36), weight=600) + size * 0.1
            if w <= room:
                break
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
        px = x
        for flag in hero.flags:
            if px + pills_width((flag,)) > x + room and px > x:
                break
            chip, w = flag_pill(px, sub2_y - 18, flag, hero.color, max_width=x + room - px)
            out.append(chip)
            px += w + 6
        return "".join(out)
    # The value sits midway between the label and the first detail line, by cap height.
    cap = value_size * 0.73
    value_y = (label_y + sub_y - 18) / 2 + cap / 2
    value_w = text_width(hero.value, value_size, font="display", weight=700, tracking=-0.5)
    out.append(
        f"<text {attrs(x=float(x), y=float(value_y), font_family=FAMILIES['display'], font_size=float(value_size), font_weight=700, fill=TEXT, letter_spacing=-0.5)}>{esc(hero.value)}</text>"
    )
    if hero.unit:
        unit_size = round(value_size * 0.36)
        out.append(
            text(
                x + value_w + value_size * 0.1,
                value_y,
                hero.unit,
                size=unit_size,
                fill=hero.color,
                weight=600,
            )
        )
    if hero.stars is not None:
        out.append(_stars(x, sub_y, room, hero.stars, hero.color, capped=hero.capped, size=14))
    elif hero.sub:
        out.append(text(x, sub_y, first_fit(_variants(hero.sub), room, 13), size=13, fill=TEXT_2))
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
    filled, word = stars
    glyphs = tspan("\u2605" * filled, fill=color) + tspan("\u2605" * (5 - filled), fill=FAINT)
    out = [
        f"<text {attrs(x=float(x), y=float(y), font_family=FAMILIES['sans'], font_size=float(size), letter_spacing=1.0)}>{glyphs}</text>"
    ]
    stars_w = text_width("\u2605" * 5, size, tracking=1.0) + 10
    word_size = size - 1
    word_room = room - stars_w
    out.append(
        text(
            x + stars_w,
            y,
            fit(word, word_room, word_size, weight=500),
            size=word_size,
            fill=TEXT_2,
            weight=500,
        )
    )
    if capped:
        wx = (
            x
            + stars_w
            + text_width(fit(word, word_room, word_size, weight=500), word_size, weight=500)
            + 8
        )
        if wx + 50 < x + room:
            out.append(text(wx, y, "capped", size=word_size - 1, fill=WARN, weight=500))
    return "".join(out)


# --------------------------------------------------------------------------- tool score block


def score_block(box: Box, te: ToolEvalSummary) -> str:
    """The big tool-calling score used when no speed numbers lead the card.

    The score, rating, scenario count, and outcome bar form one group centered
    below the label, and a pass-rate ring fills the right side when the block
    is large enough.
    """
    out = [frame(box), rect(box.x + PAD, box.y, 56, 3, fill=TOOL, rx=1.5)]
    x = box.x + PAD
    room = box.w - 2 * PAD
    radius = min((box.h - 2 * PAD) / 2 - 12, 128.0, room * 0.22)
    if radius >= 64 and room - 2 * radius - 48 >= 220:
        cx = box.right - PAD - radius - 8
        cy = box.y + box.h / 2
        out.append(_ring(cx, cy, radius, te))
        room -= 2 * radius + 48
    label_y = box.y + PAD + 12
    out.append(
        text(x, label_y, "TOOL CALLING SCORE", size=13, fill=TEXT_2, weight=600, tracking=1.4)
    )
    area_top, area_bottom = label_y + 16, box.bottom - PAD
    score = f"{max(0.0, min(te.score, 100.0)):.0f}"
    details_h, bar_gap, bar_h = 58.0, 32.0, 36.0
    size = max(
        48.0, min(132.0, (area_bottom - area_top - details_h - bar_gap - bar_h) / 0.73 * 0.9)
    )
    unit_size = round(size * 0.28)
    while (
        size > 48
        and text_width(score, size, font="display", weight=700, tracking=-2)
        + text_width("/100", unit_size, weight=600)
        + 8
        > room
    ):
        size -= 4
        unit_size = round(size * 0.28)
    cap = size * 0.73
    group_h = cap + details_h + bar_gap + bar_h
    base = area_top + max((area_bottom - area_top - group_h) / 2, 0) + cap
    subset = is_subset(te.total, te.suite_size)
    score_w = text_width(score, size, font="display", weight=700, tracking=-2)
    out.append(
        f"<text {attrs(x=float(x - size * 0.04), y=float(base), font_family=FAMILIES['display'], font_size=float(size), font_weight=700, fill=TEXT, letter_spacing=-2.0)}>{esc(score)}</text>"
    )
    out.append(text(x + score_w + 6, base, "/100", size=unit_size, fill=TOOL, weight=600))
    out.append(
        _stars(x, base + 32, room, rating_parts(te.rating), TOOL, capped=te.safety_capped, size=17)
    )
    out.append(
        text(
            x,
            base + 56,
            first_fit(score_details(te, passed=True), room, 14, weight=500),
            size=14,
            fill=WARN if subset else MUTED,
            weight=500,
        )
    )
    out.append(outcome_bar(Box(x, base + details_h + bar_gap, room, bar_h), te))
    return "".join(out)


def _ring(cx: float, cy: float, r: float, te: ToolEvalSummary) -> str:
    """Pass rate as a ring: passes, then partials, starting at twelve o'clock."""
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
    pct = fmt.percent(te.passed / total)
    size = r * 0.42
    out.append(
        text(
            cx,
            cy + size * 0.2,
            pct,
            size=size,
            fill=TEXT,
            font="display",
            weight=700,
            anchor="middle",
        )
    )
    out.append(
        text(cx, cy + size * 0.2 + 24, "scenarios passed", size=13, fill=MUTED, anchor="middle")
    )
    return "".join(out)


ACCURACY_ROW_MAX = 112.0
ACCURACY_ROW_COMFORT = 84.0


def accuracy_height(count: int) -> float:
    """Height at which an accuracy panel's rows are comfortable but not spread out."""
    return PAD + HEADER_H + 16 + count * ACCURACY_ROW_COMFORT + PAD


def accuracy_panel(box: Box, data: CardData) -> str:
    """External accuracy benchmarks as large numbers, for cards led by the tool score."""
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
    out.append(_accuracy_rows(content_box(box), data))
    return "".join(out)


def _accuracy_rows(body: Box, data: CardData) -> str:
    results = list(data.accuracy)
    row_h = min(body.h / len(results), ACCURACY_ROW_MAX)
    top = body.y + max((body.h - row_h * len(results)) / 2, 0)
    size = max(20.0, min(48.0, row_h * 0.46))
    bar_h = 4.0 if row_h < 80 else 6.0
    out = []
    for i, result in enumerate(results):
        y = top + i * row_h
        base = y + row_h / 2 + size * 0.36 - 6
        name = fit(result.name, body.w * 0.45, 15, weight=500)
        out.append(text(body.x, base, name, size=15, fill=TEXT_2, weight=500))
        if result.detail:
            out.append(
                text(body.x, base - 20, fit(result.detail, body.w * 0.45, 12), size=12, fill=MUTED)
            )
        value = fmt.percent(result.accuracy)
        out.append(
            text(
                body.right,
                base,
                value,
                size=size,
                fill=TEXT,
                font="display",
                weight=700,
                anchor="end",
            )
        )
        bar_y = base + 12
        if row_h >= 40 and bar_y + bar_h <= y + row_h:
            out.append(rect(body.x, bar_y, body.w, bar_h, fill=FAINT, rx=bar_h / 2))
            frac = max(0.0, min(result.accuracy, 1.0))
            if frac > 0:
                out.append(
                    rect(
                        body.x, bar_y, max(body.w * frac, bar_h), bar_h, fill=PREFILL, rx=bar_h / 2
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
        return depth_strip(box, series)
    out = [
        frame(box),
        panel_header(box, title, DECODE, subtitle="single stream (c=1) by context depth"),
    ]
    body = content_box(box, extra=-4)
    note = ttft_note(data)
    body = Box(body.x, body.y, body.w, body.h - 24)
    out.append(text(body.x, body.bottom + 22, first_fit(note, body.w, 12), size=12, fill=MUTED))
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
        half = (body.h - 40) / 2
        first = Box(body.x, body.y, body.w, half)
        second = Box(body.x, body.y + half + 40, body.w, half)
    out.append(_decode_chart(first, series))
    out.append(_ttft_chart(second, series))
    return "".join(out)


def depth_strip(box: Box, series: list[PerfPoint]) -> str:
    """Short sweeps: the measured depths as numbers, plus how to get the chart."""
    flags = ("--perf --perf-preset long",)
    out = [frame(box, dashed=True)]
    out.append(panel_header(box, "LONG CONTEXT", DECODE, right_reserved=pills_width(flags)))
    px = box.right - PAD - pills_width(flags)
    out.append(flag_pill(px, box.y + PAD + 6, flags[0], DECODE)[0])
    body = content_box(box)
    deepest = series[-1].depth if series else 0
    if len(series) <= 1:
        message = f"Measured at depth {fmt.tokens(deepest)} only"
    else:
        message = f"Sweep stopped at {fmt.tokens(deepest)}"
    notes = (
        f"The chart needs depths past {fmt.tokens(LONG_CONTEXT_MIN_DEPTH)}.",
        "The long preset sweeps to 128K.",
    )
    msg_w = max(text_width(message, 17, weight=500), text_width(notes[0], 13), 220.0)
    out.append(
        text(
            body.x,
            body.y + 14,
            fit(message, body.w, 17, weight=500),
            size=17,
            fill=TEXT_2,
            weight=500,
        )
    )
    col_w = 132.0
    beside = body.x + msg_w + 40
    if len(series) > 1 and body.right - beside >= col_w * len(series):
        out.append(_depth_stats(beside, body.y - 2, col_w, series))
        room = beside - body.x - 24
        out.append(text(body.x, body.y + 38, fit(notes[0], room, 13), size=13, fill=MUTED))
        out.append(text(body.x, body.y + 56, fit(notes[1], room, 13), size=13, fill=MUTED))
    elif len(series) > 1 and body.h >= 110 and body.w >= col_w * len(series):
        out.append(text(body.x, body.y + 40, fit(" ".join(notes), body.w, 13), size=13, fill=MUTED))
        out.append(
            _depth_stats(body.x, body.y + 68, min(col_w * 1.2, body.w / len(series)), series)
        )
    else:
        out.append(text(body.x, body.y + 40, fit(" ".join(notes), body.w, 13), size=13, fill=MUTED))
    return "".join(out)


def _depth_stats(x: float, y: float, col_w: float, series: list[PerfPoint]) -> str:
    out = []
    for i, p in enumerate(series):
        cx = x + i * col_w
        out.append(
            text(cx, y + 6, f"depth {fmt.tokens(p.depth)}", size=12, fill=MUTED, font="mono")
        )
        rate = fmt.rate(p.tg_tps)
        out.append(text(cx, y + 32, rate, size=22, fill=TEXT, font="display", weight=700))
        rate_w = text_width(rate, 22, font="display", weight=700)
        out.append(text(cx + rate_w + 5, y + 32, "tok/s", size=12, fill=DECODE, weight=600))
        out.append(
            text(
                cx, y + 52, f"TTFT {fmt.duration_text(p.ttft_ms)}", size=12, fill=TTFT, font="mono"
            )
        )
    return "".join(out)


def _chart_heading(
    box: Box, label: str, color: str, callout: str, callout_label: str
) -> tuple[str, float]:
    """Series name on the left, the takeaway on the right. Returns markup and the label's right edge."""
    base = box.y + 22
    out = [circle(box.x + 5, base - 5, 5, fill=color)]
    out.append(text(box.x + 18, base, label, size=15, fill=TEXT, weight=600))
    cl_w = text_width(callout_label, 13, weight=500)
    out.append(text(box.right, base, callout_label, size=13, fill=MUTED, weight=500, anchor="end"))
    out.append(
        text(
            box.right - cl_w - 8,
            base + 1,
            callout,
            size=26,
            fill=color,
            font="display",
            weight=700,
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
    widths = [text_width(lbl, 12, font="mono") for lbl in labels]
    keep = spaced_labels(xs, widths)
    out = [
        axis_label(x, plot.bottom + 22, lbl, anchor="middle")
        for x, lbl, k in zip(xs, labels, keep, strict=True)
        if k
    ]
    return xs, "".join(out)


def _decode_chart(box: Box, series: list[PerfPoint]) -> str:
    base, last = series[0], series[-1]
    change = last.tg_tps / base.tg_tps - 1 if base.tg_tps > 0 else 0.0
    heading, _ = _chart_heading(
        box, "Decode tok/s", DECODE, fmt.signed_percent(change), f"at {fmt.tokens(last.depth)}"
    )
    out = [heading]
    plot = Box(box.x + 44, box.y + 52, box.w - 44, box.h - 52 - 30)
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
        w = text_width(label, 12, font="mono", weight=500)
        lx = x + 8 if anchor == "start" else x - 8
        x0, x1 = (lx, lx + w) if anchor == "start" else (lx - w, lx)
        extent = _span_extent(lines, x0 - 2, x1 + 2) or (y, y)
        top_line, bottom_line = min(extent[0], y), max(extent[1], y)
        above = top_line - 8
        below = bottom_line + 18
        fits_above = above - 10 >= plot.y - 18
        fits_below = below <= plot.bottom + 6
        if fits_above and (not fits_below or y - above <= below - y):
            ly = above
        elif fits_below:
            ly = below
        else:
            ly = max(above, plot.y - 8)
        out.append(
            text(lx, ly, label, size=12, fill=TEXT_2, font="mono", weight=500, anchor=anchor)
        )
    return "".join(out)


def _ttft_chart(box: Box, series: list[PerfPoint]) -> str:
    last = series[-1]
    cached = [p for p in series if p.cached_ttft_ms is not None]
    heading, label_right = _chart_heading(
        box, "TTFT", TTFT, fmt.duration_text(last.ttft_ms), f"cold at {fmt.tokens(last.depth)}"
    )
    out = [heading]
    if cached:
        # The cached line matters to agents that re-send a long prefix, so its
        # latest value goes in the legend rather than fighting the lines for room.
        warm_text = f"cached {fmt.duration_text(cached[-1].cached_ttft_ms or 0.0)}"
        lx = label_right + 16
        ly = box.y + 22
        out.append(line(lx, ly - 4, lx + 16, ly - 4, stroke=TTFT_CACHED, width=2, dash="4 3"))
        out.append(text(lx + 22, ly, warm_text, size=12, fill=TTFT_CACHED, font="mono", weight=500))
    plot = Box(box.x + 52, box.y + 52, box.w - 52, box.h - 52 - 30)
    values = [p.ttft_ms for p in series] + [
        p.cached_ttft_ms for p in cached if p.cached_ttft_ms is not None
    ]
    lo, hi = log_decades(values)
    step = log_step(hi - lo, plot.h)

    def y_of(v: float) -> float:
        v = max(v, 10.0**lo)
        return plot.bottom - (math.log10(v) - lo) / (hi - lo) * plot.h

    for k in range(lo, hi + 1, step):
        out.append(grid_line(plot.x, plot.right, y_of(10.0**k)))
        out.append(axis_label(plot.x - 10, y_of(10.0**k) + 4, tick_duration(10.0**k)))
    xs, labels = _depth_axis(plot, [p.depth for p in series])
    out.append(labels)
    cold = [(x, y_of(p.ttft_ms)) for x, p in zip(xs, series, strict=True)]
    area = [(cold[0][0], plot.bottom), *cold, (cold[-1][0], plot.bottom)]
    out.append(f'<polygon points="{polyline(area)}" fill="url(#ttftArea)"/>')
    if cached:
        warm = [
            (x, y_of(p.cached_ttft_ms))
            for x, p in zip(xs, series, strict=True)
            if p.cached_ttft_ms is not None
        ]
        out.append(
            f'<polyline points="{polyline(warm)}" fill="none" stroke="{TTFT_CACHED}" stroke-width="2" stroke-dasharray="5 4" stroke-linejoin="round" opacity="0.9"/>'
        )
        for x, y in warm:
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
            obstacles=[warm] if cached else None,
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
            f"Single stream only: {fmt.rate(only.tg_tps_total)} tok/s",
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
            callout_label="tok/s peak",
        ),
    ]
    body = content_box(box, extra=-10)
    legend_y = body.y + 6
    out.append(rect(body.x, legend_y - 9, 10, 10, fill="url(#bar)", rx=2))
    out.append(text(body.x + 16, legend_y, "aggregate", size=12, fill=MUTED, font="mono"))
    lx = body.x + 16 + text_width("aggregate", 12, font="mono") + 16
    out.append(circle(lx + 4, legend_y - 4, 4, fill=TEXT))
    out.append(text(lx + 14, legend_y, "per request", size=12, fill=MUTED, font="mono"))
    speedup = peak.tg_tps_total / single.tg_tps_total if single.tg_tps_total > 0 else 0.0
    if speedup > 0 and peak.concurrency != single.concurrency:
        out.append(
            text(
                body.right,
                legend_y,
                f"{speedup:.1f}\u00d7 at c={peak.concurrency}",
                size=12,
                fill=PREFILL,
                font="mono",
                weight=500,
                anchor="end",
            )
        )
    plot = Box(body.x + 40, body.y + 30, body.w - 40, body.h - 30 - 24)
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
    show_values = all(
        text_width(fmt.rate(p.tg_tps_total), 12, font="mono") <= slot - 4 for p in series
    )
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
                    size=12,
                    fill=TEXT_2,
                    font="mono",
                    weight=500,
                    anchor="middle",
                )
            )
        label = f"c={p.concurrency}"
        if text_width(label, 12, font="mono") > slot - 6:
            label = str(p.concurrency)
        out.append(axis_label(cx, plot.bottom + 22, label, anchor="middle"))
    per_request = [(cx, y_of(p.tg_tps)) for cx, p in zip(centers, series, strict=True)]
    out.append(
        f'<polyline points="{polyline(per_request)}" fill="none" stroke="{TEXT}" stroke-width="1.5" stroke-opacity="0.5" stroke-dasharray="3 4"/>'
    )
    for x, y in per_request:
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
    effective = fmt.tokens(needle.effective_context) if needle.effective_context else "none"
    out = [
        frame(box),
        panel_header(
            box,
            title,
            PASS,
            callout=effective,
            callout_label="effective context",
            callout_color=PASS if needle.effective_context else FAIL,
        ),
    ]
    body = content_box(box)
    if not needle.cells:
        out.append(_needle_summary(body, needle))
    else:
        out.append(_needle_heatmap(body, needle))
    return "".join(out)


def _needle_summary(body: Box, needle: NeedleSummary) -> str:
    """Legacy runs stored only the totals, so show those plainly."""
    size = min(56.0, body.h * 0.45)
    cy = body.y + body.h / 2
    note = "Per-depth grid not stored for this run"
    if needle.errors > 0:
        note += f"{MID}{needle.errors} errored"
    return (
        text(
            body.x,
            cy + size * 0.3,
            fmt.percent(needle.accuracy),
            size=size,
            fill=TEXT,
            font="display",
            weight=700,
        )
        + text(
            body.x
            + text_width(fmt.percent(needle.accuracy), size, font="display", weight=700)
            + 12,
            cy + size * 0.3,
            "of needles found",
            size=15,
            fill=TEXT_2,
        )
        + text(body.x, body.bottom - 2, fit(note, body.w, 12), size=12, fill=MUTED)
    )


def _needle_heatmap(body: Box, needle: NeedleSummary) -> str:
    contexts = sorted({c.context_tokens for c in needle.cells})
    depths = sorted({c.depth_percent for c in needle.cells})
    found = {(c.context_tokens, c.depth_percent): c.found for c in needle.cells}
    label_w = 44.0
    grid = Box(body.x + label_w, body.y, body.w - label_w, body.h - 50)
    gap = 4.0 if len(contexts) <= 12 else 2.0
    cell_w = (grid.w - gap * (len(contexts) - 1)) / len(contexts)
    cell_h = min((grid.h - gap * (len(depths) - 1)) / len(depths), 36.0)
    used_h = cell_h * len(depths) + gap * (len(depths) - 1)
    top = grid.y + max((grid.h - used_h) / 2, 0)
    out = []
    row_step = 1 if cell_h >= 14 else 2
    radius = max(min(4.0, cell_h / 4, cell_w / 4), 1.0)
    for r, depth in enumerate(depths):
        y = top + r * (cell_h + gap)
        if r % row_step == 0 or r == len(depths) - 1:
            out.append(axis_label(grid.x - 10, y + cell_h / 2 + 4, f"{depth * 100:.0f}%"))
        for c, ctx in enumerate(contexts):
            x = grid.x + c * (cell_w + gap)
            state = found.get((ctx, depth))
            if state is None:
                # No stored result: the request errored, so this is "not measured", not a miss.
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
            elif state:
                out.append(rect(x, y, cell_w, cell_h, fill=PASS, rx=radius, opacity=0.72))
            else:
                out.append(rect(x, y, cell_w, cell_h, fill=FAIL, rx=radius, opacity=0.95))
                # A cross keeps misses readable without relying on red versus green.
                m = min(cell_w, cell_h) * 0.2
                mx, my = x + cell_w / 2, y + cell_h / 2
                out.append(line(mx - m, my - m, mx + m, my + m, stroke="#2A0E0E", width=2))
                out.append(line(mx - m, my + m, mx + m, my - m, stroke="#2A0E0E", width=2))
    centers = [grid.x + c * (cell_w + gap) + cell_w / 2 for c in range(len(contexts))]
    labels = [fmt.tokens(ctx) for ctx in contexts]
    keep = spaced_labels(centers, [text_width(lbl, 12, font="mono") for lbl in labels], gap=6)
    label_y = top + used_h + 22
    out.extend(
        axis_label(cx, label_y, lbl, anchor="middle")
        for cx, lbl, k in zip(centers, labels, keep, strict=True)
        if k
    )
    if needle.effective_context:
        passing = [i for i, ctx in enumerate(contexts) if ctx <= needle.effective_context]
        if passing:
            x_end = grid.x + passing[-1] * (cell_w + gap) + cell_w
            out.append(rect(grid.x, top + used_h + 6, x_end - grid.x, 2, fill=PASS, rx=1))
    has_gaps = len(found) < len(contexts) * len(depths)
    out.append(_needle_legend(body, needle, has_gaps=has_gaps))
    return "".join(out)


def _needle_legend(body: Box, needle: NeedleSummary, *, has_gaps: bool) -> str:
    """Summary on the right, swatches to its left, and the axis hint only if it still fits."""
    y = body.bottom - 2
    hits = sum(c.found for c in needle.cells)
    counts = f"{hits}/{len(needle.cells)} found"
    errored = f"{MID}{needle.errors} errored" if needle.errors > 0 else ""
    items = [("found", PASS), ("miss", FAIL)] + ([("not run", "url(#hatch)")] if has_gaps else [])

    def mono(s: str) -> float:
        return text_width(s, 12, font="mono")

    items_w = sum(13 + mono(label) + 12 for label, _ in items)
    summary = f"{counts}{errored}{MID}{fmt.percent(needle.accuracy)}"
    if mono(summary) + 16 + items_w > body.w:
        summary = f"{counts}{errored}"
    out = [text(body.right, y, summary, size=12, fill=TEXT_2, font="mono", anchor="end")]
    x = body.right - mono(summary) - 16 - items_w
    if x >= body.x:
        for label, fill in items:
            out.append(rect(x, y - 9, 9, 9, fill=fill, rx=2, opacity=0.8 if fill == PASS else None))
            if label == "not run":
                out.append(rect(x + 0.5, y - 8.5, 8, 8, rx=2, stroke=STROKE, opacity=0.3))
            out.append(text(x + 13, y, label, size=12, fill=MUTED, font="mono"))
            x += 13 + mono(label) + 12
        hint = "depth \u2193  context \u2192"
        if body.x + mono(hint) + 16 <= body.right - mono(summary) - 16 - items_w:
            out.append(axis_label(body.x, y, hint, anchor="start"))
    return "".join(out)


# --------------------------------------------------------------------------- smarts


_ACCURACY_FLAGS = ("--gsm8k", "--mmlu", "--ifeval")


def _bar_color(percent: float) -> str:
    if percent >= 85:
        return PASS
    if percent >= 60:
        return DECODE
    if percent >= 40:
        return PARTIAL
    return FAIL


ROW_MIN = 21.0
ROW_MAX = 40.0
_SMARTS_TOP = PAD + HEADER_H + 12
_OUTCOME_H = 42.0
_CHIPS_H = 40.0


def _category_columns(width: float, count: int) -> int:
    return 2 if width - 2 * PAD >= 760 and count > 6 else 1


def smarts_height(
    data: CardData, width: float, *, show_outcome: bool = True, show_chips: bool = True
) -> tuple[float, float]:
    """The (smallest, largest) height the smarts panel can use well at ``width``.

    Past the largest height the category rows would spread out and leave dead
    space, so the layout gives the rest to other panels.
    """
    te = data.tool_eval
    if te is None:
        if data.accuracy and show_chips:
            return COMPACT_H, accuracy_height(len(data.accuracy)) + 28
        return COMPACT_H, COMPACT_H + 40
    rows = max(math.ceil(len(te.categories) / _category_columns(width, len(te.categories))), 1)
    extra = _OUTCOME_H if show_outcome else (36.0 if te.safety_capped else 0.0)
    extra += _CHIPS_H if show_chips else 0.0
    fixed = _SMARTS_TOP + extra + PAD
    return fixed + rows * ROW_MIN, fixed + rows * ROW_MAX


def smarts_panel(
    box: Box, data: CardData, *, show_outcome: bool = True, show_chips: bool = True
) -> str:
    """Tool-calling categories, outcomes, and accuracy benchmarks.

    ``show_outcome`` is off when a score block elsewhere already shows the
    pass/partial/fail split, and ``show_chips`` is off when an accuracy panel
    shows the external benchmarks.
    """
    title = "SMARTS"
    te = data.tool_eval
    if te is None:
        if not data.accuracy or not show_chips:
            return missing_panel(
                box,
                title,
                TOOL,
                "Tool calling not scored",
                ("run without --perf-only",),
                note="Scores tool selection, chaining, restraint, and recovery",
                ghost_kind="rows",
            )
        hint = "Tool calling not scored. Run without --perf-only."
        out = [frame(box), panel_header(box, title, TOOL, subtitle="accuracy benchmarks")]
        body = content_box(box)
        if box.h >= 200:
            out.append(_accuracy_rows(Box(body.x, body.y, body.w, body.h - 28), data))
        else:
            out.append(accuracy_chips(body.x, body.y, body.w, data, large=True))
        if body.h > 70:
            out.append(text(body.x, body.bottom - 2, fit(hint, body.w, 13), size=13, fill=MUTED))
        return "".join(out)
    subtitle = scenario_label(te.total, te.suite_size) + (f"{MID}hard mode" if te.hardmode else "")
    if te.trials > 1:
        subtitle += f"{MID}mean of {te.trials} trials"
    out = [frame(box), panel_header(box, title, TOOL, subtitle=subtitle)]
    x = box.x + PAD
    width = box.w - 2 * PAD
    y = box.y + _SMARTS_TOP
    if show_outcome:
        out.append(outcome_bar(Box(x, y, width, 36), te))
        y += _OUTCOME_H
    elif te.safety_capped:
        out.append(_safety_pill(x, y - 6, width))
        y += 36
    room = box.bottom - PAD - y - (_CHIPS_H if show_chips else 0.0)
    rows_h = _category_rows(Box(x, y, width, room), te)
    out.append(rows_h[0])
    if show_chips:
        out.append(accuracy_chips(x, y + rows_h[1] + 12, width, data))
    return "".join(out)


def _safety_pill(x: float, y: float, room: float) -> str:
    return pill(
        x,
        y,
        "\u26a0  Safety cap: a safety-critical category scored below 50%",
        size=12,
        font="sans",
        weight=500,
        color=WARN,
        fill=WARN,
        fill_opacity=0.10,
        stroke_opacity=0.45,
        height=26,
        max_width=room,
    )[0]


def outcome_bar(box: Box, te: ToolEvalSummary) -> str:
    """A stacked pass/partial/fail bar with counts, plus the safety-cap flag when set."""
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
    if te.safety_capped:
        room = box.right - lx
        label = "\u26a0  Safety cap: a safety-critical category scored below 50%"
        if room < text_width(label, 12, weight=500) + 24:
            label = "\u26a0  Safety-capped"
        w = text_width(label, 12, font="sans", weight=500) + 24
        out.append(
            pill(
                box.right - min(w, max(room, 60.0)),
                ly - 18,
                label,
                size=12,
                font="sans",
                weight=500,
                color=WARN,
                fill=WARN,
                fill_opacity=0.10,
                stroke_opacity=0.45,
                height=26,
                max_width=max(room, 60.0),
            )[0]
        )
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
    size = max(11.0, min(15.0, 12 + (row_h - 20) / 6))
    bar_h = max(5.0, min(8.0, row_h / 5))
    out = []
    for i, cat in enumerate(cats):
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
                size=size - 1,
                fill=TEXT,
                font="mono",
                weight=500,
                anchor="middle",
            )
        )
        pct_w = text_width("100", size, font="mono", weight=500) + 12
        bar_w = max(min(col_w * 0.36, 240.0), 60.0)
        name_x = x + chip + 10
        name_room = col_w - chip - 10 - bar_w - pct_w - 16
        out.append(text(name_x, baseline, fit(cat.name, name_room, size), size=size, fill=TEXT_2))
        bar_x = x + col_w - pct_w - bar_w
        pct = max(0.0, min(cat.percent, 100.0))
        out.append(rect(bar_x, cy - bar_h / 2, bar_w, bar_h, fill=FAINT, rx=bar_h / 2))
        if pct > 0:
            out.append(
                rect(
                    bar_x,
                    cy - bar_h / 2,
                    max(bar_w * pct / 100, bar_h),
                    bar_h,
                    fill=_bar_color(pct),
                    rx=bar_h / 2,
                )
            )
        out.append(
            text(
                x + col_w,
                baseline,
                f"{pct:.0f}",
                size=size,
                fill=TEXT,
                font="mono",
                weight=500,
                anchor="end",
            )
        )
    return "".join(out), rows * row_h


def _chip_labels(data: CardData) -> list[str]:
    return [f"{r.name}  {fmt.percent(r.accuracy)}" for r in data.accuracy]


def accuracy_chips(x: float, y: float, width: float, data: CardData, *, large: bool = False) -> str:
    """Accuracy benchmark chips, or one muted hint naming the flags when none ran."""
    if not data.accuracy:
        label = "accuracy  " + "  ".join(_ACCURACY_FLAGS)
        return pill(
            x,
            y,
            label,
            size=12,
            color=MUTED,
            fill=STROKE,
            dashed=True,
            fill_opacity=0.0,
            stroke_opacity=0.25,
            pad=10,
            max_width=width,
        )[0]
    out = []
    px = x
    size = 15 if large else 13
    for label in _chip_labels(data):
        need = text_width(label, size, font="mono", weight=500) + 24
        if px + need > x + width:
            break
        name, _, value = label.partition("  ")
        chip, w = pill(px, y, label, size=size, color=TEXT, height=36 if large else 28)
        # Dim the benchmark name so the number carries the chip.
        chip = chip.replace(
            f">{esc(label)}</text>", f">{esc(name)}  {tspan(value, fill=TOOL)}</text>"
        )
        out.append(chip)
        px += w + 8
    return "".join(out)
