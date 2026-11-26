"""Hero tiles, tool-calling score block, accuracy panels, and smarts panels."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tool_eval_bench.domain.card import (
    AccuracyResult,
    CardData,
    ToolEvalSummary,
)
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    accuracy_is_thin,
    accuracy_note,
    is_subset,
    needs_caution,
    paired_wordings,
    rating_parts,
    scope_label,
    score_details,
)
from tool_eval_bench.storage.reports.card._fonts import text_width
from tool_eval_bench.storage.reports.card._primitives import (
    HEADER_H,
    HINT_H,
    LABEL,
    MID,
    _flag_row,
    _pill_shape,
    bar_color,
    content_box,
    frame,
    missing_panel,
    panel_header,
    pill,
    warn_pill,
)
from tool_eval_bench.storage.reports.card._svg import (
    Run,
    attrs,
    circle,
    first_fit,
    fit,
    rect,
    runs_text,
    runs_width,
    text,
)
from tool_eval_bench.storage.reports.card._theme import (
    FAIL,
    FAINT,
    MUTED,
    PAD,
    PARTIAL,
    PASS,
    PREFILL,
    STROKE,
    TEXT,
    TEXT_2,
    TOOL,
    WARN,
    Box,
)

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
    has_score = math.isfinite(te.score)
    score = fmt.score(te.score) if has_score else "—"
    details_h, bar_gap, bar_h = 58.0, 32.0, 36.0
    size = max(
        48.0, min(132.0, (area_bottom - area_top - details_h - bar_gap - bar_h) / 0.73 * 0.9)
    )

    def score_runs(size: float) -> list[Run]:
        if not has_score:
            return [Run(score, size, MUTED, "display", 700)]
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
    if not has_score:
        out.append(warn_pill(x, base + 13, te.rating or "Withheld", max_width=room)[0])
        details = score_details(te, passed=not ring)
        color = WARN if needs_caution(te) else MUTED
    elif is_subset(te):
        pill_label = paired_wordings(te, [scope_label(te)])[0]
        out.append(warn_pill(x, base + 13, pill_label, max_width=room)[0])
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
