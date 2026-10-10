"""Card composition: header, hero row, panels, and footer for each layout.

Panels declare the height range they can use well, and a small allocator
shares the free space between them. Chart panels absorb spare height; content
panels such as the category list stop growing once their rows are comfortable,
so a sparse session never leaves a half-empty panel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tool_eval_bench.domain.card import CardData, CardLayout
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    base_point,
    cached_ttft_label,
    concurrency_series,
    deepest_point,
    display_date,
    engine_label,
    has_long_context,
    is_subset,
    plottable,
    pp_tg_label,
    preset_label,
    rating_parts,
    score_details,
    single_stream,
    version_label,
)
from tool_eval_bench.storage.reports.card._fonts import FAMILIES, text_width
from tool_eval_bench.storage.reports.card._panels import (
    COMPACT_H,
    MID,
    Hero,
    accuracy_height,
    accuracy_panel,
    concurrency_panel,
    hero_tile,
    hero_value_size,
    long_context_panel,
    missing_panel,
    needle_panel,
    pill,
    score_block,
    smarts_height,
    smarts_panel,
)
from tool_eval_bench.storage.reports.card._svg import (
    ELLIPSIS,
    attrs,
    clean,
    esc,
    fit,
    line,
    rect,
    text,
)
from tool_eval_bench.storage.reports.card._theme import (
    DECODE,
    GAP,
    MUTED,
    PREFILL,
    STROKE,
    TEXT,
    TEXT_2,
    TOOL,
    TTFT,
    WARN,
    Box,
    background,
    defs,
)

SIZES: dict[str, tuple[int, int]] = {"portrait": (1080, 1350), "landscape": (1600, 900)}
REPO = "github.com/SeraphimSerapis/tool-eval-bench"


def render(data: CardData, layout: CardLayout) -> str:
    if layout not in SIZES:
        raise ValueError(f"unknown card layout: {layout!r}")
    width, height = SIZES[layout]
    data = plottable(data)
    body = (
        _portrait(data, width, height) if layout == "portrait" else _landscape(data, width, height)
    )
    title = esc(f"{clean(data.model)} benchmark card")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f"<title>{title}</title>"
        "<style>text{font-variant-numeric:tabular-nums;font-kerning:normal}</style>"
        f"{defs(width, height)}{background(width, height)}{body}</svg>"
    )


# --------------------------------------------------------------------------- allocation


@dataclass(frozen=True)
class Slot:
    """A panel (or a row of panels) and the height range it can use well.

    ``sink`` slots take whatever is left once every slot reaches its maximum.
    """

    draw: Callable[[Box], str]
    min_h: float
    max_h: float
    sink: bool = False


def allocate(available: float, slots: list[Slot]) -> list[float]:
    heights = [s.min_h for s in slots]
    left = available - sum(heights)
    if left < 0:
        scale = available / sum(heights) if sum(heights) else 0.0
        return [h * scale for h in heights]
    # Grow every slot evenly toward its maximum, then hand the rest to sinks.
    while left > 0.5:
        growable = [i for i, s in enumerate(slots) if heights[i] < s.max_h - 0.01]
        if not growable:
            break
        share = left / len(growable)
        for i in growable:
            add = min(share, slots[i].max_h - heights[i])
            heights[i] += add
            left -= add
    if left > 0.5:
        sinks = [i for i, s in enumerate(slots) if s.sink] or list(range(len(slots)))
        for i in sinks:
            heights[i] += left / len(sinks)
    return [round(h) for h in heights]


def stack(box: Box, slots: list[Slot]) -> str:
    heights = allocate(box.h - GAP * (len(slots) - 1), slots)
    out = []
    y = box.y
    for slot, h in zip(slots, heights, strict=True):
        out.append(slot.draw(Box(box.x, y, box.w, h)))
        y += h + GAP
    return "".join(out)


def row(slots: list[Slot], weights: list[float] | None = None) -> Slot:
    """Side-by-side slots that share one height."""
    weights = weights or [1.0] * len(slots)

    def draw(box: Box) -> str:
        total = sum(weights)
        free = box.w - GAP * (len(slots) - 1)
        out = []
        x = box.x
        for slot, weight in zip(slots, weights, strict=True):
            w = round(free * weight / total)
            out.append(slot.draw(Box(x, box.y, w, box.h)))
            x += w + GAP
        return "".join(out)

    return Slot(
        draw,
        max(s.min_h for s in slots),
        max(s.max_h for s in slots),
        any(s.sink for s in slots),
    )


def _long_slot(data: CardData, width: float) -> Slot:
    draw = lambda b: long_context_panel(b, data)  # noqa: E731
    if has_long_context(data):
        return Slot(draw, 256, 420, sink=True)
    if width >= 700:
        return Slot(draw, COMPACT_H, COMPACT_H + 24)
    return Slot(draw, 200, 230)


def _concurrency_slot(data: CardData) -> Slot:
    draw = lambda b: concurrency_panel(b, data)  # noqa: E731
    if len(concurrency_series(data)) >= 2:
        return Slot(draw, 220, 340, sink=True)
    return Slot(draw, COMPACT_H, COMPACT_H + 24)


def _needle_slot(data: CardData) -> Slot:
    draw = lambda b: needle_panel(b, data)  # noqa: E731
    if data.needle is not None and data.needle.cells:
        return Slot(draw, 220, 360, sink=True)
    if data.needle is not None:
        return Slot(draw, 150, 220)
    return Slot(draw, COMPACT_H, COMPACT_H + 24, sink=True)


def _smarts_slot(
    data: CardData, width: float, *, show_outcome: bool = True, show_chips: bool = True
) -> Slot:
    low, high = smarts_height(data, width, show_outcome=show_outcome, show_chips=show_chips)
    empty = data.tool_eval is None and not (data.accuracy and show_chips)
    return Slot(
        lambda b: smarts_panel(b, data, show_outcome=show_outcome, show_chips=show_chips),
        low,
        high,
        sink=empty,
    )


def _tool_missing_slot(data: CardData) -> Slot:
    return Slot(
        lambda b: smarts_panel(b, data, show_chips=False), COMPACT_H, COMPACT_H + 40, sink=True
    )


def _accuracy_slot(data: CardData) -> Slot:
    high = accuracy_height(len(data.accuracy)) if data.accuracy else 300
    return Slot(lambda b: accuracy_panel(b, data), min(200, high), max(high, 200))


def _speed_missing_slot() -> Slot:
    return Slot(
        lambda b: missing_panel(
            b,
            "SPEED",
            DECODE,
            "Speed not measured",
            ("--perf --perf-preset long",),
            note="Decode, prefill, TTFT, and long-context scaling",
            ghost_kind="line",
        ),
        COMPACT_H,
        COMPACT_H + 24,
        sink=True,
    )


def _score_slot(data: CardData) -> Slot:
    te = data.tool_eval
    if te is None:
        return Slot(
            lambda b: missing_panel(
                b,
                "TOOL CALLING SCORE",
                TOOL,
                "Tool calling not scored",
                ("run without --perf-only",),
            ),
            COMPACT_H,
            COMPACT_H + 24,
        )
    return Slot(lambda b: score_block(b, te), 240, 420)


# --------------------------------------------------------------------------- shared pieces


def _heroes(data: CardData) -> list[Hero]:
    base = base_point(data)
    deep = deepest_point(data)
    heroes: list[Hero] = []
    if base is None or deep is None:
        heroes += [
            Hero("DECODE", DECODE, flags=("--perf",)),
            Hero("PREFILL", PREFILL, flags=("--perf",)),
            Hero("TIME TO FIRST TOKEN", TTFT, flags=("--perf",)),
        ]
    else:
        where = f"depth {fmt.tokens(base.depth)}"
        tg = f"{MID}tg {data.perf_tg:,}" if data.perf_tg else ""
        pp = f"{MID}pp {data.perf_pp:,}" if data.perf_pp else ""
        heroes.append(
            Hero("DECODE", DECODE, fmt.rate(base.tg_tps), "tok/s", "single stream", f"{where}{tg}")
        )
        pp_sub = "estimated" if base.pp_estimated else "prompt processing"
        heroes.append(
            Hero("PREFILL", PREFILL, fmt.rate(base.pp_tps), "tok/s", pp_sub, f"{where}{pp}")
        )
        number, unit = fmt.duration(deep.ttft_ms)
        label = f"TTFT @ {fmt.tokens(deep.depth)}" if deep.depth else "TIME TO FIRST TOKEN"
        cold = (
            ("cold, full context prefill", "cold, full prefill")
            if deep.depth
            else "cold, full prefill"
        )
        cached = (
            cached_ttft_label(data, fmt.duration_text(deep.cached_ttft_ms))
            if deep.cached_ttft_ms is not None
            else None
        )
        heroes.append(Hero(label, TTFT, number, unit, cold, cached))
    heroes.append(_tool_hero(data))
    return heroes


def _tool_hero(data: CardData) -> Hero:
    te = data.tool_eval
    if te is None:
        return Hero("TOOL CALLING", TOOL, flags=("drop --perf-only",))
    score = max(0.0, min(te.score, 100.0))
    return Hero(
        "TOOL CALLING",
        TOOL,
        f"{score:.0f}",
        "/100",
        stars=rating_parts(te.rating),
        capped=te.safety_capped,
        sub2=tuple(score_details(te)),
        sub2_color=WARN if is_subset(te.total, te.suite_size) else MUTED,
    )


def _hero_boxes(x: float, y: float, width: float, height: float, count: int) -> list[Box]:
    w = (width - GAP * (count - 1)) / count
    return [Box(x + i * (w + GAP), y, w, height) for i in range(count)]


def _hero_row(boxes: list[Box], heroes: list[Hero]) -> str:
    size = hero_value_size(heroes, boxes, max_size=56)
    return "".join(hero_tile(b, h, size) for b, h in zip(boxes, heroes, strict=True))


def _brand_row(x: float, y: float, right: float, data: CardData) -> str:
    out = [rect(x, y, 20, 20, fill="url(#brand)", rx=6)]
    out.append(rect(x + 6, y + 6, 8, 8, fill="#05070D", rx=2, opacity=0.85))
    out.append(
        text(
            x + 32,
            y + 15,
            "TOOL-EVAL-BENCH",
            size=13,
            fill=TEXT_2,
            font="mono",
            weight=500,
            tracking=2.6,
        )
    )
    out.append(
        text(
            right,
            y + 15,
            display_date(data.created_at),
            size=13,
            fill=MUTED,
            font="mono",
            anchor="end",
        )
    )
    return "".join(out)


def _model_lines(
    model: str, width: float, max_size: int, min_size: int, max_lines: int
) -> tuple[int, list[str]]:
    """Largest size that fits on one line, else a two-line break, else an ellipsis."""
    model = clean(model) or "unknown model"

    def w(s: str, size: int) -> float:
        return text_width(s, size, font="display", weight=700, tracking=-1.0)

    for size in range(max_size, min_size - 1, -2):
        if w(model, size) <= width:
            return size, [model]
    if max_lines >= 2:
        breaks = [i + 1 for i, ch in enumerate(model) if ch in "/-_ ." and 0 < i < len(model) - 1]
        for size in range(max_size, min_size - 1, -2):
            best: tuple[float, list[str]] | None = None
            for b in breaks:
                a, c = model[:b].rstrip(), model[b:].lstrip()
                worst = max(w(a, size), w(c, size))
                if worst <= width and (best is None or worst < best[0]):
                    best = (worst, [a, c])
            if best is not None:
                return size, best[1]
        first = ""
        for b in reversed(breaks):
            if w(model[:b], min_size) <= width:
                first = model[:b]
                break
        if not first:
            first = fit(model, width, min_size, font="display", weight=700, tracking=-1.0).rstrip(
                ELLIPSIS
            )
        rest = model[len(first) :].lstrip()
        second = fit(rest, width, min_size, font="display", weight=700, tracking=-1.0, middle=True)
        return min_size, [first.rstrip(), second]
    return min_size, [
        fit(model, width, min_size, font="display", weight=700, tracking=-1.0, middle=True)
    ]


def _model_name(x: float, y: float, data: CardData, size: int, lines: list[str]) -> str:
    """Model name with the organisation prefix (``org/``) dimmed."""
    model = clean(data.model)
    org = model[: model.rfind("/") + 1] if "/" in model else ""
    out = []
    for i, line_text in enumerate(lines):
        baseline = y + i * size * 1.08
        if org and i == 0 and line_text.startswith(org):
            body = f'<tspan fill="{TEXT_2}" fill-opacity="0.62">{esc(org)}</tspan>{esc(line_text[len(org) :])}'
        else:
            body = esc(line_text)
        out.append(
            f"<text {attrs(x=float(x), y=float(baseline), font_family=FAMILIES['display'], font_size=float(size), font_weight=700, fill=TEXT, letter_spacing=-1.0)}>{body}</text>"
        )
    return "".join(out)


CHIP_GAP = 8.0
CHIP_PAD = 14.0
CHIP_MIN = 90.0


def chip_labels(data: CardData, width: float) -> list[str]:
    """Header chip texts that fit ``width`` together, shortened as needed."""
    context = fmt.tokens(data.max_context) if data.max_context else None
    labels = [
        clean(label)
        for label in (
            engine_label(data.engine, data.engine_version),
            data.quantization,
            data.hardware,
            f"{context} context" if context else None,
        )
        if label and clean(label)
    ]

    def chip_w(label: str) -> float:
        return text_width(label, 14, weight=500) + 2 * CHIP_PAD

    def total(items: list[str]) -> float:
        return sum(chip_w(lbl) for lbl in items) + CHIP_GAP * max(len(items) - 1, 0)

    if context and total(labels) > width:
        labels[-1] = f"{context} ctx"
    while labels and total(labels) > width:
        gaps = CHIP_GAP * (len(labels) - 1)
        if sum(min(chip_w(lbl), CHIP_MIN) for lbl in labels) + gaps > width:
            labels.pop()
            continue
        # The widest cap at which every chip still fits, so long labels share the cut.
        lo, hi = CHIP_MIN, max(chip_w(lbl) for lbl in labels)
        for _ in range(24):
            mid = (lo + hi) / 2
            if sum(min(chip_w(lbl), mid) for lbl in labels) + gaps <= width:
                lo = mid
            else:
                hi = mid
        labels = [
            lbl if chip_w(lbl) <= lo else fit(lbl, lo - 2 * CHIP_PAD, 14, weight=500)
            for lbl in labels
        ]
    return labels


def _chips(x: float, y: float, width: float, data: CardData) -> str:
    out = []
    px = x
    for label in chip_labels(data, width):
        chip, w = pill(
            px, y, label, size=14, font="sans", weight=500, color=TEXT_2, height=32, pad=CHIP_PAD
        )
        out.append(chip)
        px += w + CHIP_GAP
    return "".join(out)


def _stamp(x: float, y: float, width: float, data: CardData) -> str:
    """The perf preset stamp, kept beside the speed numbers so short and long cards are not confused."""
    label = preset_label(data)
    if label is None:
        return ""
    chip, w = pill(
        x,
        y,
        label,
        size=12,
        font="mono",
        weight=500,
        color=DECODE,
        fill=DECODE,
        fill_opacity=0.10,
        stroke_opacity=0.45,
        height=28,
        tracking=1.2,
    )
    series = single_stream(data)
    levels = concurrency_series(data)
    parts = [pp_tg_label(data)]
    if len(series) > 1:
        parts.append(f"depth {fmt.tokens(series[0].depth)}\u2013{fmt.tokens(series[-1].depth)}")
    if len(levels) > 1:
        parts.append(f"c={levels[0].concurrency}\u2013{levels[-1].concurrency}")
    detail = MID.join(p for p in parts if p)
    if not detail:
        return chip
    room = width - w - 12
    return chip + text(
        x + w + 12, y + 19, fit(detail, room, 13, font="mono"), size=13, fill=MUTED, font="mono"
    )


def footer_receipt(data: CardData, room: float) -> str:
    """Version, seed, and config fingerprint; the preset lives in the stamp by the speed numbers."""
    rest = []
    if data.seed is not None:
        rest.append(f"seed {data.seed}")
    if data.fingerprint:
        rest.append(f"cfg {clean(data.fingerprint)[:12]}")
    version = clean(data.tool_version)
    heads = [f"tool-eval-bench {version_label(version)}"]
    if "+" in version:
        # The local segment (+g1234abcd) identifies a build; drop it before truncating.
        heads.append(f"tool-eval-bench {version_label(version.split('+', 1)[0])}")
    for head in heads:
        receipt = MID.join([head, *rest])
        if text_width(receipt, 12, font="mono") <= room:
            return receipt
    head_room = room - text_width(MID + MID.join(rest), 12, font="mono") if rest else room
    head = fit(heads[-1], max(head_room, 120.0), 12, font="mono")
    return fit(MID.join([head, *rest]), room, 12, font="mono")


def _footer(x: float, y: float, right: float, data: CardData) -> str:
    room = right - x - text_width(REPO, 12, font="mono") - 32
    receipt = footer_receipt(data, room)
    return (
        line(x, y - 24, right, y - 24, stroke=STROKE, opacity=0.08)
        + text(x, y, receipt, size=12, fill=MUTED, font="mono")
        + text(right, y, REPO, size=12, fill=TEXT_2, font="mono", anchor="end")
    )


# --------------------------------------------------------------------------- portrait


def _portrait(data: CardData, width: int, height: int) -> str:
    margin = 56
    x = margin
    inner = width - 2 * margin
    right = width - margin
    out = [_brand_row(x, 44, right, data)]
    size, lines = _model_lines(data.model, inner, 64, 34, 1)
    out.append(_model_name(x, 124, data, size, lines))
    out.append(_chips(x, 142, inner, data))
    footer_y = height - 40
    bottom = footer_y - 40
    if data.perf:
        out.append(_stamp(x, 186, inner, data))
        out.append(_hero_row(_hero_boxes(x, 226, inner, 152, 4), _heroes(data)))
        top = 226 + 152 + GAP
        slots = [
            _long_slot(data, inner),
            row([_concurrency_slot(data), _needle_slot(data)]),
            _smarts_slot(data, inner),
        ]
    else:
        # Without speed numbers the score leads, and accuracy gets its own panel.
        top = 198
        slots = [
            row([_score_slot(data), _accuracy_slot(data)], [3, 2]),
            _smarts_slot(data, inner, show_outcome=False, show_chips=False),
            row([_speed_missing_slot(), _needle_slot(data)]),
        ]
        if data.tool_eval is None:
            slots.pop(1)
    out.append(stack(Box(x, top, inner, bottom - top), slots))
    out.append(_footer(x, footer_y, right, data))
    return "".join(out)


# --------------------------------------------------------------------------- landscape


def _landscape(data: CardData, width: int, height: int) -> str:
    margin = 48
    x = margin
    right = width - margin
    left_w = 520
    area_x = x + left_w + 24
    area_w = right - area_x
    col_w = (area_w - 24) / 2
    footer_y = height - 36
    bottom = footer_y - 40
    top = 44
    out = [_brand_row(x, top, x + left_w, data)]
    size, lines = _model_lines(data.model, left_w, 52, 30, 2)
    first_baseline = top + 36 + size * 0.78
    out.append(_model_name(x, first_baseline, data, size, lines))
    chips_y = first_baseline + (len(lines) - 1) * size * 1.08 + 20
    out.append(_chips(x, chips_y, left_w, data))
    below_chips = chips_y + 32 + 24
    if data.perf:
        out.append(_stamp(x, below_chips, left_w, data))
        tiles_y = below_chips + 40
        tile_h = 152
        boxes = _hero_boxes(x, tiles_y, left_w, tile_h, 2) + _hero_boxes(
            x, tiles_y + tile_h + GAP, left_w, tile_h, 2
        )
        out.append(_hero_row(boxes, _heroes(data)))
        conc_y = tiles_y + 2 * tile_h + 2 * GAP
        out.append(concurrency_panel(Box(x, conc_y, left_w, bottom - conc_y), data))
        r1 = Box(area_x, top, col_w, bottom - top)
        r2 = Box(area_x + col_w + 24, top, col_w, bottom - top)
        if has_long_context(data):
            out.append(long_context_panel(r1, data))
            out.append(stack(r2, [_smarts_slot(data, col_w), _needle_slot(data)]))
        else:
            out.append(stack(r1, [_long_slot(data, col_w), _needle_slot(data)]))
            if data.tool_eval is None and data.accuracy:
                out.append(stack(r2, [_accuracy_slot(data), _tool_missing_slot(data)]))
            else:
                smarts = _smarts_slot(data, col_w)
                out.append(stack(r2, [Slot(smarts.draw, smarts.min_h, smarts.max_h, sink=True)]))
    else:
        left = Box(x, below_chips, left_w, bottom - below_chips)
        out.append(stack(left, [_score_slot(data), _accuracy_slot(data)]))
        area = Box(area_x, top, area_w, bottom - top)
        slots = [row([_speed_missing_slot(), _needle_slot(data)])]
        if data.tool_eval is not None:
            slots.insert(0, _smarts_slot(data, area_w, show_outcome=False, show_chips=False))
        out.append(stack(area, slots))
    out.append(_footer(x, footer_y, right, data))
    return "".join(out)
