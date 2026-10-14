"""Card composition: header, hero row, panels, and footer for each layout.

Panels declare the height range they can use well, and a small allocator
shares the free space between them. Data panels get spare height first;
anything that was not measured collapses into a compact hint strip that names
the flag to run, so a sparse session is not mostly placeholders.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from tool_eval_bench.domain.card import CardData, CardLayout, ToolEvalSummary
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    base_point,
    cached_ttft_label,
    cap_note,
    concurrency_series,
    deepest_point,
    display_date,
    engine_label,
    has_long_context,
    is_subset,
    needs_caution,
    perf_warnings,
    plottable,
    pp_tg_label,
    preset_label,
    rating_parts,
    scope_label,
    score_details,
    single_stream,
    stream_concurrency,
    suite_tag,
    version_label,
    window_capped,
)
from tool_eval_bench.storage.reports.card._fonts import FAMILIES, text_width
from tool_eval_bench.storage.reports.card._panels import (
    HINT_H,
    MID,
    GhostKind,
    Hero,
    accuracy_height,
    accuracy_min_height,
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
    Run,
    attrs,
    clean,
    esc,
    fit,
    line,
    rect,
    runs_text,
    runs_width,
    text,
    tspan,
)
from tool_eval_bench.storage.reports.card._theme import (
    DECODE,
    GAP,
    MUTED,
    PASS,
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
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f"<title>{title}</title>"
        "<style>text{font-variant-numeric:tabular-nums;font-kerning:normal}</style>"
        f"{defs(width, height)}{background(width, height)}{body}</svg>"
    )
    return scope_ids(svg)


_ID = re.compile(r'id="([A-Za-z][\w-]*)"')
_URL = re.compile(r'="url\(#([A-Za-z][\w-]*)\)"')


def scope_ids(svg: str) -> str:
    """Prefix every id with a hash of the card, so two cards inlined in one page do not collide.

    The prefix is derived from the content, so the same card always renders
    the same bytes. Escaping guarantees ``id="`` and ``="url(#`` only occur in
    markup, never in text from the run.
    """
    prefix = "c" + hashlib.sha256(svg.encode()).hexdigest()[:8]
    ids = set(_ID.findall(svg))
    svg = _ID.sub(lambda m: f'id="{prefix}-{m.group(1)}"', svg)
    return _URL.sub(
        lambda m: f'="url(#{prefix}-{m.group(1)})"' if m.group(1) in ids else m.group(0), svg
    )


# --------------------------------------------------------------------------- allocation


@dataclass(frozen=True)
class Slot:
    """A panel (or a row of panels) and the height range it can use well.

    Space left once every data slot reaches its maximum goes to ``sink`` slots,
    else to data slots that are not ``rigid``, and only then to ``hint`` strips,
    which grow into placeholder panels when there is nothing else to show.
    """

    draw: Callable[[Box], str]
    min_h: float
    max_h: float
    sink: bool = False
    hint: bool = False
    rigid: bool = False
    floor: float | None = None

    @property
    def floor_h(self) -> float:
        return self.min_h if self.floor is None else min(self.floor, self.min_h)


def allocate(available: float, slots: list[Slot]) -> list[float]:
    heights = [s.min_h for s in slots]
    left = available - sum(heights)
    if left < 0:
        # Short on room: every slot gives up a share of its slack above its floor.
        slack = [s.min_h - s.floor_h for s in slots]
        if sum(slack) >= -left:
            ratio = -left / sum(slack)
            return [round(h - d * ratio) for h, d in zip(heights, slack, strict=True)]
        floors = [s.floor_h for s in slots]
        scale = available / sum(floors) if sum(floors) else 0.0
        return [round(h * scale) for h in floors]
    while left > 0.5:
        growable = [i for i, s in enumerate(slots) if not s.hint and heights[i] < s.max_h - 0.01]
        if not growable:
            break
        share = left / len(growable)
        for i in growable:
            add = min(share, slots[i].max_h - heights[i])
            heights[i] += add
            left -= add
    if left > 0.5:
        takers = (
            [i for i, s in enumerate(slots) if s.sink]
            or [i for i, s in enumerate(slots) if not s.hint and not s.rigid]
            or [i for i, s in enumerate(slots) if s.hint]
            or list(range(len(slots)))
        )
        for i in takers:
            heights[i] += left / len(takers)
    return [round(h) for h in heights]


def stack(box: Box, slots: list[Slot]) -> str:
    if not slots:
        return ""
    heights = allocate(box.h - GAP * (len(slots) - 1), slots)
    out = []
    y = box.y
    for slot, h in zip(slots, heights, strict=True):
        out.append(slot.draw(Box(box.x, y, box.w, h)))
        y += h + GAP
    return "".join(out)


def row(slots: list[Slot], weights: list[float] | None = None) -> Slot:
    """Side-by-side slots that share one height."""
    if len(slots) == 1:
        return slots[0]
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
        sink=any(s.sink for s in slots),
        hint=all(s.hint for s in slots),
        rigid=all(s.rigid for s in slots),
        floor=max(s.floor_h for s in slots),
    )


# --------------------------------------------------------------------------- hints


@dataclass(frozen=True)
class Hint:
    """Something the session did not measure and the flag that measures it."""

    title: str
    color: str
    message: str
    flags: tuple[str, ...]
    note: str
    ghost: GhostKind


SPEED_HINT = Hint(
    "SPEED",
    DECODE,
    "Speed not measured",
    ("--perf --perf-preset long",),
    "Decode, prefill, and TTFT by context",
    "line",
)
NEEDLE_HINT = Hint(
    "NEEDLE IN A HAYSTACK",
    PASS,
    "Retrieval not measured",
    ("--needle",),
    "A fact hidden at every depth",
    "heatmap",
)
# Only cards without speed numbers show this hint; with them, the tool-calling
# hero tile says it instead. Those cards come from a ``--gsm8k-only``-style run.
TOOL_HINT = Hint(
    "TOOL CALLING",
    TOOL,
    "Tool calling not scored",
    ("drop --*-only",),
    "Tool choice, chaining, recovery",
    "rows",
)
ACCURACY_HINT = Hint(
    "ACCURACY",
    PREFILL,
    "Accuracy not measured",
    ("--gsm8k", "--mmlu", "--ifeval"),
    "Math, knowledge, instructions",
    "bars",
)
HINT_MIN_W = 300.0


def _hint_slot(hint: Hint) -> Slot:
    return Slot(
        lambda b: missing_panel(
            b,
            hint.title,
            hint.color,
            hint.message,
            hint.flags,
            note=hint.note,
            ghost_kind=hint.ghost,
        ),
        HINT_H,
        HINT_H,
        hint=True,
        rigid=True,
    )


def hint_rows(hints: list[Hint], width: float, *, alone: bool = False) -> list[Slot]:
    """Hint strips, up to three to a row, balanced across rows.

    With no data panel beside them (``alone``) each hint gets its own row, so
    the placeholders they grow into are wide rather than tall.
    """
    if not hints:
        return []
    per_row = 1 if alone else max(1, min(3, int((width + GAP) // (HINT_MIN_W + GAP))))
    rows = -(-len(hints) // per_row)
    size = -(-len(hints) // rows)
    return [row([_hint_slot(h) for h in hints[i : i + size]]) for i in range(0, len(hints), size)]


# --------------------------------------------------------------------------- slots


def _long_slot(data: CardData) -> Slot:
    draw = lambda b: long_context_panel(b, data)  # noqa: E731
    if has_long_context(data):
        return Slot(draw, 300, 460, sink=True, floor=250)
    return Slot(draw, 144, 152, rigid=True)


def _concurrency_slot(data: CardData, *, paired: bool = True) -> Slot:
    """``paired`` means it shares a row with needle data; alone, bars stop growing at 340."""
    draw = lambda b: concurrency_panel(b, data)  # noqa: E731
    if len(concurrency_series(data)) >= 2:
        return Slot(draw, 200, 340, sink=paired, rigid=not paired, floor=172)
    return Slot(draw, 128, 152, rigid=True)


def _needle_slot(data: CardData) -> Slot | None:
    """The needle panel, or None when retrieval was not measured (a hint instead)."""
    if data.needle is None:
        return None
    draw = lambda b: needle_panel(b, data)  # noqa: E731
    if data.needle.cells:
        return Slot(draw, 200, 360, sink=True, floor=172)
    return Slot(draw, 150, 220, floor=128)


def _smarts_slot(
    data: CardData, width: float, *, show_outcome: bool = True, show_chips: bool = True
) -> Slot:
    floor, low, high = smarts_height(data, width, show_outcome=show_outcome, show_chips=show_chips)
    return Slot(
        lambda b: smarts_panel(b, data, show_outcome=show_outcome, show_chips=show_chips),
        low,
        high,
        rigid=True,
        floor=floor,
    )


def _accuracy_slot(data: CardData, *, sink: bool = False, rigid: bool = False) -> Slot:
    count = len(data.accuracy)
    return Slot(
        lambda b: accuracy_panel(b, data),
        accuracy_min_height(count),
        accuracy_height(count),
        sink=sink,
        rigid=rigid,
        floor=accuracy_min_height(count) - 8 * count,
    )


def _score_slot(te: ToolEvalSummary, *, sink: bool = True) -> Slot:
    """``sink=False`` caps the block at its comfortable size; spare room then goes elsewhere."""
    return Slot(lambda b: score_block(b, te), 260, 440, sink=sink, rigid=not sink, floor=240)


def _evidence(data: CardData, width: float, hints: list[Hint]) -> Slot | None:
    """Tool calling and accuracy below the speed panels.

    With tool eval, the smarts panel carries accuracy as chips (or a hint chip
    naming the flags). Without it, accuracy gets its own panel; the tool-calling
    hero tile already says the scenarios were not run, so no second placeholder
    repeats it.
    """
    if data.tool_eval is not None:
        return _smarts_slot(data, width)
    if data.accuracy:
        return _accuracy_slot(data)
    hints.append(ACCURACY_HINT)
    return None


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
        # Without a c=1 measurement the lowest level stands in, and says so.
        level = stream_concurrency(data)
        where = f"depth {fmt.tokens(base.depth)}"
        tg = f"{MID}tg {data.perf_tg:,}" if data.perf_tg else ""
        pp = f"{MID}pp {data.perf_pp:,}" if data.perf_pp else ""
        if level == 1:
            variants = ["single stream"]
        else:
            variants = [f"per request at c={level}", f"at c={level}"]
        if base.tg_estimated:
            variants = [*(f"{v}, estimated" for v in variants), "estimated"]
        heroes.append(
            Hero(
                "DECODE",
                DECODE,
                fmt.rate(base.tg_tps),
                "tok/s",
                tuple(variants),
                sub2=f"{where}{tg}",
            )
        )
        if base.pp_estimated:
            pp_sub: str | tuple[str, ...] = "estimated"
        elif level == 1:
            pp_sub = "prompt processing"
        else:
            pp_sub = (f"prompt processing at c={level}", f"at c={level}")
        heroes.append(
            Hero("PREFILL", PREFILL, fmt.rate(base.pp_tps), "tok/s", pp_sub, sub2=f"{where}{pp}")
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
        heroes.append(Hero(label, TTFT, number, unit, cold, sub2=cached))
    heroes.append(_tool_hero(data))
    return heroes


def _tool_hero(data: CardData) -> Hero:
    te = data.tool_eval
    if te is None:
        return Hero("TOOL CALLING", TOOL, flags=("drop --perf-only",))
    score = fmt.score(te.score)
    if is_subset(te):
        # A rating earned on a few scenarios is not a rating of the model: no stars.
        extra = [f"{te.passed} of {te.total} passed"]
        if te.trials > 1:
            extra.append(f"mean of {te.trials} trials")
        if te.safety_capped:
            extra.append("safety-capped")
        tag = suite_tag(te)
        scopes = (
            scope_label(te),
            *([f"subset: {te.total} of {te.suite_size}{MID}{tag}"] if tag else []),
            scope_label(te, tag=False),
            f"subset: {te.total} of {te.suite_size}",
        )
        return Hero(
            "TOOL CALLING",
            TOOL,
            score,
            "/100",
            sub=tuple(dict.fromkeys(scopes)),
            sub_color=WARN,
            sub2=(MID.join(extra), extra[0]),
        )
    return Hero(
        "TOOL CALLING",
        TOOL,
        score,
        "/100",
        stars=rating_parts(te.rating),
        capped=te.safety_capped,
        sub2=tuple(score_details(te)),
        sub2_color=WARN if needs_caution(te) else MUTED,
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
            body = tspan(org, fill=TEXT_2, fill_opacity=0.62) + esc(line_text[len(org) :])
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


def stamp_runs(data: CardData, room: float) -> list[Run]:
    """The sweep shape beside the preset stamp, and any reason it is partial.

    Warnings outrank the shape: detail parts drop from the end before a
    warning is shortened.
    """
    series = single_stream(data)
    levels = concurrency_series(data)
    parts = [p for p in (pp_tg_label(data),) if p]
    if len(series) > 1:
        parts.append(f"depth {fmt.tokens(series[0].depth)}\u2013{fmt.tokens(series[-1].depth)}")
    if len(levels) > 1:
        parts.append(f"c={levels[0].concurrency}\u2013{levels[-1].concurrency}")
    warnings = perf_warnings(data)
    capped = window_capped(data)
    warn_options = [
        MID.join([*warnings, cap_note(data)] if capped else warnings),
        MID.join([*warnings, f"capped at {fmt.tokens(series[-1].depth)}"] if capped else warnings),
    ]

    def build(keep: int, warn: str) -> list[Run]:
        runs = []
        if keep:
            runs.append(Run(MID.join(parts[:keep]), 13, MUTED, "mono"))
        if warn:
            runs.append(Run(warn, 13, WARN, "mono", 500, gap=12 if keep else 0))
        return runs

    for warn in dict.fromkeys(warn_options):
        for keep in range(len(parts), -1, -1):
            runs = build(keep, warn)
            if not runs or runs_width(runs) <= room:
                return runs
    warn = warn_options[-1]
    return (
        [Run(fit(warn, room, 13, font="mono", weight=500), 13, WARN, "mono", 500)] if warn else []
    )


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
    runs = stamp_runs(data, width - w - 12)
    if not runs:
        return chip
    return chip + runs_text(x + w + 12, y + 19, runs)


def footer_receipt(data: CardData, room: float) -> str:
    """Version, seed, and config fingerprint; the preset lives in the stamp by the speed numbers."""
    rest = []
    if data.seed is not None:
        rest.append(f"seed {data.seed}")
    if data.fingerprint:
        rest.append(f"cfg {clean(data.fingerprint)[:12]}")
    head = f"tool-eval-bench {version_label(clean(data.tool_version))}"
    receipt = MID.join([head, *rest])
    if text_width(receipt, 12, font="mono") <= room:
        return receipt
    head_room = room - text_width(MID + MID.join(rest), 12, font="mono") if rest else room
    head = fit(head, max(head_room, 120.0), 12, font="mono")
    return fit(MID.join([head, *rest]), room, 12, font="mono")


def _footer(x: float, y: float, right: float, data: CardData) -> str:
    room = right - x - text_width(REPO, 12, font="mono") - 32
    receipt = footer_receipt(data, room)
    return (
        line(x, y - 22, right, y - 22, stroke=STROKE, opacity=0.08)
        + text(x, y, receipt, size=12, fill=MUTED, font="mono")
        + text(right, y, REPO, size=12, fill=TEXT_2, font="mono", anchor="end")
    )


# --------------------------------------------------------------------------- portrait


def _portrait(data: CardData, width: int, height: int) -> str:
    margin = 56
    x = margin
    inner = width - 2 * margin
    right = width - margin
    out = [_brand_row(x, 40, right, data)]
    size, lines = _model_lines(data.model, inner, 60, 34, 1)
    out.append(_model_name(x, 114, data, size, lines))
    out.append(_chips(x, 132, inner, data))
    footer_y = height - 32
    bottom = footer_y - 38
    hints: list[Hint] = []
    slots: list[Slot] = []
    if data.perf:
        out.append(_stamp(x, 180, inner, data))
        out.append(_hero_row(_hero_boxes(x, 220, inner, 152, 4), _heroes(data)))
        top = 220 + 152 + GAP
        slots.append(_long_slot(data))
        # Hints sit beside a data panel here: a full-width strip costs height the charts need.
        needle = _needle_slot(data)
        conc = _concurrency_slot(data, paired=needle is not None)
        slots.append(
            row([conc, needle]) if needle else row([conc, _hint_slot(NEEDLE_HINT)], [2, 1])
        )
        if data.tool_eval is not None:
            slots.append(_smarts_slot(data, inner))
        elif data.accuracy:
            # The tool-calling hero already says it was not measured; a second placeholder
            # would only take width from the accuracy bars.
            slots.append(_accuracy_slot(data))
        else:
            hints.append(ACCURACY_HINT)
    else:
        # Without speed numbers the score leads, and accuracy gets its own panel.
        top = 188
        te = data.tool_eval
        hints.append(SPEED_HINT)
        if te is not None and data.accuracy:
            slots.append(
                row([_score_slot(te, sink=False), _accuracy_slot(data, rigid=True)], [3, 2])
            )
        elif te is not None:
            slots.append(_score_slot(te, sink=False))
        elif data.accuracy:
            slots.append(_accuracy_slot(data, sink=True))
        if te is not None:
            slots.append(_smarts_slot(data, inner, show_outcome=False, show_chips=False))
        needle = _needle_slot(data)
        if needle is not None:
            slots.append(needle)
        else:
            hints.append(NEEDLE_HINT)
        if te is None:
            hints.append(TOOL_HINT)
        if not data.accuracy:
            hints.append(ACCURACY_HINT)
    slots += hint_rows(hints, inner, alone=not slots)
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
    footer_y = height - 32
    bottom = footer_y - 38
    top = 40
    out = [_brand_row(x, top, x + left_w, data)]
    size, lines = _model_lines(data.model, left_w, 52, 30, 2)
    first_baseline = top + 36 + size * 0.78
    out.append(_model_name(x, first_baseline, data, size, lines))
    chips_y = first_baseline + (len(lines) - 1) * size * 1.08 + 20
    out.append(_chips(x, chips_y, left_w, data))
    below_chips = chips_y + 32 + 24
    area = Box(area_x, top, area_w, bottom - top)
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
        out.append(_landscape_perf_area(data, area, col_w))
    else:
        out.append(
            _landscape_score_area(data, Box(x, below_chips, left_w, bottom - below_chips), area)
        )
    out.append(_footer(x, footer_y, right, data))
    return "".join(out)


def _landscape_perf_area(data: CardData, area: Box, col_w: float) -> str:
    hints: list[Hint] = []
    needle = _needle_slot(data)
    if has_long_context(data):
        first = Box(area.x, area.y, col_w, area.h)
        second = Box(area.right - col_w, area.y, col_w, area.h)
        slots: list[Slot] = []
        evidence = _evidence(data, col_w, hints)
        if evidence is not None:
            slots.append(evidence)
        if needle is not None:
            slots.append(needle)
        else:
            hints.insert(0, NEEDLE_HINT)
        slots += hint_rows(hints, col_w, alone=not slots)
        return long_context_panel(first, data) + stack(second, slots)
    # Too shallow to chart: the depth strip spans the area, and data panels share the row below.
    if needle is None:
        hints.append(NEEDLE_HINT)
    pair_w = col_w if needle is not None else area.w
    evidence = _evidence(data, pair_w, hints)
    pair = [s for s in (needle, evidence) if s is not None]
    slots = [_long_slot(data)]
    if pair:
        slots.append(row(pair))
    slots += hint_rows(hints, area.w)
    return stack(area, slots)


def _landscape_score_area(data: CardData, left: Box, area: Box) -> str:
    te = data.tool_eval
    hints: list[Hint] = [SPEED_HINT]
    left_slots: list[Slot] = []
    if te is not None:
        left_slots.append(_score_slot(te))
    if data.accuracy:
        left_slots.append(_accuracy_slot(data, sink=te is None))
    left_hints: list[Hint] = []
    if te is None:
        (hints if left_slots else left_hints).append(TOOL_HINT)
    if not data.accuracy:
        (hints if left_slots else left_hints).append(ACCURACY_HINT)
    slots: list[Slot] = []
    if te is not None:
        slots.append(_smarts_slot(data, area.w, show_outcome=False, show_chips=False))
    needle = _needle_slot(data)
    if needle is not None:
        slots.append(needle)
    else:
        hints.insert(1, NEEDLE_HINT)
    slots += hint_rows(hints, area.w, alone=not slots)
    left_slots += hint_rows(left_hints, left.w, alone=not left_slots)
    return stack(left, left_slots) + stack(area, slots)
