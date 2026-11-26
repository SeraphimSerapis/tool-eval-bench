"""Needle-in-a-Haystack retrieval heatmap and summary panel."""

from __future__ import annotations

import math

from tool_eval_bench.domain.card import CardData, NeedleSummary
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._primitives import (
    LABEL,
    MID,
    axis_label,
    content_box,
    frame,
    missing_panel,
    mono_width,
    panel_header,
    spaced_labels,
)
from tool_eval_bench.storage.reports.card._svg import Run, fit, line, rect, runs_text, text
from tool_eval_bench.storage.reports.card._theme import (
    FAIL,
    FAINT,
    MUTED,
    PASS,
    STROKE,
    TEXT,
    TEXT_2,
    Box,
)

_CELL_MIN = 8.0
_CELL_MAX = 36.0


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
