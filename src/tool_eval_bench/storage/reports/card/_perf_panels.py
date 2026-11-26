"""Long-context throughput charts and concurrency scaling panel."""

from __future__ import annotations

import math

from tool_eval_bench.domain.card import CardData, PerfPoint
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    cap_note,
    concurrency_series,
    has_long_context,
    single_stream,
    stream_concurrency,
    ttft_note,
    window_capped,
)
from tool_eval_bench.storage.reports.card._fonts import text_width
from tool_eval_bench.storage.reports.card._primitives import (
    DOT_FILL,
    LABEL,
    _flag_row,
    axis_label,
    content_box,
    depth_position,
    frame,
    grid_line,
    log_decades,
    log_step,
    missing_panel,
    mono_width,
    nice_ticks,
    panel_header,
    pills_width,
    spaced_labels,
    time_ticks,
)
from tool_eval_bench.storage.reports.card._svg import (
    Run,
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
    MUTED,
    PAD,
    PREFILL,
    STROKE,
    TEXT,
    TEXT_2,
    TTFT,
    TTFT_CACHED,
    WARN,
    Box,
)

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
