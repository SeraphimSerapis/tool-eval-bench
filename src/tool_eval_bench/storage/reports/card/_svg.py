"""Minimal SVG building blocks.

Every piece of text that reaches the document goes through :func:`esc`, which
also strips characters XML 1.0 cannot carry. Attribute values are escaped the
same way, so an untrusted model name cannot open a tag or an attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from tool_eval_bench.storage.reports.card._fonts import FAMILIES, FontKey, text_width

# Control characters and lone surrogates make a document unparseable. A
# translate table rather than a regex class, whose control-character ranges
# static analysis reads as accidental.
_INVALID_XML = dict.fromkeys(
    [
        *range(0x00, 0x09),
        0x0B,
        0x0C,
        *range(0x0E, 0x20),
        0x7F,
        *range(0xD800, 0xE000),
        0xFFFE,
        0xFFFF,
    ]
)
ELLIPSIS = "\u2026"


def clean(text: str) -> str:
    """Drop characters XML cannot represent and collapse whitespace runs."""
    return " ".join(text.translate(_INVALID_XML).split())


def esc(text: str) -> str:
    return escape(clean(text), quote=True)


def num(value: float) -> str:
    """A coordinate with at most two decimals, so output is stable and compact."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def attrs(**values: object) -> str:
    parts = []
    for key, value in values.items():
        if value is None:
            continue
        name = key.rstrip("_").replace("_", "-")
        if isinstance(value, float):
            text = num(value)
        else:
            text = str(value)
        parts.append(f'{name}="{escape(text, quote=True)}"')
    return " ".join(parts)


def first_fit(
    candidates: list[str],
    max_width: float,
    size: float,
    *,
    font: FontKey = "sans",
    weight: int = 400,
) -> str:
    """The first wording that fits, longest first; the last one is shortened if none do."""
    for candidate in candidates:
        if text_width(clean(candidate), size, font=font, weight=weight) <= max_width:
            return clean(candidate)
    return fit(candidates[-1], max_width, size, font=font, weight=weight)


def fit(
    text: str,
    max_width: float,
    size: float,
    *,
    font: FontKey = "sans",
    weight: int = 400,
    tracking: float = 0.0,
    middle: bool = False,
) -> str:
    """Shorten ``text`` with an ellipsis until it fits ``max_width``.

    ``middle`` keeps both ends, which suits model names whose quantization
    suffix matters as much as their family prefix.
    """
    text = clean(text)

    def width(candidate: str) -> float:
        return text_width(candidate, size, font=font, weight=weight, tracking=tracking)

    if width(text) <= max_width:
        return text
    if middle:
        lo, hi = 0, len(text)
        best = ELLIPSIS
        while lo <= hi:
            keep = (lo + hi) // 2
            head = text[: (keep + 1) // 2].rstrip()
            tail = text[len(text) - keep // 2 :].lstrip() if keep // 2 else ""
            candidate = head + ELLIPSIS + tail
            if width(candidate) <= max_width:
                best = candidate
                lo = keep + 1
            else:
                hi = keep - 1
        return best
    lo, hi = 0, len(text)
    best = ELLIPSIS
    while lo <= hi:
        keep = (lo + hi) // 2
        candidate = text[:keep].rstrip() + ELLIPSIS
        if width(candidate) <= max_width:
            best = candidate
            lo = keep + 1
        else:
            hi = keep - 1
    return best


def text(
    x: float,
    y: float,
    content: str,
    *,
    size: float,
    fill: str,
    font: FontKey = "sans",
    weight: int = 400,
    anchor: str = "start",
    tracking: float = 0.0,
    opacity: float | None = None,
) -> str:
    """A single-line text element."""
    return (
        f"<text {attrs(x=float(x), y=float(y), font_family=FAMILIES[font], font_size=float(size), font_weight=weight, fill=fill, text_anchor=None if anchor == 'start' else anchor, letter_spacing=float(tracking) if tracking else None, opacity=opacity)}>"
        f"{esc(content)}</text>"
    )


@dataclass(frozen=True)
class Run:
    """One styled stretch of a :func:`runs_text` line. ``gap`` is space before it."""

    content: str
    size: float
    fill: str
    font: FontKey = "sans"
    weight: int = 400
    tracking: float = 0.0
    gap: float = 0.0

    @property
    def width(self) -> float:
        return self.gap + text_width(
            self.content, self.size, font=self.font, weight=self.weight, tracking=self.tracking
        )


def runs_width(runs: list[Run]) -> float:
    return sum(r.width for r in runs)


def runs_text(x: float, y: float, runs: list[Run], *, anchor: str = "start") -> str:
    """Several styles on one baseline in one ``<text>``, such as a number and its unit.

    The renderer lays the pieces out from real glyph widths, so a unit stays
    beside its number even when a viewer substitutes a different font.
    """
    first, rest = runs[0], runs[1:]
    spans = "".join(
        f"<tspan {attrs(dx=float(r.gap) if r.gap else None, font_family=FAMILIES[r.font], font_size=float(r.size), font_weight=r.weight, fill=r.fill, letter_spacing=float(r.tracking) if r.tracking or first.tracking else None)}>{esc(r.content)}</tspan>"
        for r in rest
    )
    return (
        f"<text {attrs(x=float(x), y=float(y), font_family=FAMILIES[first.font], font_size=float(first.size), font_weight=first.weight, fill=first.fill, text_anchor=None if anchor == 'start' else anchor, letter_spacing=float(first.tracking) if first.tracking else None)}>"
        f"{esc(first.content)}{spans}</text>"
    )


def tspan(content: str, **values: object) -> str:
    return f"<tspan {attrs(**values)}>{esc(content)}</tspan>"


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "none",
    rx: float = 0,
    stroke: str | None = None,
    stroke_width: float | None = None,
    opacity: float | None = None,
    dash: str | None = None,
) -> str:
    return f"<rect {attrs(x=float(x), y=float(y), width=float(max(w, 0)), height=float(max(h, 0)), rx=float(rx) if rx else None, fill=fill, stroke=stroke, stroke_width=stroke_width, stroke_dasharray=dash, opacity=opacity)}/>"


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 1.0,
    opacity: float | None = None,
    dash: str | None = None,
) -> str:
    return f"<line {attrs(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), stroke=stroke, stroke_width=float(width), stroke_dasharray=dash, opacity=opacity)}/>"


def circle(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str | None = None,
    stroke_width: float | None = None,
    opacity: float | None = None,
) -> str:
    return f"<circle {attrs(cx=float(cx), cy=float(cy), r=float(r), fill=fill, stroke=stroke, stroke_width=stroke_width, opacity=opacity)}/>"


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{num(x)},{num(y)}" for x, y in points)
