"""Bundled card fonts and the glyph metrics used to fit text.

The card is laid out in Python, so long strings have to be measured before they
are placed. Reading advance widths straight from the bundled TTFs keeps the
estimate honest without a font library: only the ``cmap`` (format 4), ``hhea``,
``hmtx``, and ``head`` tables are needed. Kerning is ignored, which slightly
overestimates width and errs on the side of not overflowing.

The fonts are subsets of upstream Inter 4.1 and JetBrains Mono 2.304 (SIL OFL
1.1); ``fonts/SOURCES.txt`` records how they were made.
"""

from __future__ import annotations

import struct
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

FontKey = Literal["sans", "display", "mono"]

_PACKAGE = "tool_eval_bench.storage.reports.card"

# (family key, weight) -> bundled file. Weights not listed here fall back to the
# nearest one, which mirrors how a renderer resolves a missing weight.
_FILES: dict[tuple[FontKey, int], str] = {
    ("sans", 400): "Inter-Regular.ttf",
    ("sans", 500): "Inter-Medium.ttf",
    ("sans", 600): "Inter-SemiBold.ttf",
    ("display", 700): "InterDisplay-Bold.ttf",
    ("mono", 400): "JetBrainsMonoNL-Regular.ttf",
    ("mono", 500): "JetBrainsMonoNL-Medium.ttf",
}

# Font stacks for the SVG. The bundled family comes first so a PNG export
# matches exactly; the rest keep a browser rendering close when Inter or
# JetBrains Mono is not installed.
FAMILIES: dict[FontKey, str] = {
    "sans": "Inter, 'Inter Variable', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    "display": "'Inter Display', Inter, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    "mono": "'JetBrains Mono NL', 'JetBrains Mono', 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace",
}


def font_file_names() -> tuple[str, ...]:
    """The bundled font files, in a stable order."""
    return tuple(sorted(_FILES.values()))


def _resolve(key: FontKey, weight: int) -> str:
    candidates = [w for (k, w) in _FILES if k == key]
    best = min(candidates, key=lambda w: (abs(w - weight), w))
    return _FILES[(key, best)]


@dataclass(frozen=True)
class _Metrics:
    units_per_em: int
    advances: dict[int, int]
    fallback: int


def _tables(data: bytes) -> dict[bytes, tuple[int, int]]:
    (num_tables,) = struct.unpack_from(">H", data, 4)
    tables = {}
    for i in range(num_tables):
        tag, _checksum, offset, length = struct.unpack_from(">4sIII", data, 12 + 16 * i)
        tables[tag] = (offset, length)
    return tables


def _cmap_format4(data: bytes, cmap_offset: int) -> dict[int, int]:
    _version, count = struct.unpack_from(">HH", data, cmap_offset)
    sub_offset = None
    for i in range(count):
        platform, encoding, offset = struct.unpack_from(">HHI", data, cmap_offset + 4 + 8 * i)
        if (platform, encoding) in ((3, 1), (0, 3)):
            start = cmap_offset + offset
            if struct.unpack_from(">H", data, start)[0] == 4:
                sub_offset = start
                break
    if sub_offset is None:
        return {}
    (seg_x2,) = struct.unpack_from(">H", data, sub_offset + 6)
    segs = seg_x2 // 2
    ends_at = sub_offset + 14
    starts_at = ends_at + seg_x2 + 2
    deltas_at = starts_at + seg_x2
    ranges_at = deltas_at + seg_x2
    mapping: dict[int, int] = {}
    for s in range(segs):
        end = struct.unpack_from(">H", data, ends_at + 2 * s)[0]
        start = struct.unpack_from(">H", data, starts_at + 2 * s)[0]
        delta = struct.unpack_from(">h", data, deltas_at + 2 * s)[0]
        range_pos = ranges_at + 2 * s
        range_offset = struct.unpack_from(">H", data, range_pos)[0]
        for code in range(start, min(end, 0xFFFE) + 1):
            if range_offset == 0:
                glyph = (code + delta) & 0xFFFF
            else:
                glyph = struct.unpack_from(
                    ">H", data, range_pos + range_offset + 2 * (code - start)
                )[0]
                if glyph:
                    glyph = (glyph + delta) & 0xFFFF
            if glyph:
                mapping[code] = glyph
    return mapping


def _parse(data: bytes) -> _Metrics:
    tables = _tables(data)
    (units_per_em,) = struct.unpack_from(">H", data, tables[b"head"][0] + 18)
    (num_metrics,) = struct.unpack_from(">H", data, tables[b"hhea"][0] + 34)
    hmtx = tables[b"hmtx"][0]
    glyph_advance = [struct.unpack_from(">H", data, hmtx + 4 * i)[0] for i in range(num_metrics)]
    cmap = _cmap_format4(data, tables[b"cmap"][0])
    last = glyph_advance[-1]
    advances = {code: glyph_advance[g] if g < num_metrics else last for code, g in cmap.items()}
    # Unknown characters (CJK, emoji) render with a system fallback font, whose
    # widths we cannot know; an em-ish width keeps the fit conservative.
    return _Metrics(units_per_em, advances, int(units_per_em * 0.9))


@lru_cache(maxsize=None)
def _metrics(file_name: str) -> _Metrics:
    data = resources.files(_PACKAGE).joinpath("fonts", file_name).read_bytes()
    return _parse(data)


def _fallback_units(ch: str, metrics: _Metrics) -> int:
    # East Asian wide and fullwidth characters are a full em in every CJK font.
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return metrics.units_per_em
    return metrics.fallback


def text_width(
    text: str, size: float, *, font: FontKey = "sans", weight: int = 400, tracking: float = 0.0
) -> float:
    """Estimated rendered width of ``text`` in pixels, including letter spacing."""
    metrics = _metrics(_resolve(font, weight))
    units = 0
    for ch in text:
        advance = metrics.advances.get(ord(ch))
        units += advance if advance is not None else _fallback_units(ch, metrics)
    return units * size / metrics.units_per_em + tracking * len(text)


@lru_cache(maxsize=1)
def _covered() -> frozenset[int]:
    codes: set[int] = set()
    for name in font_file_names():
        codes.update(_metrics(name).advances)
    return frozenset(codes)


def unsupported_characters(text: str) -> str:
    """Characters in ``text`` that no bundled font can draw, in first-seen order."""
    covered = _covered()
    seen: dict[str, None] = {}
    for ch in text:
        if ord(ch) > 0x7E and ord(ch) not in covered and not ch.isspace():
            seen.setdefault(ch, None)
    return "".join(seen)
