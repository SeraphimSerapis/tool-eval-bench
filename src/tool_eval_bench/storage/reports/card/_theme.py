"""Palette, spacing, and shared SVG definitions for the card."""

from __future__ import annotations

from dataclasses import dataclass

# Spacing follows an 8px grid; GAP is the gutter between panels.
UNIT = 8
GAP = 16
PAD = 24
RADIUS = 20

BG_TOP = "#080B14"
BG_BOTTOM = "#05070D"
PANEL_TOP = "#111827"
PANEL_BOTTOM = "#0C111D"
STROKE = "#FFFFFF"

TEXT = "#F3F5FA"
TEXT_2 = "#A9B2C4"
MUTED = "#6E788C"
FAINT = "#262E3E"
GRID = "#FFFFFF"

DECODE = "#22D3EE"
PREFILL = "#A78BFA"
TTFT = "#FBBF24"
TTFT_CACHED = "#FDE68A"
TOOL = "#34D399"
PASS = "#34D399"
# Scores between solid and strong; distinct from the decode cyan so a bar never
# reads as a speed series.
GOOD = "#A3E635"
PARTIAL = "#FBBF24"
FAIL = "#F87171"
WARN = "#FBBF24"
BRAND_A = "#22D3EE"
BRAND_B = "#A78BFA"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def inset(self, dx: float, dy: float | None = None) -> Box:
        dy = dx if dy is None else dy
        return Box(self.x + dx, self.y + dy, self.w - 2 * dx, self.h - 2 * dy)


def defs(width: int, height: int) -> str:
    """Gradients, the background grid, and the glow filter. IDs are fixed so output is stable."""
    return f"""<defs>
<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{BG_TOP}"/><stop offset="1" stop-color="{BG_BOTTOM}"/>
</linearGradient>
<radialGradient id="glowA" cx="0.12" cy="0.02" r="0.62">
<stop offset="0" stop-color="#7C3AED" stop-opacity="0.30"/><stop offset="1" stop-color="#7C3AED" stop-opacity="0"/>
</radialGradient>
<radialGradient id="glowB" cx="0.96" cy="0.30" r="0.55">
<stop offset="0" stop-color="#06B6D4" stop-opacity="0.18"/><stop offset="1" stop-color="#06B6D4" stop-opacity="0"/>
</radialGradient>
<radialGradient id="glowC" cx="0.5" cy="1.05" r="0.6">
<stop offset="0" stop-color="#10B981" stop-opacity="0.10"/><stop offset="1" stop-color="#10B981" stop-opacity="0"/>
</radialGradient>
<radialGradient id="gridFade" cx="0.5" cy="0.0" r="1.0">
<stop offset="0" stop-color="#FFFFFF" stop-opacity="1"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>
</radialGradient>
<mask id="gridMask"><rect width="{width}" height="{height}" fill="url(#gridFade)"/></mask>
<pattern id="grid" width="{UNIT * 5}" height="{UNIT * 5}" patternUnits="userSpaceOnUse">
<path d="M{UNIT * 5} 0H0V{UNIT * 5}" fill="none" stroke="{GRID}" stroke-opacity="0.045" stroke-width="1"/>
</pattern>
<linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{PANEL_TOP}" stop-opacity="0.82"/><stop offset="1" stop-color="{PANEL_BOTTOM}" stop-opacity="0.82"/>
</linearGradient>
<linearGradient id="panelEdge" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{STROKE}" stop-opacity="0.14"/><stop offset="1" stop-color="{STROKE}" stop-opacity="0.04"/>
</linearGradient>
<linearGradient id="brand" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="{BRAND_A}"/><stop offset="1" stop-color="{BRAND_B}"/>
</linearGradient>
<linearGradient id="decodeArea" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{DECODE}" stop-opacity="0.32"/><stop offset="1" stop-color="{DECODE}" stop-opacity="0"/>
</linearGradient>
<linearGradient id="ttftArea" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{TTFT}" stop-opacity="0.22"/><stop offset="1" stop-color="{TTFT}" stop-opacity="0"/>
</linearGradient>
<linearGradient id="bar" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{DECODE}" stop-opacity="0.95"/><stop offset="1" stop-color="{DECODE}" stop-opacity="0.30"/>
</linearGradient>
<filter id="glow" x="-20%" y="-40%" width="140%" height="180%">
<feGaussianBlur stdDeviation="5" result="b"/>
<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="6" height="6" fill="{STROKE}" fill-opacity="0.03"/>
<path d="M0 0V6" stroke="{STROKE}" stroke-opacity="0.22" stroke-width="2"/>
</pattern>
</defs>"""


def background(width: int, height: int) -> str:
    return (
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>'
        f'<rect width="{width}" height="{height}" fill="url(#grid)" mask="url(#gridMask)"/>'
        f'<rect width="{width}" height="{height}" fill="url(#glowA)"/>'
        f'<rect width="{width}" height="{height}" fill="url(#glowB)"/>'
        f'<rect width="{width}" height="{height}" fill="url(#glowC)"/>'
    )
