"""Number formatting for the card.

Every function returns display text only. Values are split into a number and a
unit where the layout sets them at different sizes.
"""

from __future__ import annotations

import math

# A typographic minus reads better than a hyphen next to large numerals.
MINUS = "\u2212"


def tokens(n: int) -> str:
    """A token count as 0, 512, 4K, 128K, 1M, or 1.5M.

    Benchmark depths are usually powers of two (131072 is "128K"), but a
    decimal value such as 100000 is shown as "100K" rather than "97.7K".
    """
    if n < 1000:
        return str(n)
    for base in (1024, 1000):
        if n >= base * base and n % (base * base // 4) == 0:
            return _trim(n / (base * base)) + "M"
    for base in (1024, 1000):
        if n < base * base and n % base == 0:
            return f"{n // base}K"
    if n >= 1_000_000:
        return _trim(round(n / 1_000_000, 1)) + "M"
    return _trim(round(n / 1000, 1)) + "K"


def _trim(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


def rate(value: float) -> str:
    """Tokens per second: 1,234 / 142 / 12.3 / 0.85."""
    if not math.isfinite(value) or value < 0:
        return "n/a"
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def duration(ms: float) -> tuple[str, str]:
    """Milliseconds as (number, unit): 850 ms, 2.4 s, 18.4 s, 412 s."""
    if not math.isfinite(ms) or ms < 0:
        return "n/a", ""
    if ms < 10:
        return f"{ms:.1f}", "ms"
    if ms < 999.5:
        return f"{ms:.0f}", "ms"
    seconds = ms / 1000
    if seconds < 99.95:
        return f"{seconds:.1f}", "s"
    return f"{seconds:,.0f}", "s"


def duration_text(ms: float) -> str:
    number, unit = duration(ms)
    return f"{number} {unit}".strip()


def _magnitude(pct: float) -> str:
    """A non-negative percentage, whole unless rounding would fake 0 or 100.

    99.98 becomes "99.9" and 0.3 becomes "0.3", so a near miss never reads as
    a perfect score or a total collapse.
    """
    whole = round(pct)
    if whole >= 100 and pct < 100:
        return f"{math.floor(pct * 10) / 10:.1f}"
    if whole == 0 and pct > 0:
        return f"{max(math.ceil(pct * 10) / 10, 0.1):.1f}"
    return f"{whole:.0f}"


def percent(fraction: float) -> str:
    """A 0-1 fraction as a percentage: 71%, 99.9%, 0.3%."""
    if not math.isfinite(fraction):
        return "n/a"
    pct = fraction * 100
    if pct < 0:
        return f"{MINUS}{_magnitude(-pct)}%"
    return f"{_magnitude(pct)}%"


def signed_percent(change: float) -> str:
    """A relative change as +12% or -32%, using a typographic minus."""
    if not math.isfinite(change):
        return "n/a"
    pct = change * 100
    if pct < 0:
        return f"{MINUS}{_magnitude(-pct)}%"
    return f"+{_magnitude(pct)}%"


def axis_rate(value: float) -> str:
    """Compact tick labels: 0, 50, 1.2K, 9K."""
    if value >= 10_000:
        return f"{value / 1000:.0f}K"
    if value >= 1000:
        return _trim(round(value / 1000, 1)) + "K"
    if value >= 10 or value == 0:
        return f"{value:.0f}"
    return _trim(round(value, 1))
