"""Formatting helpers shared by the comparison report generators.

Both generators render the same kinds of cell: a percentage, a signed delta, an
escaped label.  These were defined identically in each file; they live here now
so the two reports cannot drift apart on how a number is displayed.
"""

from __future__ import annotations

import html
import re


def _r(pat, txt, group=1, fl=0):
    m = re.search(pat, txt, fl)
    return m.group(group).strip() if m else ""


def _tv(field, txt, strip_bt=False):
    m = re.search(rf"\*\*{re.escape(field)}\*\*\s*\|\s*(.+)", txt)
    if not m:
        return ""
    v = m.group(1).strip()
    return v.strip("`") if strip_bt else v


def dname(d: dict) -> str:
    return d["model_api"] or d["model_name"]


def esc(s: str) -> str:
    """Escape a value for HTML text or double-quoted attribute context.

    Every string that originates in a parsed Markdown report must pass through
    here: the reports are shared between people, so an attacker-authored report
    must not be able to inject markup into the comparison page.
    """
    return html.escape(str(s), quote=True)


def sign(v: int) -> str:
    return f"{'+' if v >= 0 else ''}{v}"


def pct_cls(w, r):
    if w > r:
        return "font-semibold text-emerald-700"
    if w < r:
        return "text-rose-600"
    return ""


def diff_display(wp, rp):
    dv = round(wp - rp)
    if dv > 0:
        return f"+{dv}", "diff-positive"
    if dv < 0:
        return f"{dv}", "diff-negative"
    return "\u2014", "text-slate-500"


def turn_time_display(wmt, rmt, winner_raw, runner_raw):
    if wmt is not None and rmt is not None:
        delta = wmt - rmt
        cls = "diff-positive" if delta <= 0 else "diff-negative"
        return f"{wmt:.1f}s", f"{rmt:.1f}s", f"{delta:+.1f}s", cls
    return winner_raw or "\u2014", runner_raw or "\u2014", "\u2014", "text-slate-500"
