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
    """Read the value cell of a two-column ``| Field | value |`` table row.

    The report writer bolds only some labels (``**Label**``), so the bold is
    optional, and the value stops at the next unescaped pipe.  Writers escape
    pipes in code spans as ``\\|`` and HTML-escape other cell text, so both are
    undone here; ``esc`` re-escapes the value on output.
    """
    m = re.search(
        rf"(?:^|\|)[ \t]*(?:\*\*)?{re.escape(field)}(?:\*\*)?[ \t]*\|[ \t]*((?:\\\||[^|\n])*?)[ \t]*(?:\||$)",
        txt,
        re.M,
    )
    if not m:
        return ""
    v = html.unescape(m.group(1).strip().replace("\\|", "|"))
    return v.strip("`") if strip_bt else v


#: Run settings the comparison pages describe.
_COMPARED_SETTINGS = ("backend", "temperature", "thinking")


def config_note(da: dict, db: dict) -> str:
    """Describe whether two runs used the same settings, as HTML-safe text.

    Says "same" only after comparing the parsed values, and claims nothing
    about settings neither report recorded.
    """
    differences = []
    shared = []
    for key in _COMPARED_SETTINGS:
        a, b = da.get(key) or "", db.get(key) or ""
        if a != b:
            differences.append(
                f"{key} {esc(a or '?')} ({esc(dname(da))}) vs {esc(b or '?')} ({esc(dname(db))})"
            )
        elif a:
            shared.append(f"{key} {esc(a)}")
    if differences:
        return "The runs used different settings: " + "; ".join(differences) + "."
    if not shared:
        return "Neither report records the backend, temperature, or thinking setting."
    return "Both runs used the same " + ", ".join(shared) + "."


def deployability_label(da: dict, db: dict) -> str:
    """Label the deployability row with the alpha the reports actually used."""
    a, b = da.get("alpha") or "", db.get("alpha") or ""
    if a and a == b:
        return f"Deployability (\u03b1={a})"
    if a and b:
        return f"Deployability (\u03b1={a} vs {b})"
    return "Deployability"


def card_labels(tie: bool) -> tuple[str, str]:
    """Return the (second card, first card) label markup for the model cards.

    On a tie neither card is crowned: both read TIED and the trophy is dropped.
    """
    if tie:
        tied = '<div class="model-label text-slate-600 tracking-widest">TIED</div>'
        return tied, tied
    runner = '<div class="model-label text-rose-700 tracking-widest">RUNNER-UP</div>'
    winner = (
        '<div class="flex items-center gap-x-2">\n'
        '              <span class="model-label text-emerald-700 tracking-widest">WINNER</span>\n'
        '              <span class="winner-badge"><i class="fa-solid fa-trophy mr-1"></i> BEST</span>\n'
        "            </div>"
    )
    return runner, winner


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
