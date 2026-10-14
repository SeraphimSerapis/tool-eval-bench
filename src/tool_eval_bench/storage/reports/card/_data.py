"""Views over :class:`CardData` that the panels share.

The card compares like with like: single-stream numbers come from the lowest
measured concurrency (normally 1), and the concurrency panel holds depth fixed
at the shallowest measured depth (normally 0).
"""

from __future__ import annotations

import math
from dataclasses import replace

from tool_eval_bench.domain.card import AccuracyResult, CardData, PerfPoint, ToolEvalSummary
from tool_eval_bench.storage.reports.card._format import tokens

# A sweep that stops at or below this depth says nothing about long context.
LONG_CONTEXT_MIN_DEPTH = 8192
# The deepest depth the long preset plans. A long run that stops short of it
# without being interrupted was capped by the served context window.
LONG_PRESET_DEPTH = 131072
# The chat-template allowance the planner adds when it fits a depth into the
# window. Storage cannot import runner.perf_presets, so a test keeps the two equal.
CHAT_TEMPLATE_HEADROOM = 256
# Accuracy results from fewer items than this show their sample size.
SMALL_SAMPLE = 100

MID = " \u00b7 "


def _measured(value: float) -> bool:
    return math.isfinite(value) and value >= 0


def plottable(data: CardData) -> CardData:
    """``data`` without measurements that cannot be drawn.

    A NaN, infinite, or negative rate, TTFT, or needle position would become an
    invalid SVG coordinate, so such a point or cell is dropped and a bad cached
    TTFT is treated as not measured. A point that decoded nothing (zero
    per-request decode) is a failed cell, not a measurement of zero speed.
    """
    points = []
    for p in data.perf:
        if not all(_measured(v) for v in (p.pp_tps, p.tg_tps, p.tg_tps_total, p.ttft_ms)):
            continue
        if p.tg_tps <= 0:
            continue
        if p.cached_ttft_ms is not None and not _measured(p.cached_ttft_ms):
            p = replace(p, cached_ttft_ms=None)
        points.append(p)
    needle = data.needle
    if needle is not None:
        cells = tuple(
            c for c in needle.cells if _measured(c.depth_percent) and c.context_tokens >= 0
        )
        needle = replace(needle, cells=cells)
    return replace(data, perf=tuple(points), needle=needle)


# --------------------------------------------------------------------------- perf


def stream_concurrency(data: CardData) -> int:
    """The concurrency single-stream numbers come from: 1, or the lowest level measured."""
    return min((p.concurrency for p in data.perf), default=1)


def single_stream(data: CardData) -> list[PerfPoint]:
    """Points at :func:`stream_concurrency`, one per depth, shallowest first."""
    level = stream_concurrency(data)
    by_depth: dict[int, PerfPoint] = {}
    for point in data.perf:
        if point.concurrency == level and point.depth not in by_depth:
            by_depth[point.depth] = point
    return [by_depth[d] for d in sorted(by_depth)]


def base_point(data: CardData) -> PerfPoint | None:
    series = single_stream(data)
    return series[0] if series else None


def deepest_point(data: CardData) -> PerfPoint | None:
    series = single_stream(data)
    return series[-1] if series else None


def concurrency_series(data: CardData) -> list[PerfPoint]:
    """Points at the shallowest depth, one per concurrency level."""
    if not data.perf:
        return []
    depth = min(p.depth for p in data.perf)
    by_level: dict[int, PerfPoint] = {}
    for point in data.perf:
        if point.depth == depth and point.concurrency not in by_level:
            by_level[point.concurrency] = point
    return [by_level[c] for c in sorted(by_level)]


def has_long_context(data: CardData) -> bool:
    series = single_stream(data)
    return len(series) >= 2 and series[-1].depth > LONG_CONTEXT_MIN_DEPTH


def deepest_depth(data: CardData) -> int:
    series = single_stream(data)
    return series[-1].depth if series else 0


def window_capped(data: CardData) -> bool:
    """A long preset that finished but stopped short of 128K because the window capped it.

    A failed cell can also leave the sweep short, so any failed cell rules the
    claim out. When the window is known, the 128K request must also not fit it,
    by the same test the planner uses to drop depths.
    """
    if not (
        data.perf_preset == "long"
        and not data.perf_incomplete
        and data.perf_failed_cells == 0
        and data.perf
        and deepest_depth(data) < LONG_PRESET_DEPTH
    ):
        return False
    if data.max_context is None:
        return True
    request = LONG_PRESET_DEPTH + (data.perf_pp or 0) + (data.perf_tg or 0)
    return request + CHAT_TEMPLATE_HEADROOM > data.max_context


def cap_note(data: CardData) -> str:
    """Why a long sweep stops where it does, for subtitles and the stamp."""
    deepest = tokens(deepest_depth(data))
    if data.max_context:
        return f"capped at {deepest} by the {tokens(data.max_context)} window"
    return f"capped at {deepest} by the context window"


def perf_warnings(data: CardData) -> list[str]:
    """Short notes that the sweep is partial, for the stamp."""
    notes = []
    if data.perf_incomplete:
        notes.append("incomplete")
    if data.perf_failed_cells > 0:
        noun = "cell" if data.perf_failed_cells == 1 else "cells"
        notes.append(f"{data.perf_failed_cells} {noun} failed")
    return notes


def preset_label(data: CardData) -> str | None:
    """The preset stamp text, or None when nothing was measured."""
    if not data.perf:
        return None
    if data.perf_preset and data.perf_preset != "custom":
        return f"{data.perf_preset.upper()} PRESET"
    return "CUSTOM SWEEP"


def pp_tg_label(data: CardData) -> str | None:
    parts = []
    if data.perf_pp is not None:
        parts.append(f"pp {data.perf_pp:,}")
    if data.perf_tg is not None:
        parts.append(f"tg {data.perf_tg:,}")
    return MID.join(parts) or None


def ttft_note(data: CardData) -> list[str]:
    """What cold and cached TTFT measure, longest wording first.

    At depth, cold TTFT prefills the whole context from scratch, while cached
    TTFT sends ``perf_pp`` new tokens on top of a context the server already
    holds. Sweeps without cached values only describe the cold number.
    """
    if not any(p.cached_ttft_ms is not None for p in data.perf):
        return ["TTFT cold: full context prefill from scratch", "TTFT: full context prefill"]
    new = f"+{tokens(data.perf_pp)} tokens" if data.perf_pp else "new tokens"
    return [
        f"TTFT cold: full context prefill{MID}cached: {new} on cached context",
        f"cold: full prefill{MID}cached: {new} on cached context",
        f"cold: full prefill{MID}cached: {new} on cached",
    ]


def cached_ttft_label(data: CardData, duration: str) -> tuple[str, ...]:
    if not data.perf_pp:
        return (f"cached context: {duration}", f"cached: {duration}")
    new = f"+{tokens(data.perf_pp)}"
    return (
        f"cached, {new} new tokens: {duration}",
        f"cached {new}: {duration}",
        f"cached: {duration}",
    )


# --------------------------------------------------------------------------- tool eval


def rating_parts(rating: str) -> tuple[int, str]:
    """Split "★★★★ Good (safety-capped)" into (4, "Good").

    The safety cap is drawn as its own warning, so the suffix is dropped here.
    """
    stars = len(rating) - len(rating.lstrip("\u2605"))
    word = rating.lstrip("\u2605").strip()
    for suffix in ("(safety-capped)",):
        if word.endswith(suffix):
            word = word[: -len(suffix)].strip()
    return min(stars, 5), word


def is_subset(te: ToolEvalSummary) -> bool:
    """Fewer scenarios selected than the suite the run draws from.

    The suite size comes from the scenario registry; ``None`` (a scenario pack,
    an older registry) means unknown, and then no subset is claimed.
    """
    return te.suite_size is not None and te.total + te.excluded < te.suite_size


def needs_caution(te: ToolEvalSummary) -> bool:
    """The score covers less than its suite: a subset, or scenarios excluded after failures."""
    return is_subset(te) or te.excluded > 0


def suite_tag(te: ToolEvalSummary) -> str | None:
    if te.scenario_pack:
        return "scenario pack"
    if te.hardmode_only:
        return "Hard Mode only"
    if te.hardmode:
        return "Hard Mode"
    return None


def scope_label(te: ToolEvalSummary, *, tag: bool = True) -> str:
    """What the score covers: "subset: 3 of 69 scenarios", "97 scenarios · Hard Mode"."""
    noun = "scenario" if te.total == 1 else "scenarios"
    if is_subset(te):
        label = f"subset: {te.total} of {te.suite_size} scenarios"
    else:
        label = f"{te.total} {noun}"
    if te.excluded > 0:
        label += f"{MID}{te.excluded} excluded"
    name = suite_tag(te) if tag else None
    return f"{label}{MID}{name}" if name else label


def score_details(te: ToolEvalSummary, *, passed: bool = False) -> list[str]:
    """Wordings for the line under a tool score, longest first.

    The scope (subset, exclusions, Hard Mode, pack) is always kept. A
    multi-trial score says it is a mean; the pass count goes first when space
    runs out, then the suite tag.
    """
    scopes = [scope_label(te)]
    tag = suite_tag(te)
    if tag:
        count = f"{te.total} of {te.suite_size}" if is_subset(te) else str(te.total)
        scopes += [f"{count}{MID}{tag}", scope_label(te, tag=False)]
    passed_text = f"{te.passed} of {te.total} passed"
    trials = (f"mean of {te.trials} trials", f"{te.trials} trials") if te.trials > 1 else ()
    options: list[str] = []
    for scope in scopes:
        if passed:
            options.append(MID.join([scope, passed_text, *trials[:1]]))
        options.extend(MID.join([scope, t]) for t in trials)
        options.append(scope)
    return list(dict.fromkeys(options))


# --------------------------------------------------------------------------- accuracy


def accuracy_note(result: AccuracyResult, *, detail: bool = True) -> str | None:
    """Why an accuracy score may be thin: errored items, a small sample, the stored detail."""
    parts = []
    if detail and result.detail:
        parts.append(result.detail)
    if result.errors > 0:
        if result.items:
            parts.append(f"{result.errors:,} of {result.items:,} errored")
        else:
            parts.append(f"{result.errors:,} errored")
    elif (
        result.items is not None and result.items < SMALL_SAMPLE and not (detail and result.detail)
    ):
        parts.append(f"n={result.items:,}")
    return MID.join(parts) or None


def accuracy_is_thin(result: AccuracyResult) -> bool:
    return result.errors > 0 or (result.items is not None and result.items < SMALL_SAMPLE)


# --------------------------------------------------------------------------- labels


def display_date(created_at: str) -> str:
    """The calendar date from an ISO timestamp, or the raw text if it is not one."""
    head = created_at.strip()[:10]
    if len(head) == 10 and head[4] == "-" and head[7] == "-" and head.replace("-", "").isdigit():
        return head
    return created_at.strip()


def public_version(version: str) -> str:
    """``version`` without its PEP 440 local segment: ``0.21.1rc1.dev292+g97e4022c6`` is ``0.21.1rc1.dev292``.

    The local segment names a commit or build; the public part, including any
    ``.devN``, says which release line it is and stays.
    """
    return version.strip().split("+", 1)[0].strip()


def version_label(version: str) -> str:
    version = public_version(version)
    return f"v{version}" if version[:1].isdigit() else version


def engine_label(engine: str | None, version: str | None) -> str | None:
    parts = [p for p in (engine, public_version(version) if version else None) if p]
    return " ".join(parts) or None
