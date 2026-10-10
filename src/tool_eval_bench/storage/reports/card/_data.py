"""Views over :class:`CardData` that the panels share.

The card compares like with like: single-stream numbers come from
concurrency 1, and the concurrency panel holds depth fixed at the shallowest
measured depth (normally 0).
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

from tool_eval_bench.domain.card import CardData, PerfPoint, ToolEvalSummary
from tool_eval_bench.storage.reports.card._format import tokens

# A sweep that stops at or below this depth says nothing about long context.
LONG_CONTEXT_MIN_DEPTH = 8192


def _measured(value: float) -> bool:
    return math.isfinite(value) and value >= 0


def plottable(data: CardData) -> CardData:
    """``data`` without measurements that cannot be drawn.

    A NaN, infinite, or negative rate, TTFT, or needle position would become an
    invalid SVG coordinate, so such a point or cell is dropped and a bad cached
    TTFT is treated as not measured.
    """
    points = []
    for p in data.perf:
        if not all(_measured(v) for v in (p.pp_tps, p.tg_tps, p.tg_tps_total, p.ttft_ms)):
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


def single_stream(data: CardData) -> list[PerfPoint]:
    """Concurrency-1 points, one per depth, shallowest first."""
    by_depth: dict[int, PerfPoint] = {}
    for point in data.perf:
        if point.concurrency == 1 and point.depth not in by_depth:
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


def preset_label(data: CardData) -> str | None:
    """The preset stamp text, or None when nothing was measured."""
    if not data.perf:
        return None
    if data.perf_preset:
        return f"{data.perf_preset.upper()} PRESET"
    return "CUSTOM SWEEP"


def pp_tg_label(data: CardData) -> str | None:
    parts = []
    if data.perf_pp is not None:
        parts.append(f"pp {data.perf_pp:,}")
    if data.perf_tg is not None:
        parts.append(f"tg {data.perf_tg:,}")
    return " \u00b7 ".join(parts) or None


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


def display_date(created_at: str) -> str:
    """The calendar date from an ISO timestamp, or the raw text if it is not one."""
    head = created_at.strip()[:10]
    if len(head) == 10 and head[4] == "-" and head[7] == "-" and head.replace("-", "").isdigit():
        return head
    return created_at.strip()


def version_label(version: str) -> str:
    version = version.strip()
    return f"v{version}" if version[:1].isdigit() else version


_DEV_SUFFIX = re.compile(r"\.?dev\d*$")


def short_version(version: str) -> str:
    """A release-style version for display: ``0.21.1rc1.dev292+g97e4022c6`` becomes ``0.21.1rc1``.

    The PEP 440 local segment and a trailing ``.devN`` identify a build, not a
    release, and only add noise on a card.
    """
    public = version.strip().split("+", 1)[0]
    return _DEV_SUFFIX.sub("", public) or public


def engine_label(engine: str | None, version: str | None) -> str | None:
    parts = [p for p in (engine, short_version(version) if version else None) if p]
    return " ".join(parts) or None


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
        f"TTFT cold: full context prefill \u00b7 cached: {new} on cached context",
        f"cold: full prefill \u00b7 cached: {new} on cached context",
        f"cold: full prefill \u00b7 cached: {new} on cached",
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


# A run with fewer scenarios than the suite it drew from is a subset, and the
# card says so next to the score. The size comes from the scenario registry via
# ``ToolEvalSummary.suite_size``; unknown means no mark.
def is_subset(total: int, suite_size: int | None) -> bool:
    return suite_size is not None and total < suite_size


def scenario_label(total: int, suite_size: int | None) -> str:
    noun = "scenario" if total == 1 else "scenarios"
    label = f"{total} {noun}"
    return f"{label} \u00b7 subset" if is_subset(total, suite_size) else label


def score_details(te: ToolEvalSummary, *, passed: bool = False) -> list[str]:
    """Wordings for the line under a tool score, longest first.

    The scenario count and subset mark are always kept. A multi-trial score
    says it is a mean; the pass count goes first when space runs out.
    """
    mid = " \u00b7 "
    label = scenario_label(te.total, te.suite_size)
    passed_text = f"{te.passed} of {te.total} passed"
    if te.trials > 1:
        trials = (f"mean of {te.trials} trials", f"{te.trials} trials")
        options = [mid.join([label, passed_text, trials[0]])] if passed else []
        return [*options, mid.join([label, trials[0]]), mid.join([label, trials[1]])]
    return [mid.join([label, passed_text]), label] if passed else [label]
