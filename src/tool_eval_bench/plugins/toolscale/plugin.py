"""Toolscale plugin: orchestration, aggregation, and report rendering.

Implements ``BenchmarkPlugin`` for the tool-count by similarity sweep from
issue #319. Each item is one streamed request with a generated tool list; the
first tool call's name is graded. See ``grid.py`` for how the lists are built
and ``docs/toolscale.md`` for the method.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, default_max_tokens
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
    raise_for_transport_error,
)
from tool_eval_bench.domain.tools_large import LARGE_TOOLSET
from tool_eval_bench.plugins.toolscale.catalog import Catalog
from tool_eval_bench.plugins.toolscale.grid import (
    CONTROL,
    CORRECT,
    CUES,
    DENSITIES,
    ERROR,
    OUTCOMES,
    SIMILARITY_LEVELS,
    ToolscaleItem,
    build_messages,
    grade,
    openai_tools,
)
from tool_eval_bench.plugins.toolscale.similarity import ToolSimilarity, text_tokens, tool_tokens
from tool_eval_bench.plugins.toolscale.stats import (
    bootstrap_mean_interval,
    sample_stddev,
    slope_per_doubling,
    wilson_interval,
)

logger = logging.getLogger(__name__)

# Relative position buckets for the "where in the list was the target" table.
_POSITION_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("first third", 0.0, 1 / 3),
    ("middle third", 1 / 3, 2 / 3),
    ("last third", 2 / 3, 1.0001),
)

# Failures listed in the Markdown report. Every item is still counted; past
# this the list stops being readable and the cell tables carry the signal.
_MAX_LISTED_FAILURES = 200


def _rating_for_accuracy(accuracy: float) -> str:
    if accuracy >= 95:
        return "★★★★★ Excellent"
    if accuracy >= 85:
        return "★★★★ Good"
    if accuracy >= 70:
        return "★★★ Adequate"
    if accuracy >= 50:
        return "★★ Weak"
    return "★ Poor"


class ToolscalePlugin(BenchmarkPlugin):
    """Tool-selection accuracy across tool count and description similarity."""

    @property
    def name(self) -> str:
        return "toolscale"

    @property
    def description(self) -> str:
        return "Tool-selection accuracy as tool count and description similarity vary"

    async def run(
        self,
        adapter: BackendAdapter,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        seed: int | None = None,
        extra_params: dict[str, Any] | None = None,
        on_progress: OnPluginProgress | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Send every item, grade its first tool call, and aggregate the grid.

        Keyword arguments: ``items`` (from :func:`grid.build_items`),
        ``catalog`` (the :class:`Catalog` they came from), and
        ``concurrency``.
        """
        items: list[ToolscaleItem] = list(kwargs.get("items") or [])
        catalog: Catalog | None = kwargs.get("catalog")
        concurrency: int = kwargs.get("concurrency", 1)
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        total = len(items)
        if total == 0:
            return BenchmarkResult(
                plugin_name="toolscale",
                score=0.0,
                score_label="0/0",
                rating=_rating_for_accuracy(0),
                details={"total": 0, "correct": 0},
            )
        if catalog is None:
            raise ValueError("catalog is required with items")

        extra: dict[str, Any] = {}
        if seed is not None:
            extra["seed"] = seed
        if extra_params:
            extra.update(extra_params)
        max_tokens = default_max_tokens(extra)

        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = [{} for _ in range(total)]
        total_tokens = 0
        progress_counter = 0
        progress_lock = asyncio.Lock()
        t_start = time.monotonic()

        def row(item: ToolscaleItem, outcome: str, picked: str | None, **extra_fields: Any) -> dict:
            return {
                "item_id": item.item_id,
                "target": item.target.name,
                "target_rank": item.target_rank,
                "domain": item.target.domain,
                "count": item.count,
                "similarity": item.similarity,
                "cue": item.cue,
                "density": item.density,
                "trial": item.trial,
                "position": round(item.position, 4),
                "query": item.query,
                "outcome": outcome,
                "picked": picked,
                "correct": outcome == CORRECT,
                "is_error": outcome == ERROR,
                **extra_fields,
            }

        async def eval_one(idx: int, item: ToolscaleItem) -> None:
            nonlocal total_tokens, progress_counter
            prompt_tokens: int | None = None
            arguments: str | None = None
            reply: str | None = None
            error: str | None = None
            try:
                async with sem:
                    response = await adapter.chat_completion(
                        model=model,
                        messages=build_messages(item),
                        tools=openai_tools(item),
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                        extra_params=extra or None,
                        # The scenario suite streams every turn; matching it keeps
                        # one tool-call parsing path and gap-based timeouts for
                        # models that think before calling.
                        stream=True,
                    )
                raise_for_transport_error(response)
                outcome, picked = grade(item, response)
                prompt_tokens = response.prompt_tokens
                total_tokens += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
                if response.tool_calls:
                    arguments = response.tool_calls[0].arguments_str[:300]
                else:
                    # What the model said instead of calling: a clarifying
                    # question and a refusal read very differently.
                    reply = (response.content or "").strip()[:300] or None
            except Exception as exc:
                logger.debug("Error on toolscale item %s: %s", _log_id(item, catalog), exc)
                outcome, picked = ERROR, None
                # Kept for the report: a run of errors is only diagnosable if
                # it says whether the endpoint timed out, refused, or rejected.
                error = f"{type(exc).__name__}: {exc}"[:200]

            results[idx] = row(
                item,
                outcome,
                picked,
                prompt_tokens=prompt_tokens,
                arguments=arguments,
                reply=reply,
                error=error,
            )
            if on_progress:
                async with progress_lock:
                    progress_counter += 1
                    await on_progress(progress_counter, total, results[idx])

        gathered = await asyncio.gather(
            *(eval_one(i, item) for i, item in enumerate(items)), return_exceptions=True
        )
        for i, exc in enumerate(gathered):
            if isinstance(exc, BaseException):
                logger.error("Toolscale item %s crashed: %s", _log_id(items[i], catalog), exc)
                if not results[i]:
                    results[i] = row(
                        items[i],
                        ERROR,
                        None,
                        prompt_tokens=None,
                        arguments=None,
                        reply=None,
                        error=f"{type(exc).__name__}: {exc}"[:200],
                    )

        details = summarize(results, items, catalog)
        duration = time.monotonic() - t_start
        accuracy = details["accuracy"]
        return BenchmarkResult(
            plugin_name="toolscale",
            score=accuracy,
            score_label=(
                f"{details['accuracy_when_called']:.1f}% accurate when called · "
                f"{details['call_rate']:.1f}% call rate"
            ),
            rating=_rating_for_accuracy(accuracy),
            details=details,
            item_results=results,
            metadata={"dataset": f"toolscale catalog v{catalog.version} ({catalog.source})"},
            duration_seconds=round(duration, 2),
            total_tokens=total_tokens,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Render the grid, effects, error mix, similarity check, and failures."""
        return render_report(result)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _accuracy(rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """``(correct, items)`` over *rows*.

    A request error stays in the denominator, as in every accuracy plugin: a
    failing endpoint must not make a cell look better. Errors are also counted
    on their own so a reader can tell them from wrong picks.
    """
    rows = list(rows)
    return sum(1 for r in rows if r["correct"]), len(rows)


def _pct(correct: int, total: int) -> float:
    return round(correct / total * 100, 2) if total else 0.0


def summarize(
    results: list[dict[str, Any]], items: list[ToolscaleItem], catalog: Catalog
) -> dict[str, Any]:
    """Everything the report and storage need, from per-item rows.

    The headline is two numbers, never one blend: the **call rate** (did the
    model act at all) and **accuracy when called** (did it pick the right
    tool). A model that asks for an optional parameter instead of calling is
    abstaining, not confusing tools, and a single accuracy figure cannot tell
    the two apart. ``accuracy`` (correct over all items) is kept as the stored
    score so runs still sort.

    The stored run keeps ``details`` but not ``item_results``, so the per-item
    outcomes are also kept here in a compact form for later paired analysis.
    A private catalog's tool names and queries never enter ``details``.
    """
    counts = sorted({r["count"] for r in results if r["similarity"] != CONTROL})
    levels = [lvl for lvl in SIMILARITY_LEVELS if any(r["similarity"] == lvl for r in results)]
    cues = [cue for cue in CUES if any(r["cue"] == cue for r in results)]
    densities = [d for d in DENSITIES if any(r["density"] == d for r in results)]
    slices = [(cue, density) for cue in cues for density in densities]
    trials = sorted({r["trial"] for r in results})
    grid_rows = [r for r in results if r["similarity"] != CONTROL]
    control_rows = [r for r in results if r["similarity"] == CONTROL]

    by_cell: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in grid_rows:
        by_cell[(r["count"], r["similarity"], r["cue"], r["density"])].append(r)

    cells: list[dict[str, Any]] = []
    for cue, density in slices:
        for count in counts:
            for level in levels:
                rows = by_cell.get((count, level, cue, density), [])
                if rows:
                    cells.append(
                        _cell_summary(
                            rows,
                            trials,
                            count=count,
                            similarity=level,
                            cue=cue,
                            density=density,
                        )
                    )

    errors = sum(1 for r in results if r["outcome"] == ERROR)
    total = len(results)
    targets = sorted({(r["target_rank"], r["target"]) for r in results})
    # Compact per-item outcomes:
    # [target_rank, count, similarity, cue, density, trial, outcome].
    compact = [
        [
            r["target_rank"],
            r["count"],
            r["similarity"],
            r["cue"],
            r["density"],
            r["trial"],
            r["outcome"],
        ]
        for r in results
    ]

    return {
        "total": total,
        "correct": sum(1 for r in results if r["correct"]),
        "errors": errors,
        "answered": total - errors,
        "completion_rate": round((total - errors) / total * 100, 2) if total else 0.0,
        "status": "incomplete" if errors else "completed",
        "incomplete": errors > 0,
        # A control-only run has no grid; its headline is the control's.
        **_headline(grid_rows or control_rows),
        "by_cue": {cue: _headline([r for r in grid_rows if r["cue"] == cue]) for cue in cues},
        "by_density": {
            density: _headline([r for r in grid_rows if r["density"] == density])
            for density in densities
        },
        "counts": counts,
        "similarities": levels,
        "cues": cues,
        "densities": densities,
        "trials": len(trials),
        "cells": cells,
        "control": {
            cue: _cell_summary([r for r in control_rows if r["cue"] == cue], trials, cue=cue)
            for cue in cues
            if any(r["cue"] == cue for r in control_rows)
        },
        "similarity_effects": _similarity_effects(grid_rows, counts, levels, slices),
        "count_effects": _count_effects(grid_rows, cells, counts, levels, slices),
        "cue_effects": _cue_effects(grid_rows, counts, levels, cues, densities),
        "density_effects": _density_effects(grid_rows, counts, levels, cues, densities),
        "by_position": _by_position(grid_rows),
        "similarity_check": similarity_check(items, catalog),
        "query_check": query_check(items, catalog),
        "catalog": {
            "version": catalog.version,
            "source": catalog.source,
            "sha256": catalog.sha256,
            "tools": len(catalog.tools),
            "families": len(catalog.families),
            "domains": len(catalog.domains),
        },
        "targets": [name if not catalog.is_custom else f"target-{rank}" for rank, name in targets],
        "item_outcomes": compact,
    }


def _headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Call rate, accuracy when called, and the combined accuracy over *rows*."""
    correct, n = _accuracy(rows)
    called = sum(1 for r in rows if r["picked"] is not None)
    errors = sum(1 for r in rows if r["outcome"] == ERROR)
    return {
        "grid_items": n,
        "grid_correct": correct,
        "grid_called": called,
        "grid_errors": errors,
        "accuracy": _pct(correct, n),
        "call_rate": _call_rate(called, n, errors),
        "accuracy_when_called": _pct(correct, called),
    }


def _call_rate(called: int, items: int, errors: int) -> float:
    """Share of answered items with a tool call.

    A request error has no model answer at all, so it would be wrong to read it
    as the model declining to act; it leaves the denominator here (it still
    counts as a miss in combined accuracy). A truncated turn stays in: the
    model spent its budget without calling.
    """
    return _pct(called, items - errors)


def _log_id(item: ToolscaleItem, catalog: Catalog) -> str:
    """An item label for logs; grid coordinates only for a private catalog."""
    if catalog.is_custom:
        return f"target-{item.target_rank}@N{item.count}/{item.similarity}/{item.cue}"
    return item.item_id


def _cell_summary(
    rows: list[dict[str, Any]], trials: list[int], **coordinates: Any
) -> dict[str, Any]:
    correct, n = _accuracy(rows)
    per_trial = []
    for trial in trials:
        tc, tn = _accuracy(r for r in rows if r["trial"] == trial)
        if tn:
            per_trial.append(_pct(tc, tn))
    prompt_tokens = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    called = sum(1 for r in rows if r["picked"] is not None)
    low, high = wilson_interval(correct, n)
    return {
        **coordinates,
        "items": n,
        "correct": correct,
        "accuracy": _pct(correct, n),
        "ci_low": low,
        "ci_high": high,
        "trial_accuracy": per_trial,
        "trial_stddev": round(sample_stddev(per_trial), 2),
        "outcomes": {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES},
        # Selection among items where the model did call a tool. A model that
        # declines to act (no call) is not confusing tools, so this isolates
        # confusion from abstention.
        "called": called,
        "call_rate": _call_rate(called, n, sum(1 for r in rows if r["outcome"] == ERROR)),
        "accuracy_when_called": _pct(correct, called),
        "mean_prompt_tokens": round(fmean(prompt_tokens)) if prompt_tokens else None,
    }


def _per_target_accuracy(rows: Iterable[dict[str, Any]]) -> dict[int, float]:
    """Mean accuracy per target across its trials, errors counted as misses."""
    hits: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        hits[r["target_rank"]].append(1 if r["correct"] else 0)
    return {rank: fmean(v) * 100 for rank, v in hits.items() if v}


def _paired_effect(
    a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Mean of per-target ``acc(a) - acc(b)`` with a bootstrap interval over targets."""
    a, b = _per_target_accuracy(a_rows), _per_target_accuracy(b_rows)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = [a[rank] - b[rank] for rank in shared]
    low, high = bootstrap_mean_interval(diffs)
    return {
        "delta": round(fmean(diffs), 2),
        "ci_low": round(low, 2),
        "ci_high": round(high, 2),
        "targets": len(shared),
    }


def _select(
    rows: list[dict[str, Any]],
    *,
    count: int,
    similarity: str,
    cue: str,
    density: str = "sparse",
) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r["count"] == count
        and r["similarity"] == similarity
        and r["cue"] == cue
        and r["density"] == density
    ]


def _similarity_effects(
    rows: list[dict[str, Any]],
    counts: list[int],
    levels: list[str],
    slices: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """At each N and slice, distinct minus each more similar tier: the cost of similarity."""
    if "distinct" not in levels:
        return []
    effects = []
    for cue, density in slices:
        for count in counts:
            base = _select(rows, count=count, similarity="distinct", cue=cue, density=density)
            for level in levels:
                if level == "distinct":
                    continue
                effect = _paired_effect(
                    base,
                    _select(rows, count=count, similarity=level, cue=cue, density=density),
                )
                if effect is not None:
                    effects.append(
                        {
                            "cue": cue,
                            "density": density,
                            "count": count,
                            "versus": level,
                            **effect,
                        }
                    )
    return effects


def _count_effects(
    rows: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    counts: list[int],
    levels: list[str],
    slices: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """For each tier and slice, smallest N minus largest N, and the slope per doubling.

    Crowded slices have no N=4 cell, so their range starts at the smallest
    count they were measured at.
    """
    effects = []
    for cue, density in slices:
        measured = sorted(
            {c["count"] for c in cells if c["cue"] == cue and c["density"] == density}
        )
        if len(measured) < 2:
            continue
        low_n, high_n = measured[0], measured[-1]
        for level in levels:
            effect = _paired_effect(
                _select(rows, count=low_n, similarity=level, cue=cue, density=density),
                _select(rows, count=high_n, similarity=level, cue=cue, density=density),
            )
            slope = slope_per_doubling(
                [
                    (c["count"], c["accuracy"])
                    for c in cells
                    if c["similarity"] == level and c["cue"] == cue and c["density"] == density
                ]
            )
            if effect is not None:
                effects.append(
                    {
                        "cue": cue,
                        "density": density,
                        "similarity": level,
                        "from_count": low_n,
                        "to_count": high_n,
                        **effect,
                        "slope_per_doubling": None if slope is None else round(slope, 2),
                    }
                )
    return effects


def _cue_effects(
    rows: list[dict[str, Any]],
    counts: list[int],
    levels: list[str],
    cues: list[str],
    densities: list[str],
) -> list[dict[str, Any]]:
    """Literal minus implied at each (N, tier): what naming the qualifier was worth."""
    if not {"literal", "implied"} <= set(cues):
        return []
    effects = []
    for density in densities:
        for count in counts:
            for level in levels:
                effect = _paired_effect(
                    _select(rows, count=count, similarity=level, cue="literal", density=density),
                    _select(rows, count=count, similarity=level, cue="implied", density=density),
                )
                if effect is not None:
                    effects.append(
                        {"density": density, "count": count, "similarity": level, **effect}
                    )
    return effects


def _density_effects(
    rows: list[dict[str, Any]],
    counts: list[int],
    levels: list[str],
    cues: list[str],
    densities: list[str],
) -> list[dict[str, Any]]:
    """Sparse minus crowded at each (N, tier, cue): the cost of a crowded catalog."""
    if not {"sparse", "crowded"} <= set(densities):
        return []
    effects = []
    for cue in cues:
        for count in counts:
            for level in levels:
                effect = _paired_effect(
                    _select(rows, count=count, similarity=level, cue=cue, density="sparse"),
                    _select(rows, count=count, similarity=level, cue=cue, density="crowded"),
                )
                if effect is not None:
                    effects.append({"cue": cue, "count": count, "similarity": level, **effect})
    return effects


def _by_position(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label, low, high in _POSITION_BUCKETS:
        bucket = [r for r in rows if low <= r["position"] < high]
        correct, n = _accuracy(bucket)
        if n:
            out.append({"bucket": label, "items": n, "accuracy": _pct(correct, n)})
    return out


def similarity_check(items: list[ToolscaleItem], catalog: Catalog) -> dict[str, Any]:
    """Measured lexical similarity of each tier, the check on the structural axis.

    ``target_to_neighbours`` is the mean similarity between a target and its
    three neighbours: it must rise from distinct to related to near.
    ``nearest_by_count`` is the target's nearest-neighbour similarity within
    the offered list at each N; at fixed tier it should stay roughly flat as
    fillers are added, which is what holding similarity fixed means.
    """
    sim = ToolSimilarity(tool.to_openai_tool() for tool in catalog.tools)
    defs = {tool.name: tool.to_openai_tool() for tool in catalog.tools}

    # Each (target, tool) pair counts once per tier, however many cells and
    # trials offered it, so a pair repeated across N does not dominate the mean.
    neighbour_scores: dict[str, list[float]] = defaultdict(list)
    neighbour_seen: set[tuple[str, str, str]] = set()
    filler_scores: list[float] = []
    filler_seen: set[tuple[str, str]] = set()
    nearest: dict[tuple[str, int], list[float]] = defaultdict(list)
    # The "flat across N" check is about the target's neighbourhood with
    # unrelated fillers, so it reads sparse lists when there are any; crowding
    # is measured separately in filler_crowding.
    has_sparse = any(i.density == "sparse" and i.similarity != CONTROL for i in items)
    for item in items:
        if item.similarity == CONTROL or (has_sparse and item.density != "sparse"):
            continue
        target = defs[item.target.name]
        roles = dict(item.roles)
        best = 0.0
        for tool in item.tools:
            if tool.name == item.target.name:
                continue
            score = sim.between(target, defs[tool.name])
            best = max(best, score)
            if roles[tool.name] == "neighbour":
                key3 = (item.similarity, item.target.name, tool.name)
                if key3 not in neighbour_seen:
                    neighbour_seen.add(key3)
                    neighbour_scores[item.similarity].append(score)
            elif (item.target.name, tool.name) not in filler_seen:
                filler_seen.add((item.target.name, tool.name))
                filler_scores.append(score)
        nearest[(item.similarity, item.count)].append(best)

    check: dict[str, Any] = {
        "metric": "tf-idf cosine over tool name, description, and parameter names",
        "target_to_neighbours": {
            level: round(fmean(scores), 3)
            for level in SIMILARITY_LEVELS
            if (scores := neighbour_scores.get(level))
        },
        "target_to_fillers": round(fmean(filler_scores), 3) if filler_scores else None,
        "nearest_by_count": [
            {"similarity": level, "count": count, "nearest": round(fmean(scores), 3)}
            for (level, count), scores in sorted(
                nearest.items(), key=lambda kv: (SIMILARITY_LEVELS.index(kv[0][0]), kv[0][1])
            )
        ],
    }
    check["filler_crowding"] = _filler_crowding(items, sim, defs)
    check["category_l_reference"] = _category_l_reference(sim)
    return check


def _filler_crowding(
    items: list[ToolscaleItem], sim: ToolSimilarity, defs: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Mean similarity of each filler to its closest other filler, per density.

    Measured on each density's largest tool list, once per target and trial,
    so the crowded axis is shown to be denser rather than assumed.
    """
    if not items:
        return {}
    # Each density's own largest list: sparse cells stop at the sparse maximum
    # while crowded ones can go further, and comparing different sizes would
    # leave sparse unmeasured.
    largest: dict[str, int] = {}
    for item in items:
        largest[item.density] = max(largest.get(item.density, 0), item.count)
    first_level = next((i.similarity for i in items if i.similarity != CONTROL), None)
    seen: set[tuple[str, int, str]] = set()
    scores: dict[str, list[float]] = defaultdict(list)
    for item in items:
        key = (item.target.name, item.trial, item.density)
        if item.count != largest[item.density] or item.similarity != first_level or key in seen:
            continue
        seen.add(key)
        fillers = [defs[name] for name, role in item.roles if role == "filler"]
        if len(fillers) < 2:
            continue
        for i, a in enumerate(fillers):
            scores[item.density].append(
                max(sim.between(a, b) for j, b in enumerate(fillers) if j != i)
            )
    return {density: round(fmean(v), 3) for density, v in scores.items() if v}


def _category_l_reference(sim: ToolSimilarity) -> dict[str, Any]:
    """Where Category L's fixed 52-tool set sits on the same similarity scale.

    TC-37 asks for ``get_weather`` from ``LARGE_TOOLSET``. Measuring its
    nearest neighbour in the catalog's TF-IDF space puts that single point on
    the toolscale chart without changing Category L.
    """
    vectors = {
        tool["function"]["name"]: sim.space.vector(tool_tokens(tool)) for tool in LARGE_TOOLSET
    }
    target = vectors["get_weather"]
    scores = {
        name: sim.space.cosine(target, v) for name, v in vectors.items() if name != "get_weather"
    }
    nearest_name = max(scores, key=lambda name: scores[name])
    return {
        "scenario": "TC-37",
        "count": len(LARGE_TOOLSET),
        "target": "get_weather",
        "nearest": nearest_name,
        "nearest_similarity": round(scores[nearest_name], 3),
        "mean_similarity": round(fmean(scores.values()), 3),
    }


def query_check(items: list[ToolscaleItem], catalog: Catalog) -> dict[str, Any]:
    """How much each cue's queries give the answer away, by word overlap.

    For every target in the run, compare its query with the target's own
    definition and with its closest family sibling. A literal query should
    lean clearly toward the target; an implied one should not, which is the
    point: the model has to infer, not match words. ``sibling_wins`` is the
    share of queries whose best-overlapping tool in the family is a sibling.
    """
    sim = ToolSimilarity(tool.to_openai_tool() for tool in catalog.tools)
    targets = {item.target.name: item.target for item in items}
    cues = [cue for cue in CUES if any(item.cue == cue for item in items)]
    out: dict[str, Any] = {}
    for cue in cues:
        margins: list[float] = []
        target_scores: list[float] = []
        for tool in targets.values():
            query = sim.space.vector(text_tokens(tool.query_for(cue)))
            own = sim.space.cosine(query, sim.vector_of(tool.to_openai_tool()))
            siblings = [
                sim.space.cosine(query, sim.vector_of(other.to_openai_tool()))
                for other in catalog.family_of(tool).variants
                if other.name != tool.name
            ]
            target_scores.append(own)
            margins.append(own - max(siblings))
        if margins:
            out[cue] = {
                "query_to_target": round(fmean(target_scores), 3),
                "margin_over_sibling": round(fmean(margins), 3),
                "sibling_wins": _pct(sum(1 for m in margins if m < 0), len(margins)),
            }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _cell(
    details: dict[str, Any], count: int, level: str, cue: str, density: str = "sparse"
) -> dict[str, Any] | None:
    for cell in details.get("cells", []):
        if (
            cell["count"] == count
            and cell["similarity"] == level
            and cell.get("cue") == cue
            and cell.get("density", "sparse") == density
        ):
            return cell
    return None


def _slice_label(cue: str, density: str, densities: list[str]) -> str:
    """``implied`` alone, or ``implied, crowded`` when the run swept densities."""
    return cue if len(densities) < 2 else f"{cue}, {density}"


def _table_text(value: Any, limit: int) -> str:
    """Model or catalog text made safe for one Markdown table cell.

    GFM splits a row on any unescaped pipe, even inside a code span, and
    CommonMark ends a line at a lone carriage return, so both are neutralised.
    """
    text = re.sub(r"[\r\n]+", " ", str(value))
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    # Escape after truncating, so a cut never separates a backslash from its pipe.
    return text.replace("|", "\\|")


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def _grid_table(
    d: dict[str, Any],
    counts: list[int],
    levels: list[str],
    cue: str,
    render: Any,
    density: str = "sparse",
) -> list[str]:
    lines = [
        "| Similarity | " + " | ".join(f"N={n}" for n in counts) + " |",
        "|---" + "|---:" * len(counts) + "|",
    ]
    for level in levels:
        row = []
        for count in counts:
            cell = _cell(d, count, level, cue, density)
            row.append("" if cell is None else render(cell))
        lines.append(f"| {level} | " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_report(result: BenchmarkResult) -> list[str]:
    """Markdown for the run report."""
    d = result.details
    counts: list[int] = d.get("counts", [])
    levels: list[str] = d.get("similarities", [])
    cues: list[str] = d.get("cues", ["literal"])
    densities: list[str] = d.get("densities", ["sparse"])
    slices = [(cue, density) for cue in cues for density in densities]
    catalog = d.get("catalog", {})
    trials = d.get("trials", 1)

    lines = [
        "## Toolscale: Tool Count x Description Similarity",
        "",
        f"**Accuracy when a tool was called:** {d.get('accuracy_when_called', 0):.1f}% "
        f"({d.get('grid_correct', 0)}/{d.get('grid_called', 0)})",
        f"**Call rate:** {d.get('call_rate', 0):.1f}% "
        f"({d.get('grid_called', 0)}/{d.get('grid_items', 0)})",
        f"**Combined accuracy (stored score):** {d.get('accuracy', 0):.1f}%",
        f"**Rating:** {result.rating}",
        f"**Catalog:** v{catalog.get('version', '?')} ({catalog.get('source', '?')}, "
        f"{catalog.get('tools', 0)} tools, sha256 `{str(catalog.get('sha256', ''))[:12]}`)",
        f"**Cues:** {', '.join(cues)} · **Fillers:** {', '.join(densities)} · **Trials:** {trials}",
        f"**Duration:** {result.duration_seconds:.1f}s",
        f"**Tokens:** {result.total_tokens:,}",
        "",
        "Each item offers the target tool, 3 neighbours from one similarity tier, and unrelated "
        "fillers up to N tools. Only the first tool call's name is graded. The two headline "
        "numbers are reported apart on purpose: accuracy when called measures tool confusion, "
        "the call rate measures whether the model acted at all (a clarifying question counts "
        "as no call). Request errors count as misses; intervals are 95% Wilson.",
        "",
    ]

    for axis, title, axis_values in (
        ("by_cue", "Cue", cues),
        ("by_density", "Fillers", densities),
    ):
        breakdown = d.get(axis, {})
        if len(axis_values) < 2 or not breakdown:
            continue
        lines.extend(
            [
                f"### Headline by {title}",
                "",
                f"| {title} | Accuracy when called | Call rate | Combined accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for value in axis_values:
            h = breakdown.get(value, {})
            lines.append(
                f"| {value} | {h.get('accuracy_when_called', 0):.1f}% "
                f"({h.get('grid_correct', 0)}/{h.get('grid_called', 0)}) | "
                f"{h.get('call_rate', 0):.1f}% | {h.get('accuracy', 0):.1f}% |"
            )
        lines.append("")

    if counts and levels:
        for cue, density in slices:
            suffix = f"{cue} cue" + (f", {density} fillers" if len(densities) > 1 else "")
            lines.extend(
                [f"### Accuracy When a Tool Was Called ({suffix})", ""]
                + _grid_table(
                    d,
                    counts,
                    levels,
                    cue,
                    lambda c: f"{c['accuracy_when_called']:.1f}% ({c['correct']}/{c['called']})",
                    density,
                )
            )
            lines.extend(
                [f"### Call Rate ({suffix})", ""]
                + _grid_table(
                    d,
                    counts,
                    levels,
                    cue,
                    lambda c: f"{c['call_rate']:.1f}% ({c['called']}/{c['items']})",
                    density,
                )
            )

            def combined(c: dict[str, Any]) -> str:
                text = f"{c['accuracy']:.1f}% ({c['ci_low']:.0f}-{c['ci_high']:.0f})"
                if trials > 1:
                    text += f" ±{c['trial_stddev']:.1f}"
                return text

            lines.extend(
                [f"### Combined Accuracy ({suffix})", ""]
                + _grid_table(d, counts, levels, cue, combined, density)
            )
        if trials > 1:
            lines.extend(["*± is the standard deviation of cell accuracy across trials.*", ""])

    controls = d.get("control") or {}
    for cue, control in controls.items():
        lines.extend(
            [
                f"**Control (N=1, target only, {cue} cue):** {control['accuracy']:.1f}% "
                f"({control['correct']}/{control['items']}); call rate "
                f"{control['call_rate']:.1f}%, truncated {control['outcomes'].get('truncated', 0)}",
                "",
            ]
        )

    cue_effects = d.get("cue_effects", [])
    if cue_effects:
        lines.extend(
            [
                "### Cue Effect",
                "",
                "Accuracy with the literal query minus accuracy with the implied query, on the "
                "identical tool list, paired by target. Positive means the model relied on the "
                "query naming the distinguishing word.",
                "",
                "| Fillers | N | Similarity | Delta (pts) | 95% CI | Targets |",
                "|---|---:|---|---:|---|---:|",
            ]
        )
        for e in cue_effects:
            lines.append(
                f"| {e.get('density', 'sparse')} | {e['count']} | {e['similarity']} | "
                f"{_signed(e['delta'])} | {_signed(e['ci_low'])} to {_signed(e['ci_high'])} | "
                f"{e['targets']} |"
            )
        lines.append("")

    density_effects = d.get("density_effects", [])
    if density_effects:
        lines.extend(
            [
                "### Density Effect",
                "",
                "Accuracy with sparse fillers minus accuracy with crowded fillers, with the "
                "same target, neighbours, and slot layout, paired by target. Positive means a "
                "catalog dense with near-duplicate clusters costs accuracy.",
                "",
                "| Cue | N | Similarity | Delta (pts) | 95% CI | Targets |",
                "|---|---:|---|---:|---|---:|",
            ]
        )
        for e in density_effects:
            lines.append(
                f"| {e['cue']} | {e['count']} | {e['similarity']} | {_signed(e['delta'])} | "
                f"{_signed(e['ci_low'])} to {_signed(e['ci_high'])} | {e['targets']} |"
            )
        lines.append("")

    sim_effects = d.get("similarity_effects", [])
    if sim_effects:
        lines.extend(
            [
                "### Similarity Effect at Fixed N",
                "",
                "Accuracy with distinct neighbours minus accuracy with more similar neighbours, "
                "paired by target. Positive means similarity costs accuracy.",
                "",
                "| Slice | N | Versus | Delta (pts) | 95% CI | Targets |",
                "|---|---:|---|---:|---|---:|",
            ]
        )
        for e in sim_effects:
            label = _slice_label(e["cue"], e.get("density", "sparse"), densities)
            lines.append(
                f"| {label} | {e['count']} | {e['versus']} | {_signed(e['delta'])} | "
                f"{_signed(e['ci_low'])} to {_signed(e['ci_high'])} | {e['targets']} |"
            )
        lines.append("")

    count_effects = d.get("count_effects", [])
    if count_effects:
        lines.extend(
            [
                "### Count Effect at Fixed Similarity",
                "",
                "Accuracy at the smallest N minus accuracy at the largest, paired by target. "
                "A positive delta means more tools cost accuracy. The slope is a least-squares "
                "fit of cell accuracy against log2(N), in points per doubling: negative means "
                "accuracy falls as the list grows.",
                "",
                "| Slice | Similarity | From N | To N | Delta (pts) | 95% CI | Slope per doubling |",
                "|---|---|---:|---:|---:|---|---:|",
            ]
        )
        for e in count_effects:
            slope = e.get("slope_per_doubling")
            label = _slice_label(e["cue"], e.get("density", "sparse"), densities)
            lines.append(
                f"| {label} | {e['similarity']} | {e['from_count']} | {e['to_count']} | "
                f"{_signed(e['delta'])} | {_signed(e['ci_low'])} to {_signed(e['ci_high'])} | "
                f"{'' if slope is None else _signed(slope)} |"
            )
        lines.append("")

    cells = d.get("cells", [])
    if cells:
        lines.extend(
            [
                "### Error Mix",
                "",
                "A wrong neighbour in the near tier is similarity confusion. A wrong filler is a "
                "count or attention failure. In the distinct tier both are unrelated tools, so "
                "their rates are the baseline.",
                "",
                "| Slice | N | Similarity | Correct | Wrong neighbour | Wrong filler | "
                "Hallucinated | No call | Truncated | Error | Prompt tokens |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cells:
            o = c["outcomes"]
            tokens = c.get("mean_prompt_tokens")
            label = _slice_label(c.get("cue", "literal"), c.get("density", "sparse"), densities)
            lines.append(
                f"| {label} | {c['count']} | {c['similarity']} | "
                f"{o['correct']} | {o['wrong_neighbour']} | {o['wrong_filler']} | "
                f"{o['hallucinated']} | {o['no_call']} | {o['truncated']} | {o['error']} | "
                f"{'' if tokens is None else f'{tokens:,}'} |"
            )
        lines.append("")

    queries = d.get("query_check") or {}
    if queries:
        lines.extend(
            [
                "### Query Check",
                "",
                "Word overlap (TF-IDF cosine) between each target's query and its own definition, "
                "and the margin over its closest family sibling. A literal query should lean "
                "toward the target; an implied query should not, so the model has to infer.",
                "",
                "| Cue | Query to target | Margin over closest sibling | Sibling overlaps more |",
                "|---|---:|---:|---:|",
            ]
        )
        for cue, q in queries.items():
            lines.append(
                f"| {cue} | {q['query_to_target']:.3f} | {q['margin_over_sibling']:+.3f} | "
                f"{q['sibling_wins']:.1f}% |"
            )
        lines.append("")

    check = d.get("similarity_check") or {}
    if check:
        lines.extend(
            [
                "### Similarity Check",
                "",
                f"Measured with {check.get('metric', 'tf-idf cosine')}. The tiers are defined by "
                "the catalog's structure; this confirms the ordering.",
                "",
                "| Tier | Target to neighbours |",
                "|---|---:|",
            ]
        )
        for level, score in check.get("target_to_neighbours", {}).items():
            lines.append(f"| {level} | {score:.3f} |")
        if check.get("target_to_fillers") is not None:
            lines.append(f"| fillers | {check['target_to_fillers']:.3f} |")
        lines.append("")
        crowding = check.get("filler_crowding") or {}
        if crowding:
            lines.extend(
                [
                    "Filler crowding, each filler's similarity to its closest other filler "
                    "on the largest list:",
                    "",
                    "| Fillers | Closest-filler similarity |",
                    "|---|---:|",
                    *(f"| {density} | {value:.3f} |" for density, value in crowding.items()),
                    "",
                ]
            )
        nearest = check.get("nearest_by_count", [])
        if nearest:
            lines.extend(
                [
                    "Target's nearest-neighbour similarity within the offered list:",
                    "",
                    "| Similarity | " + " | ".join(f"N={n}" for n in counts) + " |",
                    "|---" + "|---:" * len(counts) + "|",
                ]
            )
            lookup = {(n["similarity"], n["count"]): n["nearest"] for n in nearest}
            for level in levels:
                values = [lookup.get((level, count)) for count in counts]
                lines.append(
                    f"| {level} | "
                    + " | ".join("" if v is None else f"{v:.3f}" for v in values)
                    + " |"
                )
            lines.append("")
        ref = check.get("category_l_reference")
        if ref:
            lines.extend(
                [
                    f"**Category L reference:** {ref['scenario']} offers {ref['count']} tools; "
                    f"its target `{ref['target']}` has nearest neighbour `{ref['nearest']}` at "
                    f"{ref['nearest_similarity']:.3f} (mean to all others "
                    f"{ref['mean_similarity']:.3f}) on the same scale.",
                    "",
                ]
            )

    positions = d.get("by_position", [])
    if positions:
        lines.extend(
            [
                "### Accuracy by Target Position",
                "",
                "| Target position | Items | Accuracy |",
                "|---|---:|---:|",
            ]
        )
        for p in positions:
            lines.append(f"| {p['bucket']} | {p['items']} | {p['accuracy']:.1f}% |")
        lines.append("")

    failures = [r for r in result.item_results if not r.get("correct")]
    if failures:
        lines.append(f"### Failed Items ({len(failures)} total)")
        lines.append("")
        if catalog.get("source") == "custom":
            lines.extend(
                [
                    "Tool names and queries are withheld because this run used a private "
                    "catalog. Counts by outcome are in the error mix above.",
                    "",
                ]
            )
            return lines
        # Wrong picks first: they are the confusion signal, and a long run of
        # clarifying questions would otherwise push them past the listing cap.
        failures.sort(key=lambda f: (f.get("picked") is None, f["item_id"]))
        lines.extend(["| Item | Query | Picked | Outcome |", "|---|---|---|---|"])
        for f in failures[:_MAX_LISTED_FAILURES]:
            query = _table_text(f.get("query", ""), 100)
            if f.get("picked"):
                picked = f"`{_table_text(f['picked'], 80)}`"
            elif f.get("error"):
                picked = f"*(request failed)* {_table_text(f['error'], 80)}"
            elif f.get("reply"):
                picked = f"*(no call)* {_table_text(f['reply'], 80)}"
            else:
                picked = "*(none)*"
            lines.append(f"| `{f['item_id']}` | {query} | {picked} | {f['outcome']} |")
        if len(failures) > _MAX_LISTED_FAILURES:
            lines.append("")
            lines.append(
                f"*{len(failures) - _MAX_LISTED_FAILURES} more failures not listed. Every item "
                "is counted in the tables above, and the stored run keeps each item's outcome.*"
            )
        lines.append("")

    return lines
