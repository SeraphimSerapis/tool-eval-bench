"""Toolscale aggregation: grid cells, paired contrasts, and similarity/query checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import product
from statistics import fmean
from typing import Any

from tool_eval_bench.domain.tools_large import LARGE_TOOLSET
from tool_eval_bench.plugins.toolscale.catalog import Catalog
from tool_eval_bench.plugins.toolscale.grid import (
    AXES,
    CONTROL,
    CUES,
    DESCRIPTIONS,
    ERROR,
    OUTCOMES,
    SIMILARITY_LEVELS,
    ToolscaleItem,
)
from tool_eval_bench.plugins.toolscale.similarity import ToolSimilarity, text_tokens, tool_tokens
from tool_eval_bench.plugins.toolscale.stats import (
    bootstrap_mean_interval,
    sample_stddev,
    slope_per_doubling,
    wilson_interval,
)

# Relative position buckets for the "where in the list was the target" table.
_POSITION_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("first third", 0.0, 1 / 3),
    ("middle third", 1 / 3, 2 / 3),
    ("last third", 2 / 3, 1.0001),
)


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
    the two apart. ``accuracy`` (correct over all grid items) is kept as the
    stored score so runs still sort.

    Cells and effects are computed per *slice*: one value of each condition
    axis (cue, filler density, description quality). An axis the run did not
    sweep has a single value and adds nothing to labels.

    The stored run keeps ``details`` but not ``item_results``, so the per-item
    outcomes are also kept here in a compact form for later paired analysis.
    A private catalog's tool names and queries never enter ``details``.
    """
    counts = sorted({r["count"] for r in results if r["similarity"] != CONTROL})
    levels = [lvl for lvl in SIMILARITY_LEVELS if any(r["similarity"] == lvl for r in results)]
    trials = sorted({r["trial"] for r in results})
    grid_rows = [r for r in results if r["similarity"] != CONTROL]
    control_rows = [r for r in results if r["similarity"] == CONTROL]
    # Axes come from the grid: control rows are always "sparse" (they have no
    # fillers), so reading them would invent a sparse slice in a crowded-only
    # run. A control-only run falls back to its control rows.
    present = _present_axes(grid_rows or control_rows)
    slices = _slices(present)

    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in grid_rows:
        by_cell[(r["count"], r["similarity"], *(r[axis] for axis, _ in AXES))].append(r)

    cells: list[dict[str, Any]] = []
    for slice_ in slices:
        for count in counts:
            for level in levels:
                key = (count, level, *(slice_[axis] for axis, _ in AXES))
                rows = by_cell.get(key, [])
                if rows:
                    cells.append(
                        _cell_summary(rows, trials, count=count, similarity=level, **slice_)
                    )

    # The control has no fillers, so it varies only by cue and descriptions.
    control: dict[str, Any] = {}
    for slice_ in _slices({**present, "density": ["sparse"]}):
        rows = [r for r in control_rows if all(r[a] == v for a, v in slice_.items())]
        if rows:
            label = _label(slice_, present, skip=("density",))
            control[label] = _cell_summary(rows, trials, **slice_)

    errors = sum(1 for r in results if r["outcome"] == ERROR)
    total = len(results)
    targets = sorted({(r["target_rank"], r["target"]) for r in results})
    # Compact per-item outcomes:
    # [target_rank, count, similarity, cue, density, descriptions, trial, outcome].
    compact = [
        [
            r["target_rank"],
            r["count"],
            r["similarity"],
            *(r[axis] for axis, _ in AXES),
            r["trial"],
            r["outcome"],
        ]
        for r in results
    ]

    sim = ToolSimilarity(tool.to_openai_tool() for tool in catalog.tools)

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
        **{
            f"by_{axis}": {
                value: _headline([r for r in grid_rows if r[axis] == value])
                for value in present[axis]
            }
            for axis, _ in AXES
        },
        "counts": counts,
        "similarities": levels,
        "cues": present["cue"],
        "densities": present["density"],
        "descriptions": present["descriptions"],
        "trials": len(trials),
        "cells": cells,
        "control": control,
        "similarity_effects": _similarity_effects(grid_rows, counts, levels, slices),
        "count_effects": _count_effects(grid_rows, cells, levels, slices),
        "cue_effects": _axis_effects(grid_rows, counts, levels, present, "cue", "literal"),
        "density_effects": _axis_effects(grid_rows, counts, levels, present, "density", "sparse"),
        "description_effects": _axis_effects(
            grid_rows, counts, levels, present, "descriptions", "clean"
        ),
        "by_position": _by_position(grid_rows),
        "similarity_check": similarity_check(items, catalog, sim=sim),
        "query_check": query_check(items, catalog, sim=sim),
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


def _present_axes(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Each condition axis's values that occur in *rows*, in canonical order."""
    return {
        axis: [value for value in values if any(r[axis] == value for r in rows)]
        for axis, values in AXES
    }


def _slices(present: dict[str, list[str]]) -> list[dict[str, str]]:
    """Every combination of one value per axis, in canonical order."""
    names = [axis for axis, _ in AXES]
    return [dict(zip(names, combo, strict=True)) for combo in product(*(present[n] for n in names))]


def _label(
    slice_: dict[str, str],
    present: dict[str, list[str]],
    skip: tuple[str, ...] = (),
    fallback: str | None = None,
) -> str:
    """A slice's values on the axes the run varied, joined: ``implied, terse``.

    With nothing varied the label is *fallback*, or the cue when none is
    given, so a plain run keeps keys such as ``"literal"``.
    """
    varied = [
        slice_[axis]
        for axis, _ in AXES
        if axis not in skip and axis in slice_ and len(present.get(axis, [])) > 1
    ]
    if varied:
        return ", ".join(varied)
    return fallback if fallback is not None else slice_.get("cue", "literal")


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
    rows: list[dict[str, Any]], *, count: int, similarity: str, **axes: str
) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r["count"] == count
        and r["similarity"] == similarity
        and all(r[axis] == value for axis, value in axes.items())
    ]


def _similarity_effects(
    rows: list[dict[str, Any]],
    counts: list[int],
    levels: list[str],
    slices: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """At each N and slice, distinct minus each more similar tier: the cost of similarity."""
    if "distinct" not in levels:
        return []
    effects = []
    for slice_ in slices:
        for count in counts:
            base = _select(rows, count=count, similarity="distinct", **slice_)
            for level in levels:
                if level == "distinct":
                    continue
                effect = _paired_effect(
                    base, _select(rows, count=count, similarity=level, **slice_)
                )
                if effect is not None:
                    effects.append({**slice_, "count": count, "versus": level, **effect})
    return effects


def _count_effects(
    rows: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    levels: list[str],
    slices: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """For each tier and slice, smallest N minus largest N, and the slope per doubling.

    Crowded slices have no N=4 cell, and counts above the sparse maximum have
    no sparse cell, so each slice's range is the counts it was measured at.
    """
    effects = []
    for slice_ in slices:
        in_slice = [c for c in cells if all(c[a] == v for a, v in slice_.items())]
        measured = sorted({c["count"] for c in in_slice})
        if len(measured) < 2:
            continue
        low_n, high_n = measured[0], measured[-1]
        for level in levels:
            effect = _paired_effect(
                _select(rows, count=low_n, similarity=level, **slice_),
                _select(rows, count=high_n, similarity=level, **slice_),
            )
            slope = slope_per_doubling(
                [(c["count"], c["accuracy"]) for c in in_slice if c["similarity"] == level]
            )
            if effect is not None:
                effects.append(
                    {
                        **slice_,
                        "similarity": level,
                        "from_count": low_n,
                        "to_count": high_n,
                        **effect,
                        "slope_per_doubling": None if slope is None else round(slope, 2),
                    }
                )
    return effects


def _axis_effects(
    rows: list[dict[str, Any]],
    counts: list[int],
    levels: list[str],
    present: dict[str, list[str]],
    axis: str,
    base: str,
) -> list[dict[str, Any]]:
    """*base* minus each other value of *axis*, at each N, tier, and other-axis slice.

    The literal-minus-implied cue effect, sparse-minus-crowded density effect,
    and clean-minus-degraded description effect are all this contrast: the
    same targets and tool lists, with only one condition changed.
    """
    if base not in present[axis] or len(present[axis]) < 2:
        return []
    others = _slices({**present, axis: [base]})
    effects = []
    for other in others:
        fixed = {a: v for a, v in other.items() if a != axis}
        for value in present[axis]:
            if value == base:
                continue
            for count in counts:
                for level in levels:
                    effect = _paired_effect(
                        _select(rows, count=count, similarity=level, **fixed, **{axis: base}),
                        _select(rows, count=count, similarity=level, **fixed, **{axis: value}),
                    )
                    if effect is not None:
                        effects.append(
                            {
                                **fixed,
                                "versus": value,
                                "count": count,
                                "similarity": level,
                                **effect,
                            }
                        )
    return effects


def _by_position(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label, low, high in _POSITION_BUCKETS:
        bucket = [r for r in rows if low <= r["position"] < high]
        correct, n = _accuracy(bucket)
        if n:
            out.append({"bucket": label, "items": n, "accuracy": _pct(correct, n)})
    return out


def similarity_check(
    items: list[ToolscaleItem],
    catalog: Catalog,
    *,
    sim: ToolSimilarity | None = None,
) -> dict[str, Any]:
    """Measured lexical similarity of each tier, the check on the structural axis.

    ``target_to_neighbours`` is the mean similarity between a target and its
    three neighbours: it must rise from distinct to related to near.
    ``nearest_by_count`` is the target's nearest-neighbour similarity within
    the offered list at each N; at fixed tier it should stay roughly flat as
    fillers are added, which is what holding similarity fixed means.
    """
    if sim is None:
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


def query_check(
    items: list[ToolscaleItem],
    catalog: Catalog,
    *,
    sim: ToolSimilarity | None = None,
) -> dict[str, Any]:
    """How much each cue's queries give the answer away, by word overlap.

    For every target in the run, compare its query with the target's own
    definition and with its closest family sibling. A literal query should
    lean clearly toward the target; an implied one should not, which is the
    point: the model has to infer, not match words. ``sibling_wins`` is the
    share of queries whose best-overlapping tool in the family is a sibling.

    Keys are the cue, plus the description level when it is not clean
    (``"implied, terse"``): degraded descriptions should shrink the lean, since
    the words that explain each variant are gone.
    """
    if sim is None:
        sim = ToolSimilarity(tool.to_openai_tool() for tool in catalog.tools)
    targets = {item.target.name: item.target for item in items}
    cues = [cue for cue in CUES if any(item.cue == cue for item in items)]
    levels = [lvl for lvl in DESCRIPTIONS if any(item.descriptions == lvl for item in items)]

    def vector(tool: Any, level: str) -> dict[str, float]:
        if level == "clean":
            return sim.vector_of(tool.to_openai_tool())
        # Degraded definitions bypass the name-keyed cache, which holds clean ones.
        return sim.space.vector(tool_tokens(tool.to_openai_tool(level)))

    out: dict[str, Any] = {}
    for cue in cues:
        for level in levels:
            margins: list[float] = []
            target_scores: list[float] = []
            for tool in targets.values():
                query = sim.space.vector(text_tokens(tool.query_for(cue)))
                own = sim.space.cosine(query, vector(tool, level))
                siblings = [
                    sim.space.cosine(query, vector(other, level))
                    for other in catalog.family_of(tool).variants
                    if other.name != tool.name
                ]
                target_scores.append(own)
                margins.append(own - max(siblings))
            if margins:
                key = cue if level == "clean" else f"{cue}, {level}"
                out[key] = {
                    "query_to_target": round(fmean(target_scores), 3),
                    "margin_over_sibling": round(fmean(margins), 3),
                    "sibling_wins": _pct(sum(1 for m in margins if m < 0), len(margins)),
                }
    return out
