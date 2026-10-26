"""Toolscale plugin: orchestration, aggregation, and report rendering.

Implements ``BenchmarkPlugin`` for the tool-count by similarity sweep from
issue #319. Each item is one streamed request with a generated tool list; the
first tool call's name is graded. See ``grid.py`` for how the lists are built
and ``docs/toolscale.md`` for the method.
"""

from __future__ import annotations

import asyncio
import logging
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
    ERROR,
    OUTCOMES,
    SIMILARITY_LEVELS,
    ToolscaleItem,
    build_messages,
    grade,
    openai_tools,
)
from tool_eval_bench.plugins.toolscale.similarity import ToolSimilarity, tool_tokens
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
                "trial": item.trial,
                "position": round(item.position, 4),
                "query": item.target.query,
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
                logger.debug("Error on toolscale item %s: %s", item.item_id, exc)
                outcome, picked = ERROR, None

            results[idx] = row(
                item,
                outcome,
                picked,
                prompt_tokens=prompt_tokens,
                arguments=arguments,
                reply=reply,
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
                logger.error("Toolscale item %s crashed: %s", items[i].item_id, exc)
                if not results[i]:
                    results[i] = row(
                        items[i], ERROR, None, prompt_tokens=None, arguments=None, reply=None
                    )

        details = summarize(results, items, catalog)
        duration = time.monotonic() - t_start
        accuracy = details["accuracy"]
        return BenchmarkResult(
            plugin_name="toolscale",
            score=accuracy,
            score_label=(
                f"{accuracy:.1f}% selection accuracy "
                f"({details['grid_correct']}/{details['grid_items']} grid items)"
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

    The stored run keeps ``details`` but not ``item_results``, so the per-item
    outcomes are also kept here in a compact form for later paired analysis.
    A private catalog's tool names and queries never enter ``details``.
    """
    counts = sorted({r["count"] for r in results if r["similarity"] != CONTROL})
    levels = [lvl for lvl in SIMILARITY_LEVELS if any(r["similarity"] == lvl for r in results)]
    trials = sorted({r["trial"] for r in results})
    grid_rows = [r for r in results if r["similarity"] != CONTROL]
    control_rows = [r for r in results if r["similarity"] == CONTROL]

    by_cell: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in grid_rows:
        by_cell[(r["count"], r["similarity"])].append(r)

    cells: list[dict[str, Any]] = []
    for count in counts:
        for level in levels:
            rows = by_cell.get((count, level), [])
            if not rows:
                continue
            correct, n = _accuracy(rows)
            per_trial = []
            for trial in trials:
                tc, tg = _accuracy(r for r in rows if r["trial"] == trial)
                if tg:
                    per_trial.append(_pct(tc, tg))
            prompt_tokens = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
            called = [r for r in rows if r["picked"] is not None]
            low, high = wilson_interval(correct, n)
            cells.append(
                {
                    "count": count,
                    "similarity": level,
                    "items": n,
                    "correct": correct,
                    "accuracy": _pct(correct, n),
                    "ci_low": low,
                    "ci_high": high,
                    "trial_accuracy": per_trial,
                    "trial_stddev": round(sample_stddev(per_trial), 2),
                    "outcomes": {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES},
                    # Selection among items where the model did call a tool. A
                    # model that declines to act (no call) is not confusing
                    # tools, so this isolates confusion from abstention.
                    "called": len(called),
                    "accuracy_when_called": _pct(correct, len(called)),
                    "mean_prompt_tokens": round(fmean(prompt_tokens)) if prompt_tokens else None,
                }
            )

    grid_correct, grid_items = _accuracy(grid_rows)
    control_correct, control_items = _accuracy(control_rows)
    errors = sum(1 for r in results if r["outcome"] == ERROR)
    total = len(results)

    control = None
    if control_rows:
        low, high = wilson_interval(control_correct, control_items)
        control = {
            "items": control_items,
            "correct": control_correct,
            "accuracy": _pct(control_correct, control_items),
            "ci_low": low,
            "ci_high": high,
            "outcomes": {o: sum(1 for r in control_rows if r["outcome"] == o) for o in OUTCOMES},
        }

    targets = sorted({(r["target_rank"], r["target"]) for r in results})
    # Compact per-item outcomes: [target_rank, count, similarity, trial, outcome].
    compact = [
        [r["target_rank"], r["count"], r["similarity"], r["trial"], r["outcome"]] for r in results
    ]

    return {
        "total": total,
        "correct": grid_correct + control_correct,
        "errors": errors,
        "answered": total - errors,
        "completion_rate": round((total - errors) / total * 100, 2) if total else 0.0,
        "status": "incomplete" if errors else "completed",
        "incomplete": errors > 0,
        "accuracy": _pct(grid_correct, grid_items),
        "grid_correct": grid_correct,
        "grid_items": grid_items,
        "counts": counts,
        "similarities": levels,
        "trials": len(trials),
        "cells": cells,
        "control": control,
        "similarity_effects": _similarity_effects(grid_rows, counts, levels),
        "count_effects": _count_effects(grid_rows, cells, counts, levels),
        "by_position": _by_position(grid_rows),
        "similarity_check": similarity_check(items, catalog),
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


def _similarity_effects(
    rows: list[dict[str, Any]], counts: list[int], levels: list[str]
) -> list[dict[str, Any]]:
    """At each N, distinct minus each more similar tier: the cost of similarity."""
    if "distinct" not in levels:
        return []
    effects = []
    for count in counts:
        base = [r for r in rows if r["count"] == count and r["similarity"] == "distinct"]
        for level in levels:
            if level == "distinct":
                continue
            other = [r for r in rows if r["count"] == count and r["similarity"] == level]
            effect = _paired_effect(base, other)
            if effect is not None:
                effects.append({"count": count, "versus": level, **effect})
    return effects


def _count_effects(
    rows: list[dict[str, Any]], cells: list[dict[str, Any]], counts: list[int], levels: list[str]
) -> list[dict[str, Any]]:
    """For each tier, smallest N minus largest N, and the slope per doubling."""
    if len(counts) < 2:
        return []
    low_n, high_n = counts[0], counts[-1]
    effects = []
    for level in levels:
        small = [r for r in rows if r["count"] == low_n and r["similarity"] == level]
        large = [r for r in rows if r["count"] == high_n and r["similarity"] == level]
        effect = _paired_effect(small, large)
        slope = slope_per_doubling(
            [(c["count"], c["accuracy"]) for c in cells if c["similarity"] == level]
        )
        if effect is not None:
            effects.append(
                {
                    "similarity": level,
                    "from_count": low_n,
                    "to_count": high_n,
                    **effect,
                    "slope_per_doubling": None if slope is None else round(slope, 2),
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
    for item in items:
        if item.similarity == CONTROL:
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
    check["category_l_reference"] = _category_l_reference(sim)
    return check


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


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _cell(details: dict[str, Any], count: int, level: str) -> dict[str, Any] | None:
    for cell in details.get("cells", []):
        if cell["count"] == count and cell["similarity"] == level:
            return cell
    return None


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def render_report(result: BenchmarkResult) -> list[str]:
    """Markdown for the run report."""
    d = result.details
    counts: list[int] = d.get("counts", [])
    levels: list[str] = d.get("similarities", [])
    catalog = d.get("catalog", {})
    trials = d.get("trials", 1)

    lines = [
        "## Toolscale: Tool Count x Description Similarity",
        "",
        f"**Selection Accuracy (grid):** {d.get('accuracy', 0):.1f}% "
        f"({d.get('grid_correct', 0)}/{d.get('grid_items', 0)})",
        f"**Rating:** {result.rating}",
        f"**Catalog:** v{catalog.get('version', '?')} ({catalog.get('source', '?')}, "
        f"{catalog.get('tools', 0)} tools, sha256 `{str(catalog.get('sha256', ''))[:12]}`)",
        f"**Trials:** {trials}",
        f"**Duration:** {result.duration_seconds:.1f}s",
        f"**Tokens:** {result.total_tokens:,}",
        "",
        "Each item offers the target tool, 3 neighbours from one similarity tier, and unrelated "
        "fillers up to N tools. Only the first tool call's name is graded. Request errors "
        "count as misses and appear in the error mix; the 95% interval is Wilson.",
        "",
    ]

    if counts and levels:
        lines.extend(
            [
                "### Accuracy Grid",
                "",
                "| Similarity | " + " | ".join(f"N={n}" for n in counts) + " |",
                "|---" + "|---:" * len(counts) + "|",
            ]
        )
        for level in levels:
            row = []
            for count in counts:
                cell = _cell(d, count, level)
                if cell is None:
                    row.append("")
                    continue
                text = f"{cell['accuracy']:.1f}% ({cell['ci_low']:.0f}-{cell['ci_high']:.0f})"
                if trials > 1:
                    text += f" ±{cell['trial_stddev']:.1f}"
                row.append(text)
            lines.append(f"| {level} | " + " | ".join(row) + " |")
        lines.append("")
        if trials > 1:
            lines.extend(["*± is the standard deviation of cell accuracy across trials.*", ""])

    if counts and levels and d.get("cells") and "called" in d["cells"][0]:
        lines.extend(
            [
                "### Accuracy When a Tool Was Called",
                "",
                "The same cells, counting only items where the model called a tool. A gap "
                "between this grid and the one above is abstention (no call), not confusion.",
                "",
                "| Similarity | " + " | ".join(f"N={n}" for n in counts) + " |",
                "|---" + "|---:" * len(counts) + "|",
            ]
        )
        for level in levels:
            row = []
            for count in counts:
                cell = _cell(d, count, level)
                row.append(
                    ""
                    if cell is None
                    else f"{cell['accuracy_when_called']:.1f}% ({cell['called']}/{cell['items']})"
                )
            lines.append(f"| {level} | " + " | ".join(row) + " |")
        lines.append("")

    control = d.get("control")
    if control:
        lines.extend(
            [
                f"**Control (N=1, target only):** {control['accuracy']:.1f}% "
                f"({control['correct']}/{control['items']}); "
                f"no call {control['outcomes'].get('no_call', 0)}, "
                f"truncated {control['outcomes'].get('truncated', 0)}",
                "",
            ]
        )

    sim_effects = d.get("similarity_effects", [])
    if sim_effects:
        lines.extend(
            [
                "### Similarity Effect at Fixed N",
                "",
                "Accuracy with distinct neighbours minus accuracy with more similar neighbours, "
                "paired by target. Positive means similarity costs accuracy.",
                "",
                "| N | Versus | Delta (pts) | 95% CI | Targets |",
                "|---:|---|---:|---|---:|",
            ]
        )
        for e in sim_effects:
            lines.append(
                f"| {e['count']} | {e['versus']} | {_signed(e['delta'])} | "
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
                "Positive means more tools cost accuracy. The slope is a least-squares fit of "
                "cell accuracy against log2(N).",
                "",
                "| Similarity | From N | To N | Delta (pts) | 95% CI | Slope per doubling |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for e in count_effects:
            slope = e.get("slope_per_doubling")
            lines.append(
                f"| {e['similarity']} | {e['from_count']} | {e['to_count']} | "
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
                "| N | Similarity | Correct | Wrong neighbour | Wrong filler | Hallucinated "
                "| No call | Truncated | Error | Prompt tokens |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cells:
            o = c["outcomes"]
            tokens = c.get("mean_prompt_tokens")
            lines.append(
                f"| {c['count']} | {c['similarity']} | {o['correct']} | {o['wrong_neighbour']} "
                f"| {o['wrong_filler']} | {o['hallucinated']} | {o['no_call']} "
                f"| {o['truncated']} | {o['error']} | {'' if tokens is None else f'{tokens:,}'} |"
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
        lines.extend(["| Item | Query | Picked | Outcome |", "|---|---|---|---|"])
        for f in failures[:_MAX_LISTED_FAILURES]:
            query = str(f.get("query", "")).replace("|", "\\|").replace("\n", " ")
            if len(query) > 100:
                query = query[:97] + "…"
            if f.get("picked"):
                picked = f"`{f['picked']}`"
            elif f.get("reply"):
                reply = str(f["reply"]).replace("|", "\\|").replace("\n", " ")
                if len(reply) > 80:
                    reply = reply[:77] + "…"
                picked = f"*(no call)* {reply}"
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
