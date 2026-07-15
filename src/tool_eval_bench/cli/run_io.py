"""Trial aggregation helpers used by CLI output modes."""

from __future__ import annotations

import json
import random
import sys
from statistics import mean, stdev
from typing import Any

from tool_eval_bench.domain.scenarios import ScenarioDefinition, ScenarioResult


async def stderr_progress_start(scenario: ScenarioDefinition, idx: int, total: int) -> None:
    """Emit a JSONL progress event when a scenario starts."""
    message = {
        "event": "scenario_start",
        "scenario_id": scenario.id,
        "title": scenario.title,
        "category": scenario.category.value,
        "index": idx,
        "total": total,
    }
    sys.stderr.write(json.dumps(message) + "\n")
    sys.stderr.flush()


async def stderr_progress_result(
    scenario: ScenarioDefinition, result: ScenarioResult, idx: int, total: int
) -> None:
    """Emit a JSONL progress event when a scenario completes."""
    message = {
        "event": "scenario_result",
        "scenario_id": scenario.id,
        "status": result.status.value,
        "points": result.points,
        "index": idx,
        "total": total,
        "duration_seconds": round(result.duration_seconds, 2),
    }
    sys.stderr.write(json.dumps(message) + "\n")
    sys.stderr.flush()


def emit_json_output(data: dict[str, Any], *, json_file: str | None = None) -> None:
    """Write a versioned result envelope to stdout or a file."""
    from tool_eval_bench.api import format_result

    envelope = format_result(data)
    text = json.dumps(envelope, indent=2, default=str)
    if json_file:
        from pathlib import Path

        output = Path(json_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        message = {
            "event": "benchmark_complete",
            "json_file": str(output),
            "final_score": envelope.get("final_score"),
        }
        sys.stderr.write(json.dumps(message) + "\n")
        sys.stderr.flush()
    else:
        print(text)


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    """Compute a deterministic percentile-bootstrap interval for the mean."""
    if len(values) <= 1:
        value = values[0] if values else 0.0
        return (value, value)

    rng = random.Random(42)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_resamples))
    alpha = 1 - ci
    low_index = int(alpha / 2 * n_resamples)
    high_index = int((1 - alpha / 2) * n_resamples) - 1
    return (round(means[low_index], 1), round(means[high_index], 1))


def median(values: list[float]) -> float:
    """Return the median of a non-empty sequence."""
    ordered = sorted(values)
    size = len(ordered)
    if size % 2 == 1:
        return ordered[size // 2]
    return (ordered[size // 2 - 1] + ordered[size // 2]) / 2


def aggregate_trials(summaries: list) -> dict:
    """Compute score, category, and reliability statistics across trials."""
    count = len(summaries)
    if count <= 1:
        return {}

    final_scores = [summary.final_score for summary in summaries]
    total_points = [summary.total_points for summary in summaries]
    ci_low, ci_high = bootstrap_ci([float(score) for score in final_scores])

    scenario_ids = [result.scenario_id for result in summaries[0].scenario_results]
    scenario_stats: dict[str, dict] = {}
    pass_at_k_count = 0
    pass_hat_k_count = 0
    for scenario_id in scenario_ids:
        points = []
        for summary in summaries:
            result = next(
                (item for item in summary.scenario_results if item.scenario_id == scenario_id),
                None,
            )
            if result:
                points.append(result.points)
        passed_once = any(point == 2 for point in points)
        passed_always = all(point == 2 for point in points)
        pass_at_k_count += passed_once
        pass_hat_k_count += passed_always
        scenario_stats[scenario_id] = {
            "mean": round(mean(points), 2),
            "stddev": round(stdev(points), 2) if len(points) > 1 else 0.0,
            "points": points,
            "pass_at_k": passed_once,
            "pass_hat_k": passed_always,
        }

    category_stats: dict[str, dict] = {}
    for category_score in summaries[0].category_scores:
        percentages = []
        for summary in summaries:
            matching = next(
                (
                    item
                    for item in summary.category_scores
                    if item.category == category_score.category
                ),
                None,
            )
            if matching:
                percentages.append(matching.percent)
        category_stats[category_score.category.value] = {
            "label": category_score.label,
            "mean_percent": round(mean(percentages), 1),
            "stddev_percent": round(stdev(percentages), 1) if len(percentages) > 1 else 0.0,
        }

    total_scenarios = len(scenario_ids)
    pass_at_k = round(100 * pass_at_k_count / total_scenarios, 1) if total_scenarios else 0.0
    pass_hat_k = round(100 * pass_hat_k_count / total_scenarios, 1) if total_scenarios else 0.0
    return {
        "trials": count,
        "final_score_mean": round(mean(final_scores), 1),
        "final_score_stddev": round(stdev(final_scores), 1),
        "final_score_median": round(median([float(score) for score in final_scores]), 1),
        "final_score_ci95": (ci_low, ci_high),
        "total_points_mean": round(mean(total_points), 1),
        "total_points_stddev": round(stdev(total_points), 1),
        "pass_at_k": pass_at_k,
        "pass_hat_k": pass_hat_k,
        "reliability_gap": round(pass_at_k - pass_hat_k, 1),
        "per_scenario": scenario_stats,
        "per_category": category_stats,
    }
