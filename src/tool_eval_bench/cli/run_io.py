"""Trial aggregation helpers used by CLI output modes."""

from __future__ import annotations

import json
import random
import sys
from statistics import mean, stdev
from typing import Any

from tool_eval_bench.domain.scenarios import AuditPhase, ScenarioDefinition, ScenarioResult


def _scenario_start_event(
    scenario: ScenarioDefinition, idx: int, total: int, *, title: str
) -> None:
    message = {
        "event": "scenario_start",
        "scenario_id": scenario.id,
        "title": title,
        "category": scenario.category.value,
        "index": idx,
        "total": total,
    }
    sys.stderr.write(json.dumps(message) + "\n")
    sys.stderr.flush()


async def stderr_progress_start(scenario: ScenarioDefinition, idx: int, total: int) -> None:
    """Emit a JSONL progress event when a scenario starts.

    A held-out pack scenario's title is withheld: it names the prompt, and
    stderr is what CI logs capture.
    """
    title = "held out" if scenario.held_out else scenario.title
    _scenario_start_event(scenario, idx, total, title=title)


async def stderr_progress_start_unredacted(
    scenario: ScenarioDefinition, idx: int, total: int
) -> None:
    """The ``scenario_start`` event with every title, for ``--include-held-out``."""
    _scenario_start_event(scenario, idx, total, title=scenario.title)


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


async def stderr_progress_audit(
    scenario: ScenarioDefinition, result: ScenarioResult, phase: AuditPhase
) -> None:
    """Emit summary-only judge events; never send raw audit evidence to stderr."""
    if not result.was_decision_audited:
        return
    audit = result.decision_audit or {}
    message: dict[str, Any] = {
        "event": "decision_audit_start" if phase == "started" else "decision_audit_result",
        "scenario_id": scenario.id,
        "model": audit.get("model"),
        "check_id": audit.get("check_id"),
    }
    if phase != "started":
        for key in (
            "status",
            "choice",
            "probabilities",
            "disagreement",
            "elapsed_ms",
            "error_type",
        ):
            if key in audit:
                message[key] = audit[key]
        if phase == "reused":
            message["reused"] = True
    sys.stderr.write(json.dumps(message) + "\n")
    sys.stderr.flush()


def emit_json_output(
    data: dict[str, Any], *, json_file: str | None = None, failed: bool = False
) -> None:
    """Write a versioned result envelope to stdout or a file.

    A *failed* run writes its error envelope the same way but emits no
    ``benchmark_complete`` event; the caller reports the failure itself.
    """
    from tool_eval_bench.api import format_result

    envelope = format_result(data)
    text = json.dumps(envelope, indent=2, default=str)
    if json_file:
        from pathlib import Path

        output = Path(json_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        if failed:
            return
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
    """Compute score, category, and reliability statistics across trials.

    Per-scenario points, Pass@k, and Pass^k read only gradable results: an
    infrastructure failure (timeout, connection error, 5xx) is excluded from a
    trial's score, so it is excluded here too rather than counted as a miss. A
    scenario no trial could grade drops out of the Pass@k and Pass^k
    denominators. Scenarios and categories come from every trial, not only the
    first.
    """
    count = len(summaries)
    if count <= 1:
        return {}

    final_scores = [summary.final_score for summary in summaries]
    total_points = [summary.total_points for summary in summaries]
    ci_low, ci_high = bootstrap_ci([float(score) for score in final_scores])

    points_by_scenario: dict[str, list[int]] = {}
    for summary in summaries:
        for result in summary.scenario_results:
            graded = points_by_scenario.setdefault(result.scenario_id, [])
            if not result.is_infrastructure_failure:
                graded.append(result.points)

    scenario_stats: dict[str, dict] = {}
    pass_at_k_count = 0
    pass_hat_k_count = 0
    for scenario_id, points in points_by_scenario.items():
        if not points:
            continue
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

    percentages_by_category: dict[Any, list[float]] = {}
    labels: dict[Any, str] = {}
    for summary in summaries:
        for category_score in summary.category_scores:
            labels.setdefault(category_score.category, category_score.label)
            percentages_by_category.setdefault(category_score.category, []).append(
                category_score.percent
            )
    category_stats: dict[str, dict] = {
        category.value: {
            "label": labels[category],
            "mean_percent": round(mean(percentages), 1),
            "stddev_percent": round(stdev(percentages), 1) if len(percentages) > 1 else 0.0,
        }
        for category, percentages in percentages_by_category.items()
    }

    total_scenarios = len(scenario_stats)
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
