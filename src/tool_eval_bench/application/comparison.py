"""Paired 3-tier statistical comparison shared by the CLI and web UI.

Binary McNemar's test collapses 3-tier scenario scores of 0, 1, and 2 into pass
vs not-pass, missing partial-credit transitions between 0 and 1. This module
computes paired percentage-point deltas, deterministic paired
percentile-bootstrap confidence intervals, and an exact two-sided binomial sign
test over 3-tier point changes while excluding transient infrastructure
failures.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from tool_eval_bench.application.leaderboard import compute_mcnemar_stats
from tool_eval_bench.domain.scenarios import INFRASTRUCTURE_FAILURE_KINDS


def is_infrastructure_failure_dict(result: Mapping[str, Any]) -> bool:
    """Whether a stored scenario result dict recorded an infrastructure failure."""
    return result.get("failure_kind") in INFRASTRUCTURE_FAILURE_KINDS


def extract_result_maps(
    results_a: Sequence[Mapping[str, Any]],
    results_b: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], list[str]]:
    """Build scenario lookup maps and ordered union of scenario IDs for two runs."""
    map_a = {str(r["scenario_id"]): r for r in results_a if "scenario_id" in r}
    map_b = {str(r["scenario_id"]): r for r in results_b if "scenario_id" in r}
    all_ids = list(dict.fromkeys([*map_a.keys(), *map_b.keys()]))
    return map_a, map_b, all_ids


def _paired_bootstrap_ci95(
    pct_deltas: Sequence[float],
    *,
    mean_pct_delta: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    """Compute a deterministic paired 95% percentile-bootstrap CI on the mean."""
    n = len(pct_deltas)
    if n == 0 or bootstrap_samples <= 0 or len(set(pct_deltas)) <= 1:
        return (mean_pct_delta, mean_pct_delta)

    rng = random.Random(seed)
    values = list(pct_deltas)
    sample_means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(bootstrap_samples))
    low_idx = int(0.025 * bootstrap_samples)
    high_idx = min(bootstrap_samples - 1, max(low_idx, int(0.975 * bootstrap_samples) - 1))
    return (sample_means[low_idx], sample_means[high_idx])


def exact_two_sided_sign_p_value(regressed_count: int, improved_count: int) -> float:
    """Compute the exact two-sided binomial sign test p-value under p = 0.5."""
    n_discordant = regressed_count + improved_count
    if n_discordant == 0:
        return 1.0
    smaller_side = min(regressed_count, improved_count)
    tail_combinations = sum(math.comb(n_discordant, idx) for idx in range(smaller_side + 1))
    return min(1.0, (2 * tail_combinations) / (1 << n_discordant))


def compute_paired_comparison(
    map_a: Mapping[str, Mapping[str, Any]],
    map_b: Mapping[str, Mapping[str, Any]],
    all_ids: Sequence[str],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute paired 3-tier quality deltas, 95% CI, sign test, and McNemar stats.

    Scenarios missing from either run are skipped, and shared scenarios where
    either run encountered a timeout, connection error, or server error are
    recorded in ``infra_excluded_ids`` and excluded from quality statistics so
    transport glitches do not masquerade as model quality changes.
    """
    unique_ids = list(dict.fromkeys(all_ids))
    infra_excluded_ids: list[str] = []
    non_infra_ids: list[str] = []
    discordant_scenario_ids: list[str] = []
    improved_ids: list[str] = []
    regressed_ids: list[str] = []
    pct_deltas: list[float] = []
    total_point_delta = 0

    for sc_id in unique_ids:
        ra = map_a.get(sc_id)
        rb = map_b.get(sc_id)
        if not ra or not rb:
            continue
        if is_infrastructure_failure_dict(ra) or is_infrastructure_failure_dict(rb):
            infra_excluded_ids.append(sc_id)
            continue

        non_infra_ids.append(sc_id)
        pts_a = int(ra.get("points") or 0)
        pts_b = int(rb.get("points") or 0)
        st_a = str(ra.get("status") or "fail")
        st_b = str(rb.get("status") or "fail")

        delta_pts = pts_b - pts_a
        total_point_delta += delta_pts
        # Each scenario is worth 2 points, so +1 point is +50 percentage points.
        pct_deltas.append(delta_pts * 50.0)

        if pts_a != pts_b or st_a != st_b:
            discordant_scenario_ids.append(sc_id)
        if delta_pts > 0:
            improved_ids.append(sc_id)
        elif delta_pts < 0:
            regressed_ids.append(sc_id)

    shared_count = len(non_infra_ids)
    mean_pct_delta = (total_point_delta / (2 * shared_count)) * 100.0 if shared_count > 0 else 0.0
    ci95_low_pct, ci95_high_pct = _paired_bootstrap_ci95(
        pct_deltas,
        mean_pct_delta=mean_pct_delta,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    sign_p_value = exact_two_sided_sign_p_value(len(regressed_ids), len(improved_ids))
    mcnemar = compute_mcnemar_stats(map_a, map_b, non_infra_ids)

    if shared_count == 0:
        verdict = "no_shared_scenarios"
    elif not discordant_scenario_ids:
        verdict = "identical"
    elif sign_p_value < 0.05 and len(improved_ids) > len(regressed_ids) and mean_pct_delta > 0:
        verdict = "significant_improvement"
    elif sign_p_value < 0.05 and len(regressed_ids) > len(improved_ids) and mean_pct_delta < 0:
        verdict = "significant_regression"
    else:
        verdict = "inconclusive_noise"

    return {
        "shared_count": shared_count,
        "infra_excluded_ids": infra_excluded_ids,
        "discordant_scenario_ids": discordant_scenario_ids,
        "improved_ids": improved_ids,
        "regressed_ids": regressed_ids,
        "mean_pct_delta": mean_pct_delta,
        "ci95_low_pct": ci95_low_pct,
        "ci95_high_pct": ci95_high_pct,
        "sign_p_value": sign_p_value,
        "mcnemar": mcnemar,
        "verdict": verdict,
    }
