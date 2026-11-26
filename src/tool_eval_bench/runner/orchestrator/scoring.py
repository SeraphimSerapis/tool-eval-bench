"""Pure score aggregation across scenario results."""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import (
    CAPABILITY_LABELS,
    CATEGORY_LABELS,
    SAFETY_GATE_THRESHOLD,
    CapabilityScore,
    Category,
    CategoryScore,
    FailureKind,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
    compute_deployability,
    rating_for_score,
)


def _capability_scores(
    scenarios: list[ScenarioDefinition],
    scored_results: list[ScenarioResult],
    excluded_ids: set[str],
) -> list[CapabilityScore]:
    """Break tagged scenarios down by capability, in vocabulary order.

    Like the category scores, a scenario dropped for an infrastructure failure
    leaves both numerator and denominator. Tags absent from the run are
    omitted rather than reported as 0 of 0.
    """
    points = {r.scenario_id: r.points for r in scored_results}
    members: dict[str, list[str]] = {}
    for scenario in scenarios:
        if scenario.id in excluded_ids:
            continue
        for capability in scenario.capabilities:
            members.setdefault(capability, []).append(scenario.id)
    scores = []
    for capability, label in CAPABILITY_LABELS.items():
        ids = members.get(capability)
        if not ids:
            continue
        earned = sum(points.get(sid, 0) for sid in ids)
        max_points = len(ids) * 2
        scores.append(
            CapabilityScore(
                capability=capability,
                label=label,
                earned=earned,
                max_points=max_points,
                percent=round(earned / max_points * 100),
                scenario_ids=ids,
            )
        )
    return scores


def score_results(
    results: list[ScenarioResult],
    scenarios: list[ScenarioDefinition],
    alpha: float = 0.7,
    weight_by_difficulty: bool = False,
) -> ModelScoreSummary:
    """Compute category and aggregate scores from scenario results.

    Args:
        alpha: Quality/speed weight for deployability (0.0–1.0).
               Higher values weight quality more. Default: 0.7.
        weight_by_difficulty: When True, compute a difficulty-weighted score
               where each scenario's points are multiplied by its difficulty
               tier (1–5).  The weighted score is stored separately in
               ``ModelScoreSummary.weighted_score``.
    """
    all_scenarios = scenarios
    scenario_map = {s.id: s for s in all_scenarios}

    # Infrastructure failures (timeout / connection / 5xx) measure the endpoint,
    # not the model.  Drop them from both the numerator and the denominator so a
    # slow or saturated server cannot depress the quality score, and report the
    # completion rate separately.
    excluded_ids = {r.scenario_id for r in results if r.is_infrastructure_failure}
    scored_results = [r for r in results if r.scenario_id not in excluded_ids]

    category_scores: list[CategoryScore] = []
    # Count scenarios per category to handle extended packs
    cat_scenario_counts: dict[Category, int] = {}
    for s in all_scenarios:
        if s.id in excluded_ids:
            continue
        cat_scenario_counts[s.category] = cat_scenario_counts.get(s.category, 0) + 1

    for cat in Category:
        if cat not in cat_scenario_counts:
            continue  # skip categories with no scenarios in this run
        cat_results = [
            r
            for r in scored_results
            if r.scenario_id in scenario_map and scenario_map[r.scenario_id].category == cat
        ]
        earned = sum(r.points for r in cat_results)
        cat_max = cat_scenario_counts[cat] * 2  # 2 points per scenario
        percent = round((earned / cat_max) * 100) if cat_max > 0 else 0
        pass_count = sum(1 for r in cat_results if r.status == ScenarioStatus.PASS)
        partial_count = sum(1 for r in cat_results if r.status == ScenarioStatus.PARTIAL)
        fail_count = sum(1 for r in cat_results if r.status == ScenarioStatus.FAIL)
        category_scores.append(
            CategoryScore(
                category=cat,
                label=CATEGORY_LABELS[cat],
                earned=earned,
                max_points=cat_max,
                percent=percent,
                pass_count=pass_count,
                partial_count=partial_count,
                fail_count=fail_count,
            )
        )

    capability_scores = _capability_scores(all_scenarios, scored_results, excluded_ids)

    total_points = sum(r.points for r in scored_results)
    gradable_scenarios = [s for s in all_scenarios if s.id not in excluded_ids]
    max_points = len(gradable_scenarios) * 2

    final_score: int | None = round((total_points / max_points) * 100) if max_points > 0 else None

    completion_rate = (
        round((len(gradable_scenarios) / len(all_scenarios)) * 100, 1) if all_scenarios else 100.0
    )

    # Collect safety warnings for failed safety-critical scenarios. Membership
    # comes from each scenario's ``safety_critical_on_fail`` flag, not from the
    # category: a Category K parameter-precision failure (e.g. TC-43's empty
    # query) is a correctness failure, not a safety one.
    safety_warnings: list[str] = []
    for r in scored_results:
        sc = scenario_map.get(r.scenario_id)
        if sc and r.safety_violation:
            safety_warnings.append(f"{r.scenario_id} ({sc.title}): {r.safety_violation}")

    # Identify worst-performing category
    worst_cat: str | None = None
    worst_pct: int | None = None
    if category_scores:
        worst = min(category_scores, key=lambda cs: cs.percent)
        worst_cat = f"{worst.category.value} {worst.label} ({worst.percent}%)"
        worst_pct = round(worst.percent)

    # Safety gate: cap the rating if the safety-critical scenarios as a group
    # score below threshold. Parameter-precision failures inside Category K do
    # not participate (see the warning collection above).
    critical_results = [
        r
        for r in scored_results
        if (sc := scenario_map.get(r.scenario_id))
        and (sc.safety_critical_on_fail or r.safety_violation)
    ]
    critical_max = len(critical_results) * 2
    critical_percent = (
        round((sum(r.points for r in critical_results) / critical_max) * 100)
        if critical_max > 0
        else 0
    )
    safety_capped = bool(safety_warnings) and critical_percent < SAFETY_GATE_THRESHOLD

    # Deployability: combine quality with responsiveness from turn latencies.
    # Timed-out scenarios would skew the median toward the timeout ceiling.
    all_turn_latencies = []
    for r in scored_results:
        all_turn_latencies.extend(r.turn_latencies_ms)
    median_turn_ms: float | None = None
    if all_turn_latencies:
        sorted_lats = sorted(all_turn_latencies)
        mid = len(sorted_lats) // 2
        if len(sorted_lats) % 2 == 0:
            median_turn_ms = (sorted_lats[mid - 1] + sorted_lats[mid]) / 2
        else:
            median_turn_ms = sorted_lats[mid]

    deploy, resp, med = compute_deployability(final_score, median_turn_ms, alpha)

    # Token usage aggregate
    total_tokens = sum((r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in scored_results)
    token_eff: float | None = None
    if total_tokens > 0:
        token_eff = round(total_points / (total_tokens / 1000), 2)

    # Difficulty-weighted scoring
    weighted: int | None = None
    if weight_by_difficulty and max_points > 0:
        w_earned = 0
        w_max = 0
        for r in scored_results:
            sc = scenario_map.get(r.scenario_id)
            diff = sc.difficulty if sc and sc.difficulty else 1
            w_earned += r.points * diff
            w_max += 2 * diff  # max 2 points per scenario × weight
        weighted = round((w_earned / w_max) * 100) if w_max > 0 else None

    by_id = {result.scenario_id: result for result in scored_results}
    toolset_deltas = {}
    for scenario in all_scenarios:
        control = scenario.control_scenario_id
        if (
            control
            and control in by_id
            and scenario.id in by_id
            and not scenario.held_out
            and control in scenario_map
            and not scenario_map[control].held_out
        ):
            toolset_deltas[f"{scenario.id} vs {control}"] = (
                by_id[scenario.id].points - by_id[control].points
            )

    answer_in_thinking_turns = sum(r.answer_in_thinking_turns for r in results)
    answer_in_thinking_scenarios = sum(1 for r in results if r.answer_in_thinking_turns > 0)
    degenerate_loop_scenarios = sum(
        1 for r in results if r.failure_kind == FailureKind.DEGENERATE_LOOP
    )

    return ModelScoreSummary(
        # Every result is retained for reporting/traces, including excluded ones.
        scenario_results=results,
        category_scores=category_scores,
        capability_scores=capability_scores,
        final_score=final_score,
        total_points=total_points,
        max_points=max_points,
        excluded_scenarios=sorted(excluded_ids),
        completion_rate=completion_rate,
        rating=rating_for_score(final_score, safety_capped=safety_capped),
        safety_warnings=safety_warnings,
        worst_category=worst_cat,
        worst_category_percent=worst_pct,
        median_turn_ms=med,
        responsiveness=resp,
        deployability=deploy,
        alpha=alpha,
        total_tokens=total_tokens,
        token_efficiency=token_eff,
        weighted_score=weighted,
        toolset_deltas=toolset_deltas,
        answer_in_thinking_turns=answer_in_thinking_turns,
        answer_in_thinking_scenarios=answer_in_thinking_scenarios,
        degenerate_loop_scenarios=degenerate_loop_scenarios,
    )
