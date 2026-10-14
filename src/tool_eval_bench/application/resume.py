"""Resume rules shared by the CLI and the web UI.

A resume finishes an interrupted run under its original run ID, so it must
never change the conditions the first half ran under, and it must never retry
a real model outcome into a better score.  The CLI rebuilds the run from its
command line and compares; the web UI rebuilds it from the stored config alone
and refuses whatever that config cannot reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tool_eval_bench.adapters.wire_format import resolve_wire_format
from tool_eval_bench.application.run_config import (
    RunSettings,
    build_run_config,
    resume_mismatches,
)
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.scenarios import ScenarioDefinition, ScenarioStatus

#: Failure kinds that say nothing about the model, so their scenarios rerun.
_INFRASTRUCTURE_FAILURES = frozenset({"timeout", "connection_error", "server_error"})

#: The CLI's ``--max-turns`` and ``--alpha`` defaults.
DEFAULT_MAX_TURNS = 8
DEFAULT_ALPHA = 0.7


def prior_results_for_resume(
    prev_run: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect the reusable scenario results of a prior run.

    A completed run carries everything in its final ``scores``.  A run that was
    interrupted never got that far, so its per-scenario checkpoints are the only
    record — without them the whole run would have to be redone.  When both
    exist, final scores win because they went through the full scoring pass.
    """
    scored = [
        r
        for r in (prev_run.get("scores") or {}).get("scenario_results") or []
        if r.get("scenario_id")
    ]
    if not checkpoints:
        return scored
    merged: dict[str, dict[str, Any]] = {
        r["scenario_id"]: r for r in checkpoints if r.get("scenario_id")
    }
    merged.update({r["scenario_id"]: r for r in scored})
    return list(merged.values())


def result_requires_rerun(result: dict[str, Any]) -> bool:
    """Return whether a checkpoint is missing, corrupt, or infrastructure-only.

    A model outcome is evidence, including ordinary failures and partials.  It
    must not be silently retried into a better score under the same run ID.
    """
    if not result.get("scenario_id"):
        return True
    if result.get("status") not in {status.value for status in ScenarioStatus}:
        return True
    if not isinstance(result.get("points"), int) or not isinstance(result.get("summary"), str):
        return True
    if not isinstance(result.get("raw_log"), str) or not result["raw_log"].strip():
        return True
    return (
        result.get("status") == ScenarioStatus.FAIL.value
        and result.get("failure_kind") in _INFRASTRUCTURE_FAILURES
    )


class ResumeRejected(ValueError):
    """The stored run cannot be resumed without options only the CLI can supply."""


@dataclass(frozen=True)
class StoredResumePlan:
    """Everything needed to finish a run using only what its stored config records."""

    settings: RunSettings
    #: The run's complete protocol, in the original order.
    scenarios: list[ScenarioDefinition]
    #: Prior outcomes that stand; they are merged back in, never rerun.
    reusable: list[dict[str, Any]]
    #: Scenarios still to execute.
    remaining: list[ScenarioDefinition]


def plan_stored_resume(
    prev_run: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    *,
    base_url: str,
    wire_format: str | None = None,
) -> StoredResumePlan:
    """Plan a resume from the stored config, rejecting anything it cannot reproduce.

    ``base_url`` is the caller's unredacted endpoint: stored configs keep only a
    redacted URL, so the endpoint identity check is what proves it is the same
    server.  The current config is built from the stored values and compared
    with :func:`resume_mismatches`, the check the CLI uses.  Conditions that
    need CLI-only inputs (context-pressure filler, a decision judge, scenario
    variants, held-out packs) never appear in the rebuilt config, so a run that
    used them is reported as a mismatch rather than resumed without them.

    Raises :class:`ResumeRejected` with a message naming what differs.
    """
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

    config = prev_run.get("config") or {}
    scenario_ids = config.get("scenario_ids")
    if not isinstance(scenario_ids, list) or not scenario_ids:
        raise ResumeRejected(
            "The run did not record its scenario list, so only the CLI can resume it."
        )
    by_id = {scenario.id: scenario for scenario in ALL_SCENARIOS_WITH_HARDMODE}
    unknown = [sid for sid in scenario_ids if sid not in by_id]
    if unknown:
        raise ResumeRejected(
            f"{len(unknown)} scenario(s) in this run are not in the public registry, which "
            "means a held-out pack. Resume it from the CLI with the same --scenario-pack."
        )
    scenarios = [by_id[sid] for sid in scenario_ids]

    try:
        resolved_format = resolve_wire_format(wire_format, base_url)
    except ValueError as exc:
        raise ResumeRejected(str(exc)) from exc

    # Defaults match the CLI's, for legacy configs that predate a key; resume
    # skips comparing keys a stored run never recorded.
    settings = RunSettings(
        model=str(config.get("model") or prev_run.get("model") or ""),
        backend=str(config.get("backend") or "unknown"),
        base_url=base_url,
        temperature=float(config.get("temperature", 0.0)),
        timeout_seconds=float(config.get("timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)),
        max_turns=int(config.get("max_turns", DEFAULT_MAX_TURNS)),
        seed=config.get("seed"),
        reference_date=config.get("reference_date"),
        concurrency=int(config.get("concurrency", 1)),
        error_rate=float(config.get("error_rate", 0.0)),
        alpha=float(config.get("alpha", DEFAULT_ALPHA)),
        extra_params=config.get("extra_params"),
        context_pressure_config=None,
        weight_by_difficulty=bool(config.get("weight_by_difficulty", False)),
        system_prompt=config.get("system_prompt"),
        decision_judge=None,
        wire_format=resolved_format,
    )
    current = build_run_config(settings, scenarios=scenarios, metadata={}, scenario_packs=None)
    mismatches = resume_mismatches(config, current, base_url=base_url, wire_format=resolved_format)
    if mismatches:
        raise ResumeRejected(
            f"This run cannot be resumed here because it differs in: {', '.join(mismatches)}. "
            "Check the base URL, or resume it from the CLI with its original options."
        )

    planned = {scenario.id for scenario in scenarios}
    reusable = [
        result
        for result in prior_results_for_resume(prev_run, checkpoints)
        if not result_requires_rerun(result) and result["scenario_id"] in planned
    ]
    done = {result["scenario_id"] for result in reusable}
    remaining = [scenario for scenario in scenarios if scenario.id not in done]
    return StoredResumePlan(
        settings=settings, scenarios=scenarios, reusable=reusable, remaining=remaining
    )
