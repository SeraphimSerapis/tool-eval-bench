"""The inputs that define a run's identity, and the persisted config built from them.

`BenchmarkService.run_benchmark` used to pass the same seventeen arguments to
`_build_run_config` twice, once before the run and once after merging resumed
results, differing only in the scenario list.  `RunSettings` captures the
parameters that do not change between those two calls so the second is a
one-liner.

This is an internal composition helper.  The service's own keyword signature is
the published API and is unchanged; nothing here appears in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.utils.ids import build_config_fingerprint
from tool_eval_bench.utils.urls import endpoint_identity
from tool_eval_bench.utils.urls import redact_url as _redact_url

#: Deployment facts that make two runs comparable.  A change in any of them puts
#: a run in a different cohort, so they are folded into the fingerprint.
COMPARISON_METADATA_KEYS = (
    "server_model_id",
    "server_model_root",
    "engine_name",
    "engine_version",
    "quantization",
    "gpu_count",
    "spec_decoding",
)


@dataclass(frozen=True)
class RunSettings:
    """Run parameters that determine the persisted config and its fingerprint.

    Frozen because two calls to :func:`build_run_config` within one run must see
    exactly the same values; a fingerprint that shifted mid-run would put the
    resumed half of a run in a different cohort from the first half.
    """

    model: str
    backend: str
    base_url: str
    temperature: float
    timeout_seconds: float
    max_turns: int
    seed: int | None
    reference_date: str | None
    concurrency: int
    error_rate: float
    alpha: float
    extra_params: dict[str, Any] | None
    context_pressure_config: dict[str, Any] | None
    weight_by_difficulty: bool


def build_run_config(
    settings: RunSettings,
    *,
    scenarios: list[ScenarioDefinition],
    metadata: dict[str, Any],
    scenario_packs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the persisted config and its deterministic comparison fingerprint.

    The fingerprint decides which runs the leaderboard groups together, so the
    key set and value shapes here are a stored contract: changing them
    re-cohorts every historical run.
    """
    config: dict[str, Any] = {
        "model": settings.model,
        "backend": settings.backend,
        "base_url": _redact_url(settings.base_url),
        "endpoint_id": endpoint_identity(settings.base_url),
        "temperature": settings.temperature,
        "timeout_seconds": settings.timeout_seconds,
        "max_turns": settings.max_turns,
        "seed": settings.seed,
        "reference_date": settings.reference_date,
        "scenario_count": len(scenarios),
        "scenario_ids": [s.id for s in scenarios],
        "concurrency": settings.concurrency,
        "error_rate": settings.error_rate,
        "alpha": settings.alpha,
        "extra_params": settings.extra_params,
        "weight_by_difficulty": settings.weight_by_difficulty,
    }
    if settings.context_pressure_config:
        config["context_pressure"] = settings.context_pressure_config
    if scenario_packs:
        config["scenario_packs"] = scenario_packs
    comparison_context = {
        key: metadata.get(key) for key in COMPARISON_METADATA_KEYS if metadata.get(key) is not None
    }
    fingerprint_config = {**config, "scenario_ids": sorted(config["scenario_ids"])}
    from tool_eval_bench import __version__

    # The fingerprint answers "are these two runs comparable?".  The scenarios and
    # evaluators are code, so two runs from different commits are not comparable
    # even when every CLI flag matches — include the code identity.
    config["config_fingerprint"] = build_config_fingerprint(
        {
            "config": fingerprint_config,
            "deployment": comparison_context,
            "tool_version": __version__,
            "git_sha": metadata.get("git_sha"),
        }
    )
    return config
