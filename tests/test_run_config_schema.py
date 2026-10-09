"""Every run setting must be wired through the one run-config schema.

A setting that reaches the run but not the schema is how resume came to ignore
context pressure (#237): the builder knew the key and the resume check did not.
These tests fail when a new setting is added half-way.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from tool_eval_bench.application.run_config import (
    COHORT_EXCLUDED_KEYS,
    CONFIG_FINGERPRINT_KEY,
    RUN_CONFIG_FIELDS,
    Presence,
    RunSettings,
    build_run_config,
    resume_mismatches,
)
from tool_eval_bench.domain.scenarios import Category, ScenarioDefinition


def _scenario(sid: str, variant: dict[str, Any] | None = None) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=sid,
        title=sid,
        category=Category.A,
        user_message="",
        description="",
        handle_tool_call=lambda state, call: None,
        evaluate=lambda state: None,  # type: ignore[arg-type,return-value]
        variant_metadata=variant or {},
    )


SCENARIOS = [_scenario("TC-02", {"seed": 0}), _scenario("TC-01")]
PACKS = [{"name": "heldout", "sha256": "ab" * 32}]

#: A run with every optional key set, so every schema row is written.
FULL = RunSettings(
    model="m",
    backend="vllm",
    base_url="http://localhost:8000/v1",
    temperature=0.0,
    timeout_seconds=120.0,
    max_turns=8,
    seed=1,
    reference_date="2026-01-01",
    concurrency=1,
    error_rate=0.0,
    alpha=0.7,
    extra_params={"top_p": 0.9},
    context_pressure_config={"ratio": 0.5, "fill_tokens_target": 1000},
    weight_by_difficulty=False,
    system_prompt="Be terse.",
    decision_judge={"set": "recommended", "checks": ["a"]},
)

#: A different value for every RunSettings field.  Adding a field without a
#: row here fails test_every_setting_has_a_perturbation, and wiring it into the
#: schema is what makes the other tests pass.
PERTURBED: dict[str, Any] = {
    "model": "other",
    "backend": "sglang",
    "base_url": "http://otherhost:9000",
    "temperature": 0.5,
    "timeout_seconds": 60.0,
    "max_turns": 4,
    "seed": 2,
    "reference_date": "2026-02-02",
    "concurrency": 2,
    "error_rate": 0.1,
    "alpha": 0.5,
    "extra_params": {"top_p": 0.5},
    "context_pressure_config": {"ratio": 0.7, "fill_tokens_target": 2000},
    "weight_by_difficulty": True,
    "system_prompt": "Be verbose.",
    "decision_judge": {"set": "all", "checks": ["a"]},
    # Reaches the config through endpoint_id: a Gemini base keeps FULL's /v1.
    "wire_format": "gemini",
}


def _build(settings: RunSettings) -> dict[str, Any]:
    return build_run_config(settings, scenarios=SCENARIOS, metadata={}, scenario_packs=PACKS)


def test_every_setting_has_a_perturbation() -> None:
    assert set(PERTURBED) == {field.name for field in dataclasses.fields(RunSettings)}


@pytest.mark.parametrize("name", list(PERTURBED))
def test_every_setting_is_persisted_and_checked_on_resume(name: str) -> None:
    base = _build(FULL)
    changed = _build(dataclasses.replace(FULL, **{name: PERTURBED[name]}))

    assert changed != base, f"{name} does not reach the stored config"
    assert resume_mismatches(changed, base), f"resume ignores a change to {name}"


def test_schema_keys_are_unique() -> None:
    keys = [field.key for field in RUN_CONFIG_FIELDS]
    assert len(keys) == len(set(keys))
    assert CONFIG_FINGERPRINT_KEY not in keys


def test_full_run_writes_every_schema_key_in_order() -> None:
    expected = [field.key for field in RUN_CONFIG_FIELDS] + [CONFIG_FINGERPRINT_KEY]
    assert list(_build(FULL)) == expected


def test_default_run_writes_only_always_present_keys() -> None:
    plain = dataclasses.replace(
        FULL, system_prompt=None, decision_judge=None, context_pressure_config=None
    )
    config = build_run_config(plain, scenarios=[_scenario("TC-01")], metadata={})
    expected = [f.key for f in RUN_CONFIG_FIELDS if f.presence is Presence.ALWAYS]
    assert list(config) == [*expected, CONFIG_FINGERPRINT_KEY]


def test_a_run_resumes_cleanly_against_itself() -> None:
    assert resume_mismatches(_build(FULL), _build(FULL)) == []


def test_cohort_drops_model_identity_and_unfingerprinted_keys() -> None:
    assert COHORT_EXCLUDED_KEYS == {
        "model",
        "base_url",
        "endpoint_id",
        "decision_judge",
        CONFIG_FINGERPRINT_KEY,
    }
