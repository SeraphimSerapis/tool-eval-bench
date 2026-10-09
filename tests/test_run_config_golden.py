"""Golden values for the persisted run config, resume check, and leaderboard cohorts.

The fingerprint decides which stored runs the leaderboard ranks together, so
every value here is a stored contract: a change re-cohorts historical runs.
These literals were captured before the run-config schema became a single
table and must stay byte-identical across refactors of that table.  Update one
only for a deliberate, changelogged change to what a run's identity means.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import tool_eval_bench
from tool_eval_bench.application.run_config import RunSettings, build_run_config
from tool_eval_bench.cli.dispatch import _resume_config_mismatches
from tool_eval_bench.cli.leaderboard import _cohort_fingerprint, _extract_leaderboard_rows
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.domain.scenarios import Category, ScenarioDefinition

GOLDEN_VERSION = "0.0.0+golden"


@pytest.fixture(autouse=True)
def _pin_tool_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fingerprint folds in the package version; pin it so a release bump
    # does not look like a contract change.
    monkeypatch.setattr(tool_eval_bench, "__version__", GOLDEN_VERSION)


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


BASE_URL = "http://localhost:8000"
JUDGE = {
    "set": "recommended",
    "base_url": "http://judge.test/v1",
    "model": "judge-model",
    "checks": ["tc89-payment-claim-v1"],
}
PRESSURE = {
    "ratio": 0.5,
    "fill_tokens": 15_190,
    "fill_tokens_target": 15_000,
    "context_size": 32_768,
}
PACK = {"name": "heldout", "sha256": "ab" * 32, "scenario_count": 2}
VARIANT = {"seed": 0, "variant": "alt-names"}
DEPLOYMENT = {
    "server_model_id": "org/model",
    "server_model_root": "/models/org/model",
    "engine_name": "vllm",
    "engine_version": "0.11.0",
    "max_model_len": 32_768,
    "quantization": "fp8",
    "gpu_count": 2,
    "slot_count": 4,
    "spec_decoding": {"method": "eagle3", "num_speculative_tokens": 3},
}


def _settings(**overrides: Any) -> RunSettings:
    values: dict[str, Any] = {
        "model": "m",
        "backend": "vllm",
        "base_url": BASE_URL,
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "context_pressure_config": None,
        "weight_by_difficulty": False,
        **overrides,
    }
    return RunSettings(**values)


TWO = [_scenario("TC-01"), _scenario("TC-02")]

#: name -> (settings overrides, scenarios, scenario_packs, metadata)
BUILD_CASES: dict[str, tuple[dict[str, Any], list[ScenarioDefinition], Any, dict[str, Any]]] = {
    "default": ({}, TWO, None, {}),
    "seed_date_extra": (
        {"seed": 42, "reference_date": "2026-03-20", "extra_params": {"top_p": 0.9}},
        TWO,
        None,
        {},
    ),
    "flags": (
        {
            "temperature": 0.6,
            "timeout_seconds": 300.0,
            "max_turns": 12,
            "concurrency": 4,
            "error_rate": 0.1,
            "alpha": 0.5,
            "weight_by_difficulty": True,
        },
        TWO,
        None,
        {},
    ),
    "system_prompt": ({"system_prompt": "Be terse."}, TWO, None, {}),
    "empty_system_prompt": ({"system_prompt": ""}, TWO, None, {}),
    "judge": ({"decision_judge": JUDGE}, TWO, None, {}),
    "variants": ({}, [_scenario("TC-01", VARIANT), _scenario("TC-02")], None, {}),
    "pressure": ({"context_pressure_config": PRESSURE}, TWO, None, {}),
    "empty_pressure": ({"context_pressure_config": {}}, TWO, None, {}),
    "packs": ({}, TWO, [PACK], {}),
    "empty_packs": ({}, TWO, [], {}),
    "everything": (
        {
            "seed": 7,
            "reference_date": "2026-01-02",
            "extra_params": {"top_k": 20},
            "system_prompt": "Be terse.",
            "decision_judge": JUDGE,
            "context_pressure_config": PRESSURE,
            "weight_by_difficulty": True,
        },
        [_scenario("TC-02", VARIANT), _scenario("TC-01")],
        [PACK],
        {**DEPLOYMENT, "git_sha": "deadbeef"},
    ),
    "credentialed_url": (
        {"base_url": "https://user:token@api.example.com:8443/v1?key=secret"},
        TWO,
        None,
        {},
    ),
    "reversed_order": ({}, list(reversed(TWO)), None, {}),
    "deployment_full": ({}, TWO, None, dict(DEPLOYMENT)),
    "deployment_partial": (
        {},
        TWO,
        None,
        {
            "server_model_id": "org/model",
            "engine_name": "vllm",
            "engine_version": None,
            "gpu_count": None,
            "not_a_comparison_key": "ignored",
        },
    ),
    "git_sha": ({}, TWO, None, {"git_sha": "deadbeef"}),
}


def _build(name: str) -> dict[str, Any]:
    overrides, scenarios, packs, metadata = BUILD_CASES[name]
    return build_run_config(
        _settings(**overrides), scenarios=scenarios, metadata=metadata, scenario_packs=packs
    )


EXPECTED_CONFIGS: dict[str, dict[str, Any]] = {
    "default": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "fc14bde869b4",
    },
    "seed_date_extra": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": 42,
        "reference_date": "2026-03-20",
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": {"top_p": 0.9},
        "weight_by_difficulty": False,
        "config_fingerprint": "fde8fae44039",
    },
    "flags": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.6,
        "timeout_seconds": 300.0,
        "max_turns": 12,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 4,
        "error_rate": 0.1,
        "alpha": 0.5,
        "extra_params": None,
        "weight_by_difficulty": True,
        "config_fingerprint": "b1f9fbf0eab7",
    },
    "system_prompt": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "system_prompt": "Be terse.",
        "config_fingerprint": "1079af8660fb",
    },
    "empty_system_prompt": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "system_prompt": "",
        "config_fingerprint": "819459c4171e",
    },
    "judge": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "decision_judge": {
            "set": "recommended",
            "base_url": "http://judge.test/v1",
            "model": "judge-model",
            "checks": ["tc89-payment-claim-v1"],
        },
        "config_fingerprint": "fc14bde869b4",
    },
    "variants": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "scenario_variants": {"TC-01": {"seed": 0, "variant": "alt-names"}},
        "config_fingerprint": "65ee65e0ae59",
    },
    "pressure": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "context_pressure": {
            "ratio": 0.5,
            "fill_tokens": 15190,
            "fill_tokens_target": 15000,
            "context_size": 32768,
        },
        "config_fingerprint": "1228b626218c",
    },
    "empty_pressure": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "fc14bde869b4",
    },
    "packs": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "scenario_packs": [
            {
                "name": "heldout",
                "sha256": "abababababababababababababababababababababababababababababababab",
                "scenario_count": 2,
            }
        ],
        "config_fingerprint": "5e97561e0357",
    },
    "empty_packs": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "fc14bde869b4",
    },
    "everything": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": 7,
        "reference_date": "2026-01-02",
        "scenario_count": 2,
        "scenario_ids": ["TC-02", "TC-01"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": {"top_k": 20},
        "weight_by_difficulty": True,
        "system_prompt": "Be terse.",
        "decision_judge": {
            "set": "recommended",
            "base_url": "http://judge.test/v1",
            "model": "judge-model",
            "checks": ["tc89-payment-claim-v1"],
        },
        "scenario_variants": {"TC-02": {"seed": 0, "variant": "alt-names"}},
        "context_pressure": {
            "ratio": 0.5,
            "fill_tokens": 15190,
            "fill_tokens_target": 15000,
            "context_size": 32768,
        },
        "scenario_packs": [
            {
                "name": "heldout",
                "sha256": "abababababababababababababababababababababababababababababababab",
                "scenario_count": 2,
            }
        ],
        "config_fingerprint": "4dcba5723b22",
    },
    "credentialed_url": {
        "model": "m",
        "backend": "vllm",
        "base_url": "https://***:8443/v1",
        "endpoint_id": "endpoint:da1104a7063f",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "476c2bec907e",
    },
    "reversed_order": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-02", "TC-01"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "fc14bde869b4",
    },
    "deployment_full": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "9a3073b8e377",
    },
    "deployment_partial": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "6054623c5bbf",
    },
    "git_sha": {
        "model": "m",
        "backend": "vllm",
        "base_url": "http://***:8000",
        "endpoint_id": "endpoint:7957b9e2c4ed",
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": ["TC-01", "TC-02"],
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "db57b9781872",
    },
}


@pytest.mark.parametrize("name", list(BUILD_CASES))
def test_persisted_config_matches_golden(name: str) -> None:
    config = _build(name)
    # Compare items in order: the stored JSON keeps insertion order.
    assert list(config.items()) == list(EXPECTED_CONFIGS[name].items())


def test_every_build_case_has_a_golden() -> None:
    assert set(EXPECTED_CONFIGS) == set(BUILD_CASES)


def test_judge_and_empty_optionals_do_not_move_the_fingerprint() -> None:
    default = EXPECTED_CONFIGS["default"]["config_fingerprint"]
    for name in ("judge", "empty_pressure", "empty_packs", "reversed_order"):
        assert EXPECTED_CONFIGS[name]["config_fingerprint"] == default, name
    assert EXPECTED_CONFIGS["deployment_partial"]["config_fingerprint"] != default


# -- resume ---------------------------------------------------------------------


def _args(*flags: str) -> argparse.Namespace:
    return _make_parser().parse_args(list(flags))


def _resume(
    previous: dict[str, Any],
    *flags: str,
    scenarios: list[ScenarioDefinition] | None = None,
    extra_params: dict[str, Any] | None = None,
    scenario_packs: list[dict[str, Any]] | None = None,
    context_pressure: dict[str, Any] | None = None,
) -> list[str]:
    return _resume_config_mismatches(
        previous,
        model="m",
        backend="vllm",
        base_url=BASE_URL,
        scenarios=TWO if scenarios is None else scenarios,
        args=_args(*flags),
        extra_params=extra_params,
        scenario_packs=scenario_packs,
        context_pressure=context_pressure,
    )


def _stored_default() -> dict[str, Any]:
    """The config a default CLI run persists, built with parser defaults."""
    args = _args()
    return build_run_config(
        _settings(
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            max_turns=args.max_turns,
            seed=args.seed,
            reference_date=args.reference_date,
            concurrency=args.parallel,
            error_rate=args.error_rate,
            alpha=args.alpha,
        ),
        scenarios=TWO,
        metadata={},
    )


#: Keys a legacy run may omit; when present, any difference refuses the resume.
SKIP_IF_ABSENT_KEYS = [
    "model",
    "backend",
    "base_url",
    "endpoint_id",
    "temperature",
    "timeout_seconds",
    "max_turns",
    "seed",
    "reference_date",
    "scenario_ids",
    "concurrency",
    "error_rate",
    "alpha",
    "extra_params",
    "weight_by_difficulty",
]


def test_resume_accepts_its_own_config() -> None:
    assert _resume(_stored_default()) == []


@pytest.mark.parametrize("key", SKIP_IF_ABSENT_KEYS)
def test_resume_names_each_changed_key(key: str) -> None:
    previous = _stored_default()
    previous[key] = "changed"
    assert _resume(previous) == [key]


@pytest.mark.parametrize("key", SKIP_IF_ABSENT_KEYS)
def test_resume_skips_a_key_a_legacy_run_never_recorded(key: str) -> None:
    previous = _stored_default()
    previous[key] = "changed"
    del previous[key]
    assert _resume(previous) == []


def test_resume_accepts_an_empty_legacy_config() -> None:
    assert _resume({}) == []


@pytest.mark.parametrize("key", ["scenario_count", "config_fingerprint", "legacy_only_key"])
def test_resume_ignores_unchecked_keys(key: str) -> None:
    assert _resume({**_stored_default(), key: "changed"}) == []


def test_resume_compares_scenario_order() -> None:
    previous = _stored_default()
    previous["scenario_ids"] = ["TC-02", "TC-01"]
    assert _resume(previous) == ["scenario_ids"]


def test_resume_reads_a_missing_system_prompt_as_built_in() -> None:
    assert _resume({}, "--system-prompt", "Be terse.") == ["system_prompt"]
    assert _resume({"system_prompt": "Be terse."}) == ["system_prompt"]
    assert _resume({"system_prompt": None}) == []
    assert _resume({"system_prompt": "Be terse."}, "--system-prompt", "Be terse.") == []


def test_resume_reads_missing_variants_as_none() -> None:
    varied = [_scenario("TC-01", VARIANT), _scenario("TC-02")]
    assert _resume({}, scenarios=varied) == ["scenario_variants"]
    assert _resume({"scenario_variants": {"TC-01": VARIANT}}) == ["scenario_variants"]
    assert _resume({"scenario_variants": {}}) == []
    assert _resume({"scenario_variants": {"TC-01": VARIANT}}, scenarios=varied) == []


JUDGE_FLAGS = ("--decision-judge-base-url", "http://judge.test/v1", "--decision-judge-model", "j")


def test_resume_reads_a_missing_judge_as_unaudited() -> None:
    assert _resume({}, *JUDGE_FLAGS) == ["decision_judge"]
    assert _resume({"decision_judge": JUDGE}) == ["decision_judge"]
    assert _resume({"decision_judge": None}) == []


def test_resume_reads_missing_pressure_as_off() -> None:
    assert _resume({}, context_pressure=PRESSURE) == ["context_pressure (was off, now 0.5)"]
    assert _resume({"context_pressure": PRESSURE}) == ["context_pressure (was 0.5, now off)"]
    assert _resume({"context_pressure": {}}, context_pressure={}) == []


def test_resume_refuses_dropping_a_pack_but_skips_a_pack_free_legacy_run() -> None:
    assert _resume({"scenario_packs": [PACK]}) == ["scenario_packs"]
    assert _resume({"scenario_packs": [PACK]}, scenario_packs=[PACK]) == []
    # A run without packs persists no key, which the check reads as "legacy,
    # unrecorded": adding a pack is caught by scenario_ids, not here.
    assert _resume({}, scenario_packs=[PACK]) == []


def test_resume_reports_every_mismatch() -> None:
    previous = {
        **_stored_default(),
        "model": "other",
        "temperature": 1.0,
        "system_prompt": "Be verbose.",
        "decision_judge": JUDGE,
        "scenario_variants": {"TC-01": VARIANT},
        "context_pressure": PRESSURE,
        "scenario_packs": [PACK],
    }
    # Only membership is pinned: the joined message's order is cosmetic.
    assert set(_resume(previous)) == {
        "model",
        "temperature",
        "system_prompt",
        "decision_judge",
        "scenario_variants",
        "context_pressure (was 0.5, now off)",
        "scenario_packs",
    }


# -- leaderboard ----------------------------------------------------------------


EXPECTED_COHORTS: dict[str, str] = {
    "default": "78bb227fb58d",
    "judged": "78bb227fb58d",
    "other_model": "78bb227fb58d",
    "legacy_key": "9bf7a01ff466",
    "unsorted_ids": "167b1b7e0748",
    "pressure": "9fbb00ee305c",
    "everything": "e1c7f4adc5dc",
}
EXPECTED_ROWS: list[tuple[str, str, str, int]] = [
    ("r2", "fc14bde869b4", "78bb227fb58d", 2),
    ("r3", "1228b626218c", "9fbb00ee305c", 1),
    ("r4", "fc14bde869b4", "78bb227fb58d", 1),
    ("r5", "0cac0195d776", "78bb227fb58d", 1),
]


def _cohort_cases() -> dict[str, dict[str, Any]]:
    stored = dict(EXPECTED_CONFIGS["default"])
    return {
        "default": stored,
        "judged": {**stored, "decision_judge": JUDGE},
        "other_model": {
            **stored,
            "model": "other",
            "base_url": "http://***:9000",
            "endpoint_id": "endpoint:other",
            "config_fingerprint": "000000000000",
        },
        "legacy_key": {**stored, "legacy_only_key": True},
        "unsorted_ids": {**stored, "scenario_ids": ["TC-02", "TC-01"]},
        "pressure": dict(EXPECTED_CONFIGS["pressure"]),
        "everything": dict(EXPECTED_CONFIGS["everything"]),
    }


def test_every_cohort_case_has_a_golden() -> None:
    assert set(EXPECTED_COHORTS) == set(_cohort_cases())


@pytest.mark.parametrize("name", list(_cohort_cases()))
def test_cohort_fingerprint_matches_golden(name: str) -> None:
    assert _cohort_fingerprint(_cohort_cases()[name]) == EXPECTED_COHORTS[name]


def _run(run_id: str, model: str, config: dict[str, Any], created_at: str) -> dict[str, Any]:
    ids = config["scenario_ids"]
    return {
        "run_id": run_id,
        "model": model,
        "status": "completed",
        "created_at": created_at,
        "config": config,
        "metadata": {"engine_name": "vllm"},
        "scores": {
            "final_score": 80,
            "completion_rate": 100,
            "scenario_results": [{"scenario_id": sid, "status": "pass"} for sid in ids],
        },
    }


def test_leaderboard_groups_stored_configs() -> None:
    default = dict(EXPECTED_CONFIGS["default"])
    legacy = {k: v for k, v in default.items() if k != "config_fingerprint"}
    runs = [
        _run("r1", "m", default, "2026-01-01T00:00:00"),
        _run("r2", "m", default, "2026-01-02T00:00:00"),
        _run("r3", "m", dict(EXPECTED_CONFIGS["pressure"]), "2026-01-03T00:00:00"),
        _run("r4", "other", {**default, "model": "other"}, "2026-01-04T00:00:00"),
        _run("r5", "m", legacy, "2026-01-05T00:00:00"),
    ]
    rows = _extract_leaderboard_rows(runs)
    assert sorted(
        (row["run_id"], row["config_fingerprint"], row["cohort_fingerprint"], row["num_runs"])
        for row in rows
    ) == sorted(EXPECTED_ROWS)
