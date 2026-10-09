"""The engine registry is consistent with itself and with what consumes it.

Behaviour per engine is pinned in ``test_engine_golden.py``. These tests guard
the registry's own shape: one profile per label, no ambiguous names, and
precedence orders that cover exactly the engines they are meant to rank.
"""

from __future__ import annotations

import argparse
from collections import Counter

import pytest

from tests.test_engine_golden import _LIVE_CASES
from tool_eval_bench import api
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.domain.engines import (
    BACKEND_LABELS,
    ENGINE_PROFILES,
    LLAMACPP,
    METRICS_IDENTITY_ORDER,
    SPEC_COUNTER_ORDER,
    STRATA,
    TENSORFOLD,
    UNKNOWN_BACKEND,
    EngineProfile,
    engine_profile,
    engine_profile_by_name,
    metrics_namespace_present,
)
from tool_eval_bench.runner.spec_live import _parse_snapshot


def _duplicates(values: list[str]) -> list[str]:
    return [value for value, count in Counter(values).items() if count > 1]


def test_labels_and_names_are_unique_and_lower_case() -> None:
    labels = [label for p in ENGINE_PROFILES for label in p.labels]

    assert _duplicates(labels) == []
    assert all(label == label.lower() for label in labels)
    assert UNKNOWN_BACKEND not in labels
    assert _duplicates([p.display_name for p in ENGINE_PROFILES]) == []


def test_declared_names_are_unambiguous_per_source() -> None:
    declared = [(source, token) for p in ENGINE_PROFILES for source, token in p.declared_names]

    assert _duplicates([f"{source}:{token}" for source, token in declared]) == []
    assert all(token == token.lower() for _, token in declared)


def test_identity_order_ranks_every_namespaced_engine_once() -> None:
    namespaced = [p for p in ENGINE_PROFILES if p.metrics_prefixes]
    prefixes = [prefix for p in ENGINE_PROFILES for prefix in p.metrics_prefixes]

    assert sorted(p.key for p in METRICS_IDENTITY_ORDER) == sorted(p.key for p in namespaced)
    assert _duplicates([p.key for p in METRICS_IDENTITY_ORDER]) == []
    assert _duplicates(prefixes) == []


def test_spec_counter_order_ranks_exactly_the_always_rendered_engines() -> None:
    always_rendered = {p.key for p in ENGINE_PROFILES if p.spec_counters_always_rendered}

    assert [p.key for p in SPEC_COUNTER_ORDER] == ["tensorfold", "llamacpp", "strata"]
    assert {p.key for p in SPEC_COUNTER_ORDER} == always_rendered
    assert all(p.metrics_prefixes and p.spec_counter_detail for p in SPEC_COUNTER_ORDER)


def test_hosted_profiles_carry_no_engine_facts() -> None:
    hosted = [p for p in ENGINE_PROFILES if p.hosted]

    assert [p.key for p in hosted] == ["gemini", "openai", "anthropic"]
    assert all(
        p == EngineProfile(key=p.key, display_name=p.display_name, hosted=True) for p in hosted
    )


# key -> (always rendered, absent draft_n is zero, fixed method, reported window)
_FACTS = {
    "vllm": (False, False, None, False),
    "litellm": (False, False, None, False),
    "llamacpp": (True, True, None, True),
    "sglang": (False, False, None, False),
    "gemini": (False, False, None, False),
    "openai": (False, False, None, False),
    "anthropic": (False, False, None, False),
    "ninfer": (False, False, None, False),
    "tensorfold": (True, False, None, False),
    "halogen": (False, False, None, False),
    "strata": (True, False, "mtp", True),
    "tabbyapi": (False, False, None, False),
}


@pytest.mark.parametrize("profile", ENGINE_PROFILES, ids=lambda p: p.key)
def test_profile_facts(profile: EngineProfile) -> None:
    assert (
        profile.spec_counters_always_rendered,
        profile.absent_draft_n_is_zero,
        profile.fixed_spec_method,
        profile.reports_request_context_window,
    ) == _FACTS[profile.key]


def test_backend_labels_list_every_engine_then_unknown() -> None:
    assert BACKEND_LABELS == (*_FACTS, UNKNOWN_BACKEND)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("llamacpp", LLAMACPP),
        ("LLAMA.CPP", LLAMACPP),
        ("llama_cpp", LLAMACPP),
        ("Strata", STRATA),
        ("unknown", None),
        ("", None),
        (None, None),
        ("llama", None),
    ],
)
def test_engine_profile_by_label(label: str | None, expected: EngineProfile | None) -> None:
    assert engine_profile(label) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [("llama.cpp", LLAMACPP), ("Strata", STRATA), ("strata", None), ("", None), (None, None)],
)
def test_engine_profile_by_display_name(name: str | None, expected: EngineProfile | None) -> None:
    assert engine_profile_by_name(name) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('strata:live_state{state="idle"} 1\n', True),
        ("vllm:x 1\nstrata:live_state 1\n", True),
        ('vllm:x{model_name="acme/strata:7b"} 1\n', False),
        ("# HELP strata:live_state x\n", False),
        ("", False),
    ],
)
def test_metrics_namespace_is_matched_at_line_start(text: str, expected: bool) -> None:
    assert metrics_namespace_present(text, STRATA) is expected


def test_spec_live_labels_are_registry_keys() -> None:
    keys = {p.key for p in ENGINE_PROFILES} | {UNKNOWN_BACKEND}
    for _, text, _ in _LIVE_CASES:
        snap = _parse_snapshot(text)
        assert snap.spec_backend in keys


def _backend_help() -> str:
    parser = _make_parser()
    (action,) = [a for a in parser._actions if "--backend" in a.option_strings]
    assert isinstance(action, argparse.Action) and action.help
    return action.help


@pytest.mark.parametrize("label", BACKEND_LABELS)
def test_hand_written_label_lists_name_every_backend(label: str) -> None:
    # The CLI help and the API docstring list labels by hand.
    assert label in _backend_help()
    if label != UNKNOWN_BACKEND:
        assert f"``{label}``" in (api.run_benchmark.__doc__ or "")


def test_declared_name_reads_the_named_source() -> None:
    assert TENSORFOLD.declared_name("owned_by") == "tensorfold"
    assert LLAMACPP.declared_name("server") == "llama.cpp"
    assert STRATA.declared_name("owned_by") is None
