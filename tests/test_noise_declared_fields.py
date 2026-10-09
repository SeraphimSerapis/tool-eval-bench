"""The noise layer adds fields around a fixture; it never replaces or drops one.

A scenario's declared value is what its evaluator and its narrative assume.
Overwriting it, as ``enrich_stock`` once did to TC-67's ``volume``, hands the
model a payload that contradicts the scenario.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tool_eval_bench.evals.noise import (
    enrich_calendar,
    enrich_code_execution,
    enrich_contacts,
    enrich_email,
    enrich_file_read,
    enrich_file_search,
    enrich_payload,
    enrich_reminder,
    enrich_search,
    enrich_stock,
    enrich_translation,
    enrich_weather,
)

Enricher = Callable[[dict[str, Any]], dict[str, Any]]

FLAT: list[tuple[Enricher, dict[str, Any]]] = [
    (enrich_weather, {"temperature": 18}),
    (enrich_file_read, {"content": "a\nb"}),
    (enrich_email, {"status": "sent"}),
    (enrich_calendar, {"event_id": "e1"}),
    (enrich_stock, {"price": 100.0}),
    (enrich_translation, {"translated": "hola"}),
    (enrich_code_execution, {"stdout": "4"}),
    (enrich_reminder, {"status": "ok"}),
    (lambda payload: enrich_payload("any_tool", payload), {"error": "boom"}),
]
LISTS: list[Enricher] = [enrich_search, enrich_file_search, enrich_contacts]


def _declared(keys: list[str]) -> dict[str, Any]:
    return {key: f"declared-{key}" for key in keys}


@pytest.mark.parametrize(("enrich", "base"), FLAT)
def test_a_declared_key_survives_every_flat_enricher(
    enrich: Enricher, base: dict[str, Any]
) -> None:
    noise_keys = [key for key in enrich(base) if key not in base]
    # error_code is normalized to a string; a declared string is kept as-is.
    payload = {**base, **_declared(noise_keys)}

    enriched = enrich(payload)

    assert enriched == payload
    assert list(enriched) == list(payload)


@pytest.mark.parametrize("enrich", LISTS)
def test_declared_keys_survive_list_enrichers_at_both_levels(enrich: Enricher) -> None:
    sample = enrich({"results": [{"name": "A"}]})
    result_noise = [key for key in sample["results"][0] if key != "name"]
    top_noise = [key for key in sample if key != "results"]
    row = {"name": "A", **_declared(result_noise)}
    payload = {"results": [row], "query": "kept", **_declared(top_noise)}

    enriched = enrich(payload)

    assert enriched["results"] == [row]
    assert {key: enriched[key] for key in payload if key != "results"} == {
        key: value for key, value in payload.items() if key != "results"
    }


@pytest.mark.parametrize("enrich", LISTS)
def test_list_enrichers_keep_top_level_keys_they_do_not_know(enrich: Enricher) -> None:
    enriched = enrich({"results": [], "query": "vegan Berlin", "next_cursor": "c2"})

    assert list(enriched)[:3] == ["results", "query", "next_cursor"]
    assert (enriched["query"], enriched["next_cursor"]) == ("vegan Berlin", "c2")


def test_tc67_volume_reaches_the_model_as_declared() -> None:
    enriched = enrich_stock({"ticker": "NVDA", "price": 875.2, "volume": "42.3M"})

    assert enriched["volume"] == "42.3M"


def test_a_missing_or_empty_error_code_still_gets_the_generic_one() -> None:
    assert enrich_payload("t", {"error": "x"})["error_code"] == "ERR_TOOL_UNAVAILABLE"
    assert enrich_payload("t", {"error": "x", "error_code": None})["error_code"] == (
        "ERR_TOOL_UNAVAILABLE"
    )
    assert enrich_payload("t", {"error": "x", "error_code": "ROOM_TAKEN"})["error_code"] == (
        "ROOM_TAKEN"
    )


def test_an_enricher_without_collisions_produces_the_same_payload_and_order() -> None:
    """Additive-only must not move a field: key order is model-visible."""
    enriched = enrich_stock({"ticker": "AAPL", "price": 100.0})

    assert list(enriched.items()) == [
        ("ticker", "AAPL"),
        ("price", 100.0),
        ("timestamp", "2026-03-20T16:00:00Z"),
        ("exchange", "NASDAQ"),
        ("volume", 52_314_800),
        ("market_cap", "2.89T"),
        ("pe_ratio", 28.4),
        ("day_high", 101.2),
        ("day_low", 98.8),
        ("week_52_high", 125.0),
        ("week_52_low", 72.0),
        ("previous_close", 98.77),
        ("after_hours", None),
        ("request_id", "req_sp_8c1d4e2a"),
    ]
