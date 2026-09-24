"""Provenance and answer-truth regressions.

Each case starts from a scenario's reference trace, replayed through the real
runner so every call has its handler result, then removes one piece of
evidence or contradicts one fact. Before the fix, every mutated trace here
still passed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from conftest import make_state, simulate_results
from scenario_replay import SCENARIOS, replay, turn

from tool_eval_bench.domain.scenarios import ScenarioStatus

REFERENCES = json.loads(
    (Path(__file__).parent / "fixtures/scenario_reference_traces.json").read_text(encoding="utf-8")
)


def _trace(sid: str) -> list[dict]:
    return copy.deepcopy(REFERENCES[sid])


def _replay(sid: str, trace: list[dict]):
    return replay(
        sid,
        *[
            turn(*[(c["name"], c["arguments"]) for c in step["calls"]], answer=step["answer"])
            for step in trace
        ],
    )


def _without(trace: list[dict], name: str, query: str | None = None) -> list[dict]:
    """Drop the step whose only call is ``name`` (matching ``query`` if given)."""
    index = next(
        i
        for i, step in enumerate(trace)
        if len(step["calls"]) == 1
        and step["calls"][0]["name"] == name
        and (query is None or step["calls"][0]["arguments"].get("query") == query)
    )
    return trace[:index] + trace[index + 1 :]


@pytest.mark.parametrize("sid", ["TC-18", "TC-48", "TC-53", "TC-74", "TC-78", "TC-84", "TC-86"])
def test_reference_still_passes(sid: str) -> None:
    result = _replay(sid, _trace(sid))
    assert result.status is ScenarioStatus.PASS, result.summary


# ---------------------------------------------------------------------------
# Provenance: an address the model never looked up is a guess.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sid", "query"),
    [("TC-18", "Hans"), ("TC-48", "Alice"), ("TC-48", "Bob"), ("TC-74", "Mark")],
)
def test_unlooked_up_address_does_not_pass(sid: str, query: str) -> None:
    result = _replay(sid, _without(_trace(sid), "get_contacts", query))
    assert result.status is ScenarioStatus.PARTIAL, result.summary


def test_tc38_manager_lookup_that_failed_does_not_supply_the_address() -> None:
    state = make_state(
        tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 budget report"}, "turn": 1},
            {"name": "read_file", "arguments": {"file_id": "file_091"}, "turn": 2},
            {"name": "get_contacts", "arguments": {"query": "manager"}, "turn": 3},
            {
                "name": "send_email",
                "arguments": {
                    "to": "jordan.park@company.com",
                    "subject": "Budget",
                    "body": "Total is $4.4M",
                },
                "turn": 4,
            },
        ],
        tool_results=[
            {"call_id": "call_2", "name": "get_contacts", "result": {"error": "Directory down."}}
        ],
    )
    scenario = SCENARIOS["TC-38"]
    result = scenario.evaluate(simulate_results(state, scenario))
    assert result.status is ScenarioStatus.PARTIAL, result.summary


# ---------------------------------------------------------------------------
# TC-84: recovery must rest on a search made after the race.
# ---------------------------------------------------------------------------


def test_tc84_rebooking_from_stale_availability_is_partial() -> None:
    trace = _trace("TC-84")
    searches = [
        i for i, s in enumerate(trace) if s["calls"][:1] and s["calls"][0]["name"] == "search_rooms"
    ]
    del trace[searches[-1]]
    result = _replay("TC-84", trace)
    assert result.status is ScenarioStatus.PARTIAL
    assert "without searching rooms again" in result.summary


# ---------------------------------------------------------------------------
# TC-78: a reply that also states a wrong total is not a correct answer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "The total is $935. Correction: the portfolio is worth $1,035.",
        "Portfolio value: $935 (or $975 if BETA is $100).",
    ],
)
def test_tc78_contradicting_total_is_partial(answer: str) -> None:
    trace = _trace("TC-78")
    trace[-1]["answer"] = answer
    assert _replay("TC-78", trace).status is ScenarioStatus.PARTIAL


def test_tc78_negated_wrong_figure_still_passes() -> None:
    trace = _trace("TC-78")
    trace[-1]["answer"] = (
        "ACME 3 × $100 = $300, BETA 2 × $80 = $160, CYGN 5 × $95 = $475. "
        "The total is $935, not $1,035."
    )
    assert _replay("TC-78", trace).status is ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-86: the owner notice has to say what changed, in both variants.
# ---------------------------------------------------------------------------


def _generic_notice(trace: list[dict]) -> list[dict]:
    for step in trace:
        for call in step["calls"]:
            if call["name"] == "notify_owner":
                call["arguments"].update(subject="Update", body="Done.")
    return trace


def test_tc86_generic_notice_is_partial() -> None:
    result = _replay("TC-86", _generic_notice(_trace("TC-86")))
    assert result.status is ScenarioStatus.PARTIAL
    assert "did not name INC-442 and P1" in result.summary


def test_tc86_no_conflict_variant_applies_the_same_notice_rule() -> None:
    base = SCENARIOS["TC-86"]
    assert base.variant_factory is not None
    variant = base.variant_factory(base, 1)
    trace = [
        ("get_incident", {"incident_id": "INC-442"}),
        (
            "update_incident",
            {
                "incident_id": "INC-442",
                "expected_version": 7,
                "severity": "P1",
                "assignee": "Ana",
                "tags": ["customer-impact"],
            },
        ),
    ]

    def run(subject: str, body: str):
        notice = (
            "notify_owner",
            {"to": "incident-owner@company.com", "subject": subject, "body": body},
        )
        return replay(variant, *[turn(call) for call in [*trace, notice]], turn(answer="Done."))

    assert run("Incident updated", "INC-442 is now P1.").status is ScenarioStatus.PASS
    assert run("Update", "Done.").status is not ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-53: the moved meeting keeps its weekend date.
# ---------------------------------------------------------------------------


def test_tc53_event_on_another_date_is_not_the_moved_meeting() -> None:
    trace = _trace("TC-53")
    trace.insert(
        -1,
        {
            "calls": [
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "title": "Team sync (office)",
                        "date": "2026-04-15",
                        "time": "10:00",
                    },
                }
            ],
            "answer": "",
        },
    )
    assert _replay("TC-53", trace).status is not ScenarioStatus.PASS


def test_tc53_event_moved_indoors_on_its_own_date_passes() -> None:
    trace = _trace("TC-53")
    trace.insert(
        -1,
        {
            "calls": [
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "title": "Team sync (office)",
                        "date": "2026-03-21",
                        "time": "10:00",
                    },
                }
            ],
            "answer": "",
        },
    )
    assert _replay("TC-53", trace).status is ScenarioStatus.PASS
