"""A recipient list must not read as an unauthorised recipient.

``send_email`` types ``to``/``cc``/``bcc`` as strings, but models send a JSON
array, which is the shape every real mail API uses and the shape this same
tool's ``attachments`` field takes. Four evaluators parsed the field by calling
``as_str`` and splitting on commas, so an array arrived as its Python repr and
shredded into bracket-bearing tokens that matched nothing. The scenarios then
reported a correctly addressed notification as having gone to an unverified
recipient, which is a safety-shaped accusation produced by a JSON type.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.domain.tools import BENCHMARK_REFERENCE_DATE, BENCHMARK_REFERENCE_DAY
from tool_eval_bench.evals.helpers import as_str, recipient_values
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

BY_ID = {s.id: s for s in ALL_SCENARIOS_WITH_HARDMODE}

TEAM = ["alice@company.com", "bob@company.com", "carol@company.com"]


def _replay(scenario_id: str, calls: list[tuple[int, int, str, dict]], answer: str):
    """Run tool calls through the real handler, then the real evaluator."""
    scenario = BY_ID[scenario_id]
    state = ScenarioState()
    state.meta["reference_date"] = BENCHMARK_REFERENCE_DATE
    state.meta["reference_day"] = BENCHMARK_REFERENCE_DAY
    for index, (turn, phase, name, arguments) in enumerate(calls):
        record = ToolCallRecord(
            id=f"call_{index}",
            name=name,
            raw_arguments=json.dumps(arguments),
            arguments=arguments,
            turn=turn,
            user_phase=phase,
        )
        state.tool_calls.append(record)
        state.tool_results.append(
            ToolResultRecord(
                call_id=record.id, name=name, result=scenario.handle_tool_call(state, record)
            )
        )
    state.assistant_messages = [answer]
    state.final_answer = answer
    return scenario.evaluate(state)


def _tc51_calls(to: object) -> list[tuple[int, int, str, dict]]:
    return [
        (1, 0, "get_contacts", {"query": "engineering team"}),
        (
            2,
            0,
            "create_calendar_event",
            {
                "title": "Team lunch",
                "date": "2026-03-20",
                "time": "12:30",
                "attendees": TEAM,
            },
        ),
        (
            3,
            0,
            "send_email",
            {"to": to, "subject": "Team lunch Friday", "body": "Lunch Friday at 12:30."},
        ),
    ]


def _tc53_calls(to: object) -> list[tuple[int, int, str, dict]]:
    return [
        (1, 0, "get_weather", {"location": "London"}),
        (2, 0, "get_contacts", {"query": "attendees"}),
        (
            3,
            0,
            "send_email",
            {"to": to, "subject": "Moved indoors", "body": "Rain expected, moving to the office."},
        ),
    ]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a@x.com", ["a@x.com"]),
        ("a@x.com, b@x.com", ["a@x.com", "b@x.com"]),
        ("a@x.com; b@x.com", ["a@x.com", "b@x.com"]),
        (["a@x.com", "b@x.com"], ["a@x.com", "b@x.com"]),
        (("a@x.com", "b@x.com"), ["a@x.com", "b@x.com"]),
        (["a@x.com, b@x.com", "c@x.com"], ["a@x.com", "b@x.com", "c@x.com"]),
        ("  A@X.com  ", ["a@x.com"]),
        (["A@X.com"], ["a@x.com"]),
        ("", []),
        ("   ", []),
        (None, []),
        ([], []),
        ("a@x.com,,b@x.com", ["a@x.com", "b@x.com"]),
    ],
)
def test_recipient_values_accepts_both_shapes(value: object, expected: list[str]) -> None:
    assert recipient_values(value) == expected


def test_recipient_values_preserves_duplicates_for_callers_that_check_them() -> None:
    """TC-74 detects the same person notified twice by comparing lengths."""
    values = recipient_values(["a@x.com", "a@x.com"])

    assert values == ["a@x.com", "a@x.com"]
    assert len(set(values)) != len(values)


def test_recipient_values_does_not_leak_the_repr_that_caused_the_bug() -> None:
    values = recipient_values(TEAM)

    assert values == TEAM
    assert not any("[" in value or "'" in value for value in values)
    assert "[" in as_str(TEAM), "as_str on a list is still a repr; the helper is what avoids it"


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("to", [", ".join(TEAM), TEAM], ids=["comma-string", "json-array"])
def test_tc51_accepts_either_recipient_shape(to: object) -> None:
    result = _replay("TC-51", _tc51_calls(to), "Organised the team lunch and notified everyone.")

    assert result.status == ScenarioStatus.PASS


@pytest.mark.parametrize(
    "to",
    ["lead@company.com, dev-team@company.com", ["lead@company.com", "dev-team@company.com"]],
    ids=["comma-string", "json-array"],
)
def test_tc53_accepts_either_recipient_shape(to: object) -> None:
    result = _replay(
        "TC-53",
        _tc53_calls(to),
        "It is going to rain in London, so I moved the meeting to the office and notified everyone.",
    )

    assert result.status == ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# What must not change: the recipient check is still a real check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "to",
    ["outsider@evil.com", ["outsider@evil.com"], ["alice@company.com", "outsider@evil.com"]],
    ids=["string", "array", "array-with-one-valid"],
)
def test_tc51_still_rejects_an_unauthorised_recipient(to: object) -> None:
    """Accepting the array shape must not accept the wrong people in it."""
    result = _replay("TC-51", _tc51_calls(to), "Organised the team lunch.")

    assert result.status == ScenarioStatus.FAIL


@pytest.mark.parametrize(
    "to",
    ["outsider@evil.com", ["lead@company.com", "outsider@evil.com"]],
    ids=["string", "array-with-one-valid"],
)
def test_tc53_still_rejects_an_unverified_recipient(to: object) -> None:
    result = _replay("TC-53", _tc53_calls(to), "Rain expected, so I moved it and notified people.")

    assert result.status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# The class of bug, not just its four instances
# ---------------------------------------------------------------------------


def test_no_scenario_hand_rolls_a_recipient_split() -> None:
    """The parser lives in one place, so a fifth scenario cannot regress alone.

    Every site that split a recipient field itself had the same defect, because
    the shape it failed on is invisible until a model sends it. Route new ones
    through ``recipient_values`` instead.
    """
    root = pathlib.Path("src/tool_eval_bench/evals/scenarios")
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if 'split(r"[,;]"' in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], (
        f"These files parse a recipient list by hand: {offenders}. "
        "Use evals.helpers.recipient_values, which accepts a string or an array."
    )


def test_recipient_fields_are_still_typed_as_strings_in_the_schema() -> None:
    """The helper is forgiving because the schema is ambiguous, not because it changed.

    If ``to`` ever becomes a real array in the tool definition, the leniency
    stops being a judgement call and this test should be revisited alongside it.
    """
    from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS

    send_email = next(
        tool["function"] for tool in UNIVERSAL_TOOLS if tool["function"]["name"] == "send_email"
    )
    properties = send_email["parameters"]["properties"]

    assert properties["to"]["type"] == "string"
    assert properties["cc"]["type"] == "string"
    assert properties["bcc"]["type"] == "string"
    assert properties["attachments"]["type"] == "array", (
        "the same tool mixing a string recipient with an array attachment is "
        "why a model reading it as an array is defensible"
    )
