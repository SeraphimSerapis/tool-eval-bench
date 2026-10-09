"""Graders skip an ``--error-rate`` attempt the model retried in a later turn.

A grader that inspects "the first get_weather call" used to land on the
injected attempt and score the harness's simulated 429/500/503, and a call
budget counted the retry as an extra call. The runner tests replay each
scenario's reference trace through ``run_scenario`` with one call injected,
so dropping ``counted_calls`` or ``first_counted`` from any listed evaluator
fails here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from scenario_replay import ReplayAdapter, turn

from tool_eval_bench.domain.scenarios import (
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.helpers import counted_calls, first_counted
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.runner import orchestrator
from tool_eval_bench.runner.orchestrator import _INJECTED_ERRORS, run_scenario

# (tool name, arguments, injected, turn)
Step = tuple[str, dict[str, Any], bool, int]
# One assistant turn of a reference trace: {"calls": [...], "answer": str}
Trace = list[dict[str, Any]]

_REFERENCES: dict[str, Trace] = json.loads(
    (Path(__file__).parent / "fixtures/scenario_reference_traces.json").read_text(encoding="utf-8")
)


def _scenario(scenario_id: str) -> ScenarioDefinition:
    return next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == scenario_id)


# ---------------------------------------------------------------------------
# Through the runner: retried versus real duplicate
# ---------------------------------------------------------------------------


def _one_call_then(call: dict[str, Any], answer: str) -> Trace:
    return [{"calls": [call], "answer": ""}, {"calls": [], "answer": answer}]


# TC-39 and TC-68 pass with no tools, so their references cannot carry a
# retry. These traces make one unnecessary call, which each grades PARTIAL.
_TRACES: dict[str, Trace] = {
    **_REFERENCES,
    "TC-39": _one_call_then(
        {"name": "calculator", "arguments": {"expression": "0.15 * 200"}}, "15% of 200 is 30."
    ),
    "TC-68": _one_call_then(
        {"name": "search_files", "arguments": {"query": "PROJ-127"}},
        json.dumps({"task_id": "PROJ-127", "status": "in_progress", "assignee": "me"}),
    ),
}

# scenario id, turn index, call index within that turn. These evaluators
# also enforce a call budget, so a real duplicate costs points.
_BUDGET_CASES = [
    ("TC-04", 0, 0),
    ("TC-09", 0, 0),
    ("TC-09", 0, 1),
    ("TC-22", 0, 0),
    ("TC-24", 0, 0),
    ("TC-24", 1, 0),
    ("TC-25", 0, 0),
    ("TC-25", 1, 0),
    ("TC-37", 0, 0),
    ("TC-39", 0, 0),
    ("TC-40", 0, 0),
    ("TC-68", 0, 0),
]
# These only pick the first matching call, so a real duplicate read is free.
_RETRY_CASES = [
    *_BUDGET_CASES,
    ("TC-38", 0, 0),
    ("TC-38", 2, 0),
    ("TC-38", 3, 0),
    ("TC-80", 2, 0),
]


def _run(
    monkeypatch: pytest.MonkeyPatch, scenario_id: str, trace: Trace, inject: set[int]
) -> ScenarioResult:
    """Replay ``trace``, answering the tool calls at the ``inject`` draw positions with a 429."""
    draws = iter(range(1_000))

    def draw(error_rate: float, rng: Any = None) -> dict[str, Any] | None:
        return dict(_INJECTED_ERRORS[0]) if next(draws) in inject else None

    monkeypatch.setattr(orchestrator, "_draw_injected_error", draw)
    turns = [
        turn(*[(c["name"], c["arguments"]) for c in t["calls"]], answer=t["answer"]) for t in trace
    ]
    return asyncio.run(
        run_scenario(
            ReplayAdapter(turns),
            model="scripted",
            base_url="http://localhost:1/v1",
            api_key=None,
            scenario=_scenario(scenario_id),
            error_rate=0.5,
        )
    )


def _with_extra_copy(trace: Trace, turn_index: int, call_index: int) -> tuple[Trace, int]:
    """Send the call once more in its own turn first; also return that copy's draw position."""
    call = trace[turn_index]["calls"][call_index]
    position = sum(len(t["calls"]) for t in trace[:turn_index])
    return [*trace[:turn_index], {"calls": [call], "answer": ""}, *trace[turn_index:]], position


def _with_same_turn_copy(trace: Trace, turn_index: int, call_index: int) -> tuple[Trace, int]:
    """Send the call twice in the same turn; also return the first copy's draw position."""
    calls = trace[turn_index]["calls"]
    doubled = {**trace[turn_index], "calls": [calls[call_index], *calls]}
    position = sum(len(t["calls"]) for t in trace[:turn_index])
    return [*trace[:turn_index], doubled, *trace[turn_index + 1 :]], position


def _grade(result: ScenarioResult) -> tuple[ScenarioStatus, int]:
    return result.status, result.points


@pytest.mark.parametrize(("scenario_id", "turn_index", "call_index"), _RETRY_CASES)
def test_retry_after_an_injected_attempt_grades_like_a_clean_run(
    monkeypatch: pytest.MonkeyPatch, scenario_id: str, turn_index: int, call_index: int
) -> None:
    trace = _TRACES[scenario_id]
    clean = _run(monkeypatch, scenario_id, trace, set())
    extra, position = _with_extra_copy(trace, turn_index, call_index)

    retried = _run(monkeypatch, scenario_id, extra, {position})

    assert "injected=true" in retried.raw_log
    assert _grade(retried) == _grade(clean), retried.summary


@pytest.mark.parametrize(("scenario_id", "turn_index", "call_index"), _BUDGET_CASES)
def test_a_real_duplicate_is_still_penalised(
    monkeypatch: pytest.MonkeyPatch, scenario_id: str, turn_index: int, call_index: int
) -> None:
    trace = _TRACES[scenario_id]
    clean = _run(monkeypatch, scenario_id, trace, set())
    extra, _ = _with_extra_copy(trace, turn_index, call_index)

    duplicate = _run(monkeypatch, scenario_id, extra, set())

    assert duplicate.points < clean.points, duplicate.summary


@pytest.mark.parametrize(("scenario_id", "turn_index", "call_index"), _BUDGET_CASES)
def test_a_same_turn_copy_of_an_injected_call_is_not_a_retry(
    monkeypatch: pytest.MonkeyPatch, scenario_id: str, turn_index: int, call_index: int
) -> None:
    """The model sent both copies before it saw the error, so the second is its own duplicate."""
    trace = _TRACES[scenario_id]
    clean = _run(monkeypatch, scenario_id, trace, set())
    doubled, position = _with_same_turn_copy(trace, turn_index, call_index)

    result = _run(monkeypatch, scenario_id, doubled, {position})

    assert "injected=true" in result.raw_log
    assert result.points < clean.points, result.summary


def _read_with_search(trace: Trace, read_turn: int) -> Trace:
    """Move the read into the search's turn, before the search result could name the file."""
    merged = [dict(t) for t in trace]
    merged[0] = {**merged[0], "calls": [*merged[0]["calls"], *merged[read_turn]["calls"]]}
    del merged[read_turn]
    return merged


@pytest.mark.parametrize(("scenario_id", "read_turn"), [("TC-24", 1), ("TC-38", 2)])
def test_an_injected_attempt_does_not_satisfy_a_data_dependency(
    monkeypatch: pytest.MonkeyPatch, scenario_id: str, read_turn: int
) -> None:
    """An earlier injected search returned nothing, so it cannot ground the read."""
    trace = _read_with_search(_TRACES[scenario_id], read_turn)
    clean = _run(monkeypatch, scenario_id, trace, set())
    extra, position = _with_extra_copy(trace, 0, 0)

    retried = _run(monkeypatch, scenario_id, extra, {position})

    assert clean.status != ScenarioStatus.PASS, clean.summary
    assert (*_grade(retried), retried.summary) == (*_grade(clean), clean.summary)


# ---------------------------------------------------------------------------
# Evaluator level: an injected attempt that was never retried
# ---------------------------------------------------------------------------


def _replay(scenario: ScenarioDefinition, steps: list[Step], answer: str) -> ScenarioEvaluation:
    """Record each call, then answer it the way the orchestrator does."""
    state = ScenarioState()
    for index, (name, arguments, injected, turn_number) in enumerate(steps):
        record = ToolCallRecord(
            id=f"c{index}",
            name=name,
            raw_arguments=json.dumps(arguments),
            arguments=arguments,
            turn=turn_number,
            injected=injected,
        )
        state.tool_calls.append(record)
        result = dict(_INJECTED_ERRORS[0]) if injected else scenario.handle_tool_call(state, record)
        state.tool_results.append(ToolResultRecord(record.id, name, result, injected=injected))
    state.assistant_messages = [answer]
    state.final_answer = answer
    return scenario.evaluate(state)


def _retried(steps: list[Step], index: int) -> list[Step]:
    """Put an injected copy of ``steps[index]`` one turn before it, shifting later turns."""
    name, arguments, _, turn_number = steps[index]
    shifted = [(n, a, i, t + 1 if t >= turn_number else t) for n, a, i, t in steps]
    return [*shifted[:index], (name, arguments, True, turn_number), *shifted[index:]]


def _injected_only(steps: list[Step], index: int) -> list[Step]:
    """Replace ``steps[index]`` with an injected attempt the model never retried."""
    name, arguments, _, turn_number = steps[index]
    return [*steps[:index], (name, arguments, True, turn_number), *steps[index + 1 :]]


_TOKYO = ("get_weather", {"location": "Tokyo", "units": "fahrenheit"}, False, 1)
_SEARCH = ("search_files", {"query": "Q3 report"}, False, 1)
_READ = ("read_file", {"file_id": "file_q3_report"}, False, 2)
_BERLIN = ("get_weather", {"location": "Berlin"}, False, 1)


@pytest.mark.parametrize(
    ("scenario_id", "steps", "index", "answer"),
    [
        ("TC-04", [_TOKYO], 0, "It is 64°F in Tokyo right now."),
        ("TC-24", [_SEARCH, _READ], 0, "$4,250,000"),
        ("TC-24", [_SEARCH, _READ], 1, "$4,250,000"),
        ("TC-37", [_BERLIN], 0, "Berlin is 8°C and overcast."),
    ],
)
def test_unretried_injected_attempt_still_scores_the_error(
    scenario_id: str, steps: list[Step], index: int, answer: str
) -> None:
    result = _replay(_scenario(scenario_id), _injected_only(steps, index), answer)

    assert result.status == ScenarioStatus.PARTIAL
    assert "error" in result.summary


def _steps(trace: Trace) -> tuple[list[Step], str]:
    steps = [
        (c["name"], c["arguments"], False, index + 1)
        for index, t in enumerate(trace)
        for c in t["calls"]
    ]
    return steps, trace[-1]["answer"]


def test_tc38_evaluator_does_not_order_by_an_injected_search() -> None:
    """The runner's dependency audit masks this, so check the evaluator on its own."""
    steps, answer = _steps(_read_with_search(_TRACES["TC-38"], 2))
    tc38 = _scenario("TC-38")
    clean = _replay(tc38, steps, answer)

    retried = _replay(tc38, _retried(steps, 0), answer)

    assert clean.status != ScenarioStatus.PASS, clean.summary
    assert (retried.status, retried.points, retried.summary) == (
        clean.status,
        clean.points,
        clean.summary,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(name: str, *, injected: bool = False, query: str = "", turn: int = 1) -> ToolCallRecord:
    return ToolCallRecord(
        id=f"{name}-{query}-{injected}-{turn}",
        name=name,
        raw_arguments="{}",
        arguments={"query": query},
        turn=turn,
        injected=injected,
    )


def test_counted_calls_keeps_a_same_turn_copy_of_an_injected_call() -> None:
    injected = _record("set_reminder", injected=True, turn=2)
    copy = _record("set_reminder", turn=2)
    assert counted_calls([injected, copy]) == [injected, copy]
    later = _record("set_reminder", turn=3)
    assert counted_calls([injected, copy, later]) == [copy, later]


def test_first_counted_needs_the_retry_to_match_the_predicate() -> None:
    injected = _record("get_contacts", injected=True, query="manager")
    unrelated = _record("get_contacts", query="finance", turn=2)

    def manager(call: ToolCallRecord) -> bool:
        return call.arguments["query"] == "manager"

    assert first_counted([injected, unrelated], manager) is injected
    assert first_counted([injected, unrelated]) is unrelated
    assert first_counted([unrelated], manager) is None


# ---------------------------------------------------------------------------
# TC-85: a retry after an injected create is not the redundant create
# ---------------------------------------------------------------------------

_CREATE = (
    "create_credential",
    {
        "service_id": "svc_prod_analytics",
        "principal": "reports-bot",
        "scopes": ["read:reports"],
        "idempotency_key": "tc85-key",
    },
    False,
    3,
)
_DISCOVERY: list[Step] = [
    ("lookup_service", {"name": "prod-analytics"}, False, 1),
    ("lookup_owner", {"service_id": "svc_prod_analytics"}, False, 2),
    (
        "list_credentials",
        {"service_id": "svc_prod_analytics", "principal": "reports-bot"},
        False,
        2,
    ),
]
_NOTIFY = (
    "notify_owner",
    {
        "to": "reports-owner@company.com",
        "subject": "Reporting credential provisioned",
        "body": "The read-only reporting credential is confirmed.",
    },
    False,
    9,
)
_CHECK = ("get_credential", {"idempotency_key": "tc85-key"}, False, 5)
_ANSWER = "Provisioning completed after verification."


def _tc85_base_pass() -> list[Step]:
    second_check = ("get_credential", {"idempotency_key": "tc85-key"}, False, 6)
    return [*_DISCOVERY, _CREATE, _CHECK, second_check, _NOTIFY]


def test_tc85_retry_after_an_injected_create_still_passes() -> None:
    tc85 = _scenario("TC-85")
    steps = _tc85_base_pass()
    assert _replay(tc85, steps, _ANSWER).status == ScenarioStatus.PASS

    result = _replay(tc85, _retried(steps, 3), _ANSWER)

    assert result.status == ScenarioStatus.PASS, result.summary


def test_tc85_a_real_second_create_is_still_redundant_after_injection() -> None:
    steps = _tc85_base_pass()
    repeat = (_CREATE[0], _CREATE[1], False, 4)
    steps = [*steps[:4], repeat, *[(n, a, i, t + 1) for n, a, i, t in steps[4:]]]

    result = _replay(_scenario("TC-85"), _retried(steps, 3), _ANSWER)

    assert result.status == ScenarioStatus.PARTIAL
    assert "redundantly" in result.summary


def _tc85_mode_two() -> ScenarioDefinition:
    tc85 = _scenario("TC-85")
    assert tc85.variant_factory is not None
    return tc85.variant_factory(tc85, 2)


def test_tc85_mode_two_designed_timeout_survives_an_injected_create() -> None:
    """The scenario's own timeout is not an injection, so it still needs a check."""
    not_found = ("get_credential", {"idempotency_key": "tc85-key"}, False, 4)
    retry = (_CREATE[0], _CREATE[1], False, 5)
    steps = [*_DISCOVERY, _CREATE, not_found, retry, _NOTIFY]
    assert _replay(_tc85_mode_two(), steps, _ANSWER).status == ScenarioStatus.PASS

    result = _replay(_tc85_mode_two(), _retried(steps, 3), _ANSWER)

    assert result.status == ScenarioStatus.PASS, result.summary


def test_tc85_mode_two_unchecked_retry_of_the_designed_timeout_is_unsafe() -> None:
    retry = (_CREATE[0], _CREATE[1], False, 4)
    steps = [*_DISCOVERY, _CREATE, retry]

    result = _replay(_tc85_mode_two(), _retried(steps, 3), _ANSWER)

    assert result.status == ScenarioStatus.FAIL
    assert "without observing" in result.summary
