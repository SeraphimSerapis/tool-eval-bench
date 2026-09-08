"""Every built-in scenario has an executable reference, plus invalid mutations."""

import json
from pathlib import Path

import pytest
from scenario_replay import SCENARIOS, replay, turn

from tool_eval_bench.domain.scenarios import ScenarioStatus

REFERENCES = json.loads(
    (Path(__file__).parent / "fixtures/scenario_reference_traces.json").read_text()
)


def responses(trace):
    return [
        turn(*[(c["name"], c["arguments"]) for c in t["calls"]], answer=t["answer"]) for t in trace
    ]


def test_reference_corpus_covers_the_complete_registry():
    assert set(REFERENCES) == set(SCENARIOS)


@pytest.mark.parametrize("sid", sorted(REFERENCES))
def test_reference_through_real_runner(sid):
    result = replay(sid, *responses(REFERENCES[sid]))
    assert result.status is ScenarioStatus.PASS, result.summary
    assert not result.safety_violation


@pytest.mark.parametrize("sid", sorted(REFERENCES))
def test_empty_work_does_not_pass(sid):
    scenario = SCENARIOS[sid]
    result = replay(sid, *(turn() for _ in range(1 + len(scenario.follow_up_messages))))
    assert result.status is not ScenarioStatus.PASS


@pytest.mark.parametrize("sid", [sid for sid in sorted(REFERENCES) if SCENARIOS[sid].dependencies])
def test_unobserved_dependency_cannot_pass(sid):
    # Collapse tool rounds inside each user phase, retaining the user boundaries.
    batched = []
    pending = []
    for response in responses(REFERENCES[sid]):
        pending.extend(response.tool_calls)
        if not response.tool_calls:
            if pending:
                tool_turn = turn()
                tool_turn.tool_calls = pending
                batched.append(tool_turn)
                pending = []
            batched.append(response)
    # A cross-user-turn dependency remains observed even after this mutation.
    names_by_batch = [{c.name for c in r.tool_calls} for r in batched]
    if not any({a, b} <= names for a, b in SCENARIOS[sid].dependencies for names in names_by_batch):
        return
    result = replay(sid, *batched)
    assert result.status is not ScenarioStatus.PASS, result.summary


@pytest.mark.parametrize("sid", [f"TC-{i}" for i in range(64, 70)])
@pytest.mark.parametrize("wrapper", ["```json\n{}\n```", "Explanation: {}", "{}\nThank you."])
def test_structured_output_requires_complete_raw_json(sid, wrapper):
    turns = responses(REFERENCES[sid])
    turns[-1].content = wrapper.format(turns[-1].content)
    assert replay(sid, *turns).status is not ScenarioStatus.PASS


@pytest.mark.parametrize("sid", ["TC-47", "TC-48", "TC-50", "TC-60"])
def test_followup_action_cannot_use_information_from_a_future_user_phase(sid):
    trace = responses(REFERENCES[sid])
    first_boundary = next(i for i, response in enumerate(trace) if not response.tool_calls)
    action_index = next(
        i
        for i, response in enumerate(trace)
        if i > first_boundary
        and any(c.name in ("send_email", "create_calendar_event") for c in response.tool_calls)
    )
    action = trace.pop(action_index)
    trace.insert(first_boundary, action)
    result = replay(sid, *trace)
    assert result.status is not ScenarioStatus.PASS
    assert result.safety_violation


def test_cancellation_prevents_a_later_send():
    trace = responses(REFERENCES["TC-49"])
    trace.insert(
        -1,
        turn(
            (
                "send_email",
                {
                    "to": "team@company.com",
                    "subject": "Deadline",
                    "body": "The deadline is extended.",
                },
            )
        ),
    )
    result = replay("TC-49", *trace)
    assert result.status is ScenarioStatus.FAIL
    assert result.safety_violation
