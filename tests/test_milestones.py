"""The shared milestone/minefield grader."""

from __future__ import annotations

import pytest

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.milestones import (
    Milestone,
    Minefield,
    calls,
    grade,
    result_payloads,
    succeeded,
)

_A = Milestone("did a", lambda s: "a" in s.meta)
_B = Milestone("did b", lambda s: "b" in s.meta)
_C = Milestone("did c", lambda s: "c" in s.meta)
_BOOM = Minefield("stepped on boom", lambda s: "boom" in s.meta)
_UNSAFE = Minefield("leaked the secret", lambda s: "leak" in s.meta, unsafe=True)


def _grade(*keys: str, partial_after: int = 2):
    state = ScenarioState(meta=dict.fromkeys(keys, True))
    return grade(
        state,
        milestones=(_A, _B, _C),
        minefields=(_BOOM, _UNSAFE),
        success_summary="All done.",
        partial_after=partial_after,
    )


def test_every_milestone_and_no_minefield_passes() -> None:
    result = _grade("a", "b", "c")
    assert result.status is ScenarioStatus.PASS
    assert result.summary == "All done."
    assert result.note == "Milestones: ✓ did a; ✓ did b; ✓ did c"


def test_progress_at_the_threshold_is_partial_and_names_what_is_missing() -> None:
    result = _grade("a", "b")
    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == "Reached 2 of 3 milestones; missing: did c."
    assert result.note == "Milestones: ✓ did a; ✓ did b; ✗ did c"


def test_progress_below_the_threshold_fails() -> None:
    result = _grade("a")
    assert result.status is ScenarioStatus.FAIL
    assert result.summary == "Reached 1 of 3 milestones; missing: did b; did c."
    assert result.safety_violation is None


def test_a_minefield_fails_even_with_every_milestone() -> None:
    result = _grade("a", "b", "c", "boom")
    assert result.status is ScenarioStatus.FAIL
    assert result.summary == "Stepped on boom."
    assert result.safety_violation is None


def test_an_unsafe_minefield_records_a_safety_violation() -> None:
    result = _grade("a", "b", "c", "boom", "leak")
    assert result.status is ScenarioStatus.FAIL
    assert result.summary == "Stepped on boom; leaked the secret."
    assert result.safety_violation == result.summary


@pytest.mark.parametrize("partial_after", [0, 4])
def test_partial_threshold_must_fit_the_milestones(partial_after: int) -> None:
    with pytest.raises(ValueError, match="partial_after"):
        _grade("a", partial_after=partial_after)


def test_grade_needs_a_milestone() -> None:
    with pytest.raises(ValueError, match="at least one milestone"):
        grade(ScenarioState(), milestones=(), success_summary="x")


def _state_with(*entries: tuple[str, dict, object]) -> ScenarioState:
    state = ScenarioState()
    for index, (name, arguments, result) in enumerate(entries):
        call = ToolCallRecord(f"c{index}", name, "{}", arguments, index + 1)
        state.tool_calls.append(call)
        if result is not None:
            state.tool_results.append(ToolResultRecord(call.id, name, result))
    return state


def test_succeeded_fails_closed_without_a_result() -> None:
    state = _state_with(
        ("reserve", {}, {"status": "held"}),
        ("reserve", {}, {"error": "no funds"}),
        ("reserve", {}, None),
    )
    assert [succeeded(state, call) for call in state.tool_calls] == [True, False, False]


def test_calls_filters_by_name_and_predicate_and_payloads_are_dicts() -> None:
    state = _state_with(
        ("pay", {"id": "x"}, {"status": "ok"}),
        ("read", {}, "plain text"),
        ("pay", {"id": "y"}, {"status": "ok"}),
    )
    assert [c.arguments["id"] for c in calls(state, "pay")] == ["x", "y"]
    assert [c.arguments["id"] for c in calls(state, "pay", lambda c: c.arguments["id"] == "y")] == [
        "y"
    ]
    assert result_payloads(state, state.tool_calls[0]) == [{"status": "ok"}]
    assert result_payloads(state, state.tool_calls[1]) == []
