"""The worked example in the contributor guide has to actually run.

A code sample in documentation rots silently: nothing imports it, so a helper
rename leaves a plausible-looking example that fails the first time someone
copies it.  This runs the one in ``docs/adding-a-scenario.md`` for real.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tool_eval_bench.domain.scenarios import (
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "adding-a-scenario.md"


@pytest.fixture(scope="module")
def example() -> dict:
    block = re.search(r"```python\n(.*?)```", GUIDE.read_text(encoding="utf-8"), re.S)
    assert block is not None, "the guide must keep a Python example"
    namespace: dict = {}
    exec(compile(block.group(1), str(GUIDE), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def call() -> ToolCallRecord:
    return ToolCallRecord(
        id="c1", name="convert_timezone", arguments={}, raw_arguments="{}", turn=1
    )


def test_the_example_defines_both_exports(example: dict) -> None:
    assert isinstance(example["SCENARIO"], ScenarioDefinition)
    assert isinstance(example["DISPLAY"], ScenarioDisplayDetail)
    assert example["SCENARIO"].difficulty in {1, 2, 3, 4, 5}


def test_the_example_handler_is_deterministic(example: dict, call: ToolCallRecord) -> None:
    handle = example["SCENARIO"].handle_tool_call
    assert handle(ScenarioState(), call) == handle(ScenarioState(), call)


def test_the_example_scores_all_three_tiers(example: dict, call: ToolCallRecord) -> None:
    evaluate = example["SCENARIO"].evaluate

    no_call = ScenarioState()

    called_but_silent = ScenarioState()
    called_but_silent.tool_calls.append(call)
    called_but_silent.final_answer = "I converted it."

    complete = ScenarioState()
    complete.tool_calls.append(call)
    complete.final_answer = "It is 00:00 in Los Angeles."

    assert evaluate(no_call).status is ScenarioStatus.FAIL
    assert evaluate(called_but_silent).status is ScenarioStatus.PARTIAL
    assert evaluate(complete).status is ScenarioStatus.PASS
