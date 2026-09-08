"""Scripted assistant turns replayed through the production scenario runner."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.scenarios import ScenarioDefinition, ScenarioResult
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.runner.orchestrator import run_scenario

SCENARIOS = {scenario.id: scenario for scenario in ALL_SCENARIOS_WITH_HARDMODE}


def turn(
    *calls: tuple[str, dict[str, Any]], answer: str = "", reasoning: str | None = None
) -> ChatCompletionResult:
    return ChatCompletionResult(
        content=answer,
        reasoning=reasoning,
        tool_calls=[
            ProviderToolCall(id=str(i), name=name, arguments_str=json.dumps(arguments))
            for i, (name, arguments) in enumerate(calls)
        ],
    )


class ReplayAdapter(BackendAdapter):
    def __init__(self, turns: list[ChatCompletionResult]) -> None:
        self.turns = copy.deepcopy(turns)
        self.position = 0
        for index, response in enumerate(self.turns):
            for call in response.tool_calls:
                call.id = f"turn_{index}_{call.id}"

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        if self.position >= len(self.turns):
            raise AssertionError("Scenario requested an unscripted assistant turn")
        result = self.turns[self.position]
        self.position += 1
        return result


def replay(scenario: str | ScenarioDefinition, *turns: ChatCompletionResult) -> ScenarioResult:
    selected = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    adapter = ReplayAdapter(list(turns))
    result = asyncio.run(
        run_scenario(
            adapter,
            model="scripted",
            base_url="http://localhost:1/v1",
            api_key=None,
            scenario=selected,
        )
    )
    assert adapter.position == len(turns), "Scenario stopped before consuming its reference trace"
    return result
