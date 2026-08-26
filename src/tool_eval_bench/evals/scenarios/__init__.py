"""The scenario registry: every benchmark scenario, grouped by pack.

Each scenario lives in its own module under one of the group packages below,
holding its tools, its ``handle_tool_call``, its evaluator, its
``ScenarioDefinition``, and its ``ScenarioDisplayDetail``.  A group package
discovers its own files, so creating the file is the whole registration.

``ALL_SCENARIOS`` is the 69 standard scenarios (categories A–O).
``ALL_SCENARIOS_WITH_HARDMODE`` adds the 19 Hard Mode scenarios (category P),
which the CLI opts into with ``--hardmode``.
"""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import ScenarioDefinition, ScenarioDisplayDetail
from tool_eval_bench.evals.scenarios import (
    adversarial,
    agentic,
    core,
    extended,
    hardmode,
    hardmode_expanded,
    hardmode_transactional,
    large_toolset,
    planning,
    structured,
)
from tool_eval_bench.evals.scenarios._registry import scenario_number

#: The original ToolCall-15 scenarios (TC-01 to TC-15), kept separately because
#: ``--short`` runs exactly this set.
SCENARIOS: list[ScenarioDefinition] = core.SCENARIOS
SCENARIO_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = core.DISPLAY_DETAILS

EXTENDED_SCENARIOS = extended.SCENARIOS
AGENTIC_SCENARIOS = agentic.SCENARIOS
LARGE_TOOLSET_SCENARIOS = large_toolset.SCENARIOS
PLANNING_SCENARIOS = planning.SCENARIOS
ADVERSARIAL_SCENARIOS = adversarial.SCENARIOS
STRUCTURED_SCENARIOS = structured.SCENARIOS

#: Hard Mode is authored as three packs but ships as one category.
HARDMODE_SCENARIOS: list[ScenarioDefinition] = sorted(
    hardmode.SCENARIOS + hardmode_expanded.SCENARIOS + hardmode_transactional.SCENARIOS,
    key=lambda s: scenario_number(s.id),
)
HARDMODE_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    **hardmode.DISPLAY_DETAILS,
    **hardmode_expanded.DISPLAY_DETAILS,
    **hardmode_transactional.DISPLAY_DETAILS,
}

ALL_SCENARIOS: list[ScenarioDefinition] = sorted(
    SCENARIOS
    + EXTENDED_SCENARIOS
    + AGENTIC_SCENARIOS
    + LARGE_TOOLSET_SCENARIOS
    + PLANNING_SCENARIOS
    + ADVERSARIAL_SCENARIOS
    + STRUCTURED_SCENARIOS,
    key=lambda s: scenario_number(s.id),
)

ALL_SCENARIOS_WITH_HARDMODE: list[ScenarioDefinition] = sorted(
    ALL_SCENARIOS + HARDMODE_SCENARIOS,
    key=lambda s: scenario_number(s.id),
)

ALL_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    **SCENARIO_DISPLAY_DETAILS,
    **extended.DISPLAY_DETAILS,
    **agentic.DISPLAY_DETAILS,
    **large_toolset.DISPLAY_DETAILS,
    **planning.DISPLAY_DETAILS,
    **adversarial.DISPLAY_DETAILS,
    **structured.DISPLAY_DETAILS,
    **HARDMODE_DISPLAY_DETAILS,
}
