"""Build a scenario group from the files in its directory.

Each ``tcNN.py`` module exports ``SCENARIO`` and ``DISPLAY``.  Registration is
the file's existence, so adding a scenario cannot half-land the way appending
to two hand-maintained lists could.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Iterable

from tool_eval_bench.domain.scenarios import ScenarioDefinition, ScenarioDisplayDetail

_SCENARIO_MODULE = re.compile(r"^tc\d+$")


def scenario_number(scenario_id: str) -> int:
    """Sort key shared by every registry: the numeric half of ``TC-NN``."""
    return int(scenario_id.split("-")[1])


def collect_group(
    package: str, search_path: Iterable[str]
) -> tuple[list[ScenarioDefinition], dict[str, ScenarioDisplayDetail]]:
    """Import every ``tcNN`` module under *package* and collect what it defines."""
    scenarios: list[ScenarioDefinition] = []
    displays: dict[str, ScenarioDisplayDetail] = {}
    names = sorted(
        name
        for _, name, _ in pkgutil.iter_modules(list(search_path))
        if _SCENARIO_MODULE.match(name)
    )
    for name in names:
        module = importlib.import_module(f"{package}.{name}")
        scenario: ScenarioDefinition = module.SCENARIO
        scenarios.append(scenario)
        displays[scenario.id] = module.DISPLAY
    scenarios.sort(key=lambda s: scenario_number(s.id))
    return scenarios, displays
