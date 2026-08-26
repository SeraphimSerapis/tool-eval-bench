"""The scenario tree is the registry.

Adding a scenario used to mean appending to ``*_SCENARIOS`` *and* to
``*_DISPLAY_DETAILS``; forgetting either failed silently.  Registration is now
the existence of a ``tcNN.py`` file, and these tests pin that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
)
from tool_eval_bench.evals.scenarios import (
    ALL_DISPLAY_DETAILS,
    ALL_SCENARIOS_WITH_HARDMODE,
)
from tool_eval_bench.evals.scenarios._registry import collect_group, scenario_number

TREE = Path(__file__).resolve().parents[1] / "src" / "tool_eval_bench" / "evals" / "scenarios"
SCENARIO_FILES = sorted(TREE.glob("*/tc*.py"))


def test_the_tree_is_not_empty() -> None:
    assert len(SCENARIO_FILES) == len(ALL_SCENARIOS_WITH_HARDMODE) == 88


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_file_registers_exactly_one_scenario(path: Path) -> None:
    """A file that defines neither SCENARIO nor DISPLAY would import but vanish."""
    module = __import__(
        f"tool_eval_bench.evals.scenarios.{path.parent.name}.{path.stem}", fromlist=["SCENARIO"]
    )
    assert isinstance(module.SCENARIO, ScenarioDefinition)
    assert isinstance(module.DISPLAY, ScenarioDisplayDetail)
    assert module.SCENARIO.id == f"TC-{path.stem[2:]}", "file name and scenario id must agree"


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_scenario_is_rated_and_categorised(path: Path) -> None:
    """An unrated scenario silently drops out of ``--weight-by-difficulty``."""
    scenario = next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == f"TC-{path.stem[2:]}")
    assert isinstance(scenario.category, Category)
    assert scenario.difficulty in {1, 2, 3, 4, 5}, f"{scenario.id} has no usable difficulty"


def test_display_details_cover_exactly_the_registered_scenarios() -> None:
    assert set(ALL_DISPLAY_DETAILS) == {s.id for s in ALL_SCENARIOS_WITH_HARDMODE}


def test_ids_are_unique_and_sorted() -> None:
    ids = [s.id for s in ALL_SCENARIOS_WITH_HARDMODE]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids, key=scenario_number)


def test_a_group_with_no_scenario_files_collects_nothing(tmp_path: Path) -> None:
    """The scan must not invent scenarios, and must ignore ``_shared`` modules."""
    scenarios, displays = collect_group("tool_eval_bench.evals.scenarios", [str(tmp_path)])
    assert scenarios == [] and displays == {}


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_a_scenario_file_reaches_no_sibling_scenario(path: Path) -> None:
    """Scenarios share code through ``_shared``, never by importing each other.

    A direct sibling import would make one scenario's edit silently change
    another's score.
    """
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("tool_eval_bench.evals.scenarios.") or module.endswith(
                "._shared"
            ), f"{path.name} imports {module}"
