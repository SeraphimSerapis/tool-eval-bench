"""Guards the scenario counts quoted in prose against the live registries.

These numbers drifted silently once already: the CLI reference claimed 15
categories and 15 Hard Mode scenarios long after the registry had grown to 16
and 19.  Prose cannot be type-checked, so it is asserted instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tool_eval_bench.domain.scenarios import Category
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, ALL_SCENARIOS_WITH_HARDMODE

ROOT = Path(__file__).parents[1]

#: Prose files that quote registry sizes.  Update when a document moves.
DOCUMENTED_SOURCES = (
    ROOT / "SKILL.md",
    ROOT / "README.md",
    ROOT / "AGENTS.md",
)

CATEGORY_COUNT = len(list(Category))
STANDARD_COUNT = len(ALL_SCENARIOS)
HARDMODE_COUNT = len(ALL_SCENARIOS_WITH_HARDMODE) - len(ALL_SCENARIOS)


def _existing_sources() -> list[Path]:
    return [path for path in DOCUMENTED_SOURCES if path.is_file()]


def test_documented_sources_are_present() -> None:
    assert _existing_sources(), "no prose sources found; DOCUMENTED_SOURCES is stale"


@pytest.mark.parametrize(
    ("pattern", "expected", "label"),
    [
        (r"(\d+)\s+categories", CATEGORY_COUNT, "category count"),
        (r"(\d+)\s+Hard Mode scenarios", HARDMODE_COUNT, "Hard Mode scenario count"),
        (r"(\d+)\s+standard deterministic scenarios", STANDARD_COUNT, "standard scenario count"),
        (r"(\d+)\s+deterministic\s+scenarios", STANDARD_COUNT, "standard scenario count"),
    ],
)
def test_prose_counts_match_the_registry(pattern: str, expected: int, label: str) -> None:
    """Every number quoted next to these phrases must match the live registry."""
    for path in _existing_sources():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(pattern, text):
            quoted = int(match.group(1))
            assert quoted == expected, (
                f"{path.name} quotes {quoted} for the {label}; the registry has {expected}"
            )
