"""Grid construction, prompting, and grading for the toolscale sweep.

Each item offers one target tool, three neighbours from one similarity tier,
and enough unrelated fillers to reach the tool count N::

    target + 3 neighbours (tier s) + (N - 4) fillers

Growing N only adds fillers, and changing s only swaps the three neighbours,
so the two effects cannot leak into each other. The draws are arranged so
every contrast is paired:

- every cell uses the same targets;
- for one (target, trial), the fillers are identical across tiers, and the
  neighbours occupy the same list positions in every tier;
- fillers are nested across N: the fillers at N=8 are the first four of
  those at N=16, and so on;
- the target sits at the same relative position in the list at every N.

N=1 is the control cell: the target alone. It separates "cannot call a tool"
from "picked the wrong tool".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.domain.tools import (
    BENCHMARK_REFERENCE_DATE,
    BENCHMARK_REFERENCE_DAY,
    SYSTEM_PROMPT,
)
from tool_eval_bench.plugins.toolscale.catalog import Catalog, CatalogTool

SIMILARITY_LEVELS: tuple[str, ...] = ("near", "related", "distinct")
"""Neighbour tiers, most to least similar to the target."""

CONTROL = "control"
"""Similarity label of the N=1 cell, which has no neighbours."""

NEIGHBOURS = 3
DEFAULT_COUNTS: tuple[int, ...] = (1, 4, 8, 16, 32, 64)

# Outcome vocabulary. Wrong picks are split by the role of the chosen tool:
# a neighbour pick in the near tier is similarity confusion, a filler pick is
# a count or attention failure.
CORRECT = "correct"
WRONG_NEIGHBOUR = "wrong_neighbour"
WRONG_FILLER = "wrong_filler"
HALLUCINATED = "hallucinated"
NO_CALL = "no_call"
TRUNCATED = "truncated"
ERROR = "error"
OUTCOMES: tuple[str, ...] = (
    CORRECT,
    WRONG_NEIGHBOUR,
    WRONG_FILLER,
    HALLUCINATED,
    NO_CALL,
    TRUNCATED,
    ERROR,
)


class GridError(ValueError):
    """The requested grid cannot be built from this catalog."""


@dataclass(frozen=True, slots=True)
class ToolscaleItem:
    """One request: a target, the tool list around it, and its grid coordinates."""

    target: CatalogTool
    target_rank: int  # index into the run's target list; pairs items across cells
    count: int
    similarity: str
    trial: int
    tools: tuple[CatalogTool, ...]  # presentation order
    roles: tuple[tuple[str, str], ...]  # (tool name, "target" | "neighbour" | "filler")

    @property
    def item_id(self) -> str:
        """Stable identifier, for example ``list_draft_invoices@N16/near/t0``."""
        return f"{self.target.name}@N{self.count}/{self.similarity}/t{self.trial}"

    @property
    def target_index(self) -> int:
        """The target's zero-based index in the tool list."""
        return next(i for i, tool in enumerate(self.tools) if tool.name == self.target.name)

    @property
    def position(self) -> float:
        """The target's relative position, 0.0 for first and 1.0 for last."""
        return self.target_index / (self.count - 1) if self.count > 1 else 0.0

    def role_of(self, name: str) -> str | None:
        """The role of tool *name* in this item, or ``None`` when it was not offered."""
        for tool_name, role in self.roles:
            if tool_name == name:
                return role
        return None


def max_count(catalog: Catalog) -> int:
    """The largest N this catalog supports.

    Fillers and the distinct-tier neighbours come from families outside the
    target's domain, at most one tool per family, and must not overlap. With F
    such families, ``(N - 4) + 3 <= F``.
    """
    total = len(catalog.families)
    outside = min(total - len(domain.families) for domain in catalog.domains)
    return outside + 1


def validate_counts(counts: list[int], catalog: Catalog) -> list[int]:
    """Sorted, de-duplicated counts, or :class:`GridError` naming the bad value."""
    if not counts:
        raise GridError("at least one tool count is needed")
    ceiling = max_count(catalog)
    for count in counts:
        if count != 1 and count < NEIGHBOURS + 1:
            raise GridError(
                f"tool count {count} is too small: use 1 (the control) or at least "
                f"{NEIGHBOURS + 1}, the target plus {NEIGHBOURS} neighbours"
            )
        if count > ceiling:
            raise GridError(f"tool count {count} exceeds this catalog's maximum of {ceiling}")
    return sorted(set(counts))


def validate_similarities(levels: list[str]) -> list[str]:
    """Known tiers in canonical order, or :class:`GridError`."""
    unknown = [level for level in levels if level not in SIMILARITY_LEVELS]
    if unknown:
        raise GridError(
            f"unknown similarity level(s) {unknown}; choose from {list(SIMILARITY_LEVELS)}"
        )
    if not levels:
        raise GridError("at least one similarity level is needed")
    return [level for level in SIMILARITY_LEVELS if level in levels]


def select_targets(catalog: Catalog, limit: int | None, seed: int) -> list[CatalogTool]:
    """Targets balanced across domains, then families, then variants.

    The default (``limit=None``) takes one random variant from every family.
    A smaller limit still spreads across domains because the order rotates
    through domains first. A larger one moves on to second variants.
    """
    rng = random.Random(f"toolscale-targets:{seed}")
    shuffled = {
        (family.domain, family.id): rng.sample(family.variants, len(family.variants))
        for family in catalog.families
    }
    rounds = max(len(v) for v in shuffled.values())
    width = max(len(domain.families) for domain in catalog.domains)
    ordered: list[CatalogTool] = []
    for variant_round in range(rounds):
        for family_index in range(width):
            for domain in catalog.domains:
                if family_index >= len(domain.families):
                    continue
                variants = shuffled[(domain.id, domain.families[family_index].id)]
                if variant_round < len(variants):
                    ordered.append(variants[variant_round])
    if limit is None:
        limit = len(catalog.families)
    if limit < 1:
        raise GridError("the target limit must be at least 1")
    return ordered[:limit]


@dataclass(frozen=True, slots=True)
class _Draw:
    """Everything random about one (target, trial), shared by all its cells."""

    neighbours: dict[str, tuple[CatalogTool, ...]]
    fillers: tuple[CatalogTool, ...]
    position: float


def _draw(catalog: Catalog, target: CatalogTool, seed: int, trial: int) -> _Draw:
    rng = random.Random(f"toolscale:{seed}:{trial}:{target.name}")
    family = catalog.family_of(target)
    domain = catalog.domain_of(target)

    near = [tool for tool in family.variants if tool.name != target.name]
    rng.shuffle(near)

    others_in_domain = [f for f in domain.families if f.id != family.id]
    related = [rng.choice(f.variants) for f in rng.sample(others_in_domain, NEIGHBOURS)]

    outside = [f for f in catalog.families if f.domain != domain.id]
    rng.shuffle(outside)
    pool = [rng.choice(f.variants) for f in outside]
    # Distinct neighbours come from the tail, fillers from the head, so the
    # two never overlap and the fillers stay a nested prefix as N grows.
    distinct = pool[-NEIGHBOURS:]
    fillers = pool[:-NEIGHBOURS]

    return _Draw(
        neighbours={
            "near": tuple(near[:NEIGHBOURS]),
            "related": tuple(related),
            "distinct": tuple(distinct),
        },
        fillers=tuple(fillers),
        position=rng.random(),
    )


def _arrange(
    target: CatalogTool,
    neighbours: tuple[CatalogTool, ...],
    fillers: tuple[CatalogTool, ...],
    *,
    position: float,
    order_seed: str,
) -> tuple[tuple[CatalogTool, ...], tuple[tuple[str, str], ...]]:
    slots = [*neighbours, *fillers]
    order = list(range(len(slots)))
    # The permutation depends on (target, trial, N) only, so the neighbour
    # slots land on the same positions whichever tier fills them.
    random.Random(order_seed).shuffle(order)
    distractors = [slots[i] for i in order]
    index = round(position * len(distractors))
    tools = (*distractors[:index], target, *distractors[index:])
    roles = (
        (target.name, "target"),
        *((tool.name, "neighbour") for tool in neighbours),
        *((tool.name, "filler") for tool in fillers),
    )
    return tools, roles


def build_items(
    catalog: Catalog,
    *,
    counts: list[int],
    similarities: list[str],
    limit: int | None = None,
    trials: int = 1,
    seed: int = 0,
) -> list[ToolscaleItem]:
    """Every item of the grid, deterministic for a given catalog and seed."""
    counts = validate_counts(counts, catalog)
    similarities = validate_similarities(similarities)
    if trials < 1:
        raise GridError("trials must be at least 1")
    targets = select_targets(catalog, limit, seed)

    items: list[ToolscaleItem] = []
    for trial in range(trials):
        for rank, target in enumerate(targets):
            draw = _draw(catalog, target, seed, trial)
            for count in counts:
                if count == 1:
                    items.append(
                        ToolscaleItem(
                            target=target,
                            target_rank=rank,
                            count=1,
                            similarity=CONTROL,
                            trial=trial,
                            tools=(target,),
                            roles=((target.name, "target"),),
                        )
                    )
                    continue
                fillers = draw.fillers[: count - 1 - NEIGHBOURS]
                order_seed = f"toolscale-order:{seed}:{trial}:{target.name}:{count}"
                for level in similarities:
                    tools, roles = _arrange(
                        target,
                        draw.neighbours[level],
                        fillers,
                        position=draw.position,
                        order_seed=order_seed,
                    )
                    items.append(
                        ToolscaleItem(
                            target=target,
                            target_rank=rank,
                            count=count,
                            similarity=level,
                            trial=trial,
                            tools=tools,
                            roles=roles,
                        )
                    )
    return items


def build_messages(item: ToolscaleItem) -> list[ChatMessage]:
    """System and user messages for *item*.

    The system prompt is the scenario suite's own, unchanged and identical in
    every cell, so nothing in the prompt hints at which tool to pick.
    """
    return [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"Benchmark context: today is {BENCHMARK_REFERENCE_DATE} "
                f"({BENCHMARK_REFERENCE_DAY}). Use this date for any relative time request."
            ),
        },
        {"role": "user", "content": item.target.query},
    ]


def grade(item: ToolscaleItem, response: ChatCompletionResult) -> tuple[str, str | None]:
    """Classify a response by its first tool call: ``(outcome, picked tool name)``.

    Only the name is graded. Arguments are recorded but add a second failure
    source that has nothing to do with selection.
    """
    if not response.tool_calls:
        if response.finish_reason == "length":
            return TRUNCATED, None
        return NO_CALL, None
    picked = response.tool_calls[0].name
    role = item.role_of(picked)
    if role == "target":
        return CORRECT, picked
    if role == "neighbour":
        return WRONG_NEIGHBOUR, picked
    if role == "filler":
        return WRONG_FILLER, picked
    return HALLUCINATED, picked


def openai_tools(item: ToolscaleItem) -> list[dict[str, Any]]:
    """The item's tool list in OpenAI function format, in presentation order."""
    return [tool.to_openai_tool() for tool in item.tools]
