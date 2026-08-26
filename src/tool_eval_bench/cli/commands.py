"""Scenario resolution helpers for the CLI.

Extracted from the monolithic ``cli/bench.py`` so that the scenario-selection
rules can be unit-tested and reused without importing the full CLI dispatch.
"""

from __future__ import annotations

import argparse

from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.evals.packs import ScenarioPack, load_scenario_packs

_PACK_CACHE_ATTR = "_resolved_scenario_packs"


def resolve_packs(args: argparse.Namespace) -> list[ScenarioPack]:
    """Load --scenario-pack directories once and memoize on the namespace.

    Reading the YAML twice would be harmless, but hashing it twice invites the
    run's attestation and its scenarios to disagree if the pack changes on disk
    mid-run.
    """
    cached: list[ScenarioPack] | None = getattr(args, _PACK_CACHE_ATTR, None)
    if cached is not None:
        return cached
    packs = load_scenario_packs(getattr(args, "scenario_pack", None))
    setattr(args, _PACK_CACHE_ATTR, packs)
    return packs


def resolve_pack_scenarios(args: argparse.Namespace) -> list[ScenarioDefinition]:
    """Flatten every loaded held-out pack into a scenario list."""
    return [scenario for pack in resolve_packs(args) for scenario in pack.scenarios]


def resolve_scenarios(args: argparse.Namespace) -> list[ScenarioDefinition]:
    """Resolve scenarios from --short, --scenarios, --categories, and --hardmode flags.

    Priority: --scenarios (individual IDs) > --categories > --short > all.
    Explicit IDs resolve against the complete public registry, so naming a
    Hard Mode ID opts that scenario in without enabling the full Hard Mode
    suite.  The default pool still excludes Hard Mode.
    --hardmode-only runs Category P scenarios exclusively.
    --hardmode adds Category P scenarios to whichever base set is selected.
    --scenario-pack appends held-out packs; --pack-only runs those alone.
    """
    from tool_eval_bench.evals.scenarios import (
        ALL_SCENARIOS,
        ALL_SCENARIOS_WITH_HARDMODE,
        HARDMODE_SCENARIOS,
        SCENARIOS,
    )

    pack_scenarios = resolve_pack_scenarios(args)

    # Determine the base scenario pool. Explicit IDs are resolved against the
    # complete public registry so a named Hard Mode scenario opts in on its
    # own. ``--hardmode-only`` and ``--pack-only`` remain restrictive pools.
    if getattr(args, "pack_only", False):
        if not pack_scenarios:
            raise ValueError("--pack-only requires at least one --scenario-pack")
        base = list(pack_scenarios)
    elif getattr(args, "hardmode_only", False):
        base = list(HARDMODE_SCENARIOS)
    elif args.scenarios:
        base = list(ALL_SCENARIOS_WITH_HARDMODE)
    elif args.short:
        base = list(SCENARIOS)
        if getattr(args, "hardmode", False):
            base.extend(HARDMODE_SCENARIOS)
    elif getattr(args, "hardmode", False):
        base = list(ALL_SCENARIOS_WITH_HARDMODE)
    else:
        base = list(ALL_SCENARIOS)

    if pack_scenarios and not getattr(args, "pack_only", False):
        public_ids = {s.id for s in base}
        collisions = sorted(s.id for s in pack_scenarios if s.id in public_ids)
        if collisions:
            raise ValueError(
                "Held-out pack scenario IDs collide with the public suite: " + ", ".join(collisions)
            )
        base.extend(pack_scenarios)

    if args.scenarios:
        requested = set(args.scenarios)
        known_ids = {scenario.id for scenario in base}
        unknown = sorted(requested - known_ids)
        if unknown:
            raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
        return [s for s in base if s.id in requested]

    if args.categories:
        cats = {c.upper() for c in args.categories}
        return [s for s in base if s.category.value in cats]

    return base


def resolve_all_scenarios_for_ids(
    scenario_ids: list[str],
) -> list[ScenarioDefinition]:
    """Resolve ScenarioDefinitions by ID from ALL known scenarios.

    Used when reconstructing merged summaries from service-returned dicts
    (e.g. after resume merge) where we need the full definitions for scoring.
    """
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

    by_id = {s.id: s for s in ALL_SCENARIOS_WITH_HARDMODE}
    return [by_id[sid] for sid in scenario_ids if sid in by_id]
