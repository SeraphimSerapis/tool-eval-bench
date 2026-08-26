"""Held-out scenario packs.

Every scenario in this repository is public: its prompt, its mock tool
responses, and its evaluator. That is good for auditability and bad for
measurement, because a published benchmark eventually leaks into training data
and a memorized answer is indistinguishable from a capable one.

A *pack* is a directory of YAML scenarios (see ``evals.yaml_loader``) kept
outside the repository. Runs can load one with ``--scenario-pack DIR``. Because
the scenarios are not public, a score against them needs two extra guarantees:

* **Attestation.** The pack's content hash is recorded in the run config and
  folded into ``config_fingerprint``, so anyone can verify that two published
  numbers were produced against the same held-out set — and that the set was not
  quietly edited between them — without seeing its contents.
* **Non-disclosure.** Scenarios loaded from a pack are marked ``held_out``, and
  reporting withholds their prompts, summaries, and traces. Publishing a score
  must not publish (and thereby burn) the scenarios that produced it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.evals.yaml_loader import load_yaml_scenarios_with_bytes


@dataclass(frozen=True)
class ScenarioPack:
    """A named, content-addressed set of scenarios loaded from a directory."""

    name: str
    path: Path
    scenarios: tuple[ScenarioDefinition, ...]
    content_hash: str

    def to_dict(self) -> dict[str, object]:
        """Attestation record for the run config — never the contents."""
        return {
            "name": self.name,
            "scenario_count": len(self.scenarios),
            "scenario_ids": [s.id for s in self.scenarios],
            "content_hash": self.content_hash,
        }


def pack_content_hash(directory: str | Path) -> str:
    """Hash every ``*.yaml`` file in *directory* by name and bytes.

    Deterministic across machines: files are visited in sorted order and hashed
    with their relative name, so renaming a file changes the hash even when the
    bytes are unchanged.
    """
    root = Path(directory)
    return _digest_files((path.name, path.read_bytes()) for path in sorted(root.glob("*.yaml")))


def _digest_files(files: Iterable[tuple[str, bytes]]) -> str:
    """Hash ``(name, bytes)`` pairs, which must already be in sorted order.

    Shared so that hashing bytes a caller already read produces exactly the
    same digest as re-reading the directory.  The pack content hash is an
    attestation, so the two paths must not be allowed to drift apart.
    """
    digest = hashlib.sha256()
    for name, payload in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def load_scenario_pack(directory: str | Path, *, held_out: bool = True) -> ScenarioPack:
    """Load a scenario pack from *directory*.

    Raises ValueError when the directory is missing, empty, or contains
    duplicate scenario IDs — a silently-empty pack would produce an "official"
    number measured against nothing.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Scenario pack directory not found: {root}")
    # One directory walk and one read per file: the bytes hashed below are the
    # same bytes the scenarios were parsed from.
    loaded = load_yaml_scenarios_with_bytes(root, held_out=held_out)
    scenarios = [scenario for scenario, _, _ in loaded]
    if not scenarios:
        raise ValueError(f"Scenario pack contains no *.yaml scenarios: {root}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen:
            duplicates.add(scenario.id)
        seen.add(scenario.id)
    if duplicates:
        raise ValueError(f"Duplicate scenario IDs in pack {root}: {', '.join(sorted(duplicates))}")
    return ScenarioPack(
        name=root.name,
        path=root,
        scenarios=tuple(scenarios),
        content_hash=_digest_files((path.name, raw) for _, path, raw in loaded),
    )


def load_scenario_packs(directories: list[str] | None) -> list[ScenarioPack]:
    """Load every requested pack, rejecting IDs that collide across packs."""
    if not directories:
        return []
    packs = [load_scenario_pack(d) for d in directories]
    seen: dict[str, str] = {}
    for pack in packs:
        for scenario in pack.scenarios:
            if scenario.id in seen:
                raise ValueError(
                    f"Scenario ID {scenario.id!r} appears in both "
                    f"{seen[scenario.id]!r} and {pack.name!r}"
                )
            seen[scenario.id] = pack.name
    return packs
