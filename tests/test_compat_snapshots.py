from __future__ import annotations

import json
from pathlib import Path

from scripts.update_compat_snapshots import legacy_cli_contract
from tool_eval_bench.schema import get_schema

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _snapshot(name: str) -> object:
    return json.loads((SNAPSHOT_DIR / name).read_text(encoding="utf-8"))


def test_schema_v4_compatibility_snapshot() -> None:
    assert get_schema() == _snapshot("schema_v4.json")


def test_legacy_cli_compatibility_snapshot() -> None:
    assert legacy_cli_contract() == _snapshot("legacy_cli.json")
