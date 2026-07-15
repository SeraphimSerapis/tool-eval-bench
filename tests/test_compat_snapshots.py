from __future__ import annotations

import json
from pathlib import Path

from scripts.update_compat_snapshots import legacy_cli_contract
from tool_eval_bench.cli.helpers import safety_gate_failed
from tool_eval_bench.schema import get_schema

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _snapshot(name: str) -> object:
    return json.loads((SNAPSHOT_DIR / name).read_text(encoding="utf-8"))


def test_schema_v4_compatibility_snapshot() -> None:
    assert get_schema() == _snapshot("schema_v4.json")


def test_legacy_cli_compatibility_snapshot() -> None:
    assert legacy_cli_contract() == _snapshot("legacy_cli.json")


def test_safety_gate_is_opt_in_and_reports_warnings(capsys) -> None:
    result = {"scores": {"safety_warnings": ["TC-60 injection"]}}
    assert safety_gate_failed(type("Args", (), {"fail_on_safety": False})(), result) is False
    assert safety_gate_failed(type("Args", (), {"fail_on_safety": True})(), result) is True
    assert "SAFETY GATE: TC-60 injection" in capsys.readouterr().err
