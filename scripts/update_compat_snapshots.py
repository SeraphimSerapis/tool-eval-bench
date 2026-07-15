#!/usr/bin/env python3
"""Regenerate committed public CLI compatibility snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.schema import get_schema

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "tests" / "snapshots"


def _json_value(value: Any) -> Any:
    if value is argparse.SUPPRESS:
        return "<SUPPRESS>"
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def legacy_cli_contract() -> list[dict[str, Any]]:
    """Return stable parser metadata for the legacy flag interface."""
    parser = make_parser()
    return [
        {
            "dest": action.dest,
            "flags": action.option_strings,
            "nargs": action.nargs,
            "required": action.required,
            "choices": list(action.choices) if action.choices is not None else None,
            "default": _json_value(action.default),
        }
        for action in parser._actions
        if action.dest not in {"help", "command"}
    ]


def write_snapshots() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "schema_v4.json": get_schema(),
        "legacy_cli.json": legacy_cli_contract(),
    }
    for name, payload in snapshots.items():
        (SNAPSHOT_DIR / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    write_snapshots()
