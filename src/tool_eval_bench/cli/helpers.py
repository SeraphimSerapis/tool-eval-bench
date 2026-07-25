"""Small CLI helper functions used across the benchmark runner.

Extracted from the monolithic ``cli/bench.py`` to improve testability and
reduce coupling. These are pure or near-pure functions with no dependency on
the large ``main()`` dispatch or scenario execution code.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from dotenv import load_dotenv


def load_dotenv_file() -> None:
    """Load .env file into os.environ (does not overwrite existing vars)."""
    load_dotenv(override=False)


def redact_url(url: str) -> str:
    """Mask the host in a URL for display.  e.g. http://192.168.10.5:8080 → http://***:8080"""
    from tool_eval_bench.utils.urls import redact_url as _redact

    return _redact(url)


def metadata_for_storage(run_context: Any | None) -> dict[str, Any]:
    """Return JSON-safe persisted metadata for a plugin benchmark run."""
    return run_context.to_dict() if run_context is not None else {}


def with_config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    """Attach a deterministic comparison fingerprint to plugin config."""
    from tool_eval_bench.utils.ids import build_config_fingerprint

    return {**config, "config_fingerprint": build_config_fingerprint(config)}


def persist_plugin_run(run_data: dict[str, Any]) -> None:
    """Persist a plugin result, surfacing mandatory-storage failures."""
    from tool_eval_bench.storage.db import RunRepository

    with RunRepository() as repo:
        repo.upsert_scenario_run(run_data)


def parse_int_list(value: str) -> list[int]:
    """Parse a space-or-comma separated list of ints."""
    return [int(x) for x in value.replace(",", " ").split() if x.strip()]


def parse_sweep_range(sweep_str: str) -> tuple[float, float]:
    """Parse 'START-END' into (start, end) floats, each clamped to [0, 1]."""
    parts = sweep_str.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid sweep range '{sweep_str}'. Expected format: START-END (e.g. 0.5-1.0)"
        )
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(
            f"Invalid sweep range '{sweep_str}'. START and END must be numbers (e.g. 0.5-1.0)"
        ) from None
    start = max(0.0, min(1.0, start))
    end = max(0.0, min(1.0, end))
    if start >= end:
        raise ValueError(f"Sweep START ({start}) must be less than END ({end})")
    return start, end


def emit_headless_error(error_code: str, message: str, *, exit_code: int = 1) -> None:
    """Emit a structured JSONL error event on stderr and exit.

    Used in headless (--json) mode so agents can parse failure reasons
    instead of getting Rich-formatted console markup.
    """
    msg = {"event": "error", "error": error_code, "message": message}
    sys.stderr.write(json.dumps(msg) + "\n")
    sys.stderr.flush()
    sys.exit(exit_code)


def prior_results_for_resume(
    prev_run: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect the reusable scenario results of a prior run.

    A completed run carries everything in its final ``scores``.  A run that was
    interrupted never got that far, so its per-scenario checkpoints are the only
    record — without them the whole run would have to be redone.  When both
    exist, final scores win because they went through the full scoring pass.
    """
    scored = [
        r
        for r in (prev_run.get("scores") or {}).get("scenario_results") or []
        if r.get("scenario_id")
    ]
    if not checkpoints:
        return scored
    merged: dict[str, dict[str, Any]] = {
        r["scenario_id"]: r for r in checkpoints if r.get("scenario_id")
    }
    merged.update({r["scenario_id"]: r for r in scored})
    return list(merged.values())


def safety_gate_failed(args: Any, result: dict[str, Any]) -> bool:
    """Return whether an opted-in run must fail on safety warnings."""
    if not getattr(args, "fail_on_safety", False):
        return False
    warnings = (result.get("scores") or {}).get("safety_warnings") or []
    if warnings:
        for warning in warnings:
            print(f"SAFETY GATE: {warning}", file=sys.stderr)
        return True
    return False
