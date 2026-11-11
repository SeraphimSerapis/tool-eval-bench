"""Small CLI helper functions used across the benchmark runner.

Extracted from the monolithic ``cli/bench.py`` to improve testability and
reduce coupling. These are pure or near-pure functions with no dependency on
the large ``main()`` dispatch or scenario execution code.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from dotenv import find_dotenv, load_dotenv

# Re-exported: the fingerprint now lives in utils, and CLI callers still import
# it from here.
from tool_eval_bench.utils.fingerprint import (
    with_config_fingerprint as with_config_fingerprint,
)
from tool_eval_bench.utils.urls import redact_url as _redact_url
from tool_eval_bench.utils.urls import redact_urls as _redact_urls


def load_dotenv_file() -> None:
    """Prefer a cwd/parent .env, then the legacy lookup; keep existing env vars."""
    load_dotenv(find_dotenv(usecwd=True) or find_dotenv(), override=False)


def redact_url(url: str) -> str:
    """Mask the host in a URL for display.  e.g. http://192.168.10.5:8080 → http://***:8080"""
    return _redact_url(url)


def metadata_for_storage(run_context: Any | None) -> dict[str, Any]:
    """Return JSON-safe persisted metadata for a plugin benchmark run.

    No caller in ``src`` uses this; it is kept because ``cli.bench`` publishes it
    as ``_metadata_for_storage``.  Mode runs build metadata in
    ``application.mode_runs.finalize_mode_run``.
    """
    return run_context.to_dict() if run_context is not None else {}


def persist_plugin_run(run_data: dict[str, Any]) -> None:
    """Persist a plugin result, surfacing mandatory-storage failures.

    No caller in ``src`` uses this; it is kept because ``cli.bench`` publishes it
    as ``_persist_plugin_run``.  Mode runs persist through
    ``application.run_queries.persist_run``.
    """
    from tool_eval_bench.application.run_queries import persist_run

    persist_run(run_data)


def parse_int_list(value: str) -> list[int]:
    """Parse a space-or-comma separated list of ints."""
    return [int(x) for x in value.replace(",", " ").split() if x.strip()]


def parse_sweep_range(sweep_str: str) -> tuple[float, float]:
    """Parse 'START-END' into (start, end) floats, each clamped to [0, 1]."""
    raw = sweep_str.strip()
    sep_idx = next(
        (idx for idx in range(1, len(raw)) if raw[idx] == "-" and raw[idx - 1] not in ("e", "E")),
        None,
    )
    if sep_idx is None:
        raise ValueError(
            f"Invalid sweep range '{sweep_str}'. Expected format: START-END (e.g. 0.5-1.0)"
        )
    left, right = raw[:sep_idx], raw[sep_idx + 1 :]
    try:
        start, end = float(left), float(right)
    except ValueError:
        raise ValueError(
            f"Invalid sweep range '{sweep_str}'. START and END must be numbers (e.g. 0.5-1.0)"
        ) from None
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(
            f"Invalid sweep range '{sweep_str}'. START and END must be finite numbers (e.g. 0.5-1.0)"
        )
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
    # --json output is shareable, so a URL in the message is redacted.
    msg = {"event": "error", "error": error_code, "message": _redact_urls(message)}
    sys.stderr.write(json.dumps(msg) + "\n")
    sys.stderr.flush()
    sys.exit(exit_code)


def safety_gate_failed(args: Any, result: dict[str, Any]) -> bool:
    """Return whether an opted-in run must fail on safety warnings."""
    if not getattr(args, "fail_on_safety", False):
        return False
    warnings = (result.get("scores") or {}).get("safety_warnings") or []
    if not warnings:
        return False
    from tool_eval_bench.cli.held_out import held_out_ids, redact_warnings

    redacted = redact_warnings(warnings, held_out_ids(args))
    if getattr(args, "json", False):
        event = {
            "event": "safety_gate_failed",
            "safety_warnings": redacted,
        }
        sys.stderr.write(json.dumps(event) + "\n")
        sys.stderr.flush()
    else:
        for warning in redacted:
            print(f"SAFETY GATE: {warning}", file=sys.stderr)
    return True


def trial_safety_warnings(results: list[dict[str, Any]]) -> list[str]:
    """Every trial's safety warnings, deduplicated in first-seen order."""
    warnings = (
        warning
        for result in results
        for warning in (result.get("scores") or {}).get("safety_warnings") or []
    )
    return list(dict.fromkeys(warnings))


def trials_safety_gate_failed(args: Any, results: list[dict[str, Any]]) -> bool:
    """Gate on all trials at once: one unsafe trial fails a multi-trial run."""
    return safety_gate_failed(args, {"scores": {"safety_warnings": trial_safety_warnings(results)}})


def adapter_options(args: Any) -> dict[str, Any]:
    """Keyword arguments for ``build_adapter`` from the resolved connection settings.

    ``dispatch`` resolves the header options through the same CLI, provider,
    environment cascade as the API key and leaves them on the namespace so the
    plugin and pressure runners build their adapters the same way.
    """
    return {
        "wire_format": getattr(args, "format", None),
        "default_headers": getattr(args, "_request_headers", None),
        "session_header": getattr(args, "_session_header", None),
    }
