"""Backend detection and run context for the Python API.

The CLI does the same work in ``cli.dispatch``, interleaved with its console
output and flags. Both call the same probes in ``utils.metadata``, so a server
gets the same backend label and engine metadata whichever entry point ran it.
"""

from __future__ import annotations

import logging
from typing import Any

from tool_eval_bench.adapters.wire_format import resolve_wire_format
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.utils import metadata

logger = logging.getLogger(__name__)

# Hosted APIs name themselves. Engine probes against a vendor endpoint are
# wasted requests, and would let a compatibility shape mislabel the run.
_HOSTED_WIRE_FORMATS = frozenset({"gemini", "anthropic"})


async def detect_backend(
    backend: str | None,
    *,
    base_url: str,
    api_key: str | None,
    wire_format: str | None,
    probe: bool,
) -> str:
    """Return the backend label to record for a run.

    An explicit label is kept as given. An empty label or any spelling of
    ``unknown`` becomes the hosted wire format when there is one, otherwise
    whatever the server identifies itself as, provided ``probe`` allows the
    requests. A server that does not identify itself is recorded as ``unknown``.

    Raises:
        ValueError: if ``wire_format`` names no known format.
    """
    resolved_format = resolve_wire_format(wire_format, base_url)
    if backend and backend.lower() != "unknown":
        return backend
    if resolved_format in _HOSTED_WIRE_FORMATS:
        return resolved_format
    if not probe:
        return "unknown"
    try:
        hint = await metadata.probe_backend_hint(base_url, api_key)
    except Exception as exc:
        logger.warning("Backend detection failed: %s", exc)
        return "unknown"
    if not hint:
        return "unknown"
    logger.info("Detected backend: %s", hint[1])
    return hint[0]


def _thinking_enabled(extra_params: dict[str, Any] | None) -> bool:
    """Read the payload ``--no-think`` produces, so this reflects what is sent.

    The CLI records its flag instead, so the two differ when thinking is
    disabled only through ``extra_params`` or ``--backend-kwargs``.
    """
    kwargs = (extra_params or {}).get("chat_template_kwargs")
    return not (isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False)


async def build_run_context(
    *,
    model: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    scenario_selector: str,
    temperature: float,
    max_turns: int,
    timeout_seconds: float,
    seed: int | None,
    parallel: int,
    error_rate: float,
    extra_params: dict[str, Any] | None,
    system_prompt: str | None,
    probe_engine: bool,
) -> RunContext | None:
    """Collect the run context, or return None so the run proceeds without it."""
    try:
        return await metadata.collect_run_context(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            seed=seed,
            scenario_selector=scenario_selector,
            parallel=parallel,
            error_rate=error_rate,
            thinking_enabled=_thinking_enabled(extra_params),
            extra_params=extra_params or None,
            system_prompt=system_prompt,
            probe_engine=probe_engine,
        )
    except Exception as exc:
        logger.warning("Failed to build RunContext: %s", exc)
        return None
