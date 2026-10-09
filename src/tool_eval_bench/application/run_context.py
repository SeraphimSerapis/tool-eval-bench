"""Backend detection and run context, shared by the CLI and the Python API.

Both entry points call these helpers, so a server gets the same backend label
and engine metadata whichever one ran it. Console output and flag handling stay
with the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tool_eval_bench.adapters.wire_format import resolve_wire_format
from tool_eval_bench.domain.models import RunContext, thinking_enabled_for
from tool_eval_bench.utils import metadata

logger = logging.getLogger(__name__)

# Hosted APIs name themselves. Engine probes against a vendor endpoint are
# wasted requests, and would let a compatibility shape mislabel the run.
_HOSTED_WIRE_FORMATS = frozenset({"gemini", "anthropic"})


@dataclass(frozen=True)
class BackendDetection:
    """The backend label to record for a run.

    ``server_name`` is the human-readable name a probe identified, such as
    ``llama.cpp``. It is None when the label did not come from a probe.
    """

    backend: str
    server_name: str | None = None


async def identify_backend(
    backend: str | None,
    *,
    base_url: str,
    api_key: str | None,
    wire_format: str,
    probe: bool,
    fallback: str = "unknown",
) -> BackendDetection:
    """Decide the backend label from what the caller knows and what the server says.

    A non-empty ``backend`` is kept as given. Otherwise a hosted Gemini or
    Anthropic ``wire_format`` (already resolved) labels the run, then the
    server's own identification when ``probe`` allows the requests, then
    ``fallback``. A probe that raises counts as no identification.
    """
    if backend:
        return BackendDetection(backend)
    if wire_format in _HOSTED_WIRE_FORMATS:
        return BackendDetection(wire_format)
    if probe:
        try:
            hint = await metadata.probe_backend_hint(base_url, api_key)
        except Exception as exc:
            logger.warning("Backend detection failed: %s", exc)
            hint = None
        if hint:
            return BackendDetection(backend=hint[0], server_name=hint[1])
    return BackendDetection(fallback)


async def detect_backend(
    backend: str | None,
    *,
    base_url: str,
    api_key: str | None,
    wire_format: str | None,
    probe: bool,
) -> str:
    """Return the backend label the Python API records for a run.

    An empty label or any spelling of ``unknown`` counts as unset, because
    ``unknown`` is the API's default. The CLI keeps an explicit
    ``--backend unknown`` as given instead.

    Raises:
        ValueError: if ``wire_format`` names no known format.
    """
    resolved_format = resolve_wire_format(wire_format, base_url)
    preset = backend if backend and backend.lower() != "unknown" else None
    detection = await identify_backend(
        preset, base_url=base_url, api_key=api_key, wire_format=resolved_format, probe=probe
    )
    if detection.server_name:
        logger.info("Detected backend: %s", detection.server_name)
    return detection.backend


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
    trials: int = 1,
    context_pressure: float | None = None,
    label: str | None = None,
) -> RunContext | None:
    """Collect the run context, or return None so the run proceeds without it.

    ``thinking_enabled`` is read from ``extra_params``, the payload actually
    sent, so disabling thinking through ``--backend-kwargs`` or ``extra_params``
    is recorded the same way as ``--no-think``.
    """
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
            trials=trials,
            parallel=parallel,
            error_rate=error_rate,
            thinking_enabled=thinking_enabled_for(extra_params),
            extra_params=extra_params or None,
            context_pressure=context_pressure,
            system_prompt=system_prompt,
            label=label,
            probe_engine=probe_engine,
        )
    except Exception as exc:
        logger.warning("Failed to build RunContext: %s", exc)
        return None
