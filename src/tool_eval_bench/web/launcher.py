"""Endpoint probing, scenario preset resolution, and background run management for the web UI."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import os
from collections.abc import Mapping
from typing import Any

import httpx

from tool_eval_bench.adapters.wire_format import resolve_wire_format
from tool_eval_bench.application.resume import plan_stored_resume
from tool_eval_bench.application.run_config import RunSettings, build_run_config
from tool_eval_bench.application.run_context import build_run_context, identify_backend
from tool_eval_bench.application.run_queries import get_run, resume_state
from tool_eval_bench.application.service import open_benchmark_service, validate_run_parameters
from tool_eval_bench.cli.commands import resolve_packs, resolve_scenarios
from tool_eval_bench.cli.model_probe import _models_request
from tool_eval_bench.cli.server import _discover_async
from tool_eval_bench.domain.engines import BACKEND_LABELS
from tool_eval_bench.domain.models import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    RUN_STATUS_INTERRUPTED,
    RUN_STATUS_RUNNING,
)
from tool_eval_bench.domain.redaction import redact_url, redact_urls
from tool_eval_bench.domain.scenarios import (
    CATEGORY_LABELS,
    Category,
    ScenarioDefinition,
)
from tool_eval_bench.evals.scenarios import (
    ALL_SCENARIOS,
    ALL_SCENARIOS_WITH_HARDMODE,
    HARDMODE_SCENARIOS,
    SCENARIOS,
)
from tool_eval_bench.schema import ARGS_SCHEMA, get_schema
from tool_eval_bench.utils.ids import build_run_id
from tool_eval_bench.utils.system_prompt import normalize_system_prompt
from tool_eval_bench.web.live import LiveRunHub, LiveRunSession
from tool_eval_bench.web.sanitize import sanitize_result

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TURNS = 8
_DEFAULT_ALPHA = 0.7


class RunConflict(Exception):
    """The request conflicts with the run's current state (HTTP 409)."""


def build_launcher_schema() -> dict[str, Any]:
    """Return presets, categories, scenarios, backends, and CLI argument schema."""
    cat_counts: dict[str, int] = {}
    for sc in ALL_SCENARIOS_WITH_HARDMODE:
        cat_counts[sc.category.value] = cat_counts.get(sc.category.value, 0) + 1

    return {
        "schema": get_schema(),
        "args": ARGS_SCHEMA,
        "backends": list(BACKEND_LABELS),
        "wire_formats": ["auto", "openai", "gemini", "anthropic"],
        "presets": [
            {
                "id": "short",
                "label": f"Core 15 (--short · {len(SCENARIOS)} scenarios)",
                "count": len(SCENARIOS),
            },
            {
                "id": "standard",
                "label": f"Standard Suite ({len(ALL_SCENARIOS)} scenarios)",
                "count": len(ALL_SCENARIOS),
            },
            {
                "id": "hardmode",
                "label": f"Standard + Hard Mode ({len(ALL_SCENARIOS_WITH_HARDMODE)} scenarios)",
                "count": len(ALL_SCENARIOS_WITH_HARDMODE),
            },
            {
                "id": "hardmode_only",
                "label": f"Hard Mode Only (Category P · {len(HARDMODE_SCENARIOS)} scenarios)",
                "count": len(HARDMODE_SCENARIOS),
            },
            {
                "id": "custom",
                "label": "Custom Categories or Scenario IDs",
                "count": None,
            },
        ],
        "categories": [
            {
                "id": cat.value,
                "label": CATEGORY_LABELS.get(cat, cat.value),
                "count": cat_counts.get(cat.value, 0),
                "hardmode": cat == Category.P,
            }
            for cat in Category
        ],
        "scenarios": [
            {
                "id": sc.id,
                "title": sc.title,
                "category": sc.category.value,
                "difficulty": sc.difficulty,
                "hardmode": sc.category == Category.P,
            }
            for sc in ALL_SCENARIOS_WITH_HARDMODE
        ],
        "defaults": {
            "base_url": os.getenv("TOOL_EVAL_BASE_URL") or "",
            "model": os.getenv("TOOL_EVAL_MODEL") or "",
            "backend": os.getenv("TOOL_EVAL_BACKEND") or "unknown",
            "temperature": 0.0,
            "timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
            "max_turns": _DEFAULT_MAX_TURNS,
            "concurrency": 1,
            "error_rate": 0.0,
            "alpha": _DEFAULT_ALPHA,
        },
    }


def _number(
    payload: Mapping[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """Read a finite number from the request, raising ValueError with the field name."""
    raw = payload.get(key)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"{key} must be a number")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and (value <= minimum if exclusive_minimum else value < minimum):
        relation = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"{key} must be {relation} {minimum:g}, got {value:g}")
    return value


def _integer(
    payload: Mapping[str, Any], key: str, default: int, *, minimum: int | None = None
) -> int:
    """Read a whole number from the request, raising ValueError with the field name."""
    value = _number(payload, key, default, minimum=minimum)
    if not value.is_integer():
        raise ValueError(f"{key} must be a whole number, got {value:g}")
    return int(value)


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    raw = payload.get(key)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _api_key(payload: Mapping[str, Any]) -> str | None:
    """The caller's key, else ``TOOL_EVAL_API_KEY``, matching the CLI's precedence."""
    return _optional_text(payload, "api_key") or os.getenv("TOOL_EVAL_API_KEY") or None


async def _fetch_models_json(
    resolved_url: str,
    models_endpoint: str,
    headers: Mapping[str, str],
    resolved_format: str,
) -> Any:
    """Query the endpoint's model listing and return the parsed JSON body."""
    async with httpx.AsyncClient(timeout=6.0) as client:
        resp = await client.get(models_endpoint, headers=headers)
        if resp.status_code == 404 and resolved_format == "openai":
            fallback = f"{resolved_url.rstrip('/')}/models"
            resp = await client.get(fallback, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def probe_endpoint_models(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    wire_format: str = "auto",
    redact: bool = False,
) -> dict[str, Any]:
    """Discover a local server or probe an explicit base_url for available models.

    With ``redact`` set the response never echoes the endpoint unmasked, so the
    browser cannot fill in a discovered URL; the user types it instead.
    """
    resolved_url = (base_url or "").strip()
    if not resolved_url:
        env_url = (os.getenv("TOOL_EVAL_BASE_URL") or "").strip()
        if env_url:
            resolved_url = env_url
        else:
            discovered = await _discover_async()
            if discovered is None:
                return {
                    "reachable": False,
                    "error": "No inference server found on localhost. Enter a Base URL.",
                    "models": [],
                }
            resolved_url = discovered[0]

    shown_url = redact_url(resolved_url) if redact else resolved_url
    effective_key = api_key or os.getenv("TOOL_EVAL_API_KEY") or None
    try:
        resolved_format = resolve_wire_format(wire_format, resolved_url)
    except ValueError as exc:
        return {"reachable": False, "error": redact_urls(str(exc)), "models": []}

    models_endpoint, headers = _models_request(resolved_url, effective_key, resolved_format)
    try:
        body = await _fetch_models_json(resolved_url, models_endpoint, headers, resolved_format)
    except Exception as exc:  # noqa: BLE001
        return {
            "reachable": False,
            "base_url": shown_url,
            # httpx quotes the full request URL, query credentials included.
            "error": redact_urls(str(exc)),
            "models": [],
        }

    listed = (body.get("data") or body.get("models") or []) if isinstance(body, dict) else []
    models: list[dict[str, str]] = []
    if isinstance(listed, list):
        for item in listed:
            if not isinstance(item, Mapping):
                continue
            api_id = str(item.get("id") or str(item.get("name", "")).removeprefix("models/"))
            if not api_id:
                continue
            root = str(
                item.get("root") or item.get("displayName") or item.get("display_name") or ""
            )
            display = root if root and root != api_id else api_id
            models.append({"id": api_id, "display_name": display})

    detection = await identify_backend(
        None,
        base_url=resolved_url,
        api_key=effective_key,
        wire_format=resolved_format,
        probe=True,
        fallback="unknown",
    )
    result: dict[str, Any] = {
        "reachable": True,
        "base_url": shown_url,
        "backend": detection.backend,
        "server_name": detection.server_name,
        "wire_format": resolved_format,
        "models": models,
    }
    if not redact:
        result["resolved_base_url"] = resolved_url
    return result


def _resolve_launcher_scenarios(
    payload: Mapping[str, Any],
) -> tuple[list[ScenarioDefinition], list[dict[str, Any]] | None, frozenset[str]]:
    """Resolve scenarios and optional pack attestations from a launch request payload."""
    preset = str(payload.get("preset") or "short").lower()
    raw_categories = payload.get("categories")
    categories = (
        [str(c).upper() for c in raw_categories if str(c).strip()]
        if isinstance(raw_categories, list)
        else None
    )
    raw_scenarios = payload.get("scenarios")
    if isinstance(raw_scenarios, str):
        scenarios_list = [
            s.strip().upper() for s in raw_scenarios.replace(",", " ").split() if s.strip()
        ]
    elif isinstance(raw_scenarios, list):
        scenarios_list = [str(s).strip().upper() for s in raw_scenarios if str(s).strip()]
    else:
        scenarios_list = None

    raw_packs = payload.get("scenario_pack")
    pack_dirs = (
        [str(p).strip() for p in raw_packs if str(p).strip()]
        if isinstance(raw_packs, list)
        else (
            [str(raw_packs).strip()] if isinstance(raw_packs, str) and raw_packs.strip() else None
        )
    )

    ns = argparse.Namespace(
        short=preset == "short",
        hardmode=preset == "hardmode"
        or bool(payload.get("hardmode", False))
        or (categories is not None and "P" in categories),
        hardmode_only=preset == "hardmode_only",
        scenarios=scenarios_list or None,
        categories=categories or None,
        scenario_pack=pack_dirs,
        pack_only=bool(payload.get("pack_only", False)),
    )
    packs = resolve_packs(ns)
    resolved = resolve_scenarios(ns)
    if not resolved:
        raise ValueError("No scenarios matched the selected filters")
    attestations = [pack.to_dict() for pack in packs] or None
    held_out = frozenset(s.id for pack in packs for s in pack.scenarios)
    return resolved, attestations, held_out


def _start_task(
    hub: LiveRunHub,
    session: LiveRunSession,
    output_dir: str | None,
    run_kwargs: dict[str, Any],
) -> None:
    """Run the benchmark in the background and report its end to the hub."""
    run_id = session.run_id
    callbacks = hub.make_callbacks(session)

    async def _execute() -> None:
        try:
            with open_benchmark_service(output_dir=output_dir) as service:
                run_data = await service.run_benchmark(**run_kwargs, **callbacks)
        except asyncio.CancelledError:
            hub.mark_interrupted(run_id, "stopped by user")
            raise
        except Exception as exc:  # noqa: BLE001
            reason = redact_urls(str(exc)) or type(exc).__name__
            logger.warning("Web benchmark run %s failed: %s", run_id, reason)
            hub.mark_interrupted(run_id, reason)
            return
        hub.mark_complete(run_id, run_data)

    session.task = asyncio.create_task(_execute(), name=f"web-run-{run_id}")


async def launch_web_run(
    hub: LiveRunHub,
    payload: Mapping[str, Any],
    *,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Validate a run request, claim a run ID, and start the benchmark in the background."""
    base_url = str(payload.get("base_url") or os.getenv("TOOL_EVAL_BASE_URL") or "").strip()
    model = str(payload.get("model") or os.getenv("TOOL_EVAL_MODEL") or "").strip()
    if not base_url:
        raise ValueError("base_url is required")
    if not model:
        raise ValueError("model is required")

    api_key = _api_key(payload)
    wire_format_raw = str(payload.get("wire_format") or "auto").strip().lower()
    wire_format = resolve_wire_format(wire_format_raw, base_url)

    backend_raw = str(payload.get("backend") or "unknown").strip().lower()
    temperature = _number(payload, "temperature", 0.0, minimum=0.0)
    timeout_seconds = _number(
        payload,
        "timeout_seconds",
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        minimum=0.0,
        exclusive_minimum=True,
    )
    max_turns = _integer(payload, "max_turns", _DEFAULT_MAX_TURNS, minimum=1)
    concurrency = _integer(payload, "concurrency", 1, minimum=1)
    error_rate = _number(payload, "error_rate", 0.0)
    alpha = _number(payload, "alpha", _DEFAULT_ALPHA)
    seed_raw = payload.get("seed")
    seed = None if seed_raw in (None, "") else _integer(payload, "seed", 0)
    weight_by_difficulty = bool(payload.get("weight_by_difficulty", False))
    label = _optional_text(payload, "label")
    system_prompt_raw = _optional_text(payload, "system_prompt")
    system_prompt = normalize_system_prompt(system_prompt_raw) if system_prompt_raw else None

    validate_run_parameters(max_turns=max_turns, error_rate=error_rate, alpha=alpha)
    scenarios, scenario_packs, held_out = _resolve_launcher_scenarios(payload)

    detection = await identify_backend(
        backend_raw if backend_raw and backend_raw != "unknown" else None,
        base_url=base_url,
        api_key=api_key,
        wire_format=wire_format,
        probe=False,
        fallback=backend_raw or "unknown",
    )
    backend = detection.backend

    run_context = await build_run_context(
        model=model,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        seed=seed,
        scenario_selector=f"web ({len(scenarios)})",
        trials=1,
        parallel=concurrency,
        error_rate=error_rate,
        extra_params=None,
        system_prompt=system_prompt,
        label=label,
        probe_engine=False,
    )
    metadata = run_context.to_dict() if run_context is not None else {}

    settings = RunSettings(
        model=model,
        backend=backend,
        base_url=base_url,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        seed=seed,
        reference_date=None,
        concurrency=concurrency,
        error_rate=error_rate,
        alpha=alpha,
        extra_params=None,
        context_pressure_config=None,
        weight_by_difficulty=weight_by_difficulty,
        system_prompt=system_prompt,
        decision_judge=None,
        wire_format=wire_format,
    )
    run_config = build_run_config(
        settings,
        scenarios=scenarios,
        metadata=metadata,
        scenario_packs=scenario_packs,
    )
    # Claimed here so the browser can subscribe before the first scenario
    # starts; the service reuses it through resume_run_id.
    run_id = build_run_id(run_config)

    session = hub.register(
        run_id,
        model=model,
        backend=backend,
        total=len(scenarios),
        held_out=held_out,
    )
    _start_task(
        hub,
        session,
        output_dir,
        {
            "model": model,
            "backend": backend,
            "base_url": base_url,
            "api_key": api_key,
            "scenarios": scenarios,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
            "max_turns": max_turns,
            "seed": seed,
            "concurrency": concurrency,
            "error_rate": error_rate,
            "alpha": alpha,
            "weight_by_difficulty": weight_by_difficulty,
            "run_context": run_context,
            "resume_run_id": run_id,
            "scenario_packs": scenario_packs,
            "wire_format": wire_format,
            "system_prompt": system_prompt,
        },
    )
    return {
        "run_id": run_id,
        "status": "running",
        "model": model,
        "backend": backend,
        "total_scenarios": len(scenarios),
    }


async def stop_web_run(hub: LiveRunHub, run_id: str) -> dict[str, Any]:
    """Cancel a benchmark this server launched.

    A run started by another process cannot be stopped from here: rewriting its
    row would not stop the process, which would later overwrite the change.
    """
    session = hub.get(run_id)
    if session is not None and session.is_live and session.task is not None:
        session.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session.task
        return {"run_id": run_id, "status": RUN_STATUS_INTERRUPTED}

    run = await asyncio.to_thread(get_run, run_id, include_traces=False)
    if run is None:
        raise KeyError(run_id)
    status = run.get("status") or "completed"
    if status == RUN_STATUS_RUNNING:
        raise RunConflict(
            "This run was started by another process, so it can only be stopped there "
            "(Ctrl+C in its terminal)."
        )
    raise RunConflict(f"Run '{run_id}' is {status}, not running")


async def resume_web_run(
    hub: LiveRunHub,
    run_id: str,
    payload: Mapping[str, Any] | None = None,
    *,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Resume an interrupted tool-call run from its saved SQLite checkpoints.

    Only runs marked ``interrupted`` qualify. A ``running`` row may belong to a
    live process, and a second writer would corrupt it; the CLI can resume one
    whose process is gone. The conditions come from the stored config and must
    pass the same compatibility check the CLI applies.
    """
    loaded = await asyncio.to_thread(resume_state, run_id)
    if loaded is None:
        raise KeyError(run_id)
    prev_run, checkpoints = loaded

    session = hub.get(run_id)
    if session is not None and session.is_live:
        raise RunConflict("This run is already running in this server")
    status = prev_run.get("status") or "completed"
    if status == RUN_STATUS_RUNNING:
        raise RunConflict(
            "This run is still marked running and may belong to a live process. If that "
            f"process is gone, resume it from the CLI: tool-eval-bench resume {run_id}"
        )
    if status != RUN_STATUS_INTERRUPTED:
        raise RunConflict(f"Only interrupted runs can be resumed; this run is {status}")
    if prev_run.get("run_type", "tool_eval") != "tool_eval":
        raise ValueError("Only tool_eval runs can be resumed")

    payload = payload or {}
    base_url = str(payload.get("base_url") or os.getenv("TOOL_EVAL_BASE_URL") or "").strip()
    if not base_url:
        raise ValueError(
            "base_url is required: the stored URL is redacted, so enter the endpoint "
            "in the launcher or set TOOL_EVAL_BASE_URL"
        )
    wire_format = _optional_text(payload, "wire_format") or "auto"
    plan = plan_stored_resume(prev_run, checkpoints, base_url=base_url, wire_format=wire_format)
    settings = plan.settings

    session = hub.register(
        run_id,
        model=settings.model,
        backend=settings.backend,
        total=len(plan.scenarios),
        preserved=[sanitize_result(result, frozenset()) for result in plan.reusable],
    )
    _start_task(
        hub,
        session,
        output_dir,
        {
            "model": settings.model,
            "backend": settings.backend,
            "base_url": base_url,
            "api_key": _api_key(payload),
            "scenarios": plan.remaining,
            "temperature": settings.temperature,
            "timeout_seconds": settings.timeout_seconds,
            "max_turns": settings.max_turns,
            "seed": settings.seed,
            "reference_date": settings.reference_date,
            "concurrency": settings.concurrency,
            "error_rate": settings.error_rate,
            "alpha": settings.alpha,
            "extra_params": settings.extra_params,
            "weight_by_difficulty": settings.weight_by_difficulty,
            "resume_run_id": run_id,
            "resume_prior_results": plan.reusable,
            "resume_scenarios": plan.scenarios,
            "wire_format": settings.wire_format,
            "system_prompt": settings.system_prompt,
        },
    )
    return {
        "run_id": run_id,
        "status": "running",
        "model": settings.model,
        "backend": settings.backend,
        "preserved_scenarios": len(plan.reusable),
        "remaining_scenarios": len(plan.remaining),
        "total_scenarios": len(plan.scenarios),
    }
