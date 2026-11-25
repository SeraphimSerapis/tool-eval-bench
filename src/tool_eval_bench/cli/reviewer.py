"""The ``--reviewer*`` flags: activation, inheritance from the main flags, and validation.

Any ``--reviewer*`` flag enables the reviewer. Each setting cascades:

1. the ``--reviewer-*`` flag;
2. ``--reviewer-provider NAME`` (``TOOL_EVAL_<NAME>_*``), or else the
   ``TOOL_EVAL_REVIEWER_BASE_URL`` / ``_MODEL`` / ``_API_KEY`` variables;
3. the main run's value.

Sampling (temperature, top-p/k, min-p, repetition penalty, thinking, seed,
timeout, ``--backend-kwargs``) always inherits. The API key, headers, and
session header inherit only when the reviewer talks to the benchmarked server
(no reviewer URL, or one with the same origin): a credential issued for one
server never travels to another. Like ``TOOL_EVAL_BASE_URL``, the environment
variables configure a reviewer but never switch one on.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections.abc import Mapping
from typing import Any

from tool_eval_bench.cli.helpers import read_prompt_option
from tool_eval_bench.cli.provider_env import ProviderSettings, resolve_provider
from tool_eval_bench.domain.review import (
    DEFAULT_REVIEW_MAX_ROUNDS,
    DEFAULT_REVIEW_SCOPE,
    REVIEWER_RESERVED_KEYS,
    ReviewerSettings,
    credential_paths,
    inheritable_extra_params,
    without_auth_headers,
    without_sampling_overrides,
)
from tool_eval_bench.utils.headers import merge_headers, parse_header_pairs
from tool_eval_bench.utils.urls import is_same_origin, validate_http_url

REVIEWER_BASE_URL_ENV = "TOOL_EVAL_REVIEWER_BASE_URL"
REVIEWER_MODEL_ENV = "TOOL_EVAL_REVIEWER_MODEL"
REVIEWER_API_KEY_ENV = "TOOL_EVAL_REVIEWER_API_KEY"

#: Every ``--reviewer*`` destination; any of them set enables the reviewer.
REVIEWER_DESTS = (
    "reviewer",
    "reviewer_base_url",
    "reviewer_model",
    "reviewer_api_key",
    "reviewer_provider",
    "reviewer_format",
    "reviewer_header",
    "reviewer_session_header",
    "reviewer_scope",
    "reviewer_max_rounds",
    "reviewer_temperature",
    "reviewer_top_p",
    "reviewer_top_k",
    "reviewer_min_p",
    "reviewer_repeat_penalty",
    "reviewer_think",
    "reviewer_seed",
    "reviewer_timeout",
    "reviewer_backend_kwargs",
    "reviewer_system_prompt",
    "reviewer_system_prompt_file",
)

#: ``--reviewer-*`` sampling flag -> the extra-params key the main flag sets.
_SAMPLING_KEYS = (
    ("reviewer_top_p", "top_p"),
    ("reviewer_top_k", "top_k"),
    ("reviewer_min_p", "min_p"),
    ("reviewer_repeat_penalty", "repetition_penalty"),
)


def reviewer_requested(args: argparse.Namespace) -> bool:
    """Whether any ``--reviewer*`` flag was given.

    Every destination but the bare ``--reviewer`` toggle defaults to None, so
    ``--reviewer-temperature 0`` and ``--reviewer-no-think`` (False) count.
    """
    if getattr(args, "reviewer", False) is True:
        return True
    return any(getattr(args, dest, None) is not None for dest in REVIEWER_DESTS[1:])


def prepare_reviewer_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate the reviewer flags before any request; no-op without a reviewer.

    Fills the URL, model, and key from the provider or environment, checks
    ranges, reads the prompt file, and parses ``--reviewer-backend-kwargs``.
    Leaves ``args._reviewer_connection`` and ``args._reviewer_backend_kwargs``
    for :func:`build_reviewer_settings`.
    """
    if not reviewer_requested(args):
        return
    try:
        provider = resolve_provider(getattr(args, "reviewer_provider", None), os.environ)
    except ValueError as exc:
        parser.error(f"--reviewer-provider: {exc}")
    connection = _connection(args, provider, parser)
    args._reviewer_connection = connection
    if connection["base_url"] is None and connection["model"] is None:
        parser.error(
            "A reviewer needs --reviewer-model (to review with another model on the same "
            f"server) or --reviewer-base-url, or {REVIEWER_MODEL_ENV} / {REVIEWER_BASE_URL_ENV}"
        )
    if connection["base_url"] is not None:
        try:
            validate_http_url(connection["base_url"], what="--reviewer-base-url")
        except ValueError as exc:
            parser.error(str(exc))
    _validate_ranges(args, parser)
    args.reviewer_system_prompt = read_prompt_option(
        getattr(args, "reviewer_system_prompt", None),
        getattr(args, "reviewer_system_prompt_file", None),
        inline_flag="--reviewer-system-prompt",
        file_flag="--reviewer-system-prompt-file",
        parser=parser,
    )
    args._reviewer_backend_kwargs = _parse_backend_kwargs(args, parser)


def _connection(
    args: argparse.Namespace, provider: ProviderSettings | None, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    """The reviewer's own URL, model, key, and headers, before any inheritance.

    A provider's or the environment's key and headers belong to the URL they
    were configured with. When ``--reviewer-base-url`` points elsewhere, only
    explicit ``--reviewer-api-key`` / ``--reviewer-header`` flags travel there.
    ``TOOL_EVAL_REVIEWER_API_KEY`` without ``TOOL_EVAL_REVIEWER_BASE_URL`` is
    bound to the benchmarked server, the one a reviewer without a URL uses;
    :func:`connection_api_key` applies that once the server is known.
    """
    if provider is not None:
        source_url: str | None = provider.base_url
        default_key = provider.api_key
        default_headers: Mapping[str, str] = provider.headers
        default_session = provider.session_header
        default_model = provider.model
    else:
        source_url = os.getenv(REVIEWER_BASE_URL_ENV) or None
        default_key = os.getenv(REVIEWER_API_KEY_ENV) or None
        default_headers = {}
        default_session = None
        default_model = os.getenv(REVIEWER_MODEL_ENV) or None
    base_url = args.reviewer_base_url or source_url
    if source_url is not None and base_url is not None and not is_same_origin(base_url, source_url):
        default_key, default_headers, default_session = None, {}, None
    unbound_key = None
    if source_url is None and provider is None:
        # No URL to bind the environment key to: it belongs to the server.
        unbound_key, default_key = default_key, None
    model = args.reviewer_model or default_model
    api_key = args.reviewer_api_key or default_key
    env_headers = default_headers
    session_header = args.reviewer_session_header or default_session
    try:
        headers = merge_headers(env_headers, parse_header_pairs(args.reviewer_header))
    except ValueError as exc:
        parser.error(f"--reviewer-header: {exc}")
    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "unbound_api_key": unbound_key,
        "headers": headers or None,
        "session_header": session_header,
    }


def connection_api_key(
    connection: Mapping[str, Any], main_base_url: str, main_api_key: str | None
) -> str | None:
    """The key the reviewer authenticates with, never one bound to another origin.

    An explicit or origin-bound key wins. The environment key without its own
    URL, and the run's key, only go to the benchmarked server.
    """
    if connection.get("api_key"):
        return str(connection["api_key"])
    if not same_server(connection.get("base_url"), main_base_url):
        return None
    unbound = connection.get("unbound_api_key")
    return str(unbound) if unbound else main_api_key


def _validate_ranges(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    rounds = getattr(args, "reviewer_max_rounds", None)
    if rounds is not None and rounds < 1:
        parser.error(f"--reviewer-max-rounds must be at least 1, got {rounds}")
    temperature = getattr(args, "reviewer_temperature", None)
    if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
        parser.error(f"--reviewer-temperature must be a finite number >= 0, got {temperature}")
    top_p = getattr(args, "reviewer_top_p", None)
    if top_p is not None and not (0.0 < top_p <= 1.0):
        parser.error(f"--reviewer-top-p must be in (0, 1], got {top_p}")
    min_p = getattr(args, "reviewer_min_p", None)
    if min_p is not None and not (0.0 <= min_p <= 1.0):
        parser.error(f"--reviewer-min-p must be between 0 and 1, got {min_p}")
    for flag, attr in (
        ("--reviewer-timeout", "reviewer_timeout"),
        ("--reviewer-repeat-penalty", "reviewer_repeat_penalty"),
    ):
        value = getattr(args, attr, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{flag} must be a finite number greater than 0, got {value}")


def _parse_backend_kwargs(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, Any] | None:
    raw = getattr(args, "reviewer_backend_kwargs", None)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"--reviewer-backend-kwargs is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        parser.error(
            f"--reviewer-backend-kwargs must be a JSON object (dict), got {type(parsed).__name__}"
        )
    reserved = sorted(REVIEWER_RESERVED_KEYS & parsed.keys())
    if reserved:
        parser.error(
            f"--reviewer-backend-kwargs must not set {', '.join(reserved)}: "
            "use --reviewer-model and --reviewer-system-prompt"
        )
    secrets = credential_paths(parsed)
    if secrets:
        parser.error(
            f"--reviewer-backend-kwargs must not carry credentials ({', '.join(secrets)}): "
            "use --reviewer-api-key or --reviewer-header"
        )
    return parsed


def reviewer_extra_params(
    args: argparse.Namespace, main_extra_params: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The reviewer's extra params: the main ones with the reviewer's flags applied.

    The main ones lose their request-structure keys first (``model``,
    ``messages``, ...; see ``REQUEST_STRUCTURE_KEYS``).

    Order matches the main flags: individual sampling flags first, then
    ``--reviewer-backend-kwargs`` deep-merged on top, where a ``null`` value
    removes the inherited key. ``--reviewer-think`` drops an inherited
    ``enable_thinking=false``; ``--reviewer-no-think`` sets it.
    """
    # An explicit reviewer temperature or seed travels as its own request
    # field, which an inherited --backend-kwargs value would otherwise override.
    extra: dict[str, Any] = copy.deepcopy(
        without_sampling_overrides(
            inheritable_extra_params(main_extra_params),
            [
                key
                for dest, key in (
                    ("reviewer_temperature", "temperature"),
                    ("reviewer_seed", "seed"),
                    ("reviewer_top_p", "top_p"),
                    ("reviewer_top_k", "top_k"),
                )
                if getattr(args, dest, None) is not None
            ],
        )
    )
    for dest, key in _SAMPLING_KEYS:
        value = getattr(args, dest, None)
        if value is not None:
            extra[key] = value
    think = getattr(args, "reviewer_think", None)
    if think is False:
        kwargs = extra.get("chat_template_kwargs")
        extra["chat_template_kwargs"] = {
            **(kwargs if isinstance(kwargs, dict) else {}),
            "enable_thinking": False,
        }
    elif think is True:
        kwargs = extra.get("chat_template_kwargs")
        if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False:
            remaining = {k: v for k, v in kwargs.items() if k != "enable_thinking"}
            if remaining:
                extra["chat_template_kwargs"] = remaining
            else:
                del extra["chat_template_kwargs"]
    overrides = getattr(args, "_reviewer_backend_kwargs", None) or {}
    for key, value in overrides.items():
        if value is None:
            extra.pop(key, None)
        elif isinstance(value, dict) and isinstance(extra.get(key), dict):
            merged = {**extra[key], **value}
            extra[key] = {k: v for k, v in merged.items() if v is not None}
        else:
            extra[key] = value
    return extra


def same_server(reviewer_url: str | None, main_url: str) -> bool:
    """Whether the reviewer talks to the benchmarked server, so its key may travel."""
    return reviewer_url is None or is_same_origin(reviewer_url, main_url)


def build_reviewer_settings(
    args: argparse.Namespace,
    *,
    model: str,
    main_base_url: str,
    main_api_key: str | None,
    main_headers: Mapping[str, str] | None,
    main_session_header: str | None,
    main_extra_params: Mapping[str, Any] | None,
) -> ReviewerSettings:
    """The reviewer the service receives. Temperature, timeout, and seed stay
    None when not given, so the service inherits the run's final values (the
    CLI scales ``--timeout`` for context pressure after this point)."""
    connection: dict[str, Any] = getattr(args, "_reviewer_connection", None) or {}
    base_url = connection.get("base_url")
    shared = same_server(base_url, main_base_url)
    api_key = connection_api_key(connection, main_base_url, main_api_key)
    headers = connection.get("headers")
    if headers is None and shared and main_headers:
        # A reviewer with a key of its own keeps the run's other headers but
        # not its Authorization, which the adapter would apply over the key.
        headers = (
            dict(main_headers) if api_key == main_api_key else without_auth_headers(main_headers)
        )
    session_header = connection.get("session_header") or (main_session_header if shared else None)
    wire_format = getattr(args, "reviewer_format", None)
    return ReviewerSettings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        wire_format=None if wire_format in (None, "auto") else wire_format,
        extra_headers=headers,
        session_header=session_header,
        temperature=getattr(args, "reviewer_temperature", None),
        timeout_seconds=getattr(args, "reviewer_timeout", None),
        seed=getattr(args, "reviewer_seed", None),
        extra_params=reviewer_extra_params(args, main_extra_params),
        scope=getattr(args, "reviewer_scope", None) or DEFAULT_REVIEW_SCOPE,
        max_rounds=getattr(args, "reviewer_max_rounds", None) or DEFAULT_REVIEW_MAX_ROUNDS,
        system_prompt=getattr(args, "reviewer_system_prompt", None),
    )
