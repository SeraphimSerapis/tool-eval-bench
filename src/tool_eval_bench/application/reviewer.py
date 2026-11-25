"""Reviewer settings: validation, inheritance from the run, and the stored config.

The CLI and the Python API both hand the service a
:class:`~tool_eval_bench.domain.review.ReviewerSettings`. This module turns it
into concrete values the runner can send, and into the ``reviewer`` block of
the persisted run config.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tool_eval_bench.adapters.wire_format import WIRE_FORMATS, resolve_wire_format
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.review import (
    CONVERSATION_KEYS,
    REVIEW_SCOPES,
    REVIEWER_RESERVED_KEYS,
    ReviewerSettings,
    credential_paths,
    inheritable_extra_params,
    without_auth_headers,
    without_sampling_overrides,
)
from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.runner.review import (
    ReviewVerdict,
    TurnReviewer,
    credential_values,
    path_credential_values,
    scrub,
    url_credential_values,
)
from tool_eval_bench.utils.system_prompt import normalize_system_prompt
from tool_eval_bench.utils.urls import endpoint_identity, redact_url, validate_http_url

#: Keys of the stored ``reviewer`` block that do not decide what the pair does:
#: the redacted URL (``endpoint_id`` identifies the endpoint) and the timeout,
#: an operational setting.
_NOT_FINGERPRINTED = frozenset({"base_url", "timeout_seconds"})

logger = logging.getLogger(__name__)

#: The loggers of the adapters a reviewer can use, plus the shared retry layer.
#: Named rather than imported, so this layer does not depend on the adapters.
_ADAPTER_LOGGERS = (
    "tool_eval_bench.adapters.openai_compat",
    "tool_eval_bench.adapters.anthropic",
    "tool_eval_bench.adapters.gemini",
    "tool_eval_bench.adapters.http_retry",
    # HTTPX logs every request's full URL at INFO, query string included.
    "httpx",
)


class _CredentialLogFilter(logging.Filter):
    """Masks known credential values in adapter log records.

    Adapters log the first bytes of an HTTP error body, and a gateway that
    echoes a rejected token would otherwise print it.
    """

    def __init__(self, secrets: Sequence[str]) -> None:
        super().__init__()
        self.secrets = list(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = scrub(message, self.secrets)
        if cleaned != message:
            record.msg, record.args = cleaned, ()
        return True


def mask_credentials_in_adapter_logs(
    reviewer: ReviewerSettings, *, api_key: str | None, headers: Mapping[str, str] | None
) -> Callable[[], None]:
    """Mask the reviewer's and the run's credentials in adapter logs; returns the undo."""
    secrets = sorted(
        {
            *credential_values(reviewer.api_key, reviewer.extra_headers),
            *credential_values(api_key, headers),
            *url_credential_values(reviewer.base_url),
        },
        key=len,
        reverse=True,
    )
    # Some gateways authenticate by a path segment, which adapter log lines
    # quote as part of the request URL.
    secrets = sorted({*secrets, *path_credential_values(reviewer.base_url)}, key=len, reverse=True)
    log_filter = _CredentialLogFilter(secrets)
    loggers = [logging.getLogger(name) for name in _ADAPTER_LOGGERS]
    for adapter_logger in loggers:
        adapter_logger.addFilter(log_filter)

    def undo() -> None:
        for adapter_logger in loggers:
            adapter_logger.removeFilter(log_filter)

    return undo


def reviewer_origin(url: str) -> str:
    """The redacted origin a reviewer is shown and stored under; see ``_stored_origin``."""
    return _stored_origin(url)


def benchmark_credentials(
    base_url: str, *, api_key: str | None, headers: Mapping[str, str] | None
) -> list[str]:
    """Every credential value the harness knows for the benchmarked endpoint."""
    return sorted(
        {
            *credential_values(api_key, headers),
            *url_credential_values(base_url),
            *path_credential_values(base_url, min_length=8),
        },
        key=len,
        reverse=True,
    )


def _stored_origin(url: str) -> str:
    """The redacted scheme, host, and port of ``url``, without path or query.

    Some gateways authenticate by a path segment, so the reviewer's path never
    reaches the stored config or the report; ``endpoint_id`` still tells
    endpoints apart.
    """
    parts = urlsplit(url)
    return redact_url(urlunsplit((parts.scheme, parts.netloc, "", "", "")))


def validate_reviewer(reviewer: ReviewerSettings) -> None:
    """Reject reviewer settings that cannot produce a meaningful run."""
    if not isinstance(reviewer, ReviewerSettings):
        raise ValueError(f"reviewer must be ReviewerSettings, got {type(reviewer).__name__}")
    if not isinstance(reviewer.model, str) or not reviewer.model.strip():
        raise ValueError("reviewer model must be a non-empty string")
    if reviewer.base_url is not None:
        validate_http_url(reviewer.base_url, what="reviewer base_url")
    if reviewer.wire_format is not None and reviewer.wire_format not in WIRE_FORMATS:
        raise ValueError(
            f"reviewer wire_format must be one of {', '.join(WIRE_FORMATS)}, "
            f"got {reviewer.wire_format!r}"
        )
    if reviewer.scope not in REVIEW_SCOPES:
        raise ValueError(
            f"reviewer scope must be one of {', '.join(REVIEW_SCOPES)}, got {reviewer.scope!r}"
        )
    rounds = reviewer.max_rounds
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError(f"reviewer max_rounds must be an integer >= 1, got {rounds!r}")
    temperature = reviewer.temperature
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError(f"reviewer temperature must be a finite number >= 0, got {temperature!r}")
    timeout = reviewer.timeout_seconds
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            f"reviewer timeout_seconds must be a finite number greater than 0, got {timeout!r}"
        )
    seed = reviewer.seed
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"reviewer seed must be an integer, got {seed!r}")
    if reviewer.system_prompt is not None:
        if not isinstance(reviewer.system_prompt, str):
            raise ValueError(
                "reviewer system_prompt must be a string, "
                f"got {type(reviewer.system_prompt).__name__}"
            )
        normalize_system_prompt(reviewer.system_prompt)
    if reviewer.extra_params is not None:
        if not isinstance(reviewer.extra_params, dict):
            raise ValueError("reviewer extra_params must be a dict")
        reserved = sorted(REVIEWER_RESERVED_KEYS & reviewer.extra_params.keys())
        if reserved:
            raise ValueError(
                "reviewer extra_params must not set "
                f"{', '.join(reserved)}: the run records the reviewer's model and prompt"
            )
        secrets = credential_paths(reviewer.extra_params)
        if secrets:
            # Extra params are stored with the run; credentials have their own fields.
            raise ValueError(
                f"reviewer extra_params must not carry credentials ({', '.join(secrets)}): "
                "use api_key or extra_headers"
            )


def resolve_reviewer(
    reviewer: ReviewerSettings,
    *,
    base_url: str,
    api_key: str | None,
    wire_format: str | None,
    extra_headers: Mapping[str, str] | None,
    session_header: str | None,
    temperature: float,
    timeout_seconds: float,
    seed: int | None,
    extra_params: dict[str, Any] | None,
) -> ReviewerSettings:
    """Fill every unset reviewer field from the benchmarked run.

    Sampling settings are always inherited; inherited ``extra_params`` lose
    their request-structure keys (see ``REQUEST_STRUCTURE_KEYS``). Connection
    settings (key, wire format, headers, session header) are inherited only
    when the reviewer has no ``base_url`` of its own, because they belong to
    the benchmarked endpoint and a credential must not travel to another server.
    """
    validate_reviewer(reviewer)
    reject_conversation_override(extra_params)
    same_endpoint = reviewer.base_url is None
    url = base_url if reviewer.base_url is None else reviewer.base_url

    def connection(own: Any, inherited: Any) -> Any:
        if own is not None:
            return own
        return inherited if same_endpoint else None

    return replace(
        reviewer,
        model=reviewer.model.strip(),
        base_url=url,
        api_key=connection(reviewer.api_key, api_key),
        wire_format=resolve_wire_format(connection(reviewer.wire_format, wire_format), url),
        extra_headers=(
            without_auth_headers(extra_headers)
            if reviewer.extra_headers is None and reviewer.api_key is not None and same_endpoint
            else connection(reviewer.extra_headers, extra_headers)
        ),
        session_header=connection(reviewer.session_header, session_header),
        temperature=temperature if reviewer.temperature is None else float(reviewer.temperature),
        timeout_seconds=(
            timeout_seconds if reviewer.timeout_seconds is None else float(reviewer.timeout_seconds)
        ),
        seed=seed if reviewer.seed is None else reviewer.seed,
        extra_params=(
            _inherited_extra_params(reviewer, extra_params)
            if reviewer.extra_params is None
            else reviewer.extra_params
        ),
        system_prompt=(
            None
            if reviewer.system_prompt is None
            else normalize_system_prompt(reviewer.system_prompt)
        ),
    )


def _inherited_extra_params(
    reviewer: ReviewerSettings, extra_params: dict[str, Any] | None
) -> dict[str, Any]:
    """The run's extra params for a reviewer, minus what the reviewer sets itself.

    A reviewer temperature or seed is sent as its own request field, which an
    inherited extra-params value would otherwise override on the wire.
    """
    explicit = [
        key
        for key, value in (("temperature", reviewer.temperature), ("seed", reviewer.seed))
        if value is not None
    ]
    return without_sampling_overrides(inheritable_extra_params(extra_params), explicit)


def reject_conversation_override(extra_params: Mapping[str, Any] | None) -> None:
    """Refuse run extra params that replace the conversation a reviewer revises.

    A ``messages`` or ``contents`` override would replace every request's
    history, the revision request included, so the reviewer's feedback would
    never reach the model while each redraw still counted as a revision.
    """
    replaced = sorted(CONVERSATION_KEYS & (extra_params or {}).keys())
    if replaced:
        raise ValueError(
            f"extra_params {', '.join(replaced)} replaces the conversation, which would drop "
            "the reviewer's revision requests; remove it to run with a reviewer"
        )


def reviewer_config(resolved: ReviewerSettings) -> dict[str, Any]:
    """The ``reviewer`` block of the stored run config. Never holds a key or header."""
    if resolved.base_url is None:
        raise ValueError("reviewer_config needs resolved settings")
    config: dict[str, Any] = {
        "model": resolved.model,
        "base_url": _stored_origin(resolved.base_url),
        "endpoint_id": endpoint_identity(
            resolved.base_url, wire_format=resolved.wire_format or "openai"
        ),
        # The endpoint identity does not hash the format, and one gateway URL
        # can serve the OpenAI and the Messages API differently.
        "wire_format": resolved.wire_format,
        "scope": resolved.scope,
        "max_rounds": resolved.max_rounds,
        "temperature": resolved.temperature,
        "timeout_seconds": resolved.timeout_seconds,
        "seed": resolved.seed,
        "extra_params": resolved.extra_params,
    }
    if resolved.system_prompt is not None:
        config["system_prompt"] = resolved.system_prompt
    return config


def reviewer_fingerprint_view(value: dict[str, Any]) -> dict[str, Any]:
    """The part of a stored ``reviewer`` block that decides what the pair does."""
    return {key: item for key, item in value.items() if key not in _NOT_FINGERPRINTED}


def reviewer_mismatch(previous: Any, current: Any) -> str | None:
    """Name the reviewer difference a user must undo to resume, or None."""
    if not previous and not current:
        return None
    if not isinstance(previous, dict) or not isinstance(current, dict):
        was = previous.get("model") if isinstance(previous, dict) else "off"
        now = current.get("model") if isinstance(current, dict) else "off"
        return f"reviewer (was {was}, now {now})"
    old, new = reviewer_fingerprint_view(previous), reviewer_fingerprint_view(current)
    changed = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
    if not changed:
        return None
    return f"reviewer {', '.join(changed)}"


def reject_held_out(scenarios: Sequence[ScenarioDefinition]) -> None:
    """Refuse to send held-out scenarios to a reviewer.

    A held-out pack stays private so that its scores stay meaningful; a
    reviewer would receive every prompt, tool result, and answer of it.
    """
    held_out = sorted(s.id for s in scenarios if s.held_out)
    if held_out:
        raise ValueError(
            "A reviewer cannot be combined with held-out scenario packs: it would receive "
            f"the private scenarios ({', '.join(held_out[:5])}"
            f"{', ...' if len(held_out) > 5 else ''})."
        )


async def check_reviewer(reviewer: TurnReviewer) -> None:
    """Send the reviewer one trivial review and raise ValueError when it cannot answer.

    Review errors never fail a scenario, so a reviewer that is misconfigured
    from the start would turn a paired run into a solo run without anyone
    noticing. The check sends the request a real review sends (prompt, sampling,
    token ceiling) about a made-up exchange. That catches a wrong URL, model, or
    key, an endpoint that returns no text, and a reviewer whose replies carry no
    verdict. A reply without a verdict gets one more attempt, since a reviewer
    that rambles once may still work; two in a row refuse the reviewer.
    """
    settings = reviewer.settings
    where = _stored_origin(settings.base_url or "")
    verdict = await _test_review(reviewer)
    if verdict.verdict != "error":
        return
    if verdict.replied:
        # A transport failure or an empty reply would fail again; a reply
        # without a verdict gets one more chance.
        logger.warning(
            "Reviewer %s at %s answered a test review without a verdict (%s); retrying once",
            settings.model,
            where,
            verdict.detail,
        )
        verdict = await _test_review(reviewer)
        if verdict.verdict != "error":
            return
    hint = (
        "; its replies carry no verdict, so every turn would run unreviewed. Check "
        "--reviewer-system-prompt or choose another reviewer model"
        if verdict.replied
        else ""
    )
    raise ValueError(
        f"Reviewer endpoint {where} failed a test review (model {settings.model!r}): "
        f"{verdict.detail or 'no answer'}{hint}"
    )


async def _test_review(reviewer: TurnReviewer) -> ReviewVerdict:
    """One review of a made-up, obviously correct exchange."""
    return await reviewer.review(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        tools=None,
        draft=ChatCompletionResult(content="2 + 2 = 4.", finish_reason="stop"),
    )
