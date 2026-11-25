"""The reviewer: a second model that checks the benchmarked model's turns.

A reviewer reads each proposed turn before the harness acts on it. It either
approves the turn or returns feedback, and on feedback the benchmarked model
revises the turn. Only the turn that is finally accepted reaches the scenario
state, so a rejected tool call never runs and a rejected reply is never graded.

The pair is the system under test: a reviewed run is stored, fingerprinted, and
ranked apart from a solo run of the same model.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from tool_eval_bench.domain.redaction import is_sensitive_key

#: Which turns the reviewer checks.
#:
#: ``turns``: every turn that calls a tool or answers.
#: ``tools``: only turns that call a tool.
#: ``final``: only replies without a tool call, the turns that end a phase.
REVIEW_SCOPES: tuple[str, ...] = ("turns", "tools", "final")
DEFAULT_REVIEW_SCOPE = "turns"
#: Reviews per turn. Each review that asks for a change triggers one revision,
#: and the turn left after the last allowed revision runs without another review.
DEFAULT_REVIEW_MAX_ROUNDS = 1

#: Request keys that shape the request itself rather than how the model samples.
#: ``--backend-kwargs`` may set them for the benchmarked model, but a reviewer
#: never inherits them: an inherited ``model`` would make the report name one
#: reviewer while the endpoint runs another, and an inherited ``messages``
#: would replace the review prompt.
REQUEST_STRUCTURE_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "stream",
        "stream_options",
        "system",
        "contents",
        "systemInstruction",
        # Native and legacy tool settings: the reviewer is sent no tools.
        "toolConfig",
        "tool_config",
        "functions",
        "function_call",
        # A Gemini context cache belongs to one model's conversation.
        "cachedContent",
        "cached_content",
    }
)

#: Credential-like keys, compared case-insensitively. A key the run sends in
#: its extra params belongs to the benchmarked endpoint, and a reviewer on
#: another server must not receive it.
CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "x-api-key",
        "key",
        "token",
        "access_token",
        "auth",
        "authorization",
        "password",
        "secret",
        "cookie",
        "set_cookie",
    }
)

#: Keys of the run's own extra params that replace the whole conversation. A
#: reviewed run sends the revision request as an extra message, which such an
#: override would silently drop, so the two cannot be combined.
CONVERSATION_KEYS = frozenset({"messages", "contents"})

#: Keys the reviewer's own extra params may not set either: they would send
#: another model than the one the run stores, or replace the review prompt.
REVIEWER_RESERVED_KEYS = frozenset({"model", "messages", "system", "contents", "systemInstruction"})

#: Structure nested inside an otherwise inheritable object: the Messages API's
#: output schema and Gemini's response schema would force the benchmarked
#: model's answer format on the reviewer's verdict.
NESTED_REQUEST_STRUCTURE_KEYS: Mapping[str, frozenset[str]] = {
    "output_config": frozenset({"format"}),
    "generationConfig": frozenset({"responseSchema", "responseJsonSchema", "responseMimeType"}),
}


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def is_credential_key(key: object) -> bool:
    """Whether ``key`` names a credential: the shared predicate or an exact match."""
    if not isinstance(key, str):
        return False
    # "X-Goog-Api-Key", "accessToken", and "access_token" name the same kind
    # of credential: split camel case, then unify the separators.
    normalized = _CAMEL_BOUNDARY.sub("_", key).lower().replace("-", "_")
    return (
        is_sensitive_key(normalized)
        or normalized in CREDENTIAL_KEYS
        or key.lower() in CREDENTIAL_KEYS
    )


def without_auth_headers(headers: Mapping[str, str] | None) -> dict[str, str] | None:
    """``headers`` without the ones that authenticate, or None when none remain.

    A reviewer with a key of its own must not inherit the run's
    ``Authorization`` header: adapters apply extra headers after the key, so
    the inherited one would win.
    """
    kept = {name: value for name, value in (headers or {}).items() if not is_credential_key(name)}
    return kept or None


def _without_credentials(value: Any) -> Any:
    """``value`` with credential-like keys removed at every depth."""
    if isinstance(value, Mapping):
        return {
            key: _without_credentials(item)
            for key, item in value.items()
            if not is_credential_key(key)
        }
    if isinstance(value, list | tuple):
        return [_without_credentials(item) for item in value]
    return value


def credential_paths(value: Any, prefix: str = "") -> list[str]:
    """The dotted paths of every credential-like key in ``value``."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}{key}"
            if is_credential_key(key):
                found.append(path)
            else:
                found.extend(credential_paths(item, f"{path}."))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.extend(credential_paths(item, f"{prefix}{index}."))
    return found


#: Where a request carries temperature and seed besides its own fields:
#: top level for the OpenAI wire, ``generationConfig`` for Gemini.
_SAMPLING_CONTAINERS = ("generationConfig",)


#: Every spelling a request may carry a sampling setting under, by its OpenAI
#: name: the top-level key and the Gemini ``generationConfig`` field.
_SAMPLING_SPELLINGS: Mapping[str, frozenset[str]] = {
    "temperature": frozenset({"temperature"}),
    "seed": frozenset({"seed"}),
    "top_p": frozenset({"top_p", "topP"}),
    "top_k": frozenset({"top_k", "topK"}),
}


def without_sampling_overrides(
    extra_params: dict[str, Any], keys: Collection[str]
) -> dict[str, Any]:
    """``extra_params`` without the sampling settings a reviewer sets itself.

    ``keys`` are OpenAI names (``temperature``, ``seed``, ``top_p``,
    ``top_k``). The reviewer sends its own values, and an inherited one, at the
    top level or nested in Gemini's ``generationConfig``, would otherwise
    override them on the wire.
    """
    names = frozenset().union(*(_SAMPLING_SPELLINGS.get(key, {key}) for key in keys))
    if not names:
        return extra_params
    cleaned = {key: value for key, value in extra_params.items() if key not in names}
    for container in _SAMPLING_CONTAINERS:
        nested = cleaned.get(container)
        if isinstance(nested, Mapping):
            kept = {key: value for key, value in nested.items() if key not in names}
            if kept:
                cleaned[container] = kept
            else:
                del cleaned[container]
    return cleaned


def inheritable_extra_params(extra_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """The run's extra params a reviewer inherits: no request structure, no credentials.

    Credential-like keys go at every depth: ``metadata.api_key`` belongs to the
    benchmarked endpoint as much as a top-level ``api_key`` does.
    """
    inherited: dict[str, Any] = {}
    for key, value in (extra_params or {}).items():
        if key in REQUEST_STRUCTURE_KEYS or is_credential_key(key):
            continue
        nested = NESTED_REQUEST_STRUCTURE_KEYS.get(key)
        if nested is not None and isinstance(value, Mapping):
            value = {k: v for k, v in value.items() if k not in nested}
            if not value:
                continue
        inherited[key] = _without_credentials(value)
    return inherited


@dataclass(frozen=True)
class ReviewerSettings:
    """How to reach the reviewer and how it samples.

    A ``None`` field inherits from the benchmarked run. Sampling settings
    (``temperature``, ``timeout_seconds``, ``seed``, ``extra_params``) are always
    inherited; ``extra_params`` without :data:`REQUEST_STRUCTURE_KEYS` and
    credential-like keys. Connection settings (``api_key``, ``wire_format``,
    ``extra_headers``, ``session_header``) are inherited only when ``base_url``
    is ``None``, that is when the reviewer uses the benchmarked endpoint: a
    credential issued for one server never travels to another.

    ``system_prompt`` ``None`` selects the built-in reviewer prompt; the run's
    own system prompt describes the assistant, not the reviewer, and is never
    inherited.
    """

    model: str
    # Hidden from repr like the key: a URL can carry credentials in userinfo
    # or the query string.
    base_url: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    wire_format: str | None = None
    extra_headers: Mapping[str, str] | None = field(default=None, repr=False)
    session_header: str | None = None
    temperature: float | None = None
    timeout_seconds: float | None = None
    seed: int | None = None
    extra_params: dict[str, Any] | None = None
    scope: str = DEFAULT_REVIEW_SCOPE
    max_rounds: int = DEFAULT_REVIEW_MAX_ROUNDS
    system_prompt: str | None = None


def review_applies(scope: str, *, has_tool_calls: bool, has_answer: bool) -> bool:
    """Whether a turn with these contents falls under ``scope``.

    A turn with neither a tool call nor a visible answer (an empty or truncated
    turn) is never reviewed: the harness already tags it, and a reviewer has
    nothing to judge.
    """
    if scope == "tools":
        return has_tool_calls
    if scope == "final":
        return has_answer and not has_tool_calls
    return has_tool_calls or has_answer
