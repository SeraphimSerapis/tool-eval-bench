"""Extra request headers a user attaches to every request, and how to parse them.

Gateways sometimes need a header the wire format does not define.  OpenCode Go,
for example, refuses a request without a stable per-conversation
``x-opencode-session`` and asks clients to identify themselves in
``User-Agent``.  Headers are static endpoint configuration, so they come in
through ``--header`` and ``TOOL_EVAL_<PROVIDER>_HEADERS`` and are attached at
adapter construction rather than per request.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping

from tool_eval_bench import __version__

#: Sent on every request unless a user header overrides it.  A benchmark that
#: shows up in a provider's logs as ``python-httpx`` is hard to tell apart from
#: everything else.
USER_AGENT = f"tool-eval-bench/{__version__}"

#: Separator between pairs in the environment form, where one variable has to
#: hold several headers.  A semicolon is not valid inside a header value.
ENV_PAIR_SEPARATOR = ";"


def parse_header_pair(value: str) -> tuple[str, str]:
    """Split one ``Name=Value`` or ``Name: Value`` spelling into a pair.

    Raises ValueError when the name is missing or contains whitespace, which
    is the usual symptom of a value that forgot its separator.
    """
    text = value.strip()
    sep = min(
        (i for i in (text.find("="), text.find(":")) if i >= 0),
        default=-1,
    )
    if sep <= 0:
        raise ValueError(f"Header {value!r} must be written as NAME=VALUE or NAME: VALUE")
    name, header_value = text[:sep].strip(), text[sep + 1 :].strip()
    if not name or any(ch.isspace() for ch in name):
        raise ValueError(f"Header {value!r} has an invalid name {name!r}")
    return name, header_value


def parse_header_pairs(values: Iterable[str] | None) -> dict[str, str]:
    """Parse repeated ``--header`` values into a mapping; later pairs win."""
    headers: dict[str, str] = {}
    for value in values or ():
        name, header_value = parse_header_pair(value)
        headers[name] = header_value
    return headers


def parse_header_env(value: str | None) -> dict[str, str]:
    """Parse the ``;``-separated environment form of several headers."""
    if not value or not value.strip():
        return {}
    return parse_header_pairs(part for part in value.split(ENV_PAIR_SEPARATOR) if part.strip())


def attach_session_id(
    headers: Mapping[str, str],
    session_header: str | None,
    conversation_id: str | None = None,
) -> dict[str, str]:
    """Return *headers* plus the session header carrying *conversation_id*.

    A missing id gets a fresh one: a single-shot request is its own
    conversation.  With no session header configured the headers pass through.
    """
    merged = dict(headers)
    if session_header:
        merged[session_header] = conversation_id or uuid.uuid4().hex
    return merged
