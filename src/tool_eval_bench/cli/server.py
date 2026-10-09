"""Server discovery and detection helpers for the CLI.

Extracted from the monolithic ``cli/bench.py`` to separate the
network-touching detection code from scenario execution. All HTTP calls use
tight timeouts and are safe to call in environments without a running server.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx

from tool_eval_bench.utils.metadata import backend_from_response
from tool_eval_bench.utils.urls import redact_url

# Ports to scan on localhost.  Order matters — first match wins.
# A listening port is not engine identity. Unknown servers keep a neutral label.
DISCOVERY_PORTS: list[tuple[int, str, str]] = [
    # (port, backend_hint, human_label)
    (8000, "unknown", "inference server"),
    (8080, "unknown", "inference server"),  # vLLM, llama.cpp, or custom
    (8081, "unknown", "inference server"),  # common alt port
    (8082, "unknown", "inference server"),  # common alt port
    (30000, "unknown", "inference server"),
    (4000, "unknown", "inference server"),
    (3000, "unknown", "inference server"),
    (11434, "unknown", "inference server"),
    (5000, "unknown", "inference server"),
]


def detect_backend_from_response(resp: Any, port: int) -> tuple[str, str]:
    """Identify model ownership or headers; never infer an engine from its port."""
    return backend_from_response(resp) or ("unknown", "inference server")


def _headless_error(error_code: str, message: str, *, exit_code: int = 1) -> None:
    """Emit a structured JSONL error event on stderr and exit.

    Re-exported for server.py callers that need structured errors during
    server discovery. Delegates to ``cli.helpers.emit_headless_error`` to
    keep the implementation in one place.
    """
    from tool_eval_bench.cli.helpers import emit_headless_error

    emit_headless_error(error_code, message, exit_code=exit_code)


def _is_model_listing(resp: httpx.Response) -> bool:
    """Whether a 200 body is a model list: a JSON object listing under ``data`` or ``models``.

    A dev server, dashboard, or proxy on a scanned port often answers any path
    with HTML or some other 200, and that is not an inference server.
    """
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and any(
        isinstance(body.get(key), list) for key in ("data", "models")
    )


async def _discover_async() -> tuple[str, str, str, int] | None:
    """Async probe loop for discover_server."""
    async with httpx.AsyncClient(timeout=3.0) as client:
        for port, _hint, _label in DISCOVERY_PORTS:
            url = f"http://localhost:{port}"
            try:
                resp = await client.get(f"{url}/v1/models")
                if resp.status_code == 404:
                    resp = await client.get(f"{url}/models")
                if resp.status_code == 200 and _is_model_listing(resp):
                    backend, server_name = detect_backend_from_response(resp, port)
                    return url, backend, server_name, port
            except httpx.HTTPError:
                # Refused, timed out, or not speaking HTTP at all: not a server.
                continue
    return None


def discover_server(
    *,
    headless: bool = False,
    console: Any = None,
    redact: bool = False,
) -> tuple[str, str] | None:
    """Probe localhost on common inference server ports.

    Returns ``(base_url, backend_hint)`` for the first port that answers
    ``GET /v1/models`` (or ``GET /models`` as fallback) with HTTP 200 and a
    JSON model list. Returns ``None`` if no server is found.

    The backend is identified from the server's response headers when
    possible, otherwise reported as an unidentified inference server.

    When *headless* is True, emits a JSONL event on stderr.
    Otherwise prints to console. *redact* (``--redact-url``) masks the host in
    the console line only: the JSONL event keeps the real URL because a
    consumer needs it to connect, as ``docs/artifacts.md`` documents.
    """
    result = asyncio.run(_discover_async())
    if result:
        base_url, backend, server_name, port = result
        if headless:
            msg = {
                "event": "server_discovered",
                "base_url": base_url,
                "backend": backend,
                "server_type": server_name,
                "port": port,
            }
            sys.stderr.write(json.dumps(msg) + "\n")
            sys.stderr.flush()
        elif console:
            shown_url = redact_url(base_url) if redact else base_url
            console.print(
                f"  [bold green]✓[/] Auto-discovered [bold]{server_name}[/] at [cyan]{shown_url}[/]"
            )
        return base_url, backend
    return None
