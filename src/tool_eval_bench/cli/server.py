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

from tool_eval_bench.utils.metadata import backend_from_models

# Ports to scan on localhost.  Order matters — first match wins.
# The backend_hint is a *guess* used only when the server doesn't identify
# itself via response headers.  Ports used by multiple backends (8080, 8081)
# get a generic "vllm" hint that the user can override with --backend.
DISCOVERY_PORTS: list[tuple[int, str, str]] = [
    # (port, backend_hint, human_label)
    (8000, "vllm", "vLLM"),
    (8080, "vllm", "inference server"),  # vLLM, llama.cpp, or custom
    (8081, "vllm", "inference server"),  # common alt port
    (8082, "vllm", "inference server"),  # common alt port
    (30000, "vllm", "SGLang"),
    (4000, "litellm", "LiteLLM"),
    (3000, "litellm", "LiteLLM"),
    (11434, "litellm", "Ollama"),
    (5000, "vllm", "TGI"),
]


def detect_backend_from_response(resp: Any, port: int) -> tuple[str, str]:
    """Identify a distinctive model owner, then check response headers.

    vLLM sets ``server: vllm``, SGLang sets ``server: sglang``,
    llama.cpp sets ``server: llama.cpp``.  Falls back to port-based hint.
    """
    try:
        owner = backend_from_models(resp.json())
    except (AttributeError, ValueError):
        owner = None
    if owner:
        return owner

    server_header = ""
    if hasattr(resp, "headers"):
        server_header = resp.headers.get("server", "").lower()

    if "vllm" in server_header:
        return "vllm", "vLLM"
    if "sglang" in server_header:
        return "vllm", "SGLang"  # SGLang uses OpenAI-compat, same adapter
    if "llama" in server_header:
        return "llamacpp", "llama.cpp"

    # Fall back to port-based hint
    for p, backend, label in DISCOVERY_PORTS:
        if p == port:
            return backend, label
    return "vllm", "inference server"


def _headless_error(error_code: str, message: str, *, exit_code: int = 1) -> None:
    """Emit a structured JSONL error event on stderr and exit.

    Re-exported for server.py callers that need structured errors during
    server discovery. Delegates to ``cli.helpers.emit_headless_error`` to
    keep the implementation in one place.
    """
    from tool_eval_bench.cli.helpers import emit_headless_error

    emit_headless_error(error_code, message, exit_code=exit_code)


async def _discover_async() -> tuple[str, str, str, int] | None:
    """Async probe loop for discover_server."""
    async with httpx.AsyncClient(timeout=3.0) as client:
        for port, _hint, _label in DISCOVERY_PORTS:
            url = f"http://localhost:{port}"
            try:
                resp = await client.get(f"{url}/v1/models")
                if resp.status_code == 404:
                    resp = await client.get(f"{url}/models")
                if resp.status_code == 200:
                    backend, server_name = detect_backend_from_response(resp, port)
                    return url, backend, server_name, port
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
    return None


def discover_server(
    *,
    headless: bool = False,
    console: Any = None,
) -> tuple[str, str] | None:
    """Probe localhost on common inference server ports.

    Returns ``(base_url, backend_hint)`` for the first port that responds
    to ``GET /v1/models`` (or ``GET /models`` as fallback) with HTTP 200.
    Returns ``None`` if no server is found.

    The backend is identified from the server's response headers when
    possible (vLLM, SGLang, and llama.cpp advertise themselves), falling
    back to a port-based guess.

    When *headless* is True, emits a JSONL event on stderr.
    Otherwise prints to console.
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
            console.print(
                f"  [bold green]✓[/] Auto-discovered [bold]{server_name}[/] at [cyan]{base_url}[/]"
            )
        return base_url, backend
    return None
