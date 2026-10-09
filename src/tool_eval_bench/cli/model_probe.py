"""Model discovery and inference-server readiness checks."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from typing import Any, NoReturn

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.adapters.anthropic import ANTHROPIC_VERSION
from tool_eval_bench.adapters.wire_format import anthropic_models_url, gemini_models_url
from tool_eval_bench.cli.helpers import emit_headless_error as _headless_error
from tool_eval_bench.domain.errors import (
    CONNECTION_FAILED,
    DETECTION_FAILED,
    HTTP_ERROR,
    INVALID_RESPONSE,
    NO_MODELS,
)
from tool_eval_bench.utils.headers import USER_AGENT, merge_headers
from tool_eval_bench.utils.urls import redact_url as _redact_url
from tool_eval_bench.utils.urls import redact_urls as _redact_urls


def _models_request(
    base_url: str,
    api_key: str | None,
    wire_format: str,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Return the model-listing URL and headers for a wire format.

    *extra_headers* are the user's and are applied last.
    """
    url, headers = _models_url_and_auth(base_url, api_key, wire_format)
    return url, merge_headers({"User-Agent": USER_AGENT}, headers, extra_headers)


def _models_url_and_auth(
    base_url: str, api_key: str | None, wire_format: str
) -> tuple[str, dict[str, str]]:
    headers: dict[str, str] = {}
    if wire_format == "gemini":
        if api_key:
            headers["x-goog-api-key"] = api_key
        return gemini_models_url(base_url), headers
    if wire_format == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if api_key:
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        return anthropic_models_url(base_url), headers
    url = base_url.rstrip("/")
    # Handle base_url that already ends with /v1
    endpoint = f"{url}/models" if url.endswith("/v1") else f"{url}/v1/models"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return endpoint, headers


def _detect_model(
    base_url: str,
    api_key: str | None,
    console: Console,
    *,
    display_url: str | None = None,
    headless: bool = False,
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Query /v1/models and auto-select or let the user pick.

    Returns (api_id, display_name).
      - api_id:       what to send in API requests (e.g. "gemma4")
      - display_name: the real model path if available (e.g. "Intel/gemma-4-31B-it-int4-AutoRound")

    When *headless* is True (e.g. ``--json`` mode), the interactive picker is
    skipped: the first available model is auto-selected and a JSONL event is
    emitted on stderr.  Failures exit with the documented codes in both modes
    (2 = connection, HTTP, or unreadable response; 3 = no models).
    """
    import httpx

    gemini = wire_format == "gemini"
    anthropic = wire_format == "anthropic"
    url = base_url.rstrip("/")
    models_endpoint, request_headers = _models_request(base_url, api_key, wire_format, headers)

    # Build a display-safe endpoint URL for console output
    show_url = display_url or base_url
    show_endpoint, _ = _models_request(show_url, None, wire_format)
    if not headless:
        console.print(f"[dim]  Querying {show_endpoint} …[/]", end=" ")

    used_fallback = False

    async def _fetch() -> tuple[httpx.Response, bool]:
        nonlocal used_fallback
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(models_endpoint, headers=request_headers)
            if resp.status_code == 404 and not (gemini or anthropic):
                fallback_url = f"{url}/models"
                resp = await client.get(fallback_url, headers=request_headers)
                used_fallback = True
            return resp, used_fallback

    try:
        resp, used_fallback = asyncio.run(_fetch())
        resp.raise_for_status()
    except httpx.ConnectError:
        if headless:
            _headless_error(
                CONNECTION_FAILED,
                f"Could not connect to {show_url}. Is the server running?",
                exit_code=2,
            )
        console.print("[bold red]✗ cannot connect[/]")
        console.print(f"\n[red]Could not connect to {show_url}. Is the server running?[/]")
        sys.exit(2)
    except httpx.HTTPStatusError as exc:
        if headless:
            _headless_error(
                HTTP_ERROR,
                f"Server returned {exc.response.status_code}. Check the URL and API key.",
                exit_code=2,
            )
        console.print(f"[bold red]✗ HTTP {exc.response.status_code}[/]")
        console.print(
            f"\n[red]Server returned {exc.response.status_code}. Check the URL and API key.[/]"
        )
        sys.exit(2)
    except Exception as exc:
        if headless:
            _headless_error(DETECTION_FAILED, str(exc), exit_code=2)
        detail = _redact_urls(str(exc)) if show_url != base_url else str(exc)
        console.print(f"[bold red]✗ {detail}[/]")
        sys.exit(2)

    if used_fallback:
        if not headless:
            console.print(
                "\n  [yellow]⚠ /v1/models returned 404, used /models fallback. "
                "Check your server configuration.[/]"
            )

    model_list: list[dict[str, Any]] = []
    problem: str | None = None
    try:
        data = resp.json()
    except ValueError:
        problem = "invalid JSON"
    else:
        # OpenAI lists under "data"; the native Gemini API lists under "models".
        listed = (data.get("data") or data.get("models") or []) if isinstance(data, dict) else None
        # The OpenAI shape is a list of objects; anything else is a bad body,
        # not a model list to guess at.
        if isinstance(listed, list) and all(isinstance(m, dict) for m in listed):
            model_list = listed
        else:
            problem = "a model list that is not a list of objects"
    if problem is not None:
        status_code = resp.status_code
        content_type = resp.headers.get("Content-Type", "unknown")
        body_snippet = resp.text[:200]
        err_msg = (
            f"Server returned {problem} from /v1/models (HTTP {status_code}, "
            f"Content-Type: {content_type}). Body snippet: {body_snippet!r}"
        )
        if headless:
            _headless_error(INVALID_RESPONSE, err_msg, exit_code=2)
        console.print("[bold red]✗ invalid response[/]")
        console.print(f"[red]{err_msg}[/]")
        sys.exit(2)

    # Build (api_id, display_name) pairs
    # vLLM: "id" is the served alias, "root" is the actual model path
    # LiteLLM/others: may not have "root"
    models: list[tuple[str, str]] = []
    for m in model_list:
        # Gemini names a model "models/gemini-3.7-flash"; the id sent in a
        # request is the trailing segment.
        api_id = m.get("id") or str(m.get("name", "")).removeprefix("models/")
        if not api_id:
            continue
        root = m.get("root") or m.get("displayName") or m.get("display_name", "")
        # Use root as display name if it differs from the alias
        display = root if root and root != api_id else api_id
        models.append((api_id, display))

    if not models:
        if headless:
            _headless_error(NO_MODELS, "The server returned an empty model list.", exit_code=3)
        console.print("[bold red]✗ no models found[/]")
        console.print("[red]The server returned an empty model list.[/]")
        sys.exit(3)

    if len(models) == 1:
        api_id, display = models[0]
        if not headless:
            if display != api_id:
                console.print(f"[bold green]✓[/] [bold]{display}[/] [dim](alias: {api_id})[/]")
            else:
                console.print(f"[bold green]✓[/] [bold]{api_id}[/]")
        return api_id, display

    # Multiple models — in headless mode, auto-select the first one
    if headless:
        api_id, display = models[0]
        msg = {
            "event": "model_auto_selected",
            "model": api_id,
            "display_name": display,
            "total_available": len(models),
            "available_models": [m[0] for m in models],
        }
        sys.stderr.write(json.dumps(msg) + "\n")
        sys.stderr.flush()
        return api_id, display

    # Multiple models — interactive: let the user choose
    console.print(f"[bold cyan]found {len(models)} models[/]")
    console.print()
    console.print("[bold]Available models:[/]")
    for i, (api_id, display) in enumerate(models, 1):
        if display != api_id:
            console.print(f"  [bold cyan]{i}[/]) {display} [dim](alias: {api_id})[/]")
        else:
            console.print(f"  [bold cyan]{i}[/]) {api_id}")
    console.print()

    while True:
        try:
            choice = input(f"Select model [1-{len(models)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                api_id, display = models[idx]
                console.print(f"\n[dim]  Selected:[/] [bold]{display}[/]\n")
                return api_id, display
            console.print(f"[red]  Please enter a number between 1 and {len(models)}.[/]")
        except (ValueError, EOFError):
            console.print(f"[red]  Please enter a number between 1 and {len(models)}.[/]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Cancelled.[/]")
            sys.exit(1)


def _probe_server(
    console: Console,
    base_url: str,
    api_key: str | None,
    *,
    display_url: str | None = None,
    headless: bool = False,
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
) -> None:
    """Check if a server is reachable and responsive, then exit.

    Useful for CI/CD pipelines and sparkrun recipes where the benchmark
    step runs right after server startup — this lets the orchestrator
    wait until the server is ready.

    Exits 0 if the server answers the model listing with a JSON object, 1 if
    it is unreachable or answers with an error, and 2 (``invalid_response``)
    if it answers 2xx with anything else, such as a proxy's HTML page.
    Console lines show *display_url* (the ``--redact-url`` form) when given.
    """
    import httpx

    endpoint, request_headers = _models_request(base_url, api_key, wire_format, headers)
    show_url = display_url or base_url

    async def _check() -> httpx.Response:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(endpoint, headers=request_headers)
            resp.raise_for_status()
            return resp

    try:
        resp = asyncio.run(_check())
    except Exception as exc:
        if headless:
            msg: dict[str, Any] = {
                "event": "probe_result",
                "status": "failed",
                "base_url": _redact_url(base_url),
                "error": _redact_urls(str(exc)) or type(exc).__name__,
            }
            sys.stderr.write(json.dumps(msg) + "\n")
            sys.stderr.flush()
        else:
            # httpx quotes the request URL in its messages.
            detail = _redact_urls(str(exc)) if show_url != base_url else str(exc)
            console.print(f"[bold red]✗[/] Server at {show_url} is not ready: {detail}")
        sys.exit(1)

    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        _report_invalid_probe_response(console, resp, base_url, show_url, headless=headless)
    listed = data.get("data")
    model_ids: list[str] = (
        [str(m["id"]) for m in listed if isinstance(m, dict) and m.get("id")]
        if isinstance(listed, list)
        else []
    )

    if headless:
        msg = {
            "event": "probe_result",
            "status": "ready",
            "base_url": _redact_url(base_url),
            "models": model_ids,
        }
        sys.stderr.write(json.dumps(msg) + "\n")
        sys.stderr.flush()
    else:
        console.print(f"[bold green]✓[/] Server at {show_url} is ready")
        if model_ids:
            console.print(f"  Models: {', '.join(model_ids)}")
    sys.exit(0)


def _report_invalid_probe_response(
    console: Console,
    resp: Any,
    base_url: str,
    show_url: str,
    *,
    headless: bool,
) -> NoReturn:
    """Exit 2 for a 2xx model listing that is not a JSON object.

    Waiting will not fix it: something answers at the URL, but not an
    inference server's model listing.
    """
    content_type = resp.headers.get("Content-Type", "unknown")
    message = (
        f"Server at {base_url} answered the model listing with a body that is not a JSON "
        f"object (HTTP {resp.status_code}, Content-Type: {content_type}). "
        f"Body snippet: {resp.text[:200]!r}"
    )
    if headless:
        event = {
            "event": "probe_result",
            "status": "failed",
            "base_url": _redact_url(base_url),
            "error_code": INVALID_RESPONSE,
            "error": _redact_urls(message),
        }
        sys.stderr.write(json.dumps(event) + "\n")
        sys.stderr.flush()
    else:
        if show_url != base_url:
            message = _redact_urls(message)
        console.print("[bold red]✗ Invalid response[/]")
        console.print(f"[red]{escape(message)}[/]")
    sys.exit(2)
