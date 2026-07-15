"""Model discovery and inference-server readiness checks."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from rich.console import Console

from tool_eval_bench.cli.helpers import emit_headless_error as _headless_error
from tool_eval_bench.domain.errors import (
    CONNECTION_FAILED,
    DETECTION_FAILED,
    HTTP_ERROR,
    INVALID_RESPONSE,
    NO_MODELS,
)


def _detect_model(
    base_url: str,
    api_key: str | None,
    console: Console,
    *,
    display_url: str | None = None,
    headless: bool = False,
) -> tuple[str, str]:
    """Query /v1/models and auto-select or let the user pick.

    Returns (api_id, display_name).
      - api_id:       what to send in API requests (e.g. "gemma4")
      - display_name: the real model path if available (e.g. "Intel/gemma-4-31B-it-int4-AutoRound")

    When *headless* is True (e.g. ``--json`` mode), the interactive picker is
    skipped: the first available model is auto-selected and a JSONL event is
    emitted on stderr.  Connection errors produce structured JSON on stderr
    and use differentiated exit codes (2 = connection, 3 = no models).
    """
    import httpx

    url = base_url.rstrip("/")
    models_endpoint = f"{url}/v1/models"
    # Handle base_url that already ends with /v1
    if url.endswith("/v1"):
        models_endpoint = f"{url}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build a display-safe endpoint URL for console output
    show_url = display_url or base_url
    show_endpoint = f"{show_url.rstrip('/')}/v1/models"
    if show_url.rstrip("/").endswith("/v1"):
        show_endpoint = f"{show_url.rstrip('/')}/models"
    if not headless:
        console.print(f"[dim]  Querying {show_endpoint} …[/]", end=" ")

    used_fallback = False

    async def _fetch() -> tuple[httpx.Response, bool]:
        nonlocal used_fallback
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(models_endpoint, headers=headers)
            if resp.status_code == 404:
                fallback_url = f"{url}/models"
                resp = await client.get(fallback_url, headers=headers)
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
        sys.exit(1)
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
        sys.exit(1)
    except Exception as exc:
        if headless:
            _headless_error(DETECTION_FAILED, str(exc), exit_code=2)
        console.print(f"[bold red]✗ {exc}[/]")
        sys.exit(1)

    if used_fallback:
        if not headless:
            console.print(
                "\n  [yellow]⚠ /v1/models returned 404, used /models fallback. "
                "Check your server configuration.[/]"
            )

    try:
        data = resp.json()
        model_list = data.get("data", [])
    except Exception:
        status_code = resp.status_code
        content_type = resp.headers.get("Content-Type", "unknown")
        body_snippet = resp.text[:200]
        err_msg = (
            f"Server returned invalid JSON from /v1/models (HTTP {status_code}, "
            f"Content-Type: {content_type}). Body snippet: {body_snippet!r}"
        )
        if headless:
            _headless_error(INVALID_RESPONSE, err_msg, exit_code=2)
        console.print("[bold red]✗ invalid response[/]")
        console.print(f"[red]{err_msg}[/]")
        sys.exit(1)

    # Build (api_id, display_name) pairs
    # vLLM: "id" is the served alias, "root" is the actual model path
    # LiteLLM/others: may not have "root"
    models: list[tuple[str, str]] = []
    for m in model_list:
        api_id = m.get("id", "")
        if not api_id:
            continue
        root = m.get("root", "")
        # Use root as display name if it differs from the alias
        display = root if root and root != api_id else api_id
        models.append((api_id, display))

    if not models:
        if headless:
            _headless_error(NO_MODELS, "The server returned an empty model list.", exit_code=3)
        console.print("[bold red]✗ no models found[/]")
        console.print("[red]The server returned an empty model list.[/]")
        sys.exit(1)

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
    headless: bool = False,
) -> None:
    """Check if a server is reachable and responsive, then exit.

    Useful for CI/CD pipelines and sparkrun recipes where the benchmark
    step runs right after server startup — this lets the orchestrator
    wait until the server is ready.

    Exits 0 if the server responds to /v1/models, exit 1 otherwise.
    """
    import httpx

    from tool_eval_bench.utils.urls import models_url

    endpoint = models_url(base_url)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _check() -> httpx.Response:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(endpoint, headers=headers)
            resp.raise_for_status()
            return resp

    try:
        resp = asyncio.run(_check())
        data = resp.json()
        model_ids: list[str] = [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
    except Exception as exc:
        if headless:
            msg: dict[str, Any] = {
                "event": "probe_result",
                "status": "failed",
                "base_url": base_url,
                "error": str(exc) or type(exc).__name__,
            }
            sys.stderr.write(json.dumps(msg) + "\n")
            sys.stderr.flush()
        else:
            console.print(f"[bold red]✗[/] Server at {base_url} is not ready: {exc}")
        sys.exit(1)

    if headless:
        msg = {
            "event": "probe_result",
            "status": "ready",
            "base_url": base_url,
            "models": model_ids,
        }
        sys.stderr.write(json.dumps(msg) + "\n")
        sys.stderr.flush()
    else:
        console.print(f"[bold green]✓[/] Server at {base_url} is ready")
        if model_ids:
            console.print(f"  Models: {', '.join(model_ids)}")
    sys.exit(0)
