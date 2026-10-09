"""Inference-server validation and warm-up helpers."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.adapters.requests import minimal_request
from tool_eval_bench.cli.helpers import emit_headless_error
from tool_eval_bench.domain.errors import CONNECTION_FAILED, INVALID_RESPONSE, MODEL_NOT_AVAILABLE
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.utils.openai_compat import (
    max_tokens_retry_payload,
    output_token_limit_reached,
    sampling_retry_payload,
)
from tool_eval_bench.utils.urls import redact_urls


def _is_json_object(response: Any) -> bool:
    """Whether a response body parses as a JSON object, as every wire format answers."""
    try:
        return isinstance(response.json(), dict)
    except ValueError:
        return False


def preflight_model_check(
    console: Console,
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    headless: bool = False,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    temperature: float = 0.0,
    extra_params: dict[str, Any] | None = None,
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
    display_url: str | None = None,
) -> None:
    """Verify that a listed model can serve a minimal completion.

    ``extra_params`` and ``timeout_seconds`` mirror the options used by the
    benchmark requests so backend-specific configuration cannot make this
    gate reject an otherwise usable endpoint.  Console lines show
    *display_url* (the ``--redact-url`` form) when given.
    """
    import httpx

    show_url = display_url or base_url
    url, payload, headers = minimal_request(
        base_url,
        model,
        api_key,
        wire_format=wire_format,
        temperature=temperature,
        max_tokens=1,
        extra_params=extra_params,
        headers=headers,
    )

    async def check() -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            retry_payload = max_tokens_retry_payload(
                payload, response.status_code, response.text
            ) or sampling_retry_payload(payload, response.status_code, response.text)
            if retry_payload is not None:
                response = await client.post(url, json=retry_payload, headers=headers)
            return response

    try:
        response = asyncio.run(check())
        if response.status_code >= 400 and not output_token_limit_reached(
            response.status_code, response.text
        ):
            body = response.text[:300].strip()
            if headless:
                emit_headless_error(
                    MODEL_NOT_AVAILABLE,
                    f"Model '{model}' is listed in /v1/models but returned "
                    f"HTTP {response.status_code} on a test request: {body}",
                    exit_code=3,
                )
            console.print("[bold red]✗ Model not available[/]")
            console.print(
                f"[red]Model '{model}' is listed in /v1/models but returned "
                f"HTTP {response.status_code} on a test request.[/]"
            )
            console.print(f"[dim]  {body}[/]")
            console.print(
                "\n[yellow]The server lists this model but cannot serve it. "
                "Check server logs for model loading errors.[/]"
            )
            sys.exit(3)
        if response.status_code < 400 and not _is_json_object(response):
            # An HTML page from a proxy or a misrouted path answers 200 too;
            # passing it would start a run whose every request fails to parse.
            content_type = response.headers.get("Content-Type", "unknown")
            message = (
                f"Pre-flight request for model '{model}' returned a body that is not a JSON "
                f"object (HTTP {response.status_code}, Content-Type: {content_type}). "
                f"Body snippet: {response.text[:200]!r}"
            )
            if headless:
                emit_headless_error(INVALID_RESPONSE, message, exit_code=2)
            if show_url != base_url:
                message = redact_urls(message)
            console.print("[bold red]✗ Invalid response[/]")
            console.print(f"[red]{escape(message)}[/]")
            sys.exit(2)
    except httpx.ConnectError:
        if headless:
            emit_headless_error(
                CONNECTION_FAILED,
                f"Could not connect to {base_url} for pre-flight check.",
                exit_code=2,
            )
        console.print(f"[bold red]✗ Cannot connect to {show_url}[/]")
        sys.exit(2)
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        if headless:
            emit_headless_error(
                MODEL_NOT_AVAILABLE,
                f"Pre-flight check failed with unexpected error: {detail}",
                exit_code=3,
            )
        if show_url != base_url:
            # httpx quotes the request URL in its messages.
            detail = redact_urls(detail)
        console.print(f"[bold red]✗ Pre-flight check failed:[/] {detail}")
        sys.exit(3)


def warmup_server(
    console: Console,
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    wire_format: str = "openai",
    temperature: float = 0.0,
    extra_params: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    display_url: str | None = None,
) -> None:
    """Prime the model server before measuring benchmark behavior.

    A failure message is redacted when *display_url* (the ``--redact-url``
    form) differs from *base_url*, as in :func:`preflight_model_check`.
    """
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.throughput import (
        WARMUP_EXTRA_PARAMS,
        WARMUP_MAX_TOKENS,
        warmup,
    )

    warmup_params = dict(extra_params or {})
    warmup_params.setdefault("chat_template_kwargs", WARMUP_EXTRA_PARAMS["chat_template_kwargs"])
    request = minimal_request(
        base_url,
        model,
        api_key,
        wire_format=wire_format,
        temperature=temperature,
        max_tokens=WARMUP_MAX_TOKENS,
        extra_params=warmup_params,
        headers=headers,
    )

    with console.status(
        "[dim]  Warming up server… (first request may be slow with speculative decoding)[/]",
        spinner="dots",
    ):
        try:
            milliseconds = asyncio.run(
                warmup(
                    base_url,
                    model,
                    api_key,
                    timeout=120.0,
                    client_factory=HTTPMeasurementClient,
                    request=request,
                )
            )
            if milliseconds > 10_000:
                console.print(
                    f"  [bold green]✓[/] Warm-up complete [dim]({milliseconds:.0f} ms — "
                    "JIT/CUDA graph compilation on first request)[/]"
                )
            else:
                console.print(
                    f"  [bold green]✓[/] Warm-up complete [dim]({milliseconds:.0f} ms)[/]"
                )
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            response = getattr(exc, "response", None)
            body = getattr(response, "text", "").strip()
            if body and body not in message:
                message = f"{message}: {body[:300]}"
            if display_url is not None and display_url != base_url:
                message = redact_urls(message)
            console.print(f"  [bold yellow]⚠[/] Warm-up failed [dim]({message})[/]")
