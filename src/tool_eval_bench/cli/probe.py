"""Inference-server validation and warm-up helpers."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.adapters.factory import build_adapter
from tool_eval_bench.adapters.requests import minimal_request
from tool_eval_bench.cli.helpers import emit_headless_error
from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.errors import CONNECTION_FAILED, INVALID_RESPONSE, MODEL_NOT_AVAILABLE
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.wire_hygiene import format_tool_parser_advice
from tool_eval_bench.runner.orchestrator import ToolCallWireProbe, probe_tool_call_parsing
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
            if show_url != base_url:
                body = redact_urls(body)
            console.print("[bold red]✗ Model not available[/]")
            console.print(
                f"[red]Model '{escape(model)}' is listed in /v1/models but returned "
                f"HTTP {response.status_code} on a test request.[/]"
            )
            console.print(f"[dim]  {escape(body)}[/]")
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
        console.print(f"[bold red]✗ Cannot connect to {escape(show_url)}[/]")
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
        console.print(f"[bold red]✗ Pre-flight check failed:[/] {escape(detail)}")
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
            console.print(f"  [bold yellow]⚠[/] Warm-up failed [dim]({escape(message)})[/]")


def check_tool_call_wire_parsing(
    console: Console,
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    backend: str | None = None,
    engine_name: str | None = None,
    headless: bool = False,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    extra_params: dict[str, Any] | None = None,
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
    display_url: str | None = None,
    adapter: BackendAdapter | None = None,
    exit_on_failure: bool = False,
) -> ToolCallWireProbe:
    """Verify that the endpoint parses a simple tool call into structured tool_calls."""
    show_url = display_url or base_url

    async def _check() -> ToolCallWireProbe:
        owned = adapter is None
        active_adapter = (
            adapter
            if adapter is not None
            else build_adapter(
                base_url,
                wire_format=wire_format,
                default_headers=headers,
            )
        )
        try:
            return await probe_tool_call_parsing(
                active_adapter,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                extra_params=extra_params,
            )
        finally:
            if owned:
                aclose = getattr(active_adapter, "aclose", None)
                if aclose is not None:
                    await aclose()

    probe = asyncio.run(_check())
    ok = probe.parsed and probe.arguments_intact
    advice = (
        None
        if ok
        else format_tool_parser_advice(
            backend=backend,
            engine_name=engine_name,
            parser_family=probe.parser_family,
        )
    )

    if headless:
        event: dict[str, Any] = {
            "event": "tool_call_probe",
            "status": "ok" if ok else "failed",
            "parsed": probe.parsed,
            "arguments_intact": probe.arguments_intact,
            "detail": redact_urls(probe.detail),
        }
        if probe.unparsed_marker is not None:
            event["unparsed_marker"] = probe.unparsed_marker
        if probe.parser_family is not None:
            event["parser_family"] = probe.parser_family
        if advice is not None:
            event["advice"] = advice
        sys.stderr.write(json.dumps(event) + "\n")
        sys.stderr.flush()
    elif ok:
        console.print(
            "  [bold green]✓[/] Tool-call wire parsing verified: "
            "structured tool_calls arrived with intact arguments"
        )
    elif probe.unparsed_marker is not None:
        console.print(
            f"  [bold yellow]⚠[/] [yellow]Unparsed tool-call output detected: endpoint emitted "
            f"{escape(probe.unparsed_marker)} in content instead of tool_calls.[/]"
        )
        if advice is not None:
            console.print(f"  [yellow]  {escape(advice)}[/]")
    else:
        shown_detail = redact_urls(probe.detail) if show_url != base_url else probe.detail
        console.print(
            f"  [bold yellow]⚠[/] [yellow]Tool-call wire check did not confirm structured "
            f"tool_calls: {escape(shown_detail)}[/]"
        )
        if advice is not None:
            console.print(f"  [yellow]  {escape(advice)}[/]")

    if exit_on_failure and not ok:
        sys.exit(1)
    return probe
