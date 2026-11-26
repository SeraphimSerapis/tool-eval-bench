"""Inference-server validation and warm-up helpers."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Mapping
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
from tool_eval_bench.runner.template_probe import (
    ChatTemplateProbe,
    probe_chat_template_fidelity,
)
from tool_eval_bench.utils.openai_compat import (
    max_tokens_retry_payload,
    output_token_limit_reached,
    sampling_retry_payload,
)
from tool_eval_bench.utils.tokenizers import extract_special_tokens, resolve_tokenizer
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


def _template_probe_advice(
    probe: ChatTemplateProbe,
    *,
    backend: str | None = None,
    engine_name: str | None = None,
) -> str | None:
    """Return concrete remediation advice for a failed chat-template probe."""
    if probe.passed:
        return None
    if probe.parser_family is not None:
        return format_tool_parser_advice(
            backend=backend,
            engine_name=engine_name,
            parser_family=probe.parser_family,
        )
    if probe.diagnosis == "tool_replay_template_error":
        return (
            "Inspect the server chat template (--chat-template on vLLM/SGLang or "
            "--jinja on llama.cpp); tool-response messages were not preserved."
        )
    if probe.diagnosis == "multi_turn_template_error":
        return (
            "Inspect the server chat template; multi-turn user/assistant history was "
            "dropped or mangled."
        )
    if probe.diagnosis == "template_token_leak":
        return (
            "Check stop tokens and chat-template or reasoning-parser settings on the "
            "serving stack; special control tokens leaked into visible output."
        )
    return None


def check_chat_template_fidelity(
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
    tokenizer: str | None = None,
    adapter: BackendAdapter | None = None,
    exit_on_failure: bool = False,
) -> ChatTemplateProbe:
    """Verify chat-template replay fidelity and absence of leaked special tokens."""
    show_url = display_url or base_url
    resolution = resolve_tokenizer(model, explicit=tokenizer)
    special_tokens = (
        extract_special_tokens(resolution.path) if resolution.path is not None else frozenset()
    )

    async def _check() -> ChatTemplateProbe:
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
            return await probe_chat_template_fidelity(
                active_adapter,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                extra_params=extra_params,
                special_tokens=special_tokens,
            )
        finally:
            if owned:
                aclose = getattr(active_adapter, "aclose", None)
                if aclose is not None:
                    await aclose()

    probe = asyncio.run(_check())
    advice = _template_probe_advice(probe, backend=backend, engine_name=engine_name)
    redact = show_url != base_url

    if headless:
        event: dict[str, Any] = {
            "event": "template_probe",
            "status": "ok" if probe.passed else "failed",
            "passed": probe.passed,
            "diagnosis": probe.diagnosis,
            "detail": redact_urls(probe.detail),
            "tokenizer_source": resolution.source,
            "tokenizer_tokens_checked": probe.tokenizer_tokens_checked,
            "checks": [
                {
                    **check.to_dict(),
                    "detail": redact_urls(check.detail),
                    **({"error": redact_urls(check.error)} if check.error is not None else {}),
                }
                for check in probe.checks
            ],
        }
        if probe.diagnoses:
            event["diagnoses"] = list(probe.diagnoses)
        if probe.leaked_tokens:
            event["leaked_tokens"] = list(probe.leaked_tokens)
        if probe.unparsed_marker is not None:
            event["unparsed_marker"] = probe.unparsed_marker
        if probe.parser_family is not None:
            event["parser_family"] = probe.parser_family
        if advice is not None:
            event["advice"] = advice
        sys.stderr.write(json.dumps(event) + "\n")
        sys.stderr.flush()
    elif probe.passed:
        console.print(
            "  [bold green]✓[/] Chat-template fidelity verified: "
            "tool replay, multi-turn history, and token hygiene passed"
        )
        if probe.tokenizer_tokens_checked > 0:
            console.print(
                f"  [dim]  Checked {probe.tokenizer_tokens_checked} tokenizer control "
                f"token(s) ({escape(resolution.source)})[/]"
            )
        for check in probe.checks:
            console.print(f"    [green]✓[/] {escape(check.name)}: [dim]{escape(check.detail)}[/]")
    else:
        shown_detail = redact_urls(probe.detail) if redact else probe.detail
        console.print(
            f"  [bold yellow]⚠[/] [yellow]Chat-template fidelity check failed "
            f"({escape(probe.diagnosis)}): {escape(shown_detail)}[/]"
        )
        if probe.tokenizer_tokens_checked > 0:
            console.print(
                f"  [dim]  Checked {probe.tokenizer_tokens_checked} tokenizer control "
                f"token(s) ({escape(resolution.source)})[/]"
            )
        for check in probe.checks:
            mark = "[green]✓[/]" if check.passed else "[red]✗[/]"
            check_detail = redact_urls(check.detail) if redact else check.detail
            console.print(f"    {mark} {escape(check.name)}: [dim]{escape(check_detail)}[/]")
        if advice is not None:
            console.print(f"  [yellow]  {escape(advice)}[/]")

    if exit_on_failure and not probe.passed:
        sys.exit(1)
    return probe


def run_probe_mode(
    args: argparse.Namespace,
    console: Console,
    endpoint: Any,
    *,
    parser: argparse.ArgumentParser | None = None,
    probe_server_fn: Callable[..., None],
    build_extra_params_fn: Callable[[argparse.Namespace, argparse.ArgumentParser], dict[str, Any]],
    asyncio_mod: Any = None,
    check_tool_call_fn: Callable[..., ToolCallWireProbe] | None = None,
    check_template_fn: Callable[..., ChatTemplateProbe] | None = None,
) -> None:
    """``--probe``: check server reachability and optional tool/template fidelity."""
    from tool_eval_bench.cli import model_probe as _model_probe
    from tool_eval_bench.cli.legacy_parser import _make_parser
    from tool_eval_bench.cli.resolve import redact_url as _redact_url

    display_url = _redact_url(endpoint.base_url) if args.redact_url else endpoint.base_url
    check_tools = bool(getattr(args, "check_tools", False))
    check_template = bool(getattr(args, "check_template", False))
    if not (check_tools or check_template):
        probe_server_fn(
            console,
            endpoint.base_url,
            endpoint.api_key,
            display_url=display_url,
            headless=args.json,
            wire_format=endpoint.wire_format,
            headers=endpoint.probe_headers,
        )
        return

    prior_asyncio = _model_probe.asyncio
    _model_probe.asyncio = asyncio_mod if asyncio_mod is not None else asyncio
    try:
        model_ids = _model_probe.probe_server_models(
            console,
            endpoint.base_url,
            endpoint.api_key,
            display_url=display_url,
            headless=args.json,
            wire_format=endpoint.wire_format,
            headers=endpoint.probe_headers,
        )
    finally:
        _model_probe.asyncio = prior_asyncio

    resolved_model = args.model or endpoint.model or (model_ids[0] if model_ids else None)
    if not resolved_model:
        if args.json:
            if check_tools:
                tool_event = {
                    "event": "tool_call_probe",
                    "status": "failed",
                    "parsed": False,
                    "arguments_intact": False,
                    "detail": "No model could be resolved from --model or /v1/models.",
                }
                sys.stderr.write(json.dumps(tool_event) + "\n")
            if check_template:
                template_event = {
                    "event": "template_probe",
                    "status": "failed",
                    "passed": False,
                    "diagnosis": "model_or_endpoint_failure",
                    "detail": "No model could be resolved from --model or /v1/models.",
                }
                sys.stderr.write(json.dumps(template_event) + "\n")
            sys.stderr.flush()
        else:
            if check_tools:
                console.print(
                    "  [bold red]✗[/] [red]Cannot verify tool-call wire parsing: "
                    "pass --model or ensure /v1/models lists at least one model.[/]"
                )
            if check_template:
                console.print(
                    "  [bold red]✗[/] [red]Cannot verify chat-template fidelity: "
                    "pass --model or ensure /v1/models lists at least one model.[/]"
                )
        sys.exit(1)

    active_parser = parser if parser is not None else _make_parser()
    extra_params = build_extra_params_fn(args, active_parser)
    tool_checker = (
        check_tool_call_fn if check_tool_call_fn is not None else check_tool_call_wire_parsing
    )
    template_checker = (
        check_template_fn if check_template_fn is not None else check_chat_template_fidelity
    )
    failed = False
    if check_tools:
        tool_probe = tool_checker(
            console,
            endpoint.base_url,
            resolved_model,
            endpoint.api_key,
            backend=endpoint.backend,
            headless=bool(args.json),
            timeout_seconds=args.timeout,
            extra_params=extra_params or None,
            wire_format=endpoint.wire_format,
            headers=endpoint.probe_headers,
            display_url=display_url,
            exit_on_failure=not check_template,
        )
        if not (tool_probe.parsed and tool_probe.arguments_intact):
            failed = True
    if check_template:
        template_probe = template_checker(
            console,
            endpoint.base_url,
            resolved_model,
            endpoint.api_key,
            backend=endpoint.backend,
            headless=bool(args.json),
            timeout_seconds=args.timeout,
            extra_params=extra_params or None,
            wire_format=endpoint.wire_format,
            headers=endpoint.probe_headers,
            display_url=display_url,
            tokenizer=getattr(args, "tokenizer", None),
            exit_on_failure=not check_tools,
        )
        if not template_probe.passed:
            failed = True
    sys.exit(1 if failed else 0)
