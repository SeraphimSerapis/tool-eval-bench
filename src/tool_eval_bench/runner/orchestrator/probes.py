"""Diagnostic tool-call and tool_choice wire probes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult
from tool_eval_bench.domain.wire_hygiene import (
    detect_unparsed_tool_call,
    format_tool_parser_advice,
)
from tool_eval_bench.runner.orchestrator.wire import DEFAULT_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# A trivial call for checking the endpoint's required-tool-call response path.
# Success does not prove server-side enforcement rather than model cooperation.
_TOOL_CHOICE_PROBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "probe_ping",
            "description": "Acknowledge the probe. Call this with any value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
]


@dataclass(frozen=True)
class ToolChoiceProbe:
    """What the forced-call probe established about the endpoint.

    ``enforced`` is a legacy name for whether a tool call came back at all,
    not proof of server-side enforcement. False also covers inconclusive
    probes; ``detail`` explains why. ``arguments_intact``
    is whether that call carried the schema's required argument: an endpoint
    whose tool parser drops arguments under a forced call produces ``{}``,
    which a scenario would otherwise grade as the model sending an empty
    expression. ``detail`` is the sentence the report shows.
    """

    enforced: bool
    arguments_intact: bool
    detail: str


@dataclass(frozen=True)
class ToolCallWireProbe:
    """What the auto tool-call wire probe established about parser hygiene."""

    parsed: bool
    arguments_intact: bool
    detail: str
    unparsed_marker: str | None = None
    parser_family: str | None = None

    @property
    def enforced(self) -> bool:
        """Compatibility alias matching ToolChoiceProbe.enforced."""
        return self.parsed


@dataclass(frozen=True)
class _PingProbeOutcome:
    """Raw outcome from the shared probe_ping request/retry loop."""

    result: ChatCompletionResult | None
    attempts: tuple[str, ...]
    error: Exception | None = None


def _strip_probe_max_tokens(extra_params: dict[str, Any] | None) -> dict[str, Any]:
    """Remove caller token ceilings so a probe owns its retry budget."""
    probe_extra = dict(extra_params or {})
    for key in ("max_tokens", "max_completion_tokens", "maxOutputTokens"):
        probe_extra.pop(key, None)
    generation_config = probe_extra.get("generationConfig")
    if isinstance(generation_config, dict) and "maxOutputTokens" in generation_config:
        probe_extra["generationConfig"] = {
            k: v for k, v in generation_config.items() if k != "maxOutputTokens"
        }
    return probe_extra


async def _run_ping_probe(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
    tool_choice: str,
    extra_params: dict[str, Any] | None,
    attempt_style: Literal["paren", "colon"],
) -> _PingProbeOutcome:
    """Execute the shared 2-budget ``probe_ping`` request loop."""
    probe_extra = _strip_probe_max_tokens(extra_params)
    attempts: list[str] = []
    result: ChatCompletionResult | None = None
    for max_tokens in (256, 4096):
        label = "probe" if not attempts else "retry"
        try:
            result = await adapter.chat_completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Call probe_ping with value 'ok'.",
                    }
                ],
                tools=_TOOL_CHOICE_PROBE_TOOLS,
                tool_choice=tool_choice,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                api_key=api_key,
                base_url=base_url,
                extra_params=probe_extra or None,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 - a failed probe is inconclusive
            if attempt_style == "paren":
                logger.warning(
                    "tool_choice=required %s failed (%s); probe inconclusive", label, exc
                )
                attempts.append(f"{label}(max_tokens={max_tokens}, request failed)")
            else:
                logger.warning("tool-call wire %s failed: %s; probe inconclusive", label, exc)
                attempts.append(f"{label}: max_tokens={max_tokens}, request failed")
            return _PingProbeOutcome(result=None, attempts=tuple(attempts), error=exc)

        if attempt_style == "paren":
            attempts.append(
                f"{label}(max_tokens={max_tokens}, finish_reason={result.finish_reason}, "
                f"completion_tokens={result.completion_tokens})"
            )
        else:
            attempts.append(
                f"{label}: max_tokens={max_tokens}, finish_reason={result.finish_reason}, "
                f"completion_tokens={result.completion_tokens}"
            )
        if (
            result.transport_error_status is not None
            or result.malformed
            or result.tool_calls
            or result.finish_reason != "length"
        ):
            break

    return _PingProbeOutcome(result=result, attempts=tuple(attempts))


def _ping_arguments_intact(result: ChatCompletionResult) -> bool:
    """Return whether the first tool call is ``probe_ping`` with a non-empty ``value``."""
    call = result.tool_calls[0]
    return call.name == "probe_ping" and bool(str(call.arguments.get("value", "")).strip())


async def probe_tool_choice_required(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
    extra_params: dict[str, Any] | None = None,
) -> ToolChoiceProbe:
    """Check for a parsed call with ``tool_choice="required"``, without gating scoring.

    The prompt agrees with the wire constraint. Contradictory instructions can
    derail reasoning models even when the scenario's own request works.
    A successful probe confirms a parsed call, not server-side enforcement;
    a failed probe cannot establish that a different prompt would fail.
    Retry truncation once because reasoning can consume the initial budget.
    """
    outcome = await _run_ping_probe(
        adapter,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        tool_choice="required",
        extra_params=extra_params,
        attempt_style="paren",
    )
    attempts_suffix = "; ".join(outcome.attempts)
    if outcome.error is not None or outcome.result is None:
        return ToolChoiceProbe(
            False,
            False,
            f"inconclusive; probe request failed ({outcome.error}); {attempts_suffix}",
        )

    result = outcome.result
    enforced = False
    arguments_intact = False
    if result.transport_error_status is not None:
        detail = f"probe request rejected with HTTP {result.transport_error_status}"
    elif result.malformed:
        detail = "inconclusive; probe response was not valid JSON"
    elif not result.tool_calls:
        unparsed_match = detect_unparsed_tool_call(result.content, _TOOL_CHOICE_PROBE_TOOLS)
        if unparsed_match is not None:
            advice = format_tool_parser_advice(parser_family=unparsed_match.parser_family)
            if unparsed_match.parser_family == "raw_json":
                detail = (
                    f"unverified; the endpoint emitted unparsed {unparsed_match.marker} "
                    f"in content instead of structured tool_calls. {advice}"
                )
            else:
                detail = (
                    f"unverified; the endpoint emitted raw unparsed tool-call markup "
                    f"{unparsed_match.marker} in content instead of structured tool_calls. "
                    f"{advice}"
                )
        elif result.finish_reason == "length":
            detail = "inconclusive; probe exhausted both token budgets without a tool call"
            # Truncated text may be prose or a call cut off before it parsed,
            # so it is reported, not read as non-enforcement.
            if result.content.strip():
                detail += ", leaving partial text"
        elif result.content.strip():
            detail = "unverified; the endpoint answered in prose instead of calling the tool"
        else:
            detail = "inconclusive; probe returned no tool call or answer"
    else:
        enforced = True
        arguments_intact = _ping_arguments_intact(result)
        if arguments_intact:
            detail = "parsed tool call observed with tool_choice='required'; probe call carried arguments"
        else:
            detail = (
                "parsed tool call observed with tool_choice='required', but it arrived with empty arguments; "
                "empty-argument calls in this scenario may be the tool parser rather than the model"
            )
    return ToolChoiceProbe(enforced, arguments_intact, f"{detail}; {attempts_suffix}")


async def probe_tool_call_parsing(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    extra_params: dict[str, Any] | None = None,
) -> ToolCallWireProbe:
    """Check whether a simple auto tool-call request returns structured tool_calls."""
    outcome = await _run_ping_probe(
        adapter,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        tool_choice="auto",
        extra_params=extra_params,
        attempt_style="colon",
    )
    attempts_suffix = "; ".join(outcome.attempts)
    if outcome.error is not None or outcome.result is None:
        return ToolCallWireProbe(
            False,
            False,
            f"inconclusive; probe request failed with {outcome.error}; {attempts_suffix}",
        )

    result = outcome.result
    parsed = False
    arguments_intact = False
    if result.transport_error_status is not None:
        detail = f"probe request rejected with HTTP {result.transport_error_status}"
    elif result.malformed:
        detail = "inconclusive; probe response was not valid JSON"
    elif not result.tool_calls:
        unparsed_match = detect_unparsed_tool_call(result.content, _TOOL_CHOICE_PROBE_TOOLS)
        if unparsed_match is not None:
            advice = format_tool_parser_advice(parser_family=unparsed_match.parser_family)
            if unparsed_match.parser_family == "raw_json":
                detail = (
                    f"unparsed tool call; the endpoint emitted {unparsed_match.marker} "
                    f"in content instead of structured tool_calls. {advice}"
                )
            else:
                detail = (
                    f"unparsed tool call; the endpoint emitted raw tool-call markup "
                    f"{unparsed_match.marker} in content instead of structured tool_calls. "
                    f"{advice}"
                )
            return ToolCallWireProbe(
                False,
                False,
                f"{detail}; {attempts_suffix}",
                unparsed_marker=unparsed_match.marker,
                parser_family=unparsed_match.parser_family,
            )
        if result.finish_reason == "length":
            detail = "inconclusive; probe exhausted both token budgets without a tool call"
            if result.content.strip():
                detail += ", leaving partial text"
        elif result.content.strip():
            detail = "no tool call; the model answered in prose instead of calling probe_ping"
        else:
            detail = "inconclusive; probe returned no tool call or answer"
    else:
        parsed = True
        arguments_intact = _ping_arguments_intact(result)
        if arguments_intact:
            detail = "structured tool_calls parsed with intact arguments"
        else:
            detail = "structured tool_calls parsed, but probe_ping arrived with empty arguments"
    return ToolCallWireProbe(parsed, arguments_intact, f"{detail}; {attempts_suffix}")


async def supports_tool_choice_required(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
    extra_params: dict[str, Any] | None = None,
) -> bool:
    """Compatibility wrapper: whether a required-tool probe returned a parsed call."""
    probe = await probe_tool_choice_required(
        adapter,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        extra_params=extra_params,
    )
    return probe.enforced
