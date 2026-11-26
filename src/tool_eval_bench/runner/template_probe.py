"""Chat-template replay fidelity and special-token leak probe.

Runs four compact, deterministic requests against a live endpoint to separate
serving-stack Jinja chat-template bugs from model capability or endpoint
failures:

1. ``tool_replay`` — replays a user -> assistant(tool_calls) -> tool(result)
   exchange and checks whether the assistant surfaces the tool result code.
2. ``tool_replay_twin`` — single-turn plain twin giving the same code in a user
   message, isolating tool-response template serialization bugs.
3. ``multi_turn`` — replays a 5-message user/assistant history carrying two
   codes across turns and asks for both.
4. ``multi_turn_twin`` — single-turn plain twin giving both codes directly,
   isolating multi-turn history serialization bugs.

Every reply is also checked for leaked tokenizer special/control tokens and
unparsed native parser markers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, ChatMessage
from tool_eval_bench.domain.wire_hygiene import (
    BUILTIN_TEMPLATE_LEAK_TOKENS,
    detect_parser_marker,
    detect_special_token_leak,
    detect_unparsed_tool_call,
    is_control_special_token,
    strip_closed_think_blocks,
)
from tool_eval_bench.utils.urls import redact_urls

logger = logging.getLogger(__name__)

_TOOL_REPLAY_CODE = "ALPHA-7942"
_MULTI_TURN_CODE_1 = "COBALT-41"
_MULTI_TURN_CODE_2 = "AMBER-88"

_INITIAL_MAX_TOKENS = 128
_RETRY_MAX_TOKENS = 2048

_TEMPLATE_PROBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_code",
            "description": "Look up the current verification code.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
]


@dataclass(frozen=True)
class _ProbeCase:
    name: str
    messages: tuple[ChatMessage, ...]
    expected: tuple[str, ...]
    tools: list[dict[str, Any]] | None = None


_PROBE_CASES: tuple[_ProbeCase, ...] = (
    _ProbeCase(
        name="tool_replay",
        messages=(
            {
                "role": "user",
                "content": "Look up the verification code using get_code.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_template_probe_1",
                        "type": "function",
                        "function": {
                            "name": "get_code",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_template_probe_1",
                "name": "get_code",
                "content": json.dumps({"code": _TOOL_REPLAY_CODE}),
            },
        ),
        expected=(_TOOL_REPLAY_CODE,),
        tools=_TEMPLATE_PROBE_TOOLS,
    ),
    _ProbeCase(
        name="tool_replay_twin",
        messages=(
            {
                "role": "user",
                "content": (
                    f"The verification code is {_TOOL_REPLAY_CODE}. "
                    "Reply with only that verification code."
                ),
            },
        ),
        expected=(_TOOL_REPLAY_CODE,),
    ),
    _ProbeCase(
        name="multi_turn",
        messages=(
            {
                "role": "user",
                "content": f"Remember the first code: {_MULTI_TURN_CODE_1}. Reply OK.",
            },
            {
                "role": "assistant",
                "content": "OK",
            },
            {
                "role": "user",
                "content": f"Remember the second code: {_MULTI_TURN_CODE_2}. Reply OK.",
            },
            {
                "role": "assistant",
                "content": "OK",
            },
            {
                "role": "user",
                "content": "Reply with both codes you were asked to remember.",
            },
        ),
        expected=(_MULTI_TURN_CODE_1, _MULTI_TURN_CODE_2),
    ),
    _ProbeCase(
        name="multi_turn_twin",
        messages=(
            {
                "role": "user",
                "content": (
                    f"The first code is {_MULTI_TURN_CODE_1} and the second code is "
                    f"{_MULTI_TURN_CODE_2}. Reply with both codes."
                ),
            },
        ),
        expected=(_MULTI_TURN_CODE_1, _MULTI_TURN_CODE_2),
    ),
)


@dataclass(frozen=True)
class TemplateProbeCheck:
    """Outcome of a single chat-template fidelity check."""

    name: str
    passed: bool
    content_matched: bool
    expected: tuple[str, ...]
    detail: str
    leaked_tokens: tuple[str, ...] = ()
    unparsed_marker: str | None = None
    parser_family: str | None = None
    response_preview: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "content_matched": self.content_matched,
            "expected": list(self.expected),
            "detail": self.detail,
        }
        if self.leaked_tokens:
            payload["leaked_tokens"] = list(self.leaked_tokens)
        if self.unparsed_marker is not None:
            payload["unparsed_marker"] = self.unparsed_marker
        if self.parser_family is not None:
            payload["parser_family"] = self.parser_family
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ChatTemplateProbe:
    """Combined results and causal diagnosis across all template fidelity checks."""

    passed: bool
    diagnosis: str
    detail: str
    checks: tuple[TemplateProbeCheck, ...]
    leaked_tokens: tuple[str, ...] = ()
    unparsed_marker: str | None = None
    parser_family: str | None = None
    diagnoses: tuple[str, ...] = ()
    tokenizer_tokens_checked: int = 0

    def check_by_name(self, name: str) -> TemplateProbeCheck | None:
        """Look up an individual check result by name."""
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "passed": self.passed,
            "diagnosis": self.diagnosis,
            "detail": self.detail,
            "checks": [check.to_dict() for check in self.checks],
            "tokenizer_tokens_checked": self.tokenizer_tokens_checked,
        }
        if self.diagnoses:
            payload["diagnoses"] = list(self.diagnoses)
        if self.leaked_tokens:
            payload["leaked_tokens"] = list(self.leaked_tokens)
        if self.unparsed_marker is not None:
            payload["unparsed_marker"] = self.unparsed_marker
        if self.parser_family is not None:
            payload["parser_family"] = self.parser_family
        return payload


def _strip_probe_max_tokens(extra_params: dict[str, Any] | None) -> dict[str, Any]:
    """Remove caller token ceilings so the probe owns its compact token budget."""
    probe_extra = dict(extra_params or {})
    for key in ("max_tokens", "max_completion_tokens", "maxOutputTokens"):
        probe_extra.pop(key, None)
    generation_config = probe_extra.get("generationConfig")
    if isinstance(generation_config, dict) and "maxOutputTokens" in generation_config:
        probe_extra["generationConfig"] = {
            k: v for k, v in generation_config.items() if k != "maxOutputTokens"
        }
    return probe_extra


def _preview(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _evaluate_reply(
    case: _ProbeCase,
    result: ChatCompletionResult,
    leak_tokens: frozenset[str],
) -> TemplateProbeCheck:
    """Evaluate one completion against expected tokens, special-token leaks, and parser markers."""
    if result.transport_error_status is not None:
        msg = f"request rejected with HTTP {result.transport_error_status}"
        return TemplateProbeCheck(
            name=case.name,
            passed=False,
            content_matched=False,
            expected=case.expected,
            detail=msg,
            error=msg,
        )
    if result.malformed:
        msg = "response was not valid JSON"
        return TemplateProbeCheck(
            name=case.name,
            passed=False,
            content_matched=False,
            expected=case.expected,
            detail=msg,
            error=msg,
        )

    raw_content = result.content or ""
    visible = strip_closed_think_blocks(raw_content)
    lower_visible = visible.lower()
    missing = [token for token in case.expected if token.lower() not in lower_visible]
    content_matched = not missing

    leaked = tuple(detect_special_token_leak(raw_content, leak_tokens))
    parser_match = (
        detect_unparsed_tool_call(raw_content, case.tools)
        if case.tools
        else detect_parser_marker(raw_content)
    )
    unparsed_marker = parser_match.marker if parser_match is not None else None
    parser_family = parser_match.parser_family if parser_match is not None else None

    passed = content_matched and not leaked and unparsed_marker is None
    issues: list[str] = []
    if not content_matched:
        if result.tool_calls and not visible.strip():
            called = ", ".join(tc.name for tc in result.tool_calls)
            issues.append(
                f"re-called tool ({called}) instead of replying with {', '.join(case.expected)}"
            )
        elif result.finish_reason == "length" and not visible.strip():
            issues.append("exhausted token budget without emitting a visible answer")
        else:
            issues.append(f"missing expected {', '.join(missing)}")
    if leaked:
        issues.append(f"leaked special token(s): {', '.join(leaked)}")
    if unparsed_marker is not None and unparsed_marker not in leaked:
        issues.append(f"emitted unparsed parser marker {unparsed_marker}")

    if passed:
        detail = f"matched {', '.join(case.expected)}"
    else:
        detail = "; ".join(issues)

    return TemplateProbeCheck(
        name=case.name,
        passed=passed,
        content_matched=content_matched,
        expected=case.expected,
        detail=detail,
        leaked_tokens=leaked,
        unparsed_marker=unparsed_marker,
        parser_family=parser_family,
        response_preview=_preview(visible or raw_content),
    )


async def _run_case(
    adapter: BackendAdapter,
    case: _ProbeCase,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
    probe_extra: dict[str, Any],
    leak_tokens: frozenset[str],
) -> TemplateProbeCheck:
    """Execute one probe case, retrying once on reasoning truncation."""
    last_check: TemplateProbeCheck | None = None
    for max_tokens in (_INITIAL_MAX_TOKENS, _RETRY_MAX_TOKENS):
        try:
            result = await adapter.chat_completion(
                model=model,
                messages=list(case.messages),
                tools=case.tools,
                tool_choice="auto" if case.tools else None,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                api_key=api_key,
                base_url=base_url,
                extra_params=probe_extra or None,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 - a failed probe reports an endpoint failure
            redacted = redact_urls(str(exc))
            logger.warning("chat-template probe %s failed: %s", case.name, redacted)
            msg = f"probe request failed: {redacted}"
            return TemplateProbeCheck(
                name=case.name,
                passed=False,
                content_matched=False,
                expected=case.expected,
                detail=msg,
                error=msg,
            )

        last_check = _evaluate_reply(case, result, leak_tokens)
        should_retry = (
            result.finish_reason == "length"
            and result.transport_error_status is None
            and not result.malformed
            and not last_check.content_matched
            and not last_check.leaked_tokens
            and last_check.unparsed_marker is None
        )
        if not should_retry:
            return last_check

    assert last_check is not None  # noqa: S101 - loop always executes at least once
    return last_check


def _diagnose(
    checks: tuple[TemplateProbeCheck, ...],
    *,
    tokenizer_tokens_checked: int,
) -> ChatTemplateProbe:
    """Derive a causal diagnosis from the 4 paired replay/twin checks."""
    by_name = {check.name: check for check in checks}
    tool_replay = by_name["tool_replay"]
    tool_twin = by_name["tool_replay_twin"]
    multi_turn = by_name["multi_turn"]
    multi_twin = by_name["multi_turn_twin"]

    all_leaked = tuple(sorted({tok for check in checks for tok in check.leaked_tokens}))
    first_marker: str | None = None
    first_family: str | None = None
    for check in checks:
        if check.unparsed_marker is not None and first_marker is None:
            first_marker = check.unparsed_marker
        if check.parser_family is not None and first_family is None:
            first_family = check.parser_family

    passed = all(check.passed for check in checks)
    if passed:
        return ChatTemplateProbe(
            passed=True,
            diagnosis="ok",
            detail="Chat-template replay and special-token hygiene checks passed.",
            checks=checks,
            tokenizer_tokens_checked=tokenizer_tokens_checked,
        )

    tool_replay_error = (not tool_replay.content_matched) and tool_twin.content_matched
    multi_turn_error = (not multi_turn.content_matched) and multi_twin.content_matched
    has_leak = bool(all_leaked) or first_marker is not None
    twin_failure = (not tool_twin.content_matched) or (not multi_twin.content_matched)

    diagnoses: list[str] = []
    reasons: list[str] = []
    if tool_replay_error:
        diagnoses.append("tool_replay_template_error")
        reasons.append(
            "server chat template dropped or mangled tool-response messages "
            "(tool_replay failed while tool_replay_twin passed)"
        )
    if multi_turn_error:
        diagnoses.append("multi_turn_template_error")
        reasons.append(
            "server chat template dropped or mangled multi-turn history "
            "(multi_turn failed while multi_turn_twin passed)"
        )
    if has_leak:
        diagnoses.append("template_token_leak")
        leak_items: list[str] = list(all_leaked)
        if first_marker is not None and first_marker not in leak_items:
            leak_items.append(first_marker)
        reasons.append(
            f"special or parser control tokens leaked into visible output ({', '.join(leak_items)})"
        )
    if twin_failure or not diagnoses:
        diagnoses.append("model_or_endpoint_failure")
        if twin_failure:
            reasons.append(
                "single-turn baseline twin check failed, pointing to a model or endpoint failure "
                "rather than a chat-template replay bug"
            )
        else:
            reasons.append("template probe check failed")

    return ChatTemplateProbe(
        passed=False,
        diagnosis=diagnoses[0],
        detail="; ".join(reasons),
        checks=checks,
        leaked_tokens=all_leaked,
        unparsed_marker=first_marker,
        parser_family=first_family,
        diagnoses=tuple(diagnoses),
        tokenizer_tokens_checked=tokenizer_tokens_checked,
    )


async def probe_chat_template_fidelity(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    extra_params: dict[str, Any] | None = None,
    special_tokens: Iterable[str] = (),
) -> ChatTemplateProbe:
    """Run paired chat-template replay and special-token leak checks against *adapter*."""
    probe_extra = _strip_probe_max_tokens(extra_params)
    caller_tokens = frozenset(
        token
        for token in special_tokens
        if isinstance(token, str) and is_control_special_token(token)
    )
    leak_tokens = BUILTIN_TEMPLATE_LEAK_TOKENS | caller_tokens

    results: list[TemplateProbeCheck] = []
    for case in _PROBE_CASES:
        results.append(
            await _run_case(
                adapter,
                case,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                probe_extra=probe_extra,
                leak_tokens=leak_tokens,
            )
        )

    return _diagnose(
        tuple(results),
        tokenizer_tokens_checked=len(caller_tokens),
    )
