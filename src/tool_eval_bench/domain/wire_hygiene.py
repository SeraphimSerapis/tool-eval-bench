"""Detect unparsed tool-call markup in assistant text and suggest server flags.

When an OpenAI-compatible server runs without its chat-template tool parser,
models still emit their native tool-call tokens or raw JSON call objects, but
those tokens arrive in ``message.content`` with ``tool_calls`` empty.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_QWEN3_CODER_FUNCTION_RE = re.compile(r"<function=[A-Za-z_][A-Za-z0-9_.-]*[ \t\r\n]*>")
_GEMMA4_CALL_RE = re.compile(
    r"(?<![A-Za-z0-9_])call:[A-Za-z_][A-Za-z0-9_.-]*(?::[A-Za-z_][A-Za-z0-9_.-]*)?\{"
)
_FENCED_TOOL_CALL_RE = re.compile(r"```tool_calls?\b")
_JSON_START_RE = re.compile(r"\{[ \t\r\n]*\"|\[[ \t\r\n]*\{")

_PIPE_CONTROL_TOKEN_RE = re.compile(
    r"^(?:<\|[^\s<>]{1,64}\|>|<\|[^\s<>]{2,64}>|<[^\s<>]{2,64}\|>|<｜[^\s<>]{1,64}｜>)$"
)
_ANGLE_CONTROL_TOKEN_RE = re.compile(r"^</?[A-Za-z][A-Za-z0-9_.:-]{2,63}>$")
_BRACKET_CONTROL_TOKEN_RE = re.compile(r"^\[/?[A-Z][A-Z0-9_.:-]{2,63}\]$")

# Standard HTML/Markdown element names that should never be flagged as leaked
# chat-template control tokens even if a tokenizer marks them special.
_COMMON_HTML_TAGS: frozenset[str] = frozenset(
    {
        "article",
        "aside",
        "audio",
        "blockquote",
        "body",
        "button",
        "canvas",
        "code",
        "col",
        "colgroup",
        "del",
        "details",
        "dialog",
        "div",
        "figure",
        "figcaption",
        "footer",
        "form",
        "head",
        "header",
        "html",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "link",
        "main",
        "mark",
        "meta",
        "nav",
        "noscript",
        "option",
        "path",
        "pre",
        "script",
        "section",
        "select",
        "small",
        "source",
        "span",
        "strong",
        "style",
        "sub",
        "summary",
        "sup",
        "svg",
        "table",
        "tbody",
        "textarea",
        "tfoot",
        "thead",
        "title",
        "var",
        "video",
    }
)

_LITERAL_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("<|tool_calls_begin|>", "<|tool_calls_begin|>", "deepseek"),
    ("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁begin｜>", "deepseek"),
    ("<|tool_call_begin|>", "<|tool_call_begin|>", "deepseek"),
    ("<｜tool▁call▁begin｜>", "<｜tool▁call▁begin｜>", "deepseek"),
    ("<|python_tag|>", "<|python_tag|>", "llama3"),
    ("[TOOL_CALLS]", "[TOOL_CALLS]", "mistral"),
    ("<|tool_call|>", "<|tool_call|>", "hermes/qwen"),
    ("<|tool_call>", "<|tool_call>", "gemma4"),
    ("<tool_call|>", "<tool_call|>", "gemma4"),
    ("<channel|>", "<channel|>", "gemma4"),
    ("<turn|>", "<turn|>", "gemma4"),
)

# Built-in chat-template, turn, channel, and tool-call control tokens checked
# even when no local HuggingFace tokenizer.json is available.
BUILTIN_TEMPLATE_LEAK_TOKENS: frozenset[str] = frozenset(
    {
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|tool_call|>",
        "<|/tool_call|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|python_tag|>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|thought>",
        "<thought|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|tool_declaration>",
        "<tool_declaration|>",
        "[INST]",
        "[/INST]",
        "[TOOL_CALLS]",
        "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]",
        "[TOOL_RESULTS]",
        "[/TOOL_RESULTS]",
        "<|tool_calls_begin|>",
        "<|tool_calls_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁calls▁end｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁call▁end｜>",
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
    }
)

PARSER_FLAG_BY_FAMILY: dict[str, str] = {
    "hermes/qwen": "hermes",
    "qwen3_coder": "qwen3_coder",
    "mistral": "mistral",
    "llama3": "llama3_json",
    "deepseek": "deepseek_v3",
    "gemma4": "gemma4",
}


@dataclass(frozen=True)
class UnparsedToolCallMatch:
    """Raw tool-call markup or JSON call payload detected in assistant text."""

    marker: str
    parser_family: str
    tool_name: str | None = None

    def display_marker(self, *, held_out: bool = False) -> str:
        """Return a marker safe for traces and reports, redacting held-out tool names."""
        if held_out and (self.tool_name is not None or self.parser_family == "raw_json"):
            return "raw JSON tool call"
        return self.marker


def advertised_tool_names(tools: Sequence[Mapping[str, Any]] | None) -> set[str]:
    """Extract advertised function names from a scenario's tool schema list."""
    if not tools:
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        fn = tool.get("function")
        if isinstance(fn, Mapping):
            name = fn.get("name")
            if isinstance(name, str) and name:
                names.add(name)
                continue
        top_name = tool.get("name")
        if isinstance(top_name, str) and top_name:
            names.add(top_name)
    return names


def _looks_like_json_schema(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("type") == "object"
        and isinstance(value.get("properties"), Mapping)
    )


def _match_json_call_object(obj: Any, advertised_names: set[str]) -> str | None:
    if not isinstance(obj, Mapping):
        return None

    name = obj.get("name")
    if isinstance(name, str) and name in advertised_names:
        has_args = "arguments" in obj and isinstance(obj["arguments"], (Mapping, str, type(None)))
        has_params = (
            "parameters" in obj
            and isinstance(obj["parameters"], (Mapping, str, type(None)))
            and not _looks_like_json_schema(obj["parameters"])
        )
        is_fn_envelope = (
            obj.get("type") == "function"
            and "description" not in obj
            and not _looks_like_json_schema(obj.get("parameters"))
        )
        if has_args or has_params or is_fn_envelope:
            return name

    fn = obj.get("function")
    if isinstance(fn, Mapping):
        fn_name = fn.get("name")
        if isinstance(fn_name, str) and fn_name in advertised_names:
            has_fn_args = "arguments" in fn and isinstance(
                fn["arguments"], (Mapping, str, type(None))
            )
            has_fn_params = (
                "parameters" in fn
                and isinstance(fn["parameters"], (Mapping, str, type(None)))
                and not _looks_like_json_schema(fn["parameters"])
            )
            is_wrapped_fn = (
                obj.get("type") == "function"
                and "description" not in fn
                and not _looks_like_json_schema(fn.get("parameters"))
            )
            if has_fn_args or has_fn_params or is_wrapped_fn:
                return fn_name

    return None


def _match_json_tool_call_value(data: Any, advertised_names: set[str]) -> str | None:
    direct = _match_json_call_object(data, advertised_names)
    if direct is not None:
        return direct

    if isinstance(data, Mapping):
        single_tc = data.get("tool_call")
        if single_tc is not None:
            matched = _match_json_call_object(single_tc, advertised_names)
            if matched is not None:
                return matched
        tc_list = data.get("tool_calls")
        if isinstance(tc_list, list):
            for item in tc_list:
                matched = _match_json_call_object(item, advertised_names)
                if matched is not None:
                    return matched
        return None

    if isinstance(data, list):
        for item in data:
            matched = _match_json_call_object(item, advertised_names)
            if matched is not None:
                return matched

    return None


def _detect_raw_json_call(text: str, advertised_names: set[str]) -> str | None:
    if not advertised_names:
        return None
    decoder = json.JSONDecoder()
    resume_at = 0
    for candidate in _JSON_START_RE.finditer(text):
        start = candidate.start()
        if start < resume_at:
            continue
        try:
            parsed, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        resume_at = end
        matched = _match_json_tool_call_value(parsed, advertised_names)
        if matched is not None:
            return matched
    return None


def strip_closed_think_blocks(text: str) -> str:
    """Remove closed <think>...</think> blocks in one linear pass."""
    lower = text.lower()
    open_idx = lower.find("<think>")
    if open_idx == -1:
        return text
    out: list[str] = []
    cursor = 0
    while open_idx != -1:
        close_idx = lower.find("</think>", open_idx + 7)
        if close_idx == -1:
            break
        out.append(text[cursor:open_idx])
        cursor = close_idx + 8
        open_idx = lower.find("<think>", cursor)
    if cursor == 0:
        return text
    out.append(text[cursor:])
    return "".join(out)


_strip_closed_think_blocks = strip_closed_think_blocks


def is_control_special_token(token: str) -> bool:
    """Return whether *token* matches chat-template/control delimiter syntax.

    Accepts angle-delimited pipe tokens (``<|...|>``, ``<|...>``, ``<...|>``,
    and fullwidth ``<｜...｜>``), non-HTML angle tokens of length >= 5
    (such as ``<start_of_turn>``, ``<end_of_turn>``, ``<eos>``), and uppercase
    bracketed control tokens of length >= 5 (such as ``[INST]``, ``[/INST]``,
    ``[TOOL_CALLS]``). Plain words, single characters, short fragments
    (length < 4), and standard HTML/Markdown tags are rejected to avoid
    false positives in normal assistant prose.
    """
    if len(token) < 4 or any(ch.isspace() for ch in token):
        return False
    if _PIPE_CONTROL_TOKEN_RE.fullmatch(token):
        return True
    if _BRACKET_CONTROL_TOKEN_RE.fullmatch(token):
        return True
    if _ANGLE_CONTROL_TOKEN_RE.fullmatch(token):
        tag = token.lstrip("</").rstrip(">").lower()
        return tag not in _COMMON_HTML_TAGS
    return False


def detect_special_token_leak(
    content: str | None,
    special_tokens: Iterable[str],
) -> list[str]:
    """Return sorted unique control/special tokens leaked into assistant content.

    Closed ``<think>...</think>`` blocks are stripped before scanning so normal
    inline reasoning tags are not misclassified as template leaks, while stray
    unclosed reasoning tags or leaked turn/channel/tool delimiters are caught.
    """
    if not content:
        return []
    cleaned = strip_closed_think_blocks(content)
    if not cleaned:
        return []
    eligible = sorted(
        {
            token
            for token in special_tokens
            if isinstance(token, str) and is_control_special_token(token)
        }
    )
    return [token for token in eligible if token in cleaned]


def _has_tool_tag(
    text: str,
    open_tag: str,
    close_tag: str,
    *,
    allow_nested_tool_call: bool = False,
) -> bool:
    """Check for a closed tool-call XML block or an unclosed tag before payload start."""
    first_open = text.find(open_tag)
    if first_open == -1:
        return False
    open_len = len(open_tag)
    if text.find(close_tag, first_open + open_len) != -1:
        return True
    n = len(text)
    prefixes: tuple[str, ...] = (
        ("{", "[", "<function=", "<tool_call>", "<tool_call ")
        if allow_nested_tool_call
        else ("{", "[", "<function=")
    )
    idx = first_open
    while idx != -1:
        pos = idx + open_len
        while pos < n and text[pos].isspace():
            pos += 1
        if text.startswith(prefixes, pos):
            return True
        idx = text.find(open_tag, idx + open_len)
    return False


def detect_parser_marker(content: str | None) -> UnparsedToolCallMatch | None:
    """Detect native tool-call or channel parser markers regardless of advertised tools."""
    if not content:
        return None
    cleaned = strip_closed_think_blocks(content)
    if not cleaned.strip():
        return None

    for needle, marker, family in _LITERAL_MARKERS:
        if needle in cleaned:
            return UnparsedToolCallMatch(marker=marker, parser_family=family)

    if _QWEN3_CODER_FUNCTION_RE.search(cleaned):
        return UnparsedToolCallMatch(marker="<function=...>", parser_family="qwen3_coder")

    if _GEMMA4_CALL_RE.search(cleaned):
        return UnparsedToolCallMatch(marker="call:<name>{...}", parser_family="gemma4")

    if _FENCED_TOOL_CALL_RE.search(cleaned):
        return UnparsedToolCallMatch(marker="```tool_call", parser_family="hermes/qwen")

    if _has_tool_tag(cleaned, "<tool_call>", "</tool_call>"):
        return UnparsedToolCallMatch(marker="<tool_call>", parser_family="hermes/qwen")

    if _has_tool_tag(cleaned, "<tool_calls>", "</tool_calls>", allow_nested_tool_call=True):
        return UnparsedToolCallMatch(marker="<tool_calls>", parser_family="hermes/qwen")

    return None


def detect_unparsed_tool_call(
    content: str | None,
    tools: Sequence[Mapping[str, Any]] | None,
) -> UnparsedToolCallMatch | None:
    """Detect raw tool-call markup or JSON tool-call envelopes leaked into content."""
    if not content or not tools:
        return None
    marker_match = detect_parser_marker(content)
    if marker_match is not None:
        return marker_match

    cleaned = strip_closed_think_blocks(content)
    if not cleaned.strip():
        return None

    advertised = advertised_tool_names(tools)
    matched_tool = _detect_raw_json_call(cleaned, advertised)
    if matched_tool is not None:
        return UnparsedToolCallMatch(
            marker=f'raw JSON call to "{matched_tool}"',
            parser_family="raw_json",
            tool_name=matched_tool,
        )

    return None


def infer_parser_family_from_text(text: str | None) -> str | None:
    """Recover the parser family from a stored diagnostic note when unambiguous."""
    if not text:
        return None
    for needle, _, family in _LITERAL_MARKERS:
        if needle in text:
            return family
    if "<function=...>" in text:
        return "qwen3_coder"
    if "call:<name>{...}" in text or _GEMMA4_CALL_RE.search(text):
        return "gemma4"
    if "```tool_call" in text or "<tool_call>" in text or "<tool_calls>" in text:
        return "hermes/qwen"
    if 'raw JSON call to "' in text or "raw JSON tool call" in text:
        return "raw_json"
    return None


def dominant_parser_family(notes: Sequence[str | None]) -> str | None:
    """Return a single parser family when observed diagnostics agree on one."""
    specific: set[str] = set()
    saw_raw_json = False
    for note in notes:
        family = infer_parser_family_from_text(note)
        if family is None:
            continue
        if family == "raw_json":
            saw_raw_json = True
        else:
            specific.add(family)
    if len(specific) == 1:
        return next(iter(specific))
    if not specific and saw_raw_json:
        return "raw_json"
    return None


def format_tool_parser_advice(
    backend: str | None = None,
    engine_name: str | None = None,
    parser_family: str | None = None,
) -> str:
    """Format concrete server-flag guidance for enabling tool-call parsing."""
    suggested = PARSER_FLAG_BY_FAMILY.get(parser_family or "", "<parser>")
    tokens = {
        (backend or "").strip().lower(),
        (engine_name or "").strip().lower(),
    }
    if "vllm" in tokens:
        return (
            "Enable server-side tool parsing on vLLM with "
            f"--enable-auto-tool-choice --tool-call-parser {suggested}."
        )
    if "sglang" in tokens:
        return f"Enable server-side tool parsing on SGLang with --tool-call-parser {suggested}."
    if tokens & {"llamacpp", "llama.cpp", "llama_cpp"}:
        return "Enable chat-template tool parsing on llama.cpp with --jinja."
    return (
        "Enable server-side tool parsing, such as "
        f"--enable-auto-tool-choice --tool-call-parser {suggested} on vLLM, "
        f"--tool-call-parser {suggested} on SGLang, or --jinja on llama.cpp."
    )
