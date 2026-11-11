"""Helpers for handling inline ``<think>...</think>`` blocks in plugin responses.

Some OpenAI-compatible servers leave reasoning traces inline inside ``content``
wrapped in ``<think>...</think>`` tags instead of populating a separate
``reasoning_content`` field.  Leaving those blocks in place pollutes format
checkers (such as IFEval's ``no_comma`` or ``json_format``) and causes answer
extractors to match intermediate candidates from the scratchpad.
"""

from __future__ import annotations

import re

from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.plugin import final_answer_text

_LEADING_THINK_RE = re.compile(
    r"^\s*<think(?:\s[^>]*)?>(.*?)</think>\s*",
    flags=re.DOTALL | re.IGNORECASE,
)
_CLOSED_THINK_RE = re.compile(
    r"<think(?:\s[^>]*)?>(.*?)</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
_OPEN_THINK_RE = re.compile(r"<think(?:\s[^>]*)?>", flags=re.IGNORECASE)


def strip_thinking(text: str, *, fallback_to_thought: bool = False) -> str | None:
    """Strip inline ``<think>...</think>`` blocks from *text*.

    When ``fallback_to_thought`` is ``False`` (used for ``finish_reason == "length"``),
    an unclosed ``<think>`` block or a closed ``<think>...</think>`` block with no
    non-whitespace answer outside it returns ``None`` so callers record the turn
    as truncated rather than grading half-written reasoning.
    """
    if "<think" not in text.lower():
        return text

    thoughts: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        thoughts.append(match.group(1))
        return ""

    cleaned = _LEADING_THINK_RE.sub(_collect, text, count=1)
    cleaned = _CLOSED_THINK_RE.sub(_collect, cleaned)

    open_match = _OPEN_THINK_RE.search(cleaned)
    if open_match is not None:
        prefix = cleaned[: open_match.start()]
        tail = cleaned[open_match.end() :]
        if prefix.strip():
            return prefix.rstrip()
        if not fallback_to_thought:
            return None
        thoughts.append(tail)
        cleaned = ""

    if cleaned.strip():
        return cleaned
    if not fallback_to_thought:
        return None
    joined = "\n".join(t for t in thoughts if t.strip())
    return joined if joined else cleaned


def plugin_answer_text(result: ChatCompletionResult) -> str | None:
    """Return the gradable final answer from *result*, or ``None`` if truncated."""
    raw = final_answer_text(result)
    if raw is None:
        return None
    return strip_thinking(raw, fallback_to_thought=(result.finish_reason != "length"))
