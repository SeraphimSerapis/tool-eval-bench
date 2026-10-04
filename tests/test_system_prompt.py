"""Canonical form of a system prompt override.

The same words must produce the same run whichever entry point supplied them, so
these rules are enforced in one place and asserted here directly.
"""

from __future__ import annotations

import pytest

from tool_eval_bench.utils.system_prompt import MAX_SYSTEM_PROMPT_BYTES, normalize_system_prompt


def test_plain_text_is_returned_unchanged() -> None:
    assert normalize_system_prompt("You are a strict evaluator.") == "You are a strict evaluator."


def test_trailing_whitespace_is_dropped() -> None:
    """A file ends in a newline, an inline flag does not: one cohort, not two."""
    assert normalize_system_prompt("Be strict.\n") == "Be strict."
    assert normalize_system_prompt("Be strict.  \n\n") == "Be strict."


def test_leading_byte_order_mark_is_dropped() -> None:
    assert normalize_system_prompt("\ufeffBe strict.") == "Be strict."


def test_internal_whitespace_is_preserved() -> None:
    """Only the edges are normalized; the prompt's own shape is the user's."""
    assert normalize_system_prompt("  Rules:\n  - one\n  - two\n") == "  Rules:\n  - one\n  - two"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t ", "\ufeff  \n"])
def test_blank_prompts_are_rejected(blank: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_system_prompt(blank)


def test_prompt_at_the_limit_is_accepted() -> None:
    assert len(normalize_system_prompt("x" * MAX_SYSTEM_PROMPT_BYTES).encode("utf-8")) == (
        MAX_SYSTEM_PROMPT_BYTES
    )


def test_oversized_prompt_is_rejected() -> None:
    """The text is persisted in the run config and metadata, so it is bounded."""
    with pytest.raises(ValueError, match="exceeds"):
        normalize_system_prompt("x" * (MAX_SYSTEM_PROMPT_BYTES + 1))


def test_the_limit_counts_utf8_bytes_not_characters() -> None:
    """A multi-byte prompt that fits in characters can still blow the byte budget."""
    with pytest.raises(ValueError, match="exceeds"):
        normalize_system_prompt("\u00e9" * MAX_SYSTEM_PROMPT_BYTES)
