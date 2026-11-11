"""Compatibility re-exports; inline ``<think>`` handling lives in ``domain.plugin``."""

from __future__ import annotations

from tool_eval_bench.domain.plugin import (
    final_answer_text,
    strip_thinking,
)
from tool_eval_bench.domain.plugin import (
    final_answer_text as plugin_answer_text,
)

__all__ = ["final_answer_text", "plugin_answer_text", "strip_thinking"]
