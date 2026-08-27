"""Needle-in-a-haystack retrieval benchmark."""

from __future__ import annotations

from tool_eval_bench.plugins.needle.haystack import (
    NeedleCase,
    build_cases,
    build_needle_messages,
    grade_response,
)
from tool_eval_bench.plugins.needle.plugin import NeedlePlugin

__all__ = [
    "NeedleCase",
    "NeedlePlugin",
    "build_cases",
    "build_needle_messages",
    "grade_response",
]
