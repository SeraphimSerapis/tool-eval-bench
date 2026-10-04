"""Decision-model benchmark: single-pass scoring of caller-supplied options."""

from __future__ import annotations

from tool_eval_bench.plugins.decision.dataset import DecisionItem, build_items
from tool_eval_bench.plugins.decision.plugin import DecisionPlugin

__all__ = ["DecisionItem", "DecisionPlugin", "build_items"]
