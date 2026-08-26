"""Core 15 ToolCall benchmark scenarios + central scenario registry.

The 15 base scenarios (TC-01 to TC-15) are defined here, ported from
ToolCall-15 (MIT License, https://github.com/stevibe/ToolCall-15).

Extended, agentic, large-toolset, and structured output scenario packs are
imported at the bottom to build ALL_SCENARIOS (69 total across 15 categories).
Hard Mode scenarios (Category P) are available via ALL_SCENARIOS_WITH_HARDMODE
(88 total across 16 categories, opt-in with ``--hardmode``).
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
