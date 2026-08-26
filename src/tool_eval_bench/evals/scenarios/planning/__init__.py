"""Autonomous planning & creative composition scenarios.

TC-51 to TC-56: 6 scenarios in Categories M (Autonomous Planning)
and N (Creative Composition) that test whether models can decompose
goals into tool chains without step-by-step guidance, and combine
tools in non-obvious ways.

TC-61: Async polling scenario (Category C expansion).
TC-62: 5-turn deep research (Category I expansion).
TC-63: Accumulating constraints (Category I expansion).
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
