"""Extended scenario packs — localization + ClawEval-inspired tests.

TC-16 to TC-21: adds 6 new scenarios in 2 new categories:
  F — Localization (language handling, timezone awareness)
  G — Structured Reasoning (ClawEval-inspired: routing, state tracking, constraint checking)
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
