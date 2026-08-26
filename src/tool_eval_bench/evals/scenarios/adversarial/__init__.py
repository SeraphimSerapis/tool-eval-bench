"""Adversarial diversity scenario pack — expanded safety testing.

TC-57 to TC-60: 4 new scenarios expanding Category K (Safety & Boundaries)
with diverse attack vectors beyond the existing prompt injection scenarios.
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
