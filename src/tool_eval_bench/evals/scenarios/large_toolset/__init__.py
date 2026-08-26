"""Large-toolset scenario pack — tests tool selection at scale.

TC-37 to TC-40: 4 scenarios in Category L (Toolset Scale) that evaluate
model ability to navigate a crowded 52-tool namespace.
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
