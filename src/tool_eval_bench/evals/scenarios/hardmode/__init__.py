"""Hard Mode scenarios — ceiling-breaking difficulty for top-scoring models.

These scenarios are NOT included in the standard 69-scenario benchmark.
Enable with ``--hardmode`` to add them to the run, or run them exclusively
with ``--hardmode --categories P``.

Design principles:
- Longer multi-turn chains with state accumulation
- Ambiguous/underspecified requests requiring clarification
- Adversarial tool definitions (near-duplicates, misleading names)
- Compositional reasoning under simultaneous constraints
- Multi-step error recovery cascades

ID range: TC-70+
Category: P (Hard Mode)
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
