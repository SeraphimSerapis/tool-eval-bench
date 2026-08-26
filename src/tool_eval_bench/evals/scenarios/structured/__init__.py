"""Category O — Structured Output scenarios (TC-64 to TC-69).

Tests whether models can produce valid JSON conforming to a given schema,
chain structured output with tool calls, handle nested schemas, and
refuse to produce output that violates schema constraints.

These scenarios use ``response_format`` with ``json_schema`` type to test
structured generation — a critical capability for agentic pipelines where
downstream systems expect machine-parseable output.
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
