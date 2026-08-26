"""Agentic scenario packs — instruction following, context tracking, code patterns, safety.

TC-22 to TC-36, TC-41 to TC-50: adds 25 scenarios across 4 categories:
  H — Instruction Following (format compliance, tool prohibition, multi-constraint)
  I — Context & State Tracking (cross-reference, state consistency, deduplication)
  J — Code-Specific Patterns (read-before-write, explain-vs-execute, chained execution)
  K — Safety & Boundaries (ambiguity, scope limitation, hallucination, prompt injection,
      contradictory parameters, missing required info, schema validation)
"""

from __future__ import annotations

from tool_eval_bench.evals.scenarios._registry import collect_group

SCENARIOS, DISPLAY_DETAILS = collect_group(__name__, __path__)
