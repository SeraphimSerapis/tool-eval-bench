"""IFEval evaluator — prompt-level and instruction-level scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tool_eval_bench.plugins._thinking import strip_thinking
from tool_eval_bench.plugins.ifeval.checkers import check_instruction


@dataclass(slots=True)
class InstructionResult:
    """Result of checking a single instruction constraint."""

    instruction_id: str
    passed: bool
    error: str | None = None


@dataclass(slots=True)
class PromptResult:
    """Result of evaluating all constraints for a single prompt."""

    prompt_pass: bool  # True only if ALL instructions passed
    instruction_results: list[InstructionResult] = field(default_factory=list)

    @property
    def instructions_passed(self) -> int:
        return sum(1 for r in self.instruction_results if r.passed)

    @property
    def instructions_total(self) -> int:
        return len(self.instruction_results)


def evaluate_prompt(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict[str, Any]],
    *,
    prompt: str | None = None,
) -> PromptResult:
    """Evaluate a model response against all constraints for a prompt.

    Parameters
    ----------
    response
        The model's raw text response.
    instruction_ids
        List of instruction IDs (e.g. ``["punctuation:no_comma", ...]``).
    kwargs_list
        List of kwargs dicts, one per instruction.
    prompt
        Optional source prompt.  Some IFEval constraints, such as
        ``detectable_format:constrained_response``, keep their contract in
        the prompt instead of in the kwargs payload.
    """
    cleaned_response = strip_thinking(response, fallback_to_thought=True) or ""
    results: list[InstructionResult] = []

    for i, inst_id in enumerate(instruction_ids):
        raw_kw = kwargs_list[i] if i < len(kwargs_list) else {}
        kw = {k: v for k, v in raw_kw.items() if v is not None} if isinstance(raw_kw, dict) else {}
        if prompt is not None:
            kw.setdefault("prompt", prompt)

        try:
            passed = check_instruction(inst_id, cleaned_response, kw)
            results.append(
                InstructionResult(
                    instruction_id=inst_id,
                    passed=passed,
                )
            )
        except KeyError:
            # Unknown instructions are unsupported constraints, not
            # successful evaluations.  Passing them would inflate benchmark
            # scores and hide gaps in the checker registry.
            results.append(
                InstructionResult(
                    instruction_id=inst_id,
                    passed=False,
                    error=f"Unknown instruction: {inst_id}",
                )
            )
        except (TypeError, ValueError, re.error) as exc:
            results.append(
                InstructionResult(
                    instruction_id=inst_id,
                    passed=False,
                    error=f"Checker error ({type(exc).__name__}): {exc}",
                )
            )

    all_passed = all(r.passed for r in results)
    return PromptResult(prompt_pass=all_passed, instruction_results=results)
