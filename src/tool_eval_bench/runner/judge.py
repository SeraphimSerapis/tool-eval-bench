"""LLM-as-Judge re-evaluation for failed scenarios.

When enabled via ``--llm-judge``, scenarios that received a FAIL verdict from
the deterministic evaluator are re-evaluated by an LLM judge. The judge can
upgrade FAIL → PARTIAL (never FAIL → PASS) when it determines the model's
behavior was partially correct despite not matching the deterministic evaluator's
string-matching patterns.

This preserves the deterministic-first design while catching false negatives
from fragile string matching and refusal detection.
"""

from __future__ import annotations

import json
import logging



from tool_eval_bench.adapters.base import BackendAdapter
from tool_eval_bench.domain.scenarios import (
    ScenarioDefinition,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a benchmark evaluation judge. Your job is to determine whether a model's
response was PARTIALLY correct, even though an automated evaluator marked it as FAIL.

You will be given:
1. The scenario description (what the model was asked to do)
2. The expected behavior (what a correct response looks like)
3. The tool calls the model made (or didn't make)
4. The model's final answer
5. The automated evaluator's failure reason

Your task: Determine if the model's behavior showed PARTIAL understanding of the
task — i.e., it made meaningful progress but didn't fully satisfy the evaluator.

Rules:
- You can ONLY upgrade a FAIL to PARTIAL (1 point). You cannot upgrade to PASS.
- If the model truly failed (wrong tool, hallucinated data, completely missed the point),
  keep the FAIL verdict.
- If the model showed the right intent but used different phrasing, different tool
  arguments that are semantically equivalent, or a valid alternative approach,
  upgrade to PARTIAL.
- Be conservative: when in doubt, keep FAIL.

Respond with EXACTLY one JSON object (no other text):
{
  "verdict": "partial" or "fail",
  "reason": "One-sentence explanation of your judgment"
}"""


def _build_judge_prompt(
    scenario: ScenarioDefinition,
    result: ScenarioResult,
    state: ScenarioState,
) -> str:
    """Build the user message for the judge LLM."""
    tool_calls_str = "No tool calls made."
    if state.tool_calls:
        calls = []
        for tc in state.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            calls.append(f"  - {tc.name}({args_str}) [turn {tc.turn}]")
        tool_calls_str = "\n".join(calls)

    tool_results_str = "No tool results."
    if state.tool_results:
        results = []
        for tr in state.tool_results:
            result_str = json.dumps(tr.result, ensure_ascii=False)
            if len(result_str) > 500:
                result_str = result_str[:497] + "..."
            results.append(f"  - {tr.name}: {result_str}")
        tool_results_str = "\n".join(results)

    return f"""\
## Scenario: {scenario.id} — {scenario.title}

**User message:** {scenario.user_message}

**Expected behavior:** {scenario.description}

**Tool calls made:**
{tool_calls_str}

**Tool results:**
{tool_results_str}

**Model's final answer:**
{state.final_answer}

**Automated evaluator verdict:** FAIL
**Evaluator reason:** {result.summary}

---

Should this be upgraded from FAIL to PARTIAL? Respond with a JSON object."""


# ---------------------------------------------------------------------------
# Judge execution
# ---------------------------------------------------------------------------

async def judge_failed_scenarios(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None = None,
    scenarios: list[ScenarioDefinition],
    results: list[ScenarioResult],
    states: dict[str, ScenarioState],
    judge_model: str | None = None,
    timeout_seconds: float = 60.0,
) -> list[ScenarioResult]:
    """Re-evaluate FAIL results using an LLM judge.

    Returns a new list of ScenarioResult with upgraded verdicts where
    the judge agreed. Original PASS and PARTIAL results are unchanged.

    Args:
        adapter: Backend adapter for API calls.
        model: The model being benchmarked (used as default judge).
        base_url: Server URL.
        api_key: Optional API key.
        scenarios: The scenario definitions.
        results: The original results from deterministic evaluation.
        states: Mapping of scenario_id → ScenarioState from the run.
        judge_model: Model to use for judging (defaults to same model).
        timeout_seconds: Request timeout for judge calls.
    """
    judge = judge_model or model
    scenario_map = {s.id: s for s in scenarios}
    updated: list[ScenarioResult] = []
    upgrades = 0

    for result in results:
        # Only re-evaluate FAILs
        if result.status != ScenarioStatus.FAIL:
            updated.append(result)
            continue

        scenario = scenario_map.get(result.scenario_id)
        state = states.get(result.scenario_id)

        if not scenario or not state:
            updated.append(result)
            continue

        try:
            judge_result = await _call_judge(
                adapter,
                judge_model=judge,
                base_url=base_url,
                api_key=api_key,
                scenario=scenario,
                result=result,
                state=state,
                timeout_seconds=timeout_seconds,
            )

            if judge_result and judge_result.get("verdict") == "partial":
                # Upgrade FAIL → PARTIAL
                upgraded = ScenarioResult(
                    scenario_id=result.scenario_id,
                    status=ScenarioStatus.PARTIAL,
                    points=1,
                    summary=result.summary,
                    note=f"[judge override] {judge_result.get('reason', 'Judge upgraded to partial.')}",
                    raw_log=result.raw_log,
                    tool_calls_made=result.tool_calls_made,
                    expected_behavior=result.expected_behavior,
                    duration_seconds=result.duration_seconds,
                    ttft_ms=result.ttft_ms,
                    turn_count=result.turn_count,
                    turn_latencies_ms=result.turn_latencies_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    tool_call_arg_bytes=result.tool_call_arg_bytes,
                )
                updated.append(upgraded)
                upgrades += 1
                logger.info(
                    "Judge upgraded %s from FAIL to PARTIAL: %s",
                    result.scenario_id,
                    judge_result.get("reason", ""),
                )
            else:
                updated.append(result)
                logger.debug(
                    "Judge confirmed FAIL for %s: %s",
                    result.scenario_id,
                    judge_result.get("reason", "no reason") if judge_result else "judge call failed",
                )

        except Exception as exc:
            logger.warning("Judge call failed for %s: %s", result.scenario_id, exc)
            updated.append(result)

    if upgrades > 0:
        logger.info("LLM judge upgraded %d scenario(s) from FAIL to PARTIAL.", upgrades)

    return updated


async def _call_judge(
    adapter: BackendAdapter,
    *,
    judge_model: str,
    base_url: str,
    api_key: str | None,
    scenario: ScenarioDefinition,
    result: ScenarioResult,
    state: ScenarioState,
    timeout_seconds: float = 60.0,
) -> dict[str, str] | None:
    """Make a single judge API call and parse the response.

    Returns {"verdict": "partial"|"fail", "reason": "..."} or None on error.
    """
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _build_judge_prompt(scenario, result, state)},
    ]

    try:
        completion = await adapter.chat_completion(
            model=judge_model,
            messages=messages,
            tools=None,
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            base_url=base_url,
        )

        content = (completion.content or "").strip()

        # Try to extract JSON from the response
        # Handle code fences
        json_match = None
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()

        data = json.loads(content)
        verdict = data.get("verdict", "").lower()
        reason = data.get("reason", "")

        if verdict in ("partial", "fail"):
            return {"verdict": verdict, "reason": reason}

        logger.warning("Judge returned unexpected verdict: %s", verdict)
        return None

    except json.JSONDecodeError:
        logger.warning("Judge returned non-JSON response: %s", content[:200])
        return None
    except Exception as exc:
        logger.warning("Judge API call failed: %s", exc)
        return None
