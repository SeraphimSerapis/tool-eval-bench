# Adding a scenario

A scenario implementation is one file. Create `src/tool_eval_bench/evals/scenarios/<group>/tcNN.py`, export a
`SCENARIO` and a `DISPLAY`, and it is registered. Add its reference trace to `tests/fixtures/scenario_reference_traces.json` so the runner contract tests cover the new registration.

## Pick a group

| Group | What lives there | IDs |
|---|---|---|
| `core/` | The original ToolCall-15 set, run by `--short` | TC-01  to  TC-15 |
| `extended/` | Reference-date and multilingual handling | TC-16  to  TC-21 |
| `agentic/` | Multi-step chains, error recovery, safety | TC-22 to TC-36, TC-41 to TC-50 |
| `large_toolset/` | Selection under 20+ tools | TC-37  to  TC-40 |
| `planning/` | Autonomous planning and creative composition | TC-51 to TC-56, TC-61 to TC-63 |
| `adversarial/` | Prompt injection and authority escalation | TC-57  to  TC-60 |
| `structured/` | JSON schema compliance | TC-64  to  TC-69 |
| `hardmode/`, `hardmode_expanded/`, `hardmode_transactional/` | Category P, opt-in with `--hardmode` | TC-70  to  TC-88 |

Take the next free number. The file name and the scenario ID must agree, and the ID must be
`TC-NN`: every registry sorts on `int(s.id.split("-")[1])`, so another shape raises at import.

## Write the file

Three parts: a mock handler that answers tool calls deterministically, an evaluator that scores the
final state, and the two module-level exports.

```python
"""TC-89: date-aware timezone conversion with an explicit tool contract."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import fail_eval, partial_eval, pass_eval

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert_timezone",
            "description": "Convert a local date and time between IANA timezones.",
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": "string"} for name in ("date", "time", "source", "target")
                },
                "required": ["date", "time", "source", "target"],
                "additionalProperties": False,
            },
        },
    }
]
EXPECTED = {
    "date": "2026-03-20",
    "time": "09:00",
    "source": "Europe/Berlin",
    "target": "America/Los_Angeles",
}


def handle(state: ScenarioState, call: ToolCallRecord):
    args = call.arguments
    if (
        call.name != "convert_timezone"
        or set(args) != set(EXPECTED)
        or not all(isinstance(value, str) for value in args.values())
    ):
        return {"error": "Supply date, time, source, and target strings to convert_timezone."}
    try:
        local = datetime.strptime(f"{args['date']} {args['time']}", "%Y-%m-%d %H:%M")
        converted = local.replace(tzinfo=ZoneInfo(args["source"])).astimezone(
            ZoneInfo(args["target"])
        )
    except (ValueError, ZoneInfoNotFoundError):
        return {"error": "Invalid date, time, or IANA timezone."}
    return {
        "date": converted.date().isoformat(),
        "time": converted.strftime("%H:%M"),
        "timezone": args["target"],
    }


def evaluate(state: ScenarioState):
    if len(state.tool_calls) != 1:
        return fail_eval("Expected exactly one timezone conversion.")
    call = state.tool_calls[0]
    if call.name != "convert_timezone" or call.arguments != EXPECTED:
        return fail_eval("The conversion arguments do not match the request.")
    results = [
        record.result
        for record in state.tool_results
        if record.call_id == call.id and record.name == call.name
    ]
    if not any(
        isinstance(result, dict)
        and result.get("time") == "01:00"
        and result.get("date") == "2026-03-20"
        for result in results
    ):
        return fail_eval("No correlated successful conversion supports the answer.")
    if not state.final_answer.strip():
        return partial_eval("Converted correctly but omitted the final answer.")
    if state.final_answer.strip() != "01:00":
        return fail_eval("The answer must contain only the converted HH:MM time.")
    return pass_eval("Converted the specified date and returned 01:00 Pacific time.")


SCENARIO = ScenarioDefinition(
    id="TC-89",
    title="Timezone conversion",
    category=Category.B,
    user_message="Use convert_timezone to convert 09:00 Europe/Berlin on March 20, 2026 to America/Los_Angeles. Return only the converted time in HH:MM format.",
    description="Use the requested timezone tool with the specified date and return its result.",
    handle_tool_call=handle,
    evaluate=evaluate,
    tools_override=TOOLS,
    difficulty=2,
)
DISPLAY = ScenarioDisplayDetail(
    "Pass for the requested conversion and raw HH:MM answer.",
    "Partial for a missing final answer; fail for wrong arguments, missing results, or an incorrect answer.",
)
```

Four things carry weight:

**The handler must be deterministic.** The same arguments must return the same result. Outputs must
respect input values; a calculator must not return the reference answer for every expression. `with_noise` adds realistic extra fields without adding randomness. Declared contact roles and departments take precedence over noise defaults.

Accept the identifiers your tools expose. If search returns both a file ID and a path, document which attachment representation is required, or accept either when it identifies the same observed file. Do not penalize a returned path while the mock reports a successful send.

**`difficulty` is required in practice.** It is a tier from 1 (trivial) to 5 (very hard). A
scenario without one is unrated and drops out of `--weight-by-difficulty` scoring.
`tests/test_scenario_registry.py` fails if you leave it off.

**Evaluators return three tiers.** `pass_eval` scores 2, `partial_eval` scores 1, `fail_eval`
scores 0. Reach for `partial_eval` when the model did the work but reported it badly. A benchmark
that only knows pass and fail cannot tell those apart.

**`DISPLAY` is what reports show** next to the score, so write it as the reader's test rather than
a restatement of the title.

## Optional fields that change how the runner behaves

`ScenarioDefinition` exposes optional conversation and grading settings. Read
`domain/scenarios.py` before using them:

- `follow_up_messages` turns the scenario multi-turn.
- `tools_override` replaces the default toolset, which is how the large-toolset scenarios present
  20+ tools.
- `tool_choice_after_first_call` forces or forbids further tool calls once the model has started.
- `preserve_reasoning_across_follow_ups` keeps reasoning blocks in the transcript between turns.
- `dependencies` declares producer/consumer tool names that need separate model turns.

Use `unsafe_eval` only for an observed unsafe action or disclosure. Ordinary incomplete
work uses `fail_eval` or `partial_eval` without a safety violation. Put endpoint observations
in `ScenarioEvaluation.diagnostics`, separate from task-correctness points.

`max_turns_override` raises the 8-turn default for a scenario that genuinely needs more rounds.

## Sharing code with other scenarios in the group

A helper used by more than one scenario in a group belongs in that group's `_shared.py`.

Do not import one scenario module from another. Helpers with the same name behave differently
across groups on purpose: `_result_matches_if_present` has five different implementations, and
`_call_index`, `_numeric_value`, `_positive_argument_contains`, and `_status_is_success` each have
two. A sibling import would let an edit to one scenario change another's score, and
`tests/test_scenario_registry.py` rejects it.

Helpers used across groups live in `evals/helpers.py`.

## Test it

Cover the pass case, the fail case, and the near-misses that separate them. The near-misses are the
point: a scenario that only passes its own happy path measures nothing.

```bash
env -u FORCE_COLOR .venv/bin/python -m pytest tests/ -m "not live" -q
tool-eval-bench run --scenarios TC-89 --dry-run
```

`--dry-run` confirms registration without touching a server.

## The YAML alternative

A lookup-shaped scenario can skip Python entirely. `evals/yaml_loader.py` loads a declarative
subset; `evals/yaml_scenarios/` holds three worked examples covering a single call, a two-call
chain, and restraint.

```yaml
id: YAML-04
title: Stock price lookup
category: A
difficulty: 1
user_message: What is AAPL trading at?
expected_tool_calls:
  - tool: get_stock_price
    arguments:
      ticker: AAPL
tool_responses:
  get_stock_price:
    - match:
        ticker: AAPL
      response:
        price: 214.30
answer_contains:
  - "214.30"
```

`answer_contains` reaches the middle tier: right tool calls but an answer that never states the
result scores PARTIAL. Everything else about the subset is deliberately narrow. It matches tool
calls positionally, cannot inspect tool results, and has no conditionals, so a scenario that needs
to react to what a tool returned belongs in Python.

Its real job is [held-out packs](scenario-packs.md), where a third party needs to author private
scenarios without shipping executable Python.
