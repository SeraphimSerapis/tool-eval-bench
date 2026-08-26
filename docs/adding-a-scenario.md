# Adding a scenario

A scenario is one file. Create `src/tool_eval_bench/evals/scenarios/<group>/tcNN.py`, export a
`SCENARIO` and a `DISPLAY`, and it is registered. Nothing else lists it.

## Pick a group

| Group | What lives there | IDs |
|---|---|---|
| `core/` | The original ToolCall-15 set, run by `--short` | TC-01 – TC-15 |
| `extended/` | Reference-date and multilingual handling | TC-16 – TC-21 |
| `agentic/` | Multi-step chains, error recovery, safety | TC-22 – TC-50, TC-62 – TC-63 |
| `large_toolset/` | Selection under 20+ tools | TC-37 – TC-40 |
| `planning/` | Autonomous planning and creative composition | TC-51 – TC-56 |
| `adversarial/` | Prompt injection and authority escalation | TC-57 – TC-60 |
| `structured/` | JSON schema compliance | TC-64 – TC-69 |
| `hardmode/`, `hardmode_expanded/`, `hardmode_transactional/` | Category P, opt-in with `--hardmode` | TC-70 – TC-88 |

Take the next free number. The file name and the scenario ID must agree, and the ID must be
`TC-NN`: every registry sorts on `int(s.id.split("-")[1])`, so another shape raises at import.

## Write the file

Three parts: a mock handler that answers tool calls deterministically, an evaluator that scores the
final state, and the two module-level exports.

```python
"""TC-89 — Timezone Conversion."""

from __future__ import annotations

from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    has_tool_call as _has_tool_call,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc89_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Answer every tool call with a fixed payload, so runs are reproducible."""
    if call.name == "convert_timezone":
        return _noise({"source": "09:00 Europe/Berlin", "target": "00:00"}, call.name)
    return _noise({"error": f"Tool {call.name} is not relevant here."}, call.name)


def _tc89_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What time is our 09:00 Berlin standup in Los Angeles?'"""
    if not _has_tool_call(state, "convert_timezone"):
        return _fail("Answered from memory instead of converting the time.")
    if not _includes_text(state.final_answer, "00:00"):
        return _partial("Converted the time but never stated the result.")
    return _pass("Converted through the tool and reported midnight Pacific.")


SCENARIO = ScenarioDefinition(
    id="TC-89",
    title="Timezone Conversion",
    category=Category.B,
    user_message="What time is our 09:00 Berlin standup in Los Angeles?",
    description="Convert a time through the tool rather than answering from memory.",
    handle_tool_call=_tc89_handle,
    evaluate=_tc89_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls convert_timezone and reports the converted time.",
    "Fail if it answers from memory or omits the result.",
)
```

Four things carry weight:

**The handler must be deterministic.** Every run gives the same tool result, so a score difference
is a model difference. `with_noise` adds realistic extra fields without adding randomness.

**`difficulty` is required in practice.** It is a tier from 1 (trivial) to 5 (very hard). A
scenario without one is unrated and drops out of `--weight-by-difficulty` scoring.
`tests/test_scenario_registry.py` fails if you leave it off.

**Evaluators return three tiers.** `pass_eval` scores 2, `partial_eval` scores 1, `fail_eval`
scores 0. Reach for `partial_eval` when the model did the work but reported it badly. A benchmark
that only knows pass and fail cannot tell those apart.

**`DISPLAY` is what reports show** next to the score, so write it as the reader's test rather than
a restatement of the title.

## Optional fields that change how the runner behaves

`ScenarioDefinition` has ten optional fields. Four of them change the conversation itself, and
are worth reading in `domain/scenarios.py` before you use one:

- `follow_up_messages` turns the scenario multi-turn.
- `tools_override` replaces the default toolset, which is how the large-toolset scenarios present
  20+ tools.
- `tool_choice_after_first_call` forces or forbids further tool calls once the model has started.
- `preserve_reasoning_across_follow_ups` keeps reasoning blocks in the transcript between turns.

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
