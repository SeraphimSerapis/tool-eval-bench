# Contributing to tool-eval-bench

Thank you for your interest in contributing! This guide covers how to set up a development environment, add new scenarios, and submit changes.

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/SeraphimSerapis/tool-eval-bench.git
cd tool-eval-bench
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Running Tests

```bash
# Full check suite
ruff check .
pytest

# Quick feedback loop
pytest --tb=short -q
```

All tests must pass and `ruff check .` must be clean before submitting a PR.

## Architecture Overview

```
src/tool_eval_bench/
  adapters/           # OpenAI-compatible HTTP adapter
  cli/                # CLI entry point & display rendering
  domain/             # Data models, tool definitions, scoring logic
  evals/              # Scenario definitions & evaluators
  runner/             # Orchestrator, service layer, throughput measurement
  storage/            # SQLite persistence, Markdown report generation
  utils/              # URL helpers, metadata collection
```

**Key rules:**
- `domain` must not import storage adapters
- `evals` depends on domain types, not concrete server logic
- `runner` orchestrates scenarios using adapter interfaces
- `cli` is the delivery layer that calls `runner.service`

## Adding a New Scenario

Scenarios are defined across several files in `src/tool_eval_bench/evals/`:

- `scenarios.py` — Core 15 scenarios (A–E) + the `ALL_SCENARIOS` registry
- `scenarios_extended.py` — Extended scenarios (F–G)
- `scenarios_agentic.py` — Agentic scenarios (H–K)
- `scenarios_large_toolset.py` — Large-toolset scenarios (L)
- `scenarios_planning.py` — Planning + creative scenarios (M–N)
- `scenarios_adversarial.py` — Adversarial safety scenarios (K extras)

Each scenario is a `ScenarioDefinition` with:

### 1. Define the scenario

```python
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)

def _my_handler(state: ScenarioState, call: ToolCallRecord):
    """Mock tool handler — returns deterministic data."""
    if call.name == "expected_tool":
        return {"result": "mock_value"}
    return {"error": f"Unknown tool: {call.name}"}

def _my_evaluator(state: ScenarioState) -> ScenarioEvaluation:
    """Evaluator — checks if the model used the correct tool correctly."""
    calls = [tc for tc in state.tool_calls if tc.name == "expected_tool"]
    if not calls:
        return ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Did not call expected_tool",
        )
    # Check arguments
    args = calls[0].arguments
    if args.get("key") == "expected_value":
        return ScenarioEvaluation(
            status=ScenarioStatus.PASS,
            points=2,
            summary="Correct tool call with correct arguments",
        )
    return ScenarioEvaluation(
        status=ScenarioStatus.PARTIAL,
        points=1,
        summary="Called correct tool but wrong arguments",
    )

MY_SCENARIO = ScenarioDefinition(
    id="TC-XX",                           # Unique ID (TC-01 through TC-99)
    title="Short descriptive title",
    category=Category.A,                  # A through N
    user_message="The prompt the model sees",
    description="What the model should do",
    handle_tool_call=_my_handler,
    evaluate=_my_evaluator,
)
```

### 2. Register the scenario

Add your scenario to the appropriate list in `scenarios.py`:

```python
# For standard scenarios (included in --short runs):
SCENARIOS.append(MY_SCENARIO)

# For extended scenarios (excluded from --short):
EXTENDED_SCENARIOS.append(MY_SCENARIO)

# ALL_SCENARIOS is computed automatically as SCENARIOS + EXTENDED_SCENARIOS
```

### 3. Write a test

Add a test in `tests/test_scenarios.py` to verify the evaluator works:

```python
def test_my_scenario_pass():
    state = ScenarioState()
    state.tool_calls.append(ToolCallRecord(
        id="tc_1", name="expected_tool",
        raw_arguments='{"key": "expected_value"}',
        arguments={"key": "expected_value"}, turn=1,
    ))
    result = MY_SCENARIO.evaluate(state)
    assert result.status == ScenarioStatus.PASS
    assert result.points == 2
```

### Scenario Design Guidelines

| Principle | Details |
|---|---|
| **Deterministic** | Mock handlers return fixed data. No randomness. |
| **Self-contained** | Each scenario has its own handler and evaluator. |
| **Clear pass/fail** | PASS = 2 points, PARTIAL = 1, FAIL = 0. No ambiguity. |
| **Multi-turn aware** | Use `state.tool_calls` to track calls across turns. |
| **Category fit** | Choose the right category (see Categories below). |

### Categories

| Category | Focus |
|---|---|
| A | Tool Selection — picking the right tool from the set |
| B | Parameter Precision — units, dates, multi-value extraction |
| C | Multi-Step Chains — chained reasoning, parallel calls |
| D | Restraint & Refusal — knowing when NOT to call tools |
| E | Error Recovery — handling failures, preserving data integrity |
| F | Localization — German, timezone awareness, translation chains |
| G | Structured Reasoning — routing, data extraction, constraint validation |
| H | Instruction Following — output format, tool prohibition, multi-constraint, tool_choice compliance |
| I | Context & State — cross-reference, state consistency, deduplication, multi-turn correction, constraint accumulation |
| J | Code Patterns — read-before-write, explain vs execute, chained conditional |
| K | Safety & Boundaries — ambiguity, scope limits, hallucination, prompt injection, authority escalation (⚠️ failures generate warnings) |
| L | Toolset Scale — 52-tool namespace, domain confusion |
| M | Autonomous Planning — goal decomposition, open-ended research, conditional workflows |
| N | Creative Composition — cross-tool synthesis, data pipelines, notification workflows |

## Scoring

- **Per-scenario:** 0 (fail), 1 (partial), or 2 (pass) points
- **Per-category:** `(earned points / max points) × 100` within each category
- **Overall score:** `(total points / total max points) × 100` — weighted by scenario count (0–100)
- **Rating thresholds:**
  - 90–100 → ★★★★★ Excellent
  - 75–89 → ★★★★ Good
  - 60–74 → ★★★ Adequate
  - 40–59 → ★★ Weak
  - 0–39 → ★ Poor
- **Worst category floor:** The lowest-scoring category is surfaced separately
- **Safety warnings:** Category K failures are explicitly called out (not auto-penalized)

## Pull Request Checklist

Before submitting:

- [ ] `ruff check .` passes with no errors
- [ ] `pytest` passes all tests
- [ ] New scenarios have evaluator tests
- [ ] README.md updated if CLI flags changed
- [ ] CHANGELOG.md updated if notable changes are made
- [ ] No credentials or API keys in committed code

## Code Style

- Line length: 100 characters (configured in `pyproject.toml`)
- Python 3.11+ type hints everywhere
- Prefer dataclasses over dicts for structured data
- Async functions for anything touching HTTP
- No global mutable state — use composition
