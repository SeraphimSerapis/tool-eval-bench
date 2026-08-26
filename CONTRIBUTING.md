# Contributing to tool-eval-bench

Thank you for helping improve `tool-eval-bench`! This guide explains how to set
up the project, choose the right checks, add or change benchmark behavior, and
prepare a pull request.

The benchmark is deliberately deterministic and auditable. A good contribution
improves the benchmark without making its scores easier to inflate, harder to
reproduce, or unclear to interpret.

For the complete repository conventions, see [`AGENTS.md`](AGENTS.md). This
document is the contributor-facing summary.

## Development setup

The supported development environments use Python 3.11–3.13 and the project
virtual environment. From a fresh checkout:

```bash
git clone https://github.com/SeraphimSerapis/tool-eval-bench.git
cd tool-eval-bench
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,perf]'
```

Install the `[hf]` extra only when working on the GSM8K, MMLU, or IFEval dataset
plugins. It is not part of the normal quality-bar setup.

Install the hooks once if you plan to commit or push from this checkout:

```bash
.venv/bin/pre-commit install
```

The configured default installs pre-commit, pre-push, and post-checkout hooks.
The post-checkout hook links the primary checkout's `.venv` into new Git
worktrees, so the documented quality commands work there without reinstalling
dependencies. Re-run the install command once after upgrading an older clone.

## Checks and feedback loops

Run a focused test while iterating, then run the complete quality bar before
opening or updating a pull request.

### Focused checks

```bash
# Run one evaluator or test module
.venv/bin/python -m pytest tests/test_evaluators_extended.py -k TC33

# Run the deterministic adapter tests
.venv/bin/python -m pytest tests/test_adapter.py

# Run all configured pre-commit checks on the current files
.venv/bin/pre-commit run --all-files
```

### Required quality bar

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
env -u FORCE_COLOR .venv/bin/python -m pytest \
  tests/ \
  --ignore=tests/test_llama_benchy.py \
  -m "not live" \
  --randomly-seed=104729
```

`FORCE_COLOR` only needs to be unset in environments that define it globally;
it can cause Rich rendering tests to emit ANSI codes into captured output.
The project venv is important: using system Python can silently skip async test
support and produce misleading results.

CI repeats the required suite with seeds `104729`, `130363`, and `155921` on
Python 3.11, 3.12, and 3.13. It also builds the Docker image, runs a wheel
smoke test, checks coverage floors, and runs the optional `llama-benchy` tests.
Those checks do not require a live inference server.

The live canary is separate and requires an OpenAI-compatible endpoint. Use it
only when you have an authorized test server:

```bash
TOOL_EVAL_CANARY_BASE_URL=http://host:port/v1 \
  .venv/bin/python -m pytest -m live tests/test_live_canary.py
```

## What a good contribution looks like

Prefer one focused change per pull request. A strong contribution generally:

- explains the problem and the intended user or benchmark impact;
- includes a regression test for changed behavior;
- tests both the intended case and plausible false positives or edge cases;
- preserves deterministic mock data and reproducible scoring;
- follows the project architecture and existing conventions;
- updates `README.md` for user-facing CLI or API changes;
- adds a changelog fragment under `changelog.d/` for notable behavior, scoring, or
  compatibility changes;
- contains no credentials, live endpoints, generated run artifacts, or unrelated
  formatting changes; and
- records the checks that were run in the pull request description.

Do not weaken an evaluator merely to make a model score higher. If a model
behavior should receive more credit, document the evidence that distinguishes it
from a misleading or hallucinated answer.

## Changing scenarios and evaluators

Scenario definitions live in `src/tool_eval_bench/evals/scenarios/`, one file
per scenario, grouped by pack: `core/`, `extended/`, `agentic/`,
`large_toolset/`, `planning/`, `adversarial/`, `structured/`, and the three
Hard Mode groups. Each group package discovers its own files, so a scenario is
registered by existing.

### Adding a new scenario

1. Create `evals/scenarios/<group>/tcNN.py` in the group that best matches what
   the scenario measures. The file name and the scenario ID must agree, and the
   ID must be `TC-NN`: every registry sorts on `int(s.id.split("-")[1])`.
2. Define a module-level `SCENARIO = ScenarioDefinition(...)` with a
   deterministic user prompt, a mock tool handler, and an evaluator returning
   PASS, PARTIAL, or FAIL. Set `difficulty` to a tier from 1 (trivial) to 5
   (very hard); a scenario without one is unrated and drops out of
   `--weight-by-difficulty` scoring.
3. Define a module-level `DISPLAY = ScenarioDisplayDetail(...)` describing the
   success and failure cases. Reports show these next to the score.
4. Set `max_turns_override` if the scenario needs more than the default 8 turns.

Nothing else registers the scenario. `tests/test_scenario_registry.py` checks
every file in the tree for all four of these and fails if one is missing.

A helper used by more than one scenario in the group belongs in that group's
`_shared.py`. Do not import one scenario module from another: helpers with the
same name behave differently across groups on purpose, and a sibling import
would let one scenario's edit change another's score.

Then add tests for the expected pass case, the failure case, and the important
near-misses, plus a description that explains what the scenario measures.

`ScenarioDefinition` carries several optional fields that change how the runner
drives the conversation: `tool_choice_after_first_call`,
`preserve_reasoning_across_follow_ups`, `checkpoint`, and `held_out`. Read the
dataclass in `domain/scenarios.py` before reaching for one.

Simple lookup-shaped scenarios can skip Python entirely and be written as YAML
under `evals/yaml_scenarios/`. See `evals/yaml_loader.py` for the supported
subset, which is intentionally narrower than the Python API.

For evaluator changes in particular:

- Score observable behavior and tool-call history, not stylistic preferences.
- Prefer contextual evidence over broad substring matching.
- Make PARTIAL represent a meaningful middle outcome, not an arbitrary fallback.
- Ensure a plausible hallucination or unsafe action cannot receive full credit.
- Keep external data, randomness, and live services out of mock handlers.
- Add a test that would have failed before the fix.

The standard score is 0 points for FAIL, 1 for PARTIAL, and 2 for PASS. Scenario
and category scoring is documented in the README; changes to scoring semantics
also need a changelog fragment, because they move results a previous run already
produced.

## Changelog entries

`CHANGELOG.md` is generated and is not edited by hand. Add a fragment instead:

```bash
.venv/bin/towncrier create --edit +short-slug.fixed.md
```

Use the issue or PR number when there is one (`72.fixed.md`), and the `+slug` form when there is
not. Types are `added`, `changed`, `fixed`, `removed`, and `security`.

This is why parallel branches stopped conflicting: previously every change edited the same lines at
the top of `CHANGELOG.md`, so any two open pull requests collided there. A file per change cannot.

Preview what your fragment will look like:

```bash
.venv/bin/towncrier build --draft --version 0.0.0
```

Format, types, and worked examples: [`changelog.d/README.md`](changelog.d/README.md).

Purely internal changes, refactors with no observable effect, and test-only changes do not need a
fragment. If a user running the benchmark would notice, it needs one.

## Architecture boundaries

Keep the layered architecture intact:

- `domain` defines core types and must not import storage adapters.
- `evals` depends on domain types and owns scenario definitions and evaluators.
- `runner` orchestrates scenarios through adapter interfaces.
- `plugins` owns external benchmark datasets, evaluation, and rendering.
- `application` composes concrete adapters, orchestration, storage, and reports.
- `cli` is the delivery layer for the application service and plugin runners.

Prefer composition over global state. Keep backend-specific behavior in
adapters, and preserve the OpenAI-compatible wire format used by vLLM, LiteLLM,
llama.cpp, and other supported endpoints.

## Other common changes

### CLI or API changes

Update the relevant README usage, API/schema snapshots when the interface
intentionally changes, tests, and a changelog fragment. Verify `--help` output for
new or changed CLI commands.

### Adapter or orchestration changes

Use deterministic HTTP mocks where possible. Cover streaming, tool calls,
errors, retries, and compatibility behavior without requiring a live model
server.

### Plugin changes

Keep dataset loading, evaluation, and report rendering inside the plugin. Use
the shared adapter and storage infrastructure rather than duplicating it.

### Documentation-only changes

Keep examples executable and update nearby commands when defaults or supported
options change. Avoid reformatting unrelated sections.

## Pull request checklist

Before submitting a pull request:

- [ ] The change is scoped to one logical purpose.
- [ ] The problem and intended behavior are explained.
- [ ] Regression tests were added or updated where behavior changed.
- [ ] Positive and negative/false-positive cases are covered.
- [ ] `.venv/bin/ruff check .` passes.
- [ ] `.venv/bin/ruff format --check .` passes.
- [ ] `.venv/bin/mypy` passes.
- [ ] The required pytest suite passes.
- [ ] README is updated when applicable.
- [ ] A changelog fragment was added under `changelog.d/` when applicable.
- [ ] No secrets, live endpoints, generated reports, or unrelated changes are
      included.
- [ ] The PR description lists the validation performed.

Thank you for contributing improvements that make model and tool-use evaluation
more trustworthy.
