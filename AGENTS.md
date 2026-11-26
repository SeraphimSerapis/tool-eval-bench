# AGENTS.md

Single-assistant deterministic tool-call benchmark covering 104 scenarios in total across 16
categories: 69 standard deterministic scenarios (TC-01 to TC-69, categories A-O) plus 35 Hard Mode
scenarios (TC-70 to TC-104, category P, opt-in via `--hardmode`, `--hardmode-only`, or explicit
`--scenarios` IDs). Also runs throughput, speculative-decoding, and context-pressure modes plus 6
plugins (`decision`, `gsm8k`, `ifeval`, `mmlu`, `needle`, `toolscale`).

## Architecture and context pointers

`tests/test_architecture_boundaries.py` enforces these recursive layer import rules:

| Layer | Allowed internal imports | Module layout notes |
|---|---|---|
| `domain` | `domain` | Core types, `engines.py`, `rubric.py`, `redaction.py`. |
| `evals`, `plugins`, `storage`, `utils` | `domain` (+ self) | `evals/helpers/` splits shared helpers; `storage/reports/card/_*` renders card panels; `plugins/toolscale/{plugin,aggregate,report}.py` separates orchestration, aggregation, and rendering. |
| `adapters` | `domain`, `utils` | Wire adapters for OpenAI-compatible, native Gemini, and native Anthropic APIs. |
| `runner` | `domain`, `evals`, `utils` | `runner/orchestrator/` owns scenario loop, probes, scoring, review hooks, and wire helpers; `runner/spec_detection.py` imports no other `runner` module; `runner/throughput.py` stays decoupled from `runner/speculative.py`; `runner/service.py` is the sole allowlisted compatibility re-export of `application.service`. |
| `application` | `adapters`, `domain`, `evals`, `runner`, `storage`, `utils` | Composes adapters, orchestration, finalization, queries, deletion, and leaderboards. |
| `cli`, `web` | Delivery layers (`FORBIDDEN_MODULES` bars `storage.db`) | `cli/dispatch.py` and sibling CLI phase modules plus `web/` route all DB reads and deletes through `application/run_queries.py` and `application/run_deletion.py`; `storage.reports` rendering is allowed. |

Context pointers (read the target before editing that branch):

- **Scenarios or evaluators:** `docs/adding-a-scenario.md`
- **Benchmark plugins:** `docs/adding-a-plugin.md`
- **Inference engines or wire adapters:** `src/tool_eval_bench/domain/engines.py`,
  `docs/adding-an-adapter.md`, `docs/backends.md`
- **Full module map and data flow:** `docs/architecture.md`
- **Contributor setup and release workflow:** `CONTRIBUTING.md`, `changelog.d/README.md`

## Non-obvious contracts

- **Self-contained scenarios.** Place each scenario in `evals/scenarios/<group>/tcNN.py` exporting
  `SCENARIO`, `DISPLAY`, and `difficulty` (1 to 5); group packages auto-discover files
  (`tests/test_scenario_registry.py`). Share group helpers in that group's `_shared.py`, keeping
  scenario modules isolated from sibling scenario modules because identical helper names carry
  different semantics across groups.
- **Persisted run config.** Declare every persisted run key once in `application/run_config.py`
  (`RUN_CONFIG_FIELDS`) with explicit presence, fingerprint, cohort, and resume rules
  (`tests/test_run_config_schema.py`).
- **CLI arguments and schema snapshots.** Register public commands in `cli/command_registry.py`
  and mirror every public parser flag in `schema.py` (`ARGS_SCHEMA`, bump `SCHEMA_VERSION`), then
  regenerate snapshots with `.venv/bin/python scripts/update_compat_snapshots.py`
  (`tests/test_api.py`).
- **Persistence order and redaction.** Write the Markdown report (`runs/YYYY/MM/`) before
  committing SQLite (`application/finalization.py`, `application/mode_runs.py`). Redact URLs and
  secrets via `domain/redaction.py`. Withhold held-out `--scenario-pack` titles, summaries, and
  traces from Markdown, JSON, and web responses while keeping full traces in SQLite.
- **Web static ES modules.** Register every `web/static/*.js` file in `_STATIC_MIME_TYPES` in
  `web/app.py`, use native ES module imports, and construct DOM nodes programmatically with zero
  HTML sinks (`innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`; enforced by
  `tests/test_web_history.py` and `tests/test_web_security.py`).
- **Documented counts.** `tests/test_documented_counts.py` asserts exact scenario and category
  counts in `README.md`, `AGENTS.md`, and `docs/cli-reference.md`.

## Rubric AST gate (`src/tool_eval_bench/domain/rubric.py`)

`tests/test_rubric_cohorts.py::test_rubric_ast_fingerprint_matches_source` hashes the
docstring-stripped normalized AST of all `evals/**/*.py` and `runner/orchestrator/**/*.py` files
against `RUBRIC_AST_FINGERPRINT`. Docstring, comment, and whitespace edits leave the digest
unchanged.

- **Verdict-changing edit** (scoring logic, prompts, translations, or noise injection): bump
  `EVAL_RUBRIC_VERSION`, `LOCALE_VERSION` (single source of truth in `domain/rubric.py`, imported
  as `int` by `evals/locales/__init__.py`), or `NOISE_VERSION`, then refresh
  `RUBRIC_AST_FINGERPRINT` from `compute_rubric_ast_fingerprint()`.
- **Strictly verdict-preserving refactor** (renames, helper extraction, typing): refresh
  `RUBRIC_AST_FINGERPRINT` only and explain why verdicts stay unchanged in the PR description.

Leaderboard cohorts group runs on these three rubric versions rather than unrelated commits, so
treat version bumps as deliberate comparability boundaries.

## Testing discipline

- **Test behavior through public seams.** Drive scenarios via `SCENARIO.handle_tool_call` and
  `SCENARIO.evaluate` (or `tests/scenario_replay.py::replay()`, or `simulate_results()` from
  `tests/conftest.py`), adapters via deterministic `httpx` mock transports, and CLI/web/storage
  through public entry points. Assert observable verdicts, persisted state, and wire payloads
  rather than private helper internals.
- **Own the subsystem's test file.** Put regression tests in the subsystem's existing test module
  (for example `tests/test_orchestrator.py`, `tests/test_evaluators_*.py`,
  `tests/test_adapter.py`). Keep catch-all PR, audit, and coverage-gap files out of `tests/`.
- **Keep cases minimal and non-redundant.** Cover pass, meaningful partial, failure/false-positive,
  and the exact regression input using `@pytest.mark.parametrize` for variants. Skip tautological
  constant, dataclass, or stdlib checks and mock-echo tests.
- **Meet coverage floors cleanly.** CI enforces 80% total branch coverage (`pyproject.toml`), 30
  per-module floors in `scripts/check_module_coverage.py` (70% to 95%), and 95% on
  `runner/llama_benchy.py` (`.coveragerc.perf`). Close coverage gaps by testing real failure modes
  and edge cases through public entry points or deleting unreachable dead code rather than padding
  assertions.
- **Keep tests tight, deterministic, and cross-platform.** Replace wall-clock sleeps with
  `threading.Event` or mocked clocks. Mock HTTP deterministically (mark live-server tests
  `@pytest.mark.live`; turn 1 streams SSE, so local HTTP mocks must handle `stream: true`). Pass
  explicit `encoding="utf-8"` on file I/O (Windows defaults to `cp1252`), use `spawn` for
  multiprocessing, and keep tests order-independent (PRs run seed `104729`; post-merge
  `test-extended` runs Python 3.11 and Windows with different seeds). Respect the two AST-enforced
  bans (`tests/test_architecture_boundaries.py`): keep `httpx.AsyncClient` unpatched at class
  level, and use public `argparse` attributes (`Action.const`) instead of private `argparse._*`
  classes.

## Quality bar and environment traps

Run every Python check with `.venv/bin/...` rather than system Python (system Python silently skips
`pytest-asyncio` tests):

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
for f in src/tool_eval_bench/web/static/*.js; do node --check "$f" || exit 1; done
env -u FORCE_COLOR .venv/bin/python -m pytest tests/ \
  --ignore=tests/test_llama_benchy.py -m "not live" \
  --randomly-seed=104729 --cov=tool_eval_bench --cov-report=json --cov-fail-under=80
.venv/bin/python scripts/check_module_coverage.py coverage.json
```

- Add `-n auto --maxprocesses=4` during local iteration, then run serially for coverage parity.
  Always include `tests/test_adapter.py` (deterministic `httpx` mocks, no live server).
- When editing `runner/llama_benchy.py` (requires `[perf]`):

```bash
env -u FORCE_COLOR .venv/bin/python -m pytest \
  tests/test_llama_benchy.py tests/test_llama_benchy_redaction.py \
  -m "not live" --randomly-seed=181081 \
  --cov=tool_eval_bench.runner.llama_benchy --cov-config=.coveragerc.perf --cov-fail-under=95
```

- When changing dependencies in `pyproject.toml`, update `uv.lock` and verify with `uv lock --check`.
- **Environment traps.** Unset `FORCE_COLOR` for `pytest` because `rich` treats even `FORCE_COLOR=0`
  as a forced terminal and breaks `StringIO` assertions. Keep `[hf]` (`datasets`) uninstalled for
  `mypy` because `datasets` triggers `import-untyped` errors against the CI
  `type: ignore[import-not-found]` annotations.

## PR policy, commits, and changelog fragments

- **Automated PR gate (`scripts/check_contribution.py`).** Any `src/**/*.py` edit requires a
  `tests/**` change. Any runtime or packaging edit (`src/**`, `pyproject.toml`, `uv.lock`,
  `Dockerfile`, `compose.yaml`, `docker-compose.yml`, `docker-compose.yaml`, `.env.example`,
  `MANIFEST.in`) requires a nonempty `changelog.d/<issue-or-+slug>.<type>.md` fragment (`added`,
  `changed`, `fixed`, `removed`, `security`) or a visible edit to an unreleased fragment. Leave
  `CHANGELOG.md` untouched (towncrier-generated).
- **Commits.** Reference fixed GitHub issues in both subject (`(#N)`) and trailer (`Closes #N`).
- **Durable docs.** Update `README.md` for user-facing CLI or API changes, and log
  costly-to-reverse architectural decisions in `docs/decisions.md`.
