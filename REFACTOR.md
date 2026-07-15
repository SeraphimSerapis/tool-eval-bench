# REFACTOR.md — CLI monolith breakdown & subcommand restructure

This document is the authoritative plan for refactoring `tool_eval_bench`'s
command surface. It tracks the work that resolves the issues raised in the
project assessment:

1. `src/tool_eval_bench/cli/bench.py` was a 3489-line monolith that mixed the
   argument parser, plugin benchmark handlers, trial math, and the entire
   `main()` dispatch into one file.
2. The CLI exposed ~90 flat flags with no subcommands, hurting discoverability
   and making the parser unmaintainable.
3. Coverage was 66% overall, with large gaps in user-facing modules
   (`compare_reports/*`, `runner/throughput.py`, `runner/speculative.py`,
   plugin dataset loaders).
4. No flakiness/isolation guardrail existed to detect order-dependence across
   the 1952-test suite.
5. Declarative restraint scenarios could falsely pass after forbidden tool calls,
   and the pilot YAML file was not guaranteed to ship in built wheels.
6. Domain/plugin and runner/service dependencies crossed the documented layer
   boundaries, coupling core contracts to concrete adapters and persistence.

## Decisions (locked)

- **CLI**: full subcommands **+ silent permanent backward-compat shim**.
- **`compare`**: unified subcommand with a `--report` flag toggling file/HTML
  mode vs run-ID/console mode.
- **Order**: monolith breakdown first; coverage work interleaved as modules
  are extracted; flakiness guardrail last.
- **Imports**: preserve stable import paths via re-exports so no test files
  change during Phase 1.
- **Flakiness**: `pytest-randomly` + markers only.
- **Compatibility**: keep `adapters.base` and `runner.service` as re-export
  surfaces for one release while ownership moves inward.

---

## Phase 0 — Benchmark integrity and packaging

1. Make YAML restraint scenarios fail on any tool call and validate all
   required fields with path-aware errors.
2. Package `evals/yaml_scenarios/*.yaml` and verify both source-tree loading
   and loading from an installed wheel.
3. Land these fixes before structural refactors so later green test runs cannot
   conceal a benchmark-integrity regression.

---

## Phase 1 — Extract `cli/bench.py` (3489 → ~100-line shell)

Each extraction moves logic into a focused module. `bench.py` keeps a
**re-export block** for every private symbol that tests import, so the 1952
tests stay green with zero test edits. The current symbols imported from
`cli.bench` by tests are:

```
_detect_model  _discover_server  _emit_json_output  _headless_error
_probe_server  _resolve_scenarios  _stderr_progress_result  _stderr_progress_start
_metadata_for_storage  _persist_plugin_run  _aggregate_trials  _make_parser
_parse_int_list  _parse_sweep_range  _redact_url  _resolve_all_scenarios_for_ids
_with_config_fingerprint  _run_pressure_sweep  _bootstrap_ci  _median
```

| Current lines | New module | Extracted contents |
|---|---|---|
| 125–533 | `cli/probe.py` | `_detect_model`, `_probe_server`, `_discover_server`, `_do_warmup`, `_preflight_model_check`, `_headless_error` |
| 534–1547 | `cli/plugin_runners.py` | `_run_gsm8k_benchmark`, `_write_gsm8k_report`, `_run_mmlu_benchmark`, `_run_ifeval_benchmark`, `_persist_plugin_run`, `_metadata_for_storage` |
| 2852–3489 | `cli/run_io.py` | `_bootstrap_ci`, `_median`, `_aggregate_trials`, `_print_trials_summary`, `_run_with_live_display`, `_emit_json_output`, `_run_json`, `_run_plain`, `_stderr_progress_*` |
| scenario + small helpers | `cli/resolve.py` | `_resolve_scenarios`, `_resolve_all_scenarios_for_ids`, `_parse_int_list`, `_parse_sweep_range`, `_with_config_fingerprint`, `_redact_url` |
| 2042–2851 (`main()` body) | `cli/dispatch.py` | the flag-gated branch sequence (temporary home until Phase 2) |

**End state of `bench.py`:** imports + re-export block + a ~5-line `main()`
that calls `dispatch.run()`.

**Verification gate (per extraction):** `ruff check .` and
`.venv/bin/python -m pytest tests/ --ignore=tests/test_llama_benchy.py` must
stay green.

### Other monolith candidates (not in this refactor)

Lower maintenance pain, noted for later:
- `evals/scenarios_agentic.py` (2264 lines) — pure scenario data/handlers.
- `storage/reports.py` (958), `compare_reports/summary.py` (977) — split after
  coverage work if still unwieldy.

---

## Phase 2 — Subcommand parser + silent backward-compat shim

Build `cli/parser.py` (new subcommand parser) and `cli/legacy_parser.py`
(retained `_make_parser`, the old flat-flag parser).

`main()` dispatch logic:
1. Try to parse `argv` as subcommands.
2. If `argv[0]` is **not** a known subcommand, fall back to
   `legacy_parser` (the flat 90-flag parser), **silently** — no warning, no
   deprecation notice, per decision (iii). Existing scripts and CI invocations
   keep working unchanged.

### Subcommand layout

```
tool-eval-bench run        [model/backend/scenario flags]   # was: default
tool-eval-bench probe                                         # was: --probe
tool-eval-bench bench      [--perf|--spec|--pressure|--pp|--tg]
tool-eval-bench spec-live                                    # was: --spec-live
tool-eval-bench plugin     {gsm8k|mmlu|ifeval} [--shots|--limit]
tool-eval-bench compare    RUN_A RUN_B                        # was: --compare
tool-eval-bench compare    --report FILE_A FILE_B -o OUT     # was: compare-report
tool-eval-bench history                                      # was: --history
tool-eval-bench leaderboard                                  # was: --leaderboard
tool-eval-bench export      [--format|--output]               # was: --export
tool-eval-bench resume      RUN_ID                            # was: --resume
```

### Unified `compare` subcommand

The two legacy behaviors are genuinely distinct:
- `--compare RUN_A RUN_B` → `cli/history.py:compare_runs` reads two **runs from
  SQLite by run ID** (`latest` shorthand supported) and prints a rich diff
  table to the **console**.
- `compare-report FILE_A FILE_B -o OUT.html` → `cli/compare_report.py:
  generate_compare_report` reads two **Markdown files from disk** and produces
  an **HTML** comparison report.

Unified form:
```
tool-eval-bench compare RUN_A RUN_B                       # console diff of stored runs
tool-eval-bench compare --report FILE_A FILE_B -o OUT    # HTML from .md files
```
- `--report` flips the positionals to **file paths** and requires `-o/--output`.
- Without `--report`, positionals are **run IDs** → console table.
- `compare-report` is retained as a **silent alias** subcommand mapping to
  `compare --report`, so existing invocations never even hit the shim.

---

## Phase 3 — Coverage gaps (interleaved with Phases 1–2)

Add tests **per extracted module** — cheaper to cover a ~200-line module than a
3489-line one. Targets (current coverage → target):

| Module | Cov | Tests to add |
|---|---|---|
| `compare_reports/summary.py` | 48% | HTML/markdown rendering; empty runs, missing-scenario edges |
| `compare_reports/tool_eval.py` | 48% | report diffing logic |
| `runner/throughput.py` | 64% | pp/tg computation, sweep math, error paths |
| `runner/speculative.py` | 72% | spec-decode acceptance-rate calc, draft/accept branches |
| `plugins/gsm8k/dataset.py` | 49% | dataset loading + answer extraction |
| `plugins/ifeval/dataset.py` | 44% | instruction parsing |
| `plugins/mmlu/dataset.py` | 51% | subject filtering + answer mapping |
| `runner/service.py` | 77% | orchestration error paths |

**Verified result:** overall branch coverage increased from 66% to 83.69%
across 2,098 tests. CI enforces an 80% floor.

---

## Phase 4 — Flakiness / isolation guardrail

1. Add `pytest-randomly` to the `[dev]` extra in `pyproject.toml` (self-
   activating; reports the seed per run for reproducibility).
2. Register markers in `pyproject.toml` `[tool.pytest.ini_options]`:
   `live`, `slow`, `integration`.
3. Mark the handful of tests that genuinely touch subprocess/network (the
   `spec_live` tests are mocked, so likely few qualify). Deselect `live` by
   default in CI.
4. Run the suite 3× with different seeds to confirm no order-dependence.

No `pytest-repeat` (randomization + markers is sufficient).

---

## Phase 5 — Architectural boundaries

1. Own the provider-neutral adapter port and result types in `domain` and keep
   `adapters.base` as a compatibility re-export.
2. Move concrete adapter/storage/report composition into `application`; keep
   `runner.service.BenchmarkService` as a compatibility re-export.
3. Add a static import-boundary test covering domain, evals, runner, plugins,
   storage, and utils without adding a runtime dependency.

---

## Phase 6 — CI and release gates

1. Run the required suite with three recorded `pytest-randomly` seeds across
   Python 3.11–3.13, excluding only tests marked `live`.
2. Run the optional llama-benchy tests in a dedicated Python 3.13 `[dev,perf]`
   job.
3. Build an isolated wheel, install it into a clean environment, and smoke-test
   version/help, subcommands, dependencies, and bundled YAML.
4. Enforce the verified 80% branch-coverage floor in CI.

---

## Documentation updates (tied to Phase 2)

- **README.md**: replace the 180-flag flat listing with the subcommand layout
  plus a "common workflows" section (compare two runs, run a plugin benchmark,
  run perf, generate an HTML comparison). Note that the legacy flat flags
  still work. Feature the unified `compare` (`--report` form).
- **CHANGELOG.md**: one entry per phase (extraction, subcommands, coverage
  guardrail).
- **AGENTS.md**: no change needed — still "one CLI", now subcommanded.

---

## Execution order & checkpoints

Each checkpoint = `ruff check . && .venv/bin/python -m pytest tests/ \
--ignore=tests/test_llama_benchy.py` green. **One commit per phase** for
bisectability.

1. **1a** `cli/resolve.py` + `cli/probe.py`
2. **1b** `cli/run_io.py`
3. **1c** `cli/plugin_runners.py`
4. **1d** `cli/dispatch.py` → `bench.py` is now a re-export shell
5. **3 (partial)** coverage on freshly-extracted modules
6. **2** `cli/parser.py` + `cli/legacy_parser.py` + thin `main()` + unified
   `compare`; manual `--help` per subcommand + one legacy invocation
7. **4** `pytest-randomly` + markers, 3 seeds
8. **Docs** README / CHANGELOG

## Status

| Phase | Status |
|---|---|
| 0 — benchmark integrity + packaging | complete |
| 1 — extract `bench.py` | complete (`bench.py` is a thin re-export shell) |
| 2 — subcommands + shim | complete |
| 3 — coverage | complete (83.69%; 2,098 tests) |
| 4 — flakiness guardrail | complete (three recorded seeds pass) |
| 5 — architecture boundaries | complete |
| 6 — CI + release gates | complete (80% floor enforced) |
| Docs | complete |
