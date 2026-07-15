# AGENTS.md

This file defines project-local conventions for all files in this repo.

## Mission

Build and evolve a local benchmark platform that evaluates **LLM quality** for agentic multi-agent systems. The core benchmark uses deterministic scenarios with mock tools, multi-turn conversation loops, and 3-tier scoring (pass/partial/fail). A pluggable architecture allows adding external benchmarks (GSM8K, MMLU, IFEval, future HumanEval, etc.) alongside the tool-call evaluation.

Primary focus:
1. **Tool-use effectiveness** — 69 standard scenarios plus 15 opt-in Hard Mode scenarios across 16 categories
2. **Multi-turn orchestration** — chained reasoning, conditional branching, error recovery
3. **Throughput benchmarking** — llama-bench style pp/tg measurement with depth/concurrency sweeps
4. **Pluggable benchmarks** — external accuracy benchmarks (GSM8K, MMLU, IFEval) via `BenchmarkPlugin` interface

The sole interface is the `tool-eval-bench` CLI. There is no web server or TUI.

## Architectural guardrails

- Keep a strict layered architecture:
  - `domain` must not import storage adapters. Defines core types (`ScenarioDefinition`, `BenchmarkPlugin`).
  - `evals` depends on domain types, not concrete server logic.
  - `runner` orchestrates scenarios using adapter interfaces.
  - `plugins` contains pluggable benchmark modules (GSM8K, MMLU, IFEval). Each plugin implements `domain.plugin.BenchmarkPlugin` and owns its own orchestration.
  - `application` composes concrete adapters, orchestration, storage, and reporting.
  - `cli` is the delivery layer that calls `application.service` and plugin runners.
    `runner.service` remains a compatibility re-export only.
- Prefer composition over global state.
- Keep adapters backend-specific and pluggable (all use OpenAI wire format).
- Scenarios are self-contained: each has its own mock handlers and evaluators.
- Plugins are self-contained: each owns its dataset loading, evaluation logic, and report rendering. Shared infrastructure (adapter, storage) lives outside plugins.

## Storage and reporting rules

- Every completed run MUST be persisted to SQLite.
- Every completed run MUST also produce a Markdown artifact under `runs/YYYY/MM/`.
- Run IDs use a UTC timestamp + short nonce-backed hash for unique execution identity.
- Comparable run configurations use a separate deterministic `config_fingerprint`.
- Markdown reports MUST include full traces for every scenario.

## Compatibility targets

- vLLM + LiteLLM + llama.cpp are supported via OpenAI-compatible endpoints.
- Any server exposing `/v1/chat/completions` with `tools` support should work.
- Non-tool benchmarks (GSM8K, MMLU, IFEval) only require `/v1/chat/completions` — `tools` support is not needed.

## Quality bar

Before claiming completion:

1. `ruff check .`
2. `ruff format --check .`
3. `.venv/bin/python -m pytest tests/ --ignore=tests/test_llama_benchy.py -m "not live" --randomly-seed=104729`

CI repeats the required suite with seeds `104729`, `130363`, and `155921`
across Python 3.11–3.13, and enforces the configured branch-coverage floor.

**Pre-commit hooks** enforce both checks automatically:

```bash
pip install -e '.[dev]'       # includes pre-commit
pre-commit install            # ruff lint on every commit
pre-commit install --hook-type pre-push  # pytest on every push
```

**Always use the project venv** (`.venv/bin/python`), not system Python.
Dev dependencies like `pytest-asyncio` are installed in the venv via `pip install -e '.[dev]'`.
The `[hf]` optional group (`pip install -e '.[hf]'`) installs the `datasets` library
for rate-limit-free HuggingFace downloads.
Running with system Python silently skips all `@pytest.mark.asyncio` tests, giving
a false sense of coverage.

Tests that require the `llama-benchy` package (`test_llama_benchy.py`) should be
excluded from the default run unless the `[perf]` optional group is installed.
CI installs `[dev,perf]` and runs them in a dedicated Python 3.13 job.

Note: `test_adapter.py` uses deterministic httpx mocks and does **not** require
a live inference server — it must be included in all test runs.

## Git conventions

- When a commit fixes a GitHub issue, the commit message **MUST** reference it
  with a `Closes #N` trailer (or `Fixes #N`) so the issue auto-closes on push.
- Use the issue number in the subject line too, e.g.:
  `fix: resolve reports path inside .venv (#9)`

## Documentation requirements

When changing architecture or API behavior, update:

- `README.md`
- `CHANGELOG.md`

Keep `CHANGELOG.md` up to date with notable changes.
