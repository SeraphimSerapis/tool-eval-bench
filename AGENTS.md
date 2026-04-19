# AGENTS.md

This file defines project-local conventions for all files in this repo.

## Mission

Build and evolve a local benchmark platform that evaluates **LLM tool-calling quality** for agentic multi-agent systems. The benchmark uses deterministic scenarios with mock tools, multi-turn conversation loops, and 3-tier scoring (pass/partial/fail).

Primary focus:
1. **Tool-use effectiveness** — 69 scenarios across 15 categories
2. **Multi-turn orchestration** — chained reasoning, conditional branching, error recovery
3. **Throughput benchmarking** — llama-bench style pp/tg measurement with depth/concurrency sweeps

The sole interface is the `tool-eval-bench` CLI. There is no web server or TUI.

## Architectural guardrails

- Keep a strict layered architecture:
  - `domain` must not import storage adapters.
  - `evals` depends on domain types, not concrete server logic.
  - `runner` orchestrates scenarios using adapter interfaces.
  - `cli` is the delivery layer that calls `runner.service`.
- Prefer composition over global state.
- Keep adapters backend-specific and pluggable (all use OpenAI wire format).
- Scenarios are self-contained: each has its own mock handlers and evaluators.

## Storage and reporting rules

- Every completed run MUST be persisted to SQLite.
- Every completed run MUST also produce a Markdown artifact under `runs/YYYY/MM/`.
- Run IDs use UTC timestamp + short deterministic hash.
- Markdown reports MUST include full traces for every scenario.

## Compatibility targets

- vLLM + LiteLLM + llama.cpp are supported via OpenAI-compatible endpoints.
- Any server exposing `/v1/chat/completions` with `tools` support should work.

## Quality bar

Before claiming completion:

1. `ruff check .`
2. `.venv/bin/python -m pytest tests/ --ignore=tests/test_adapter.py`

**Always use the project venv** (`.venv/bin/python`), not system Python.
Dev dependencies like `pytest-asyncio` are installed in the venv via `pip install -e '.[dev]'`.
Running with system Python silently skips all `@pytest.mark.asyncio` tests, giving
a false sense of coverage.

Tests that require a live inference server (`test_adapter.py`) should be excluded
from automated runs. `test_llama_benchy.py` may also be excluded if `llama-benchy`
is not installed (it's in the `[perf]` optional group).

If checks are not possible, explicitly state what was not run and why.

## Documentation requirements

When changing architecture or API behavior, update:

- `README.md`
- `CHANGELOG.md`

Keep `CHANGELOG.md` up to date with notable changes.
