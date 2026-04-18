# Changelog

All notable changes to `tool-eval-bench` are documented here.

## [1.2.1] — 2026-04-18

### Changed

- **Coherence check enabled by default** — llama-benchy's coherence check now runs
  before benchmarking to verify the model is producing sensible output. Previously
  `--skip-coherence` was the default, which could mask broken models.
- `--skip-coherence` CLI flag added for environments that cannot reach `gutenberg.org`
  (air-gapped / firewalled hosts).

### Fixed

- **Ruff lint errors in test suite** — removed 5 unused imports and converted 2 lambda
  assignments to `def` statements in `tests/test_context_pressure.py`.

## [1.2.0] — 2026-04-18

### Added

- **llama-benchy as default throughput benchmark** — `--perf` / `--perf-only` now delegate
  throughput measurement to [llama-benchy](https://github.com/eugr/llama-benchy),
  a dedicated llama-bench style benchmarking tool for OpenAI-compatible endpoints.
  llama-benchy provides more accurate pp/tg measurement using HuggingFace tokenizers,
  multi-run statistics, proper latency estimation, and cache-busting.
- `--perf-legacy` / `--perf-legacy-only` — the previous built-in throughput benchmark
  is still available for environments without external dependencies.
- `--benchy-runs N` — number of measurement iterations per test point (default: 3).
- `--benchy-latency-mode` — latency measurement method (`api`, `generation`, `none`).
- `--benchy-args` — pass-through for arbitrary llama-benchy flags (e.g. `--benchy-args='--no-warmup --book-url URL'`).
- **`[perf]` optional dependency** — `pip install tool-eval-bench[perf]` bundles llama-benchy,
  eliminating the need for `uvx` and avoiding first-run download delays.
- **Rich progress bar** for llama-benchy runs — replaces raw stdout dump with a live
  progress bar showing warmup → latency → per-run progress with elapsed time.
- **Real-time streaming** — `PYTHONUNBUFFERED=1` forces subprocess output to stream
  line-by-line instead of buffering until exit.

### Changed

- **Dynamic table columns** — `Test` column width is computed from data, `Conc` is now
  a compact standalone `c` column (`c1`, `c2`, `c4`). Handles arbitrarily large depth
  and concurrency values (262144, 100+) without truncation.
- **Weakest category display** — the `Weakest:` line is now hidden when all categories
  score 100%, keeping the panel clean for perfect results.
- **Noise suppression** — PyTorch and HF Hub warnings from the subprocess are filtered
  from display output via env vars (`TRANSFORMERS_NO_ADVISORY_WARNINGS`,
  `HF_HUB_DISABLE_IMPLICIT_TOKEN`) and an output line filter.

### Fixed

- **Tokenizer mismatch** — pass `--tokenizer` with the full HuggingFace model ID when
  the API model name is a served alias (e.g. `Qwen3.6-35B` vs `Qwen/Qwen3.6-35B-A3B-FP8`),
  so llama-benchy loads the correct tokenizer instead of falling back to `gpt2`.
- **Gutenberg book download crash** — added `--skip-coherence` flag to avoid llama-benchy
  crashing when the machine cannot reach `gutenberg.org` (common on air-gapped/firewalled hosts).
  *(Note: v1.2.1 re-enabled coherence by default; use `--skip-coherence` to opt out.)*
- **Multi-value argument format** — use space-separated values (`--depth 0 4096 8192`)
  instead of repeated flags (`--depth 0 --depth 4096 --depth 8192`) to match
  llama-benchy's `nargs='+'` argparse convention. Previously only the last value was used.

## [1.1.0] — 2026-04-17

### Added

- **Context pressure** (`--context-pressure`) — pre-fill the context window with
  alternating user/assistant filler turns before each scenario to test tool-calling
  quality under context pressure. Auto-detects context window size from `/v1/models`
  (`max_model_len` on vLLM); use `--context-size` to override.
- **Cache-busting filler** — filler content draws from 12 diverse paragraph styles
  (tech docs, meeting notes, code reviews, etc.), shuffled per run, with random
  noise tokens (ticket IDs, timestamps, IPs, versions) injected at sentence
  boundaries and unique nonce prefixes per chunk. This defeats vLLM/llama.cpp
  prefix caching for accurate pressure measurement.
- `--context-size` flag to manually specify context window size when auto-detection
  is unavailable.
- Progress bar during context pressure fill.

## [1.0.0] — 2026-04-17

### Initial Public Release

**63 deterministic scenarios** across **14 categories** (A–N) for evaluating
LLM tool-calling quality in agentic workflows.

### Features

- **Tool-call quality benchmark** — 63 scenarios testing tool selection,
  parameter precision, multi-step chains, error recovery, safety boundaries,
  autonomous planning, creative composition, and more.
- **3-tier scoring** — each scenario scored as pass (2 pts), partial (1 pt),
  or fail (0 pts) with deterministic evaluators.
- **Safety gating** — Category K failures cap the rating at ★★★ Adequate
  regardless of the overall numeric score.
- **Throughput benchmark** (`--perf`) — llama-bench style pp/tg measurement
  with configurable context depth and concurrency sweeps.
- **Speculative decoding benchmark** (`--spec-bench`) — measures effective t/s,
  acceptance rate (α), and speedup ratio for MTP/draft/ngram/eagle methods.
- **Multi-trial statistics** (`--trials N`) — mean ± stddev, 95% bootstrap CI,
  Pass@k / Pass^k reliability metrics.
- **Error injection** (`--error-rate`) — simulate HTTP 429/500/503 errors to
  test model robustness under failure conditions.
- **Deployability scoring** — composite quality × responsiveness metric with
  configurable weight (`--alpha`).
- **Deterministic payload noise** — all mock tool responses enriched with
  realistic metadata (timestamps, IDs, nested objects) to test signal extraction.
- **Run persistence** — SQLite storage + Markdown reports with full traces.
- **Run comparison** — `--diff`, `--compare`, `--history` for tracking
  model performance over time.
- **Backend support** — any OpenAI-compatible `/v1/chat/completions` endpoint:
  vLLM, LiteLLM, llama.cpp.
- **Model auto-detection** — queries `/v1/models` and presents an interactive
  picker when multiple models are available.

### Scenario Categories

| Category | Scenarios | Focus |
|---|---|---|
| A — Tool Selection | 3 | Picking the right tool |
| B — Parameter Precision | 3 | Correct types, units, dates |
| C — Multi-Step Chains | 4 | Chained reasoning, parallel calls |
| D — Restraint & Refusal | 3 | Knowing when NOT to call tools |
| E — Error Recovery | 3 | Handling failures gracefully |
| F — Localization | 3 | German, timezone, translation |
| G — Structured Reasoning | 3 | Routing, extraction, validation |
| H — Instruction Following | 5 | Format compliance, tool_choice |
| I — Context & State | 10 | Multi-turn correction, accumulation |
| J — Code Patterns | 3 | Read-before-write, explain vs execute |
| K — Safety & Boundaries | 13 | Injection, escalation, hallucination |
| L — Toolset Scale | 4 | 52-tool namespace selection |
| M — Autonomous Planning | 3 | Goal decomposition, research |
| N — Creative Composition | 3 | Cross-tool synthesis, pipelines |

### Credits

Scenario methodology adapted from [ToolCall-15](https://github.com/stevibe/ToolCall-15)
by [stevibe](https://x.com/stevibe) (MIT License).
