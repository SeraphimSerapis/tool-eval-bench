# Changelog

All notable changes to `tool-eval-bench` are documented here.

## [1.3.1] — 2026-04-20

### Fixed

- **Context pressure first-scenario failure** (Issue #4) — when `--context-pressure` was
  used, the first scenario in a run would consistently fail while subsequent scenarios
  passed.  Root cause: the same filler messages were reused identically across all
  scenarios, allowing the inference server's prefix cache (enabled by default in vLLM) to
  give later scenarios a free performance boost.  The first scenario — which had to compute
  the full filler prefix from scratch — bore the full cost alone.  Fix: inject a unique
  per-scenario nonce (`[scenario:TC-XX]`) into the first filler message via deep copy,
  ensuring every scenario presents a unique token prefix and faces identical evaluation
  conditions.

## [1.3.0] — 2026-04-19

### Added

- **Category O — Structured Output** (TC-64 to TC-69) — 6 new scenarios testing JSON
  schema compliance, tool-to-schema chaining, nested schemas with arrays of objects,
  enum-constrained fields, schema violation resistance (`additionalProperties: false`),
  and multi-tool synthesis into complex nested output. Total: **69 scenarios across 15 categories.**

- **`--leaderboard` CLI command** — beautiful, screenshottable Rich table ranking all
  benchmarked models. Per-category heatmap with color-coded scores (90+ green → <40 red),
  medal rankings (🥇🥈🥉), pass/partial/fail breakdown, and a legend panel.

- **`--export csv|json` CLI command** — export all stored benchmark results in normalized
  CSV or JSON format for programmatic consumption. Supports `--export-output FILE` for
  file output. Includes per-category scores, token usage, and run metadata.

- **`--llm-judge` CLI flag** — optional LLM-as-judge re-evaluation for FAIL results.
  Uses a secondary LLM call to catch false negatives from deterministic string-matching
  evaluators. Can only upgrade FAIL → PARTIAL (never FAIL → PASS). Configurable via
  `--judge-model MODEL`. Flags judge overrides as `[judge override]` in notes.

- **Per-tool-call argument tracking** — `ScenarioResult.tool_call_arg_bytes` now tracks
  the total serialized size of all tool call arguments, enabling efficiency analysis.
  Included in JSON output and reports when non-zero.

- **Experimental async tool orchestration** (`--experimental-async`) — WIP module
  providing `AsyncToolExecutor` with progress tracking, intermediate results, cancellation,
  and failure simulation. Non-breaking — existing scenarios are unchanged. Building blocks
  for future streaming/partial-result scenarios.

- **`--redact-url` CLI flag** — masks the server URL in all display output
  (e.g. `http://192.168.10.5:8080` → `http://***:8080`). Useful for screenshots,
  recordings, and demos where you don't want to expose internal IPs. The actual
  API connection is unaffected.

### Changed

- Scenario count increased from 63 to 69 (6 new structured output scenarios).
- Category count increased from 14 to 15 (new Category O: Structured Output).
- Max points increased from 126 to 138.
- Leaderboard table now shows scenario count (`N`) column to flag partial runs
  (`--short` / `--categories`) that aren't comparable to full 69-scenario runs.

### Fixed

- **Structured output schemas now sent to model API** — `response_format_override`
  is wired through the orchestrator to the adapter. Previously the schemas were
  defined but never passed to the backend (dead code).
- **Schemas embedded in user messages** — all Category O user messages now include
  the full JSON schema text, so models see the schema regardless of whether the
  backend supports `response_format`.
- **TC-68 no longer uses `response_format`** — Schema Violation Resistance now tests
  whether the *model* respects `additionalProperties: false` on its own, not whether
  the *server* enforces the constraint (which would make the test trivially passable).
- **`response_format` deferred on tool-calling turns** — the orchestrator no longer
  sends `response_format` and `tools` together on turn 1, preventing crashes on
  backends (llama.cpp, older vLLM) that reject the combination.
- **Judge upgrade preserves `tool_call_arg_bytes`** — the FAIL→PARTIAL upgrade in
  `runner/judge.py` no longer silently drops the argument size telemetry field.
- **`--llm-judge` and `--experimental-async` now show explicit WIP warnings** instead
  of silently doing nothing when used.
- **`--categories` help text updated** from A–N to A–O with Category O: Structured Output.
- **TC-06 text match hardened** — replaced brittle exact-string comparison with
  case-insensitive `includes_text` to prevent false negatives from minor punctuation.
- **TC-15 error-path consistency** — added `7450.4` to acceptable values in the
  error-injection fallback path (already present in the non-error path).
- **TC-32 de-duplicated from TC-12** — changed prompt from "Delete all my emails
  from last week" to "Clear out all the spam and junk messages from my inbox",
  eliminating near-identical wording while preserving the `send_email` distractor test.
- **TC-49 cancellation fairness** — prompt now says "Don't send it yet" explicitly,
  making the evaluator fair. Downgraded single-email-sent from FAIL to PARTIAL since
  the orchestrator processes Turn 1 fully before injecting the cancellation.
- **TC-55 "budget" ambiguity resolved** — both files are now revenue reports from
  different regions (NA + EMEA), so summing them is unambiguous. Previously, revenue
  + expenses ≠ "total budget" and a model computing net profit would be unfairly penalized.
- **TC-62 stale "8-turn" references** — all internal strings now consistently say
  "6-turn" to match the actual turn count (1 initial + 4 follow-ups).

## [1.2.2] — 2026-04-18

### Added

- **`--backend-kwargs` CLI option** — pass arbitrary JSON-encoded parameters directly
  to the backend API payload (e.g. `--backend-kwargs '{"temperature": 0.6, "top_p": 0.9}'`).
  Deep-merges with existing convenience flags (`--no-think`, `--top-p`, etc.); `--backend-kwargs`
  wins on conflict. Supports any server-specific parameter including `chat_template_kwargs`.
- **`--categories` CLI option** — run only scenarios from specific categories
  (e.g. `--categories K A J`). Letters A–O map to the 15 benchmark categories.
  Enables targeted evaluation for different model profiles (Instruct vs Thinking mode).
- **Context budget visualization** — when using `--context-pressure`, the CLI now displays
  a budget breakdown showing fill tokens, tool definition size (with tool count), output
  reserve, and remaining headroom. Helps diagnose scenarios failing under pressure.
- **`--metrics-url` CLI option** — direct URL to Prometheus `/metrics` for spec-decode
  acceptance rate. Required when the API runs behind a proxy (e.g. LiteLLM) that doesn't
  forward the backend's `/metrics` endpoint
  (e.g. `--metrics-url http://vllm-host:8080/metrics`).
- **Improved spec-bench messaging** — the "acceptance rate unavailable" notice is now
  clearly informational (not an error) and explains how to enable `/metrics` per backend.

### Fixed

- **TC-15 false failure** (Issue #1) — the evaluator required the exact substring
  `"population of iceland"` in the search query, rejecting valid phrasings like
  `"Iceland population 2026"`. Now checks for `"population"` and `"iceland"` independently.
- **Weather scenarios failing under context pressure** (Issue #2) — `_RESERVED_FOR_SCENARIO`
  was 2,500 tokens, which didn't account for tool definitions counted by the server against
  the context window. The 52-tool LARGE_TOOLSET alone consumes ~6,000 tokens. Increased to
  8,000 tokens to prevent context overflow.

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
