# Changelog

All notable changes to `tool-eval-bench` are documented here.

## [Unreleased]

### Fixed

- **Pre-flight model availability check (#19)** — when a server lists a model
  in `/v1/models` but fails to actually serve it (e.g. vLLM returns 400
  "Model not found" on inference), the benchmark previously produced
  misleading scores (1 passed, 11 partial, 72 failed) because 4xx responses
  were treated as "model returned no tool calls" by the adapter.  A new
  `_preflight_model_check()` sends a trivial 1-token chat completion after
  model detection and before warm-up.  If the server returns 4xx/5xx, the
  benchmark aborts with a clear error (exit code 3) instead of running
  84 scenarios against a broken endpoint.  New `MODEL_NOT_AVAILABLE` error
  code added to `domain/errors.py` for structured `--json` output.
- **Streamed tool-call arguments repair for `--stream-interval > 1` (#18)**
  — when vLLM is launched with `--stream-interval` set to a value higher
  than 1, tool-call argument tokens are batched into larger SSE chunks.
  In some cases the server's own tool-call parser does not detect the
  closing brace within a batch, causing the accumulated arguments string
  to be missing its final `}` or have unbalanced quotes.  The streaming
  adapter now applies a `_repair_streamed_tool_args()` function that
  closes unterminated strings and unbalanced braces/brackets before
  building `ProviderToolCall` objects, ensuring arguments are parseable
  regardless of the server's stream-interval setting.

### Changed

- **TC-48 evaluator tightened** — models that merge CC correctly but skip
  `get_contacts` (using bare names like `"Alice"` instead of resolved email
  addresses) are now downgraded from pass to partial.  Models that resolve
  contacts via `get_contacts` and ask for email content clarification (instead
  of fabricating) now receive partial credit instead of a hard fail.
- **TC-84 contact mock made query-aware (#16)** — the `get_contacts` handler
  now filters results by the search query for more realistic log output.
  No change to evaluation logic.

## [2.0.7] — 2026-06-22

### Fixed

- **`--perf` OOM prevention (#14)** — the llama-benchy subprocess no longer
  eats all available RAM. Three root causes addressed:
  - **Coherence check disabled by default** — llama-benchy's coherence check
    loads a model for perplexity evaluation, which consumed 25GB+ RAM in
    seconds.  tool-eval-bench already has 74 scenarios for quality evaluation;
    the coherence check is redundant.  `skip_coherence` now defaults to `True`
    when invoked from the CLI.
  - **No `--tokenizer` passed to subprocess** — the model's filesystem path
    (e.g. `Qwen/Qwen3.6-35B-A3B-FP8` or a HuggingFace cache path) was being
    passed as `--tokenizer`, causing transformers to load large tokenizer/model
    data.  llama-benchy's gpt2 fallback is sufficient for prompt construction.
  - **Offline env vars** — `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are
    now set in the subprocess environment to prevent any accidental large
    downloads.  The OOM detection (SIGKILL/exit-137/MemoryError) from 2.0.6
    remains as a safety net.

- **4xx HTTP errors classified as `wrong_args` not `model_crash`** — the
  `_classify_runtime_error` function now returns `FailureKind.WRONG_ARGS` for
  4xx `HTTPStatusError` instead of `MODEL_CRASH`, since 400/422 typically
  means the model generated malformed tool-call arguments.

- **Dead code in parallel crash path** — the `isinstance(exc, BaseException)`
  conditional in `run_all_scenarios` was always `True` (we only enter the
  branch when `isinstance` is already confirmed).  Simplified to a direct
  `_classify_runtime_error(exc)` call.

- **Placeholder URL removed from OOM error** — the OOM error message
  previously pointed to `https://github.com/eugr/llama-benchy/issues/XX`
  (a placeholder).  Now suggests `--perf-legacy-only` as a fallback.

- **UTF-8 encoding for leaderboard export files** — `export_runs` now opens
  output files with `encoding="utf-8"` to prevent `UnicodeEncodeError` on
  Windows for model names with non-ASCII characters (e.g. rating stars).

- **YAML loader error messages include file path** — missing `id`/`category`
  fields and YAML parse errors now report the file path, making it easier to
  debug broken scenario files.

- **Windows drive-letter paths shortened in leaderboard** —
  `_shorten_model_name` now handles `C:\Users\…\models\my-model` and UNC paths
  (`\\server\share\…`), not just Unix absolute paths and HuggingFace cache
  paths.

### Improved

- **`on_output` type tightened** — the `run_llama_benchy` callback parameter
  is now typed as `Callable[[str], None] | None` instead of `Any | None`.

- **Redundant condition removed in `compute_fill_budget`** — the
  `chunk_with_overhead > 0` check was always `True` (the value is a compile-time
  constant).  Simplified for readability.

- **CLI test coverage** — added `tests/test_cli_bench.py` with 44 unit tests
  covering scenario resolution, backend detection from response headers,
  sweep-range parsing, argument parsing, JSON output, and plugin-run
  persistence.
- **Backend metadata probing tests** — added `tests/test_metadata.py` with 27
  mocked tests for `/v1/models`, `/version`, `/health`, `/props`, and
  quantization inference, raising `utils/metadata.py` coverage from ~29% to
  ~91%.
- **Failure taxonomy** — added `failure_kind` to `ScenarioEvaluation` and
  `ScenarioResult`, with runtime-error classification (timeout,
  connection_error, server_error, model_crash) and heuristic evaluator-failure
  classification (wrong_tool, wrong_args, missing_step, forbidden_action).
  Failure kinds are rendered in Markdown reports and round-trip through
  `to_dict()` / `from_dict()`.
- **YAML scenario loader pilot** — added `evals/yaml_loader.py` and a sample
  declarative scenario under `evals/yaml_scenarios/`. Simple scenarios can now
  be authored as YAML files with expected tool calls and response rules.
  Added `pyyaml>=6.0` as a core dependency.
- **CLI refactor (part 1)** — extracted small CLI helpers and server-discovery
  code from the 4,477-line `cli/bench.py` into new modules:
  `cli/helpers.py` (dotenv, URL redaction, JSON output, sweep/int parsing,
  plugin run persistence, headless errors), `cli/commands.py` (scenario
  resolution), and `cli/server.py` (port discovery, backend detection).
  `bench.py` now re-exports the old names for backward compatibility and
  shrank by ~200 lines. Existing tests were updated where the patch path
  changed.
- **CLI refactor (part 2)** — extracted throughput, speculative-decoding, and
  context-pressure runners from `cli/bench.py` into new modules:
  `cli/perf.py` (`run_throughput`, `run_llama_benchy`),
  `cli/spec_bench.py` (`run_spec_bench`), and
  `cli/pressure.py` (`run_pressure_sweep`). Helpers are injected as
  parameters to avoid circular imports. `bench.py` shrank from 4,285 → 3,352
  lines (total reduction of 1,125 lines from the original 4,477). Two
  integration tests in `test_context_pressure.py` were updated to use the new
  patch paths and helper signatures. Plugin benchmark runners
  (`_run_gsm8k_benchmark`, `_run_mmlu_benchmark`, `_run_ifeval_benchmark`)
  remain in `bench.py` for now — they're tightly coupled to the orchestrator
  and better suited to a dedicated refactor pass.
- **Backend metadata coverage** — `tests/test_metadata.py` (27 tests) covers
  the model probing paths with mocked `httpx.AsyncClient` clients. The
  existing `tests/test_hf_utils.py` already covered the dataset downloader
  retry, resume, and HuggingFace integration paths. `utils/metadata.py`
  coverage rises from ~29% to ~91%.

## [2.0.6] — 2026-06-07

### Fixed

- **KV cache capping skipped for hybrid-attention models** — models like
  Qwen3.6-35B-A3B use a mix of linear/mamba and full-attention layers;
  vLLM's hybrid KV cache manager maps physical blocks to larger logical
  token coverage, so `num_gpu_blocks × block_size` is *not* the effective
  max context length.  Previously the tool would incorrectly cap a 256K
  context to ~32K on these models.  The fix detects hybrid models via
  `mamba_cache_mode` in `/metrics` and trusts the server's `max_model_len`.
  Standard full-attention models continue to be capped correctly.

- **Markdown report Title column showed summary instead of scenario title**
  ([#13](https://github.com/SeraphimSerapis/tool-eval-bench/issues/13)) —
  the Scenario Results table in `.md` reports used the first sentence of the
  evaluation summary for the Title column, making Title and Summary identical.
  Now correctly displays the `ScenarioDefinition.title` (e.g. "Direct
  Specialist Match" instead of "Used get_weather with Berlin only").

- **Token K display uses binary convention** — context pressure display
  now divides by 1024 instead of 1000 to match the LLM industry convention
  (262144 tokens → 256K, not 262K).  Consistent across the summary line
  and budget breakdown.

## [2.0.5] — 2026-06-07

### Fixed

- **Context pressure budget display clarified** — `--context-pressure 1`
  now explicitly reports that the percentage applies to the available fill
  budget, and the displayed scenario headroom no longer double-counts tool
  schema tokens.

## [2.0.4] — 2026-06-02

### Added

- **`--hardmode-only` CLI flag** — run only the 15 Category P Hard Mode
  scenarios. Equivalent to `--hardmode --categories P` but more discoverable.
  Registered in `ARGS_SCHEMA` for programmatic consumers.

### Improved

- **Enriched benchmark reports** — GSM8K, MMLU, and IFEval Markdown reports now
  include:
  - **Error Analysis** section categorizing failures (no answer extracted, wrong
    answer, server errors) for immediate pattern recognition.
  - **Full failure tables** — all failures shown (no more 20-item cap).
    Collapsible `<details>` wrapper when >30 failures for readability.
  - **Question/prompt text** — 120-char excerpt in failure table.
  - **Model response text** — 200-char excerpt in table, 500-char in detailed
    samples. Storage increased from 500→1000 chars.
  - **5 Detailed Failure Samples** — full question + full model response for
    manual inspection and debugging.

### Fixed

- **Empty model responses for reasoning models** — GSM8K, MMLU, and IFEval
  now fall back to `reasoning_content` when `content` is empty. Reasoning
  models (Step-3.7-Flash, DeepSeek-R1, Qwen3) return thinking in a separate
  field; when the model fails to produce a final answer, `content` is empty but
  `reasoning` has the full chain-of-thought. The fix improves both answer
  extraction (the evaluator can now search reasoning text for patterns) and
  report diagnostics (detailed samples show the thinking instead of "(empty)").

- **15 new report rendering tests** — MMLU and IFEval now have `TestReportRendering`
  classes matching GSM8K's coverage. 3 new `--hardmode-only` tests in
  `TestResolveScenarios`. Total test count: **1,765**.

## [2.0.3] — 2026-06-02

### Improved

- **Server errors no longer silently tank accuracy** — API timeouts, connection
  failures, and other server errors under high `--parallel` are now tracked
  separately from genuinely wrong answers. Accuracy is calculated from the
  questions that actually received a response.
- **Live progress shows ⚠ error count** — the real-time stats line now shows
  `✓ 132  ✗ 2  ⚠ 66` when errors occur, making it clear what's a wrong answer
  vs. what's a server failure.
- **Error summary in final output** — when errors occur, a yellow warning line
  explains the count and that they are excluded from accuracy.
- **Noisy `Error on question N:` logs suppressed** — downgraded from `WARNING`
  to `DEBUG`. Under `--parallel 16`, dozens of server timeouts are expected
  behavior, not alarming warnings.

### Fixed

- **`RuntimeError: Event loop is closed` after GSM8K / MMLU / IFEval completes** —
  `asyncio.run(adapter.aclose())` was called after `asyncio.run(run())` had
  already closed the event loop. The httpx client's connections were still bound
  to the dead loop, causing a crash on cleanup. Moved `adapter.aclose()` inside
  the `run()` coroutine so it closes on the same event loop.
- **Laggy progress updates for MMLU and IFEval** — both plugins used an O(n)
  scan (`sum(1 for r in results if r)`) with no lock to count completions on
  every progress tick. Replaced with an atomic `progress_counter` +
  `asyncio.Lock`, matching the pattern GSM8K already used.

## [2.0.1] — 2026-06-01

### Added

- **Expanded Hard Mode pack** — Added ten opt-in Category P scenarios
  (`TC-75` through `TC-84`) for missing-parameter detection, unavailable
  capabilities, irrelevant-tool restraint, independent and dependency-aware
  calls, transactional state safety, tool-output prompt injection, stale
  memory, strict JSON chaining, and long-horizon recovery.

- **Hard Mode diagnostics** — Scenario results now record informational
  same-turn parallel tool-call telemetry and optional per-call state
  checkpoints. Parallel execution is not required for correctness, preserving
  compatibility with backends such as llama.cpp.

### Fixed

- **`--parallel` ignored by GSM8K, MMLU, and IFEval** — the `--parallel N`
  flag only applied to the tool-call scenario orchestrator; plugin benchmarks
  always ran sequentially (`concurrency=1`). Now all three plugin `run()`
  calls receive `concurrency=args.parallel`, enabling concurrent API requests.
  The plugins already had semaphore-based concurrency internally — only the
  CLI wiring was missing.


## [2.0.0] — 2026-05-31

### Changed (Benchmark Integrity — 2.0 Readiness)

- **Resume merges into original run** — `--resume <RUN_ID>` now reuses the
  original run ID and merges prior passed results with new results, producing
  a complete, comparable run instead of a partial fragment. Resumed runs are
  rescored through the standard aggregation path and reports contain merged
  traces.

- **Leaderboard comparability guards** — Runs are now grouped by
  deterministic `config_fingerprint` instead of model alone.  Fingerprints
  include the scenario set, scoring options, and deployment metadata. A
  `Config` column replaces the old `N` column, showing `backend/scenarios`.

- **Plugin results persisted to SQLite** — GSM8K, MMLU, and IFEval results
  are now stored in the `scenario_runs` table with `run_type` column
  (`gsm8k`, `mmlu`, `ifeval`). `RunContext` metadata is serialized explicitly
  and persistence errors are surfaced. Schema migration is automatic.

- **Run ID uniqueness** — Timestamps now use microsecond resolution; a random
  4-byte nonce is mixed into the hash to prevent collisions. Deterministic
  `config_fingerprint` values provide a separate comparison identity.

- **TC-64 no longer sends tools** — The "Simple Schema Compliance" scenario
  now sets `tools_override=[]` so no tools are sent to the model.  The
  orchestrator correctly distinguishes `None` (use defaults) from `[]`
  (explicitly no tools).

- **Error injection is reproducible** — When `--seed` is set, error injection
  uses a per-scenario seeded `random.Random` instance, ensuring deterministic
  injection patterns regardless of execution order or Python hash seed.

- **`output_dir` docstring fixed** — The API docstring now correctly states
  that `output_dir` controls Markdown reports only, not the database.

- **`test_adapter.py` included in CI** — The 30 adapter tests use httpx mocks
  (no network), so they now run in all test suites.  Test count: 1,706.

- **Resume config validation** — `--resume` now validates model and backend
  match the prior run before proceeding.  Mismatches abort with a clear error.

- **Resume display scoring** — The live display now shows the merged total
  score after resume, not just the rerun subset score.

- **Legacy resume trace safety** — Prior passes without `raw_log` traces are
  automatically rerun for full-trace compliance instead of silently producing
  blank trace sections.

- **Benchmark revision fingerprinting** — `config_fingerprint` now includes
  `tool_eval_bench.__version__`, preventing cross-version runs from being
  grouped as comparable on the leaderboard.

- **Standalone mode persistence** — `--perf-only`, `--perf-legacy-only`,
  `--spec-bench`, and context-pressure sweeps now persist to SQLite, satisfying
  the project rule that every completed run is stored.

- **Plugin fingerprint enrichment** — GSM8K, MMLU, and IFEval fingerprints
  now include temperature, seed, shuffle, and subjects parameters.

- **`--compare` warns on incomparable runs** — McNemar analysis now warns
  when runs have different config fingerprints.

- **`--weight-by-difficulty` in live display** — The live display and
  multi-trial scoring now respect the weighted scoring flag.

- **SCHEMA_VERSION bumped to 2** — Reflects new CLI arguments added in 2.0.

- **CI tests Python 3.13** — Test matrix expanded to 3.11, 3.12, and 3.13.

- **Release checklist** — Added `RELEASING.md` with documented workflow for
  wheel, sdist, install-smoke, tag, and publish.

### Added

- **McNemar's significance test** in `--compare` — Automatically computes
  whether differences between two runs are statistically significant using
  McNemar's chi-squared test with continuity correction.  No external
  dependencies (uses stdlib `math.erfc`).  Reports p-value, discordant
  pair count, and direction.

- **Difficulty tier classification** — All 74 scenarios now have a
  `difficulty` rating (1–5 scale: trivial → very hard).  Distribution:
  4 trivial, 17 easy, 31 moderate, 20 hard, 2 very hard.  Field is
  available on `ScenarioDefinition.difficulty` for downstream reporting.

- **Difficulty in reports** — Markdown reports now include a `Diff` column
  with star ratings (★–★★★★★) in the scenario results table, plus a
  "Performance by Difficulty" summary section showing pass rates per tier.
  The `--dry-run` output also shows difficulty alongside each scenario.

- **Difficulty-weighted scoring** (`--weight-by-difficulty`) — Optional CLI
  flag that multiplies each scenario's points by its difficulty tier (1–5)
  before computing the final score.  The weighted score is shown in reports,
  CLI output, and JSON alongside the standard unweighted score.

- **Run resume** (`--resume <RUN_ID>`) — Resume a previous run by skipping
  scenarios that already passed.  Loads completed results from SQLite and
  re-runs only the failed/partial scenarios.  Use `--history` to find run IDs.

- **Pluggable benchmark abstraction** (`domain/plugin.py`) — new `BenchmarkPlugin` ABC
  and `BenchmarkResult` dataclass that allow adding external benchmark modules (GSM8K,
  future MMLU, HumanEval, etc.) alongside the existing tool-call evaluation. Plugins
  share infrastructure (adapter, storage, reporting) but own their own orchestration.
  Plugin registry at `plugins/registry.py` provides `get_plugin()` and `available_plugins()`.

- **GSM8K benchmark plugin** (`--gsm8k` / `--gsm8k-only`) — Grade School Math 8K accuracy
  evaluation using the `openai/gsm8k` dataset (1,319 test questions). Features:
  - **8-shot chain-of-thought** prompting by default (configurable: `--gsm8k-shots 0-8`)
  - **Automatic dataset download** from HuggingFace Datasets Server API on first use,
    cached locally to `data/gsm8k/test.jsonl` (no `datasets` library dependency)
  - **Multi-strategy answer extraction**: standard `#### N` marker → "the answer is N"
    pattern → last number fallback, with comma/currency/whitespace normalization
  - **Rich progress display** with live accuracy percentage during evaluation
  - **Markdown report generation** with accuracy stats, extraction method breakdown,
    and failed-question traces
  - `--gsm8k-limit N` to control question count (default: 200, `0` = all 1,319)
  - `--gsm8k-shuffle` with `--seed` for reproducible random ordering
  - Star ratings mapped from accuracy: ★★★★★ (≥90%) to ★ (< 40%)
  - CLI flags follow existing patterns (`--gsm8k` adds to tool-eval, `--gsm8k-only` skips it)
  - **Visible dataset download**: first run shows a Rich spinner with live row count
    during download from HuggingFace; subsequent runs show a quick cache-hit message

- **65 new tests** — 25 evaluator tests (answer extraction/comparison), 30 dataset/prompts/
  rating/report-rendering tests, 6 plugin interface tests, 4 CLI schema entries.

- **MMLU benchmark plugin** (`--mmlu` / `--mmlu-only`) — Massive Multitask Language
  Understanding evaluation using the `cais/mmlu` dataset (14,042 test questions across
  57 subjects in 4 categories). Features:
  - **5-shot per-subject prompting** using dev-split exemplars (configurable: `--mmlu-shots 0-5`)
  - **Automatic dataset download** from HuggingFace Datasets Server API, cached to
    `data/mmlu/test.jsonl` and `data/mmlu/dev.jsonl`
  - **Multi-strategy answer extraction**: exact single letter → "the answer is X" pattern →
    first standalone A/B/C/D letter
  - **Per-category breakdown** (STEM, Humanities, Social Sciences, Other) in reports
  - **Subject and category filtering**: `--mmlu-subjects STEM,abstract_algebra`
  - `--mmlu-limit N` to control question count (default: 500, `0` = all 14,042)
  - Rich progress display with live accuracy during evaluation

- **IFEval benchmark plugin** (`--ifeval` / `--ifeval-only`) — Instruction Following
  Evaluation using the `google/IFEval` dataset (541 prompts, 25 constraint types).
  Features:
  - **25 deterministic constraint checkers**: word/sentence/paragraph count, keyword
    existence/frequency/forbidden, JSON format, bullet lists, highlighted sections,
    title detection, no-comma, uppercase/lowercase/title-case, end phrase, quotation,
    repeat prompt, two responses, postscript, language detection, and more
  - **Dual accuracy metrics**: prompt-level (all constraints must pass) and instruction-level
    (individual constraint pass rate)
  - **Per-constraint-type breakdown** in reports (sorted by accuracy, worst first)
  - All evaluation is purely programmatic — no LLM-as-judge
  - `--ifeval-limit N` to control prompt count (default: all 541)
  - Rich progress display with live prompt/instruction accuracy

- **HuggingFace `datasets` library fast path** — all three plugins (GSM8K, MMLU, IFEval)
  now try loading datasets via `from datasets import load_dataset` first, which downloads
  directly from the HuggingFace git repo (no datasets-server API, no 429 rate limits).
  Falls back to the REST API with retry/resume if `datasets` is not installed.
  Install with: `pip install tool-eval-bench[hf]`

- **Resumable downloads** — REST API downloads now use incremental partial cache files
  (`*.partial.jsonl`). On 429 failure, progress is saved automatically. Re-running the
  command resumes from where it stopped instead of starting from scratch.

- **Live question display** — all three benchmark progress bars now show the last
  completed question/prompt with ✓/✗ verdict, answer vs expected, and a truncated
  snippet of the question text. Gives users something interesting to watch during
  long evaluation runs.

- **105 new tests** — 34 MMLU tests (answer extraction, evaluation, subject mapping,
  prompt building, ratings), 56 IFEval tests (all 25 constraint types, evaluator,
  registry, edge cases), 15 HF utils tests (download/resume, partial cache,
  `datasets` library integration). Total test count: **1,660**.
## [1.8.0] — 2026-05-19

### Removed

- **Interactive TUI (`-i/--interactive`)** — the Textual-based TUI (`tui/` package,
  `textual` optional dependency, `pip install tool-eval-bench[tui]`) has been removed.
  The project's stated interface is the CLI; shipping a second UI surface increases
  maintenance without benefit to the benchmark mission (AGENTS.md: "no TUI").
  The Rich-based live monitors (`--spec-live`, `--no-live`) are unaffected — they run
  inline in the terminal and have no external dependency.

### Changed

- **`ARGS_SCHEMA` now covers all public CLI args** — `schema.py` previously documented
  ~25 of the ~40+ public flags.  The schema now matches the parser exactly: every
  public argument is present, and a new drift-detection test
  (`TestArgsSchema::test_all_parser_args_in_schema_or_hidden`) will fail if they
  diverge in the future.
- **`_make_parser()` extracted from `main()`** — the argparse parser is now built by a
  standalone function, making it inspectable by tests and external tools without
  consuming `sys.argv`.

### Added

- **Golden-trace evaluator contract tests** (`tests/test_evaluator_contract.py`) —
  PASS/FAIL/PARTIAL golden traces for all 15 base scenarios (TC-01 to TC-15),
  including paraphrased refusals, malformed-but-common JSON arguments, wrong-order
  tool calls, and injection-leakage detection.  Protects scoring semantics from
  accidental changes to evaluator logic.


## [1.7.0] — 2026-05-11

### Added

- **Ctrl+R session reset in `--spec-live`** — press Ctrl+R to reset all session
  counters, sparkline history, and sticky gauges without restarting the monitor.
  A brief "⟳ Session reset" flash banner confirms the reset for 3 poll cycles.
  Useful for isolating workload-specific measurements (e.g., switching prompts
  mid-session).  The helper text at the bottom now shows `Ctrl+R reset · Ctrl+C exit`.
- **Reliable draft model detection** — `--spec-live` now probes `/v1/models`
  and `/version` at startup to detect draft model names and speculative decoding
  configuration.  Previously relied on Prometheus label heuristics that rarely
  matched real vLLM deployments.  When `/v1/models` returns 2+ model entries,
  the non-primary model is identified as the draft model and displayed in the
  header (`▸ Qwen3-35B  ← Qwen3-0.6B`).  If vLLM's `/version` endpoint
  exposes `speculative_config`, the method and `num_speculative_tokens` are also
  extracted.  The `--spec-method` CLI flag still takes highest priority.
- **High-k per-position scaling** — increased `max_positions` from 16 to 64 for
  setups with many speculative tokens (e.g., k=20, k=32).  The horizontal bar
  layout already auto-wraps to multiple rows; this just removes the artificial cap.
- **13 new tests** — covering `ServerSpecInfo`, `probe_server_spec_info` with
  mocked `/v1/models` responses, dashboard rendering with `ServerSpecInfo` (draft
  model priority, reset flash, Ctrl+R hint), and high-k position scaling (20 and
  32 positions).  Total test count: **1,424**.

### Fixed

- **Context pressure sweep alternating pass/fail** — when using
  `--context-pressure-sweep`, adjacent pressure levels produced a perfectly
  deterministic ✅/❌/✅/❌ alternating pattern regardless of model or server.
  Root cause: the sweep shared a single `OpenAICompatibleAdapter` across
  multiple `asyncio.run()` calls.  `httpx.AsyncClient` is bound to the event
  loop it was created in; when `asyncio.run()` closes that loop, the client
  becomes unusable but reports `is_closed=False`.  The next level reuses the
  stale client → instant `RuntimeError: Event loop is closed` → scenario FAIL.
  The failure causes the client to be GC'd, so the *next* level gets a fresh
  one and PASSes — producing perfect alternation.
  Fix: create a fresh adapter per sweep level.  Additionally, fill budgets are
  now quantised to chunk boundaries (`_TOKENS_PER_FILLER_CHUNK + 20`) and
  `build_pressure_messages()` / `calibrate_pressure_messages()` accept a `seed`
  parameter for fully deterministic, reproducible sweeps when `--seed` is set.

- **Context pressure single-run timeout** — when using `--context-pressure`
  with large fills (e.g. 182K tokens at 75% of a 260K context), the default
  60-second timeout was too short for prefill, causing scenarios to fail with
  a timeout.  The sweep path already auto-scaled timeouts but the single-run
  path did not.  Fix: apply the same auto-scaling formula
  (`120s base + 60s per 50K fill tokens`) to the single-run path.

## [1.6.0] — 2026-05-07

### Added

- **Public programmatic API** (`tool_eval_bench.api`) — new `run_benchmark()` async
  function for headless/library invocation by external integrators (e.g. sparkrun).
  Returns a versioned JSON-serializable dict with `schema_version` and promoted
  Spark Arena fields (`final_score`, `rating`, `safety_warnings`, `deployability`,
  `responsiveness`, `total_scenarios`).  Persistence is opt-in via `persist=False`
  for callers that handle their own storage.
- **`--json-file PATH`** CLI flag — write JSON results to a file instead of stdout
  (implies `--json`).  Keeps stdout clean for subprocess consumers.  Emits a
  `benchmark_complete` JSONL event on stderr when done.
- **JSONL progress events on stderr** — when `--json` is active, structured progress
  events (`scenario_start`, `scenario_result`) are emitted as one-line JSON objects
  on stderr for real-time progress tracking by orchestrators.
- **Machine-readable args schema** (`tool_eval_bench.schema`) — `ARGS_SCHEMA` list
  and `get_schema()` function for external tools to validate benchmark configuration.
  Also re-exported from `tool_eval_bench.api.ARGS_SCHEMA`.
- **Convenience re-export** — `from tool_eval_bench import run_benchmark` works
  as a shorthand for the `api.run_benchmark()` function.
- **Server auto-discovery** — when `--base-url` is omitted (and no env var is set),
  the CLI probes localhost on common inference server ports (8000, 8080, 8081, 8082,
  30000, 4000, 3000, 11434, 5000) and auto-selects the first responding server.
  Backend is identified via HTTP response header sniffing, with port-based
  fallback hints.  In `--json` mode, emits a `server_discovered` JSONL event.
- **`--probe` readiness check** — verify that a server is reachable and exit.
  Exits 0 if the server responds to `/v1/models`, exit 1 otherwise.  Emits
  a `probe_result` JSONL event in `--json` mode.  Useful for CI/CD pipelines
  and sparkrun recipes where the benchmark runs right after server startup.
- **Headless model auto-selection** — in `--json` mode, when multiple models
  are served, the first model is auto-selected instead of blocking on
  `input()`.  Emits a `model_auto_selected` JSONL event on stderr.
- **Structured headless errors** — connection failures, HTTP errors, and
  empty model lists emit JSONL error events on stderr in `--json` mode
  instead of Rich-formatted console markup.
- **Differentiated exit codes** — exit 2 for connection/HTTP errors,
  exit 3 for no-models-found (previously all exit 1).
- **`SKILL.md`** — comprehensive agent guide covering zero-config usage,
  JSON output schema, JSONL progress events, exit codes, programmatic API,
  result interpretation, and common pitfalls.
- **`py.typed` marker** — package is now recognized as typed by mypy/pyright.
- **`--dry-run` flag** — lists which scenarios would run, with category breakdown
  and estimated time, then exits (no server connection needed).  In `--json` mode,
  outputs a machine-readable JSON document.
- **Structured error taxonomy** (`tool_eval_bench.domain.errors`) — canonical
  error code constants (`CONNECTION_FAILED`, `HTTP_ERROR`, `DETECTION_FAILED`,
  `INVALID_RESPONSE`, `NO_MODELS`, `NO_SERVER`) used by all headless JSONL error
  events.  Integrators can exhaustively match on these values.
- **`RunRepository` context manager** — supports `with RunRepository() as repo:`
  for automatic cleanup of SQLite connections.
- **17 new tests** — persistence bypass, backend detection, async re-export,
  error constants, context manager, async_tools JSON safety, dry-run scenarios.
  Total test count: **1,397**.

### Fixed

- **`BenchmarkService` persistence bypass** — `repo or RunRepository()` silently
  replaced `None` with a default, defeating `persist=False`.  Now uses a sentinel
  pattern to distinguish "not provided" from "explicitly None".
- **Probe URL 404 fallback was a no-op** — when `base_url` ended with `/v1`, the
  fallback retried the same URL.  Now uses shared `utils/urls.py` for consistent
  URL construction.
- **`benchmark_complete` JSONL event emitted `null` for `final_score`** — was
  reading from the wrong nested path (`scores.final_score`) instead of the
  promoted top-level field.
- **`__init__.py` re-export was sync returning a coroutine** — callers expecting
  `asyncio.run(run_benchmark(...))` got a doubly-wrapped coroutine.  Now properly
  `async`.

### Changed

- **`BenchmarkService` persistence is now optional** — `repo` and `reporter`
  constructor arguments accept `None` to skip SQLite and Markdown writes.  This
  supports the `persist=False` path in the public API without breaking existing
  CLI behavior (which always passes concrete instances).
- **Warmup and WIP warnings suppressed in `--json` mode** — the server warmup
  request and `--llm-judge`/`--experimental-async` warnings no longer print to
  stdout when `--json` is active, keeping stdout clean for JSON parsing.
- **`.env` isolation verified** — `load_dotenv(override=False)` ensures that
  environment variables set by the calling process (e.g., an agent) are never
  overridden by a `.env` file.  CLI flags take priority over env vars.
- **Backend detection uses response headers** — `_detect_backend_from_response()`
  inspects the `Server` HTTP header to identify vLLM, SGLang, and llama.cpp,
  falling back to port-based hints only when headers are inconclusive.
- **Filler text replaced** — the Gatsby excerpt in `throughput.py` was replaced
  with original LLM-inference themed text (no copyright concern).
- **Large-toolset detection uses category check** — replaced fragile scenario-ID
  string parsing with semantic `Category.L` membership check.
- **Global `_mtp_warned` eliminated** — moved into `TokenizerConfig` as a
  per-run instance attribute for thread/library safety.
- **Silent exception handlers annotated** — 6 bare `except Exception:` blocks
  across core modules now include `logger.debug` calls for debuggability.
- **`async_tools.py` uses `json.dumps` consistently** — replaced fragile f-string
  JSON construction with `json.dumps()` in all branches of `format_async_status()`.
  A quote character in an error message previously produced invalid JSON.

## [1.5.1] — 2026-05-04

### Added

- **`--spec-method` works with `--spec-live`** — the method badge in the
  dashboard header can now be set explicitly via `--spec-method dflash` (or
  `mtp`, `eagle`, `ngram`, `draft`).  This is necessary because vLLM doesn't
  expose the speculative decoding method in its Prometheus `/metrics` output,
  making auto-detection impossible for most setups.  `dflash` was also added
  as a new choice alongside the existing `auto`, `mtp`, `draft`, `ngram`,
  and `eagle` options.
- **Draft model name in header** — if Prometheus metric labels contain
  `model_name` values for multiple models (target + draft), the dashboard
  header now shows the draft model name: `▸ Qwen3.6-27B  ← Qwen3-0.6B`.
- **`draft_flash` regex pattern** — method detection now matches `draft_flash`
  and `draft flash` in addition to `dflash`, in case future vLLM versions
  expose the method string in metric labels.
- **`mlp_speculator` method detection** — added pattern and badge for IBM's
  MLP speculator method.
- **10 new tests** — covering `draft_flash` detection, `mlp_speculator`
  detection/label, model name extraction from Prometheus labels, and
  multi-row horizontal bar scaling (6, 12 positions, narrow terminal).
  Total test count: **1,403**.

### Fixed

- **Per-position bars with >6 spec tokens** — increased `max_positions` from
  8 to 16.  The horizontal bar layout now **auto-wraps to multiple rows** when
  there are too many positions for the terminal width (minimum 14 chars per
  cell).  For example, `k=12` at 100 columns renders as 2 rows of 6.

## [1.5.0] — 2026-05-03

### Added

- **Alternate screen buffer for `--spec-live`** — the dashboard now enters the
  terminal's alternate screen buffer (like htop, vim, less) for a clean,
  full-terminal canvas.  Previous terminal output is completely hidden while the
  dashboard is active and restored on exit (Ctrl+C).  This eliminates visual
  clutter from prior command output or log lines.
- **Session-relative metrics** — all cumulative values (acceptance rate, τ,
  per-position rates, session counters) now start from zero when the dashboard
  opens.  A baseline snapshot is captured on first scrape and all metrics are
  computed as deltas from that baseline.  This lets you observe how different
  workloads actually perform during each monitoring session.
- **Per-position acceptance from vLLM counters** — fixed parsing of per-position
  acceptance data.  vLLM v1 exposes `spec_decode_num_accepted_tokens_per_pos_total`
  (a counter per position), not the rate gauge we were looking for.  The parser
  now reads both counter and gauge formats: counters are converted to rates via
  `counter[pos] / num_drafts`, and gauge rates (if present) take priority.
- **Full-width horizontal per-position display** — moved per-position acceptance
  from a cramped left-column vertical panel to a full-width horizontal row at the
  bottom of the dashboard.  Each position shows an inline bar with percentage
  (`p0 ████ 83%  p1 ███ 64% ...`), making the data readable at any terminal width.
- **Method badge always visible** — the speculative decoding method badge
  (`⟨ Draft Flash ⟩`, `⟨ MTP ⟩`, `⟨ EAGLE ⟩`, etc.) now always appears in the
  dashboard header when spec decode is active.  Previously, servers that didn't
  include method keywords in their Prometheus output got no badge.  Unknown
  methods now show `⟨ Speculative Decoding ⟩`.
- **Rolling Averages shown immediately** — the Rolling Averages panel is now
  visible from the first poll with 0.0 values, rather than waiting for 5+
  samples to appear.
- **Session α always visible** — Session acceptance rate row in Engine & Session
  starts at 0.0% immediately, rather than appearing only after the first draft.
- **7 new per-position counter tests** — covering counter parsing, rate
  computation from counters/num_drafts, monotonic decay, gauge-takes-priority,
  zero-drafts safety, and underscore prefix variants.
  Total test count: **1,393**.

### Fixed

- **KV Cache truncation at narrow terminals** — the KV cache fill bar and
  percentage text overflowed at half terminal width.  Reduced label from
  "KV Cache Fill" to "KV Cache", made bar width dynamic (`max(6, min(10,
  col_w - 20))`), reduced padding from 2 to 1, and switched to `.0f` format.
- **Per-position labels truncated to `...`** — in the old vertical layout, the
  `p0`, `p1` position labels were being truncated to `...` because the column
  was too narrow.  The new horizontal layout eliminates this entirely.
- **Pre-populated values from server history** — per-position rates and
  acceptance rate showed all-time server values on dashboard start instead of
  session-relative data.  Now properly cleared until new session data arrives.

### Changed

- **Speculative decoding config in `--spec-live` dashboard** — the live monitor
  now detects and displays the active speculative decoding method (dflash,
  MTP, EAGLE, EAGLE-3, N-Gram, or draft model) as a color-coded badge in the
  dashboard header.  The inferred `num_speculative_tokens` (k) is shown in the
  acceptance rate annotation and the metrics panel.  Method detection scans
  Prometheus `/metrics` text for keyword hints (HELP lines, labels, method
  names) and falls back to "Speculative Decoding" when spec decode counters are
  present but no specific method is identified.
- **Per-position acceptance decay analysis** — when the server exposes
  per-position acceptance rates (vLLM), the Per-Position Acceptance panel now
  includes: effective positions count (positions with >20% acceptance),
  50% drop point, and geometric decay rate (γ/pos).  Provides at-a-glance
  insight into how quickly draft quality degrades across positions.
- **Method-specific efficiency insights** — the efficiency insight line now
  accounts for the detected spec decode method: MTP models get contextual
  guidance ("acceptance at N% is typical for MTP"), dflash models with high
  draft tokens and low utilization get targeted reduction suggestions with the
  current `num_speculative_tokens` value displayed.

## [1.4.3.1] — 2026-04-26

### Fixed

- **Reports and DB created inside `.venv/` instead of project directory** (Issue #9) —
  `_default_reports_root()` and `_default_db_path()` resolved paths relative to the
  installed package location (`__file__`), which — when installed via `pip install -e .`
  or `pip install .` — points inside `.venv/lib/python3.x/site-packages/…`. Walking up
  four parent directories from there lands in `.venv/`, not the project root. Changed
  both functions to use `Path.cwd()` so reports go to `./runs/` and the database to
  `./data/benchmarks.sqlite` relative to wherever the CLI is invoked.
- **`--spec-live` session counters show server-lifetime totals** — the baseline
  snapshot (used to compute session-relative Accepted/Drafted counts) was only
  captured when the first scrape had *no* spec-decode counters. When the server
  already had counters (the normal case — vLLM had processed prior requests), the
  baseline was never set and the dashboard showed cumulative server-lifetime numbers
  instead of session-relative ones.

### Added

- **`--output-dir DIR` CLI flag** — specify a custom directory for Markdown report
  files (scenario, throughput, spec-decode, and cross-trial summary reports). When
  omitted, reports default to `./runs/` in the current working directory. The tool
  still generates filenames automatically (`<run_id>.md` under `YYYY/MM/` subfolders).

## [1.4.3] — 2026-04-25

### Fixed

- **Scientific notation breaks Prometheus parsing** — cumulative counters that
  vLLM reports in scientific notation (e.g. `1.378e+06`) were silently dropped
  by the regex patterns in both `spec_live.py` and `speculative.py`, causing
  inflated prefix cache hit rates and zero throughput readings. All `_NUM`
  capture groups now handle `\d+(?:\.\d+)?(?:[eE][+-]?\d+)?`.
- **KV cache metric always 0 in `--spec-live`** — the scraper treated `0.0` as
  "metric not present" and fell back to the sentinel `None`. Changed to an
  explicit `None` sentinel so a genuine 0% fill is rendered correctly.
- **KV cache fill stuck at 0 on vLLM ≥0.8** — added fallback to the legacy
  `gpu_cache_usage_perc` gauge when `kv_cache_usage_perc` is absent.
- **Spec-bench results table truncated on narrow terminals** — removed
  `expand=True` (table now auto-sizes to content), added `min_width` to
  columns that were clipping (`α %`, `Draft t/s`, `TTFT ms`), shortened
  `Window` → `Win` and clarified `TTFT` → `TTFT ms`.
- **Prometheus warning runs into first result** — added a blank line after the
  server-wide aggregates warning in `--spec-bench` output.

### Changed

- **Merged Draft Efficiency gauge into Acceptance Rate** — the `--spec-live`
  dashboard previously showed two separate gauge bars (Acceptance Rate and
  Draft Efficiency) that displayed nearly identical percentages with small
  draft windows (MTP, `num_speculative_tokens=1`). Consolidated into a single
  `ACCEPTANCE RATE` bar with `τ=X.X/N` annotation, saving vertical space.
- **Version stamp in benchmark summary** — the final `Benchmark Complete` panel
  and all Markdown reports now include `tool-eval-bench vX.Y.Z` for
  reproducibility (Issue #6).

### Added

- **35 new evaluator tests** — edge-case coverage for TC-51 through TC-63
  (planning, composition, adversarial categories): clarification detection,
  single-constraint partial scoring, both-sources-no-synthesis, email-not-to-CFO,
  and more. Total test count: **1,240** (up from 1,205).
- **Regression tests for Prometheus fixes** — scientific notation parsing,
  KV cache `None` sentinel fallback (3 branches), counter-derived throughput,
  and prefix cache hit rate math in both `spec_live.py` and `speculative.py`.

## [1.4.2] — 2026-04-24

### Added

- **`--hardmode` ceiling-breaking scenarios** — 5 new Hard Mode scenarios
  (Category P, TC-70 to TC-74) that challenge models beyond the standard 69-scenario
  suite. Designed for models that score 100% on the vanilla benchmark:
  - **TC-70**: Adversarial near-duplicate tool definitions (Europe-only vs global weather)
  - **TC-71**: Ambiguous recipient resolution (3 matching contacts → must clarify)
  - **TC-72**: Cascading error recovery (corrupted file → alternative → email chain)
  - **TC-73**: Multi-constraint composition (search + 3 filters + contact + email)
  - **TC-74**: Stateful multi-turn corrections (4 follow-ups modifying event details)
  - Hard Mode scenarios are opt-in (`--hardmode`) and excluded from the base score
    to maintain comparability with existing results.
  - Use `--hardmode --categories P` to run only Hard Mode, or combine with
    `--context-pressure` for maximum difficulty.

- **Draft efficiency metrics in `--spec-bench`** — three new computed metrics that
  surface actionable tuning signals for speculative decoding:
  - **Waste ratio**: fraction of drafted tokens rejected by the verifier (1 − α).
    Color-coded in CLI output: green ≤20%, yellow ≤50%, red >50%.
  - **Draft window**: average tokens drafted per speculative step — reveals the
    configured `num_speculative_tokens` setting. Compare with τ (acceptance length)
    to see window utilization.
  - **Draft t/s**: rate at which draft tokens are generated, regardless of acceptance.
    Compare with effective t/s to quantify draft overhead.
  - **Window utilization insight**: CLI prints `τ/window` utilization percentage and
    automatically suggests reducing `num_speculative_tokens` when utilization drops
    below 50%.
  - **Draft Efficiency section in Markdown reports** with utilization table and
    tuning recommendation.
  - All metrics derived from existing Prometheus counter deltas — no new server
    requirements.

- **`--spec-live` live speculative decoding monitor** — a real-time Rich Live
  terminal dashboard that continuously polls the server's Prometheus `/metrics`
  endpoint and renders:
  - **Acceptance rate gauge** with color gradient (red → green)
  - **Draft efficiency gauge** showing τ/window utilization with auto-tuning hints
    (suggests optimal `num_speculative_tokens` when utilization drops below 30%)
  - **Per-position acceptance waterfall** — bar chart showing acceptance rate
    decay across 8 draft positions
  - **Throughput sparklines** — rolling 60-second history for accept rate, gen t/s,
    accepted t/s, and waste ratio with min/max range annotations
  - **Rolling averages panel** — session-level mean α, gen t/s, and accepted t/s
    (appears after 5+ data points)
  - **Engine status** — GPU KV cache usage, prefix cache hit rate, running/waiting
    requests, prompt t/s
  - **Session totals** — cumulative accepted/drafted tokens with session-wide α
  - Activity indicator (pulsing ◉/◎) and uptime/poll counter
  - Session summary panel printed on exit (Ctrl+C) with mean ± std, peak values
  - Configurable poll interval via `--spec-live-interval` (default: 1s)
  - Works with `--metrics-url` for proxied setups (LiteLLM → vLLM)
  - New modules: `cli/spec_live_display.py` (Rich rendering) and
    `runner/spec_live.py` (Prometheus scraping and delta computation)

### Fixed

- **`--spec-live` sticky gauges** — Gen t/s, Prompt t/s, and KV cache gauges
  now retain the last non-zero reading between vLLM's ~10-second Prometheus
  update intervals, eliminating the flicker-to-zero behavior. Per-position
  acceptance panel shows a helpful note when MTP servers don't expose
  per-position rates.

## [1.4.1] — 2026-04-24

### Fixed

- **HTTP 5xx errors no longer swallowed by adapter** — the `OpenAICompatibleAdapter`
  previously caught all `httpx.HTTPStatusError` exceptions (including 500 Server Error)
  and returned a "graceful" `ChatCompletionResult`.  This caused genuine server failures
  to be silently absorbed, producing false-positive benchmark results.  Now only **4xx
  errors** (malformed tool-call arguments, common with vLLM) are caught gracefully;
  **5xx errors** are re-raised so the benchmark correctly fails on server-side issues.
  Applied to both `_non_stream_request` and `_stream_request` paths.

- **TC-11 / TC-35 eval messages disambiguated** — both scenarios tested "unnecessary
  calculator use" but their pass/partial/fail messages were nearly identical, making it
  hard to tell them apart in reports.  TC-11 messages now emphasize **arithmetic
  restraint** ("mental math was sufficient"), while TC-35 messages emphasize **critical
  thinking about nonsensical requests** ("K→K is an identity conversion, not a real
  task").  Display details updated accordingly.

### Added

- **77 new unit tests** (`test_coverage_gaps.py`) closing coverage gaps across 6 modules:
  - `runner/speculative.py` — `scrape_spec_metrics`, `detect_spec_decoding` (all method
    inference paths: eagle/ngram/mtp/draft_model), `_metrics_url`, `_get_prompt_for_type`,
    `SpecDecodeSample` edge cases (zero tokens, zero baseline)
  - `runner/async_tools.py` — full `AsyncToolExecutor` lifecycle (register, start, poll,
    cancel, failure simulation), `format_async_status` for all 5 status types, and
    `create_example_async_specs`
  - `evals/noise.py` — all 11 enrichment functions + `enrich_payload` dispatcher
    (known tool, unknown tool, error payload, non-dict passthrough, calculator)
  - `storage/db.py` — `get_latest`, `get_scenario_results`, model-filtered `list`,
    upsert-updates-existing, `__del__` safety net
  - `storage/reports.py` — spec-decode report (with/without acceptance rate),
    `_render_run_context` (engine info, quantization, context pressure, extra params,
    server model root), scenario report with `RunContext`/deployability/context pressure,
    throughput report with `RunContext`

- **12 new adapter tests** (`test_adapter.py`) reaching 100% adapter coverage:
  - Streaming SSE accumulation (content, tool-calls, reasoning, usage/token counting)
  - 4xx graceful return vs 5xx propagation (both stream and non-stream)
  - `response_format` and `extra_params` serialization
  - Malformed JSON chunks and empty choice segments in SSE streams

### Changed

- **Total test count**: 1054 → **1143** (+89 tests)
- **Coverage improvements**:
  - `adapters/openai_compat.py`: 55% → **100%**
  - `evals/noise.py`: 78% → **100%**
  - `runner/async_tools.py`: 72% → **100%**
  - `runner/speculative.py`: 63% → **75%**
  - `storage/db.py`: 80% → **96%**
  - `storage/reports.py`: 64% → **88%**
  - Overall: 54% → **58%**

## [1.4.0] — 2026-04-22

### Added

- **Run context metadata in reports** (Issue #6) — benchmark reports and SQLite
  records now include full execution context: tool-eval-bench version, git SHA,
  CLI parameters (temperature, seed, max_turns, timeout, parallel, error_rate,
  thinking mode, extra_params), and best-effort inference engine probing (vLLM
  version, llama.cpp build, LiteLLM version, max_model_len, quantization, GPU
  count).  Reports render two new tables: **Run Context** (all CLI parameters)
  and **Inference Engine** (server-side metadata).  Engine probes are best-effort
  with tight timeouts — failures produce graceful `None` fields, never crashes.
- **Version stamp in reports and display** — the tool-eval-bench version and git
  SHA now appear in Markdown report headers and the Rich live display panel.
- **Engine auto-detection in CLI** — detected engine name, version, quantization,
  context length, and model root are printed as `🔍` lines before the benchmark
  starts (suppressed in `--json` mode).
- **Enriched `--history` output** — the history table now includes a Context column
  showing tool version, backend, engine, temperature (if non-default), and
  quantization.  Old runs without metadata show `—` gracefully.
- **Enriched `--compare` output** — the comparison header panel now shows per-run
  context details (engine version, model root, quantization, host, etc.) so you
  can see *what changed* between two runs at a glance.
- **URL redaction on by default in reports** — server URLs are now automatically
  redacted (`http://***:8000`) in persisted Markdown reports for privacy.  The
  `--redact-url` CLI flag continues to control terminal display separately.
- **`--skip-tool-eval` CLI flag** — skip tool-call scenarios entirely, useful for
  running only `--spec-bench` or `--perf` without the 69 scenario evaluation.
  Example: `tool-eval-bench --spec-bench --skip-tool-eval`.
- **`--no-probe-engine` CLI flag** — disable the HTTP-based engine detection
  probes (`/version`, `/health`, `/v1/models`) for environments where these
  endpoints are slow, unavailable, or behind auth.
- **Metadata in `--export csv|json`** — exported data now includes `tool_version`,
  `engine_name`, `engine_version`, `quantization`, `max_model_len`, `temperature`,
  and `server_model_root` from the run metadata.
- **RunContext in throughput reports** — `--perf-only` and `--perf-legacy-only`
  reports now include the full Run Context and Inference Engine sections.

- **Interactive TUI mode** (`-i` / `--interactive`) — a full Textual-based terminal
  UI for configuring and running benchmarks.  Three screens: **Configure** (server
  connection, model picker, benchmark mode checkboxes, category filter, sampling
  presets, run control), **Running** (live scenario progress grid with per-row
  status updates and progress bar), and **Results** (tabbed view with scores,
  category breakdown, run history, and model leaderboard).  Requires the new
  `[tui]` optional dependency: `pip install tool-eval-bench[tui]`.
- **TUI sampling params** — configure screen now exposes Top-P, Top-K, Min-P, and
  Repeat Penalty in a 2-column grid alongside Temperature.  Values are threaded
  through to the backend as `extra_params`.
- **`__main__.py`** — `python -m tool_eval_bench` now works as an alternative to
  the `tool-eval-bench` console script.

### Fixed

- **TUI benchmark status stuck on PENDING** — the running screen now correctly
  updates scenario status, points, and timing as each test completes.  Root cause:
  `update_cell` was referencing column indices instead of column keys, and the
  callback structure didn't reliably push updates to the Textual UI thread.
- **TUI running scenario not highlighted** — the currently executing test is now
  visually indicated via cursor movement to the active row, and the previous
  "running" badge is cleared when a new scenario starts.
- **TUI scrollbar artifacts** — reduced scrollbar width to 1 character globally
  (`scrollbar-size-vertical: 1`) to eliminate rendering glitches on the vertical
  scrollbar.
- **TUI hover color changes** — disabled background color changes on hover for
  checkboxes and containers, which caused confusing visual artifacts when mousing
  over the configure screen.
- **TUI benchmark mode labels cut off** — mode checkboxes (`Tool-Call Scenarios`,
  `Throughput (llama-benchy)`, `Spec-Decode`) now use `width: 1fr` instead of
  `width: auto` so labels are never truncated regardless of terminal width.
- **TUI category grid text truncation** — category checkboxes now use `width: 1fr`
  per grid cell, and the grid switches from 3 columns to 2 on terminals narrower
  than 90 columns.
- **TUI requires too much scrolling** — tightened padding throughout all three
  screens (reduced top/bottom margins, section spacing, and button bar padding)
  to fit more content in smaller terminal windows.

- **Spec-bench acceptance rate always showing `—`** — Prometheus regex patterns for
  `spec_decode_*` counters did not account for the `{engine="0",model_name="..."}` label
  block that vLLM includes between the metric name and value.  All three regexes now
  accept an optional `{...}` label group, fixing acceptance rate (α), acceptance length
  (τ), and speedup ratio display for vLLM servers.
- **Spec-bench table truncated on narrow terminals** — removed `expand=True` (table now
  auto-sizes to content), dropped redundant Stream t/s column, conditionally hide Speedup
  column when no `--baseline-tgs` is provided, shortened header labels (`α %`, `τ len`,
  `TTFT`, `Total ms`), and use compact depth notation (`4K`, `8K`).  Table now fits
  cleanly at 80 columns.
- **Legacy throughput table truncated on narrow terminals** — removed `expand=True` from
  the built-in `--perf-legacy` table for parity with the spec-bench table fix above.
- **Trial aggregation wrong with `--categories`** — `_run_plain` multi-trial path
  re-imported `ALL_SCENARIOS`/`SCENARIOS` and scored against the full set instead of
  respecting the `--categories` / `--short` filter.  Now uses `_resolve_scenarios(args)`
  consistently.
- **`python -m tool_eval_bench` failed** — added `__main__.py` so the package can be
  invoked as `python -m tool_eval_bench` (previously only the `tool-eval-bench` console
  script worked).
- **Benchmark crash after TC-63: `unhashable type: 'list'`** (Issue #5) — the
  structured output evaluators (TC-64 to TC-69) performed set membership checks
  like `data.get("genre") not in valid_genres`, which raises `TypeError` when a
  model returns a list value (e.g. `"genre": ["sci-fi"]`) instead of a scalar
  string.  Fixed by validating the type with `isinstance(val, str)` before the
  set lookup.  Additionally, the post-loop evaluation call in the orchestrator
  was outside the existing `try/except` block, so any evaluator exception would
  crash the entire benchmark run instead of being recorded as a FAIL.  The
  evaluation phase is now wrapped in its own `try/except` as a safety net.
- **Test suite hardening** — resolved 6 classes of systemic test bugs that had
  accumulated across `test_display.py`, `test_history.py`, `test_leaderboard_display.py`,
  and `test_judge.py`:
- **vLLM 400 crash on malformed tool-call arguments** — when a model (e.g. Gemma 4)
  emits truncated JSON in tool-call arguments, vLLM's `_postprocess_messages` crashes
  with `json.JSONDecodeError` on the next turn.  Two-layer fix:
  1. `_repair_json_str()` in the orchestrator closes unterminated strings and
     brackets before arguments are sent back in conversation history.
  2. The adapter catches `httpx.HTTPStatusError` (400/422) and returns a
     graceful `[server error N]` result instead of crashing the scenario.
- **`.opencode/` removed from repo and git history** — leaked IDE directory
  purged with `git filter-branch`, added to `.gitignore`.
  - Console IO capture: replaced `Console(file=MagicMock())` with
    `Console(file=StringIO(), width=200, no_color=True)` to get real string output.
  - Mock paths: corrected 36 `patch()` targets from `cli.*.RunRepository` to
    `storage.db.RunRepository` (the actual import site).
  - `sys.exit` mocking: added `side_effect=SystemExit` so execution halts correctly.
  - Rich markup assertions: handle `[bold]2[/]/2` variant alongside plain `2/2`.
  - Test data alignment: fixed sort order, computed-vs-fixture fields, stdout
    capture for CSV export, and MagicMock `.error` attribute truthiness.
- **Resource leak in export tests** — `open(file).read()` without closing replaced
  with proper `with open(file) as f:` context managers.
- **Async teardown warnings** — suppressed `RuntimeWarning: coroutine was never
  awaited` and `PytestUnraisableExceptionWarning` via `pyproject.toml`
  `filterwarnings`.  These are garbage-collection artifacts from mocked async
  adapters and do not indicate real bugs.
- **Duplicate `Panel` import in legacy throughput** — removed redundant
  `from rich.panel import Panel` that was already imported at function scope.

### Changed

- **`redact_url` moved to shared utility** — `_redact_url` was inlined in `cli/bench.py`
  and had to be imported by `utils/metadata.py`, violating the layered architecture
  (domain/utils must not import CLI).  Moved to `utils/urls.redact_url()` and the CLI
  now delegates to it.

- **CLI flag grouping** — reorganized 45 flat `--help` flags into 10 logical
  argument groups: connection, sampling, scenario selection, run control, output,
  throughput benchmark, speculative decoding benchmark, context pressure, and
  history & comparison.  The `--help` output is now scannable instead of a wall of
  text.  Zero breaking changes — all flags work identically.
- **WIP flags hidden** — `--llm-judge`, `--judge-model`, and `--experimental-async`
  are suppressed from `--help` output since they currently have no effect.  The flags
  still work (printing a WIP warning) for users who already have them in scripts.
- **Help text tightened** — most flag descriptions shortened to one line, removing
  redundant examples and verbose explanations that inflated `--help` from ~130 to
  ~90 lines.
- **Import standardization** — hoisted ~90 redundant function-level imports to
  top-level across 4 test files (`test_display.py`, `test_history.py`,
  `test_leaderboard_display.py`, `test_judge.py`).  Eliminates duplicated
  `from tool_eval_bench.cli.* import ...` inside every test method.
- **`test_judge.py` cleanup** — replaced 14 `__import__("tool_eval_bench.runner.judge",
  fromlist=[...])` hacks with a clean top-level
  `from tool_eval_bench.runner.judge import judge_failed_scenarios`.


## [1.3.1] — 2026-04-20

### Added

- **`--context-pressure-sweep START-END`** — run scenarios at increasing context pressure
  levels and report the breaking point.  Example:
  `--context-pressure-sweep 0.9-1.0 --sweep-steps 10 --scenarios TC-61 TC-64`
  runs 11 levels (90% → 100%) and shows a compact Rich panel with per-scenario
  pass/fail status, bar chart, and the exact pressure ratio where the model starts
  failing.  Early-stops after 2 consecutive all-fail levels.
- **`--sweep-steps N`** — control granularity of the pressure sweep (default: 5
  intervals = 6 test levels).

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
- **Context pressure ratio=1.0 overflow** — increased `_RESERVED_FOR_SCENARIO` from 8,000
  to 12,000 tokens.  The extra 4K margin absorbs token estimation error (char→token
  approximation) so that `--context-pressure 1.0` can succeed on multi-turn scenarios
  instead of silently overflowing the context window.
- **`rating_for_score` safety-cap gap** — when `safety_capped=True` and `score < 60`,
  the function previously fell through to regular ratings with no safety indication.
  Now returns `★★ Weak (safety-capped)` and `★ Poor (safety-capped)` at all score
  levels, ensuring the safety concern is always visible in the rating string.
- **Defensive token sum** — `score_results()` now uses `(r.prompt_tokens or 0)` to
  guard against potential `None` values in token aggregation.
- **Trace code block language specifier** — Markdown reports now use `` ```text ``
  instead of bare `` ``` `` for trace sections, preventing report corruption when
  model output contains triple backticks.

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
