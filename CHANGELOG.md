# Changelog

All notable changes to `tool-eval-bench` are documented here.

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
