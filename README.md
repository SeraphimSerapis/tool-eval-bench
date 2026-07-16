# tool-eval-bench

A **tool-calling quality benchmark** for evaluating LLM tool-use in agentic workflows across open-weight model serving stacks (**vLLM**, **LiteLLM**, **llama.cpp**). Also includes pluggable accuracy benchmarks (**GSM8K**, **MMLU**, **IFEval**) via the same OpenAI-compatible endpoints.

Inspired by [ToolCall-15](https://github.com/stevibe/ToolCall-15), this tool runs **69 deterministic scenarios** (+ 15 opt-in Hard Mode) through OpenAI-compatible `/chat/completions` endpoints, scores each result as **pass**, **partial**, or **fail**, and produces detailed trace reports. Mock tool responses include realistic payload noise (extra metadata, timestamps, nested objects) to test whether models can extract relevant fields from noisy API responses. It also includes an integrated **throughput benchmark** (llama-bench style) for measuring prefill and token generation speed.

![tool-eval-bench benchmark output](docs/images/benchmark-output.png)

> **Scope.** tool-eval-bench measures *tool-calling quality* — whether a model picks the right tool, passes the right parameters, chains tools correctly, and handles errors and safety boundaries. It is not a full agentic system benchmark (see [Related Work](#related-work) for how it compares to BFCL, PinchBench, and Claw-Eval).

## What It Measures

### Tool-Call Quality (69 scenarios across 15 categories)

| Category | Scenarios | What It Tests |
|---|---|---|
| **A — Tool Selection** | TC-01 – TC-03 | Picking the right tool from 12 options |
| **B — Parameter Precision** | TC-04 – TC-06 | Getting parameters right (units, dates, multi-value) |
| **C — Multi-Step Chains** | TC-07 – TC-09, TC-61 | Chained reasoning, data threading, parallel calls, async polling |
| **D — Restraint & Refusal** | TC-10 – TC-12 | Knowing when NOT to call tools |
| **E — Error Recovery** | TC-13 – TC-15 | Handling failures and preserving data integrity |
| **F — Localization** | TC-16 – TC-18 | German language, timezone awareness, translate+forward |
| **G — Structured Reasoning** | TC-19 – TC-21 | Message routing, data extraction, constraint validation |
| **H — Instruction Following** | TC-22 – TC-24, TC-44 – TC-45 | Output format, tool prohibition, multi-constraint, tool_choice compliance |
| **I — Context & State** | TC-25 – TC-27, TC-46 – TC-50, TC-62 – TC-63 | Cross-reference, state consistency, multi-turn correction, 6-turn chains, constraint accumulation |
| **J — Code Patterns** | TC-28 – TC-30 | Read-before-write, explain vs execute, chained conditional |
| **K — Safety & Boundaries** | TC-31 – TC-36, TC-41 – TC-43, TC-57 – TC-60 | Ambiguity, prompt injection (file/search/system/sleeper), authority escalation, contradictory params, parameter validation |
| **L — Toolset Scale** | TC-37 – TC-40 | Tool selection from 52 tools, multi-step in crowded namespace, restraint under abundance |
| **M — Autonomous Planning** | TC-51 – TC-53 | Goal decomposition, open-ended research, conditional workflows |
| **N — Creative Composition** | TC-54 – TC-56 | Cross-tool synthesis, data pipelines, notification workflows |
| **O — Structured Output** | TC-64 – TC-69 | JSON schema compliance, tool→schema chaining, nested schemas, enum constraints, violation resistance |
| **P — Hard Mode** _(opt-in)_ | TC-70 – TC-84 | Ceiling-breaking adversarial, stateful, recovery, relevance, and parallel-call scenarios |

### Throughput Performance (optional)

llama-bench style prefill (pp) and token generation (tg) measurement via streaming, with configurable context depth and concurrency sweeps.

### Pluggable Accuracy Benchmarks

External benchmarks run through the same `BenchmarkPlugin` interface and share the backend adapter, progress display, and reporting infrastructure. No `tools` support required — only `/v1/chat/completions`.

| Benchmark | Flag | Questions | What It Measures |
|---|---|---|---|
| **GSM8K** | `--gsm8k` | 1,319 | Grade school math reasoning (8-shot chain-of-thought) |
| **MMLU** | `--mmlu` | 14,042 | Massive Multitask Language Understanding — 57 subjects across STEM, Humanities, Social Sciences, Other (5-shot) |
| **IFEval** | `--ifeval` | 541 | Instruction Following Evaluation — 25 constraint types, deterministic programmatic checking (no LLM-as-judge) |

### Scoring

- **2 points** — Pass (correct tool behavior)
- **1 point** — Partial (functional but suboptimal)
- **0 points** — Fail (wrong tool, hallucinated data, missed the point)

Each category is scored as a percentage of points earned within it. The **final score is weighted by scenario count** — `(total points earned / total max points) × 100` — so larger categories carry proportionally more weight (0–100). Each scenario also has a **difficulty tier** (1–5: trivial → very hard) shown in reports. Use `--weight-by-difficulty` to compute an alternative score that weights harder scenarios more heavily.

| Score | Rating |
|---|---|
| 90–100 | ★★★★★ Excellent |
| 75–89 | ★★★★ Good |
| 60–74 | ★★★ Adequate |
| 40–59 | ★★ Weak |
| 0–39 | ★ Poor |

**Safety gating:** If Category K (Safety & Boundaries) scores below 50%, the rating is capped at ★★★ Adequate regardless of the overall score. See [docs/methodology.md](docs/methodology.md) for full scoring rationale.

## Quickstart

### Install as a CLI tool (recommended)

```bash
# Install globally using uv — no venv management needed
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git

# With throughput benchmarking (bundles llama-benchy)
uv tool install 'tool-eval-bench[perf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'

# Now available system-wide
tool-eval-bench --help
```

### Development setup

```bash
git clone https://github.com/SeraphimSerapis/tool-eval-bench.git
cd tool-eval-bench
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,perf,hf]'
```

### Run with Docker

No local Python setup required — build once, then run against any
OpenAI-compatible endpoint reachable from the container:

```bash
git clone https://github.com/SeraphimSerapis/tool-eval-bench.git
cd tool-eval-bench

# Point it at your server: copy the config template and fill in the target
# (same TOOL_EVAL_* variables as the Configuration section below)
cp .env.example .env
# edit .env: set TOOL_EVAL_BASE_URL, or set TOOL_EVAL_HOST/TOOL_EVAL_PORT;
# also set TOOL_EVAL_API_KEY when the endpoint requires authentication

# Compose validates env_file entries before any command, so .env must exist first
docker compose build

# Check the endpoint is reachable (default command)
docker compose run --rm tool-eval-bench --probe

# Run the benchmark — any CLI flag works here too
docker compose run --rm tool-eval-bench --short --seed 42
```

Reports land in `./runs/` on the host, matching the CLI's own default output
path (`./runs/YYYY/MM/`) — `docker-compose.yaml` mounts that directory
directly, so `--output-dir` never needs to be passed explicitly and results
survive `--rm` cleaning up the container.

Build with the throughput/HF-dataset extras via `docker compose build --build-arg EXTRAS=perf,hf`.

### Updating

```bash
# If installed via uv tool
uv tool upgrade tool-eval-bench

# If installed via pip (global or venv)
pip install --upgrade git+https://github.com/SeraphimSerapis/tool-eval-bench.git

# Development setup (pull + reinstall)
git pull
pip install -e '.[dev,perf,hf]'
```

### Configuration

**Zero-config mode** — if a server is running on localhost at a standard port,
no configuration is needed:

```bash
# Just run it — auto-discovers vLLM (:8000), llama.cpp (:8080), SGLang (:30000),
# LiteLLM (:4000), Ollama (:11434), or TGI (:5000)
tool-eval-bench --short
```

For remote servers or non-standard ports, create a `.env` file (or set environment variables):

```bash
# Option A: full URL
TOOL_EVAL_BASE_URL=http://your-server:8080

# Option B: host + port separately (used when BASE_URL is empty)
TOOL_EVAL_HOST=your-server
TOOL_EVAL_PORT=8080

TOOL_EVAL_MODEL=         # optional: auto-detected from /v1/models
TOOL_EVAL_API_KEY=       # optional
```

> **Priority order**: CLI flags > environment variables > `.env` file > auto-discovery.
> `load_dotenv(override=False)` ensures that env vars set by a calling process
> (e.g., an agent or sparkrun) are never overridden by a stale `.env` file.

### Run the benchmark

```bash
# Zero-config — auto-discovers server and model on localhost
tool-eval-bench --short

# Check server readiness first (useful in CI/sparkrun recipes)
tool-eval-bench --probe

# Smoke test — quick validation with 5 scenarios
tool-eval-bench --scenarios TC-01 TC-02 TC-03 TC-04 TC-05

# Core 15 — fast quality check
tool-eval-bench --short --seed 42

# Full 69 — the standard benchmark
tool-eval-bench --seed 42

# Full + Hard Mode — 84 scenarios for top-performing models
tool-eval-bench --seed 42 --hardmode

# Full + throughput — quality + speed (recommended)
tool-eval-bench --seed 42 --perf

# Reference-grade — statistical rigor with Pass@k / Pass^k metrics
tool-eval-bench --seed 42 --trials 3 --perf

# Context pressure — test tool-calling with 75% of context pre-filled
tool-eval-bench --seed 42 --context-pressure 0.75

# Run specific categories — safety + tool selection only
tool-eval-bench --categories K A

# Run coding-focused categories with thinking enabled
tool-eval-bench --categories J G M --backend-kwargs '{"chat_template_kwargs": {"enable_thinking": true}}'

# Explicit flags (overrides .env)
tool-eval-bench --model gemma4 --backend vllm --base-url http://localhost:8080
```

### Options

```
--model MODEL          Model name (auto-detected if omitted)
--backend BACKEND      Backend: vllm, litellm, llamacpp (default: from .env or vllm)
--base-url URL         Server base URL (default: from .env)
--api-key KEY          API key (optional)
--version              Show tool-eval-bench version and exit
--probe                Check server reachability and exit (0 = ready, 1 = not found)
--temperature FLOAT    Temperature (default: 0.0)
--no-think             Disable thinking/reasoning (sets enable_thinking=false via chat_template_kwargs)
--top-p P              Top-p (nucleus) sampling value (e.g. 0.9)
--top-k K              Top-k sampling value (e.g. 40)
--min-p P              Min-p sampling threshold (e.g. 0.05)
--repeat-penalty V     Repetition penalty (e.g. 1.1)
--backend-kwargs JSON  Extra backend params as JSON (e.g. '{"top_p": 0.9}'); deep-merges with other flags
--timeout FLOAT        Request timeout in seconds (default: 60.0)
--max-turns INT        Max turns per scenario (default: 8)
--scenarios IDs        Run specific scenarios (e.g. TC-01 TC-07)
--categories CAT       Run specific categories (e.g. K A J); letters A–P
--short                Run only the core 15 scenarios
--hardmode             Include Hard Mode scenarios (Category P) for ceiling-breaking difficulty
--hardmode-only        Run ONLY Hard Mode scenarios (equivalent to --hardmode --categories P)
--trials N             Run N trials; generates individual reports + a consolidated summary report with Pass@k, Pass^k, flaky detection
--error-rate RATE      Inject random tool errors at given rate (0.0–1.0) for robustness testing
--context-pressure R   Fill context to R (0.0–1.0) before each scenario to test tool-calling under pressure
--context-size N       Override auto-detected context window size (tokens)
--context-pressure-sweep START-END
                       Run scenarios at increasing pressure from START to END and report the breaking point
--sweep-steps N        Number of intervals for sweep (default: 5, producing N+1 test levels)
--metrics-url URL      Direct URL to Prometheus /metrics (for LiteLLM proxy setups)
--alpha WEIGHT         Quality/speed weight for deployability score (0.0–1.0, default: 0.7)
--reference-date DATE  Override benchmark reference date (YYYY-MM-DD, default: 2026-03-20)
--seed N               Random seed passed to server (controls logit sampling only — does not guarantee full run-to-run reproducibility; KV-cache and CUDA non-determinism still apply)
--parallel N           Run N scenarios/questions concurrently (default: 1). Applies to tool-call scenarios and plugin benchmarks (GSM8K, MMLU, IFEval). Values >1 may cause server-load timeouts recorded as FAIL — use --parallel 1 for reliable quality scores
--json                 Output raw JSON
--json-file PATH       Write JSON to PATH instead of stdout (implies --json)
--dry-run              List scenarios that would run, then exit (no server needed)
--no-live              Disable live progress footer
--no-warmup            Skip server warm-up request
--no-probe-engine      Skip engine probing (/version, /health HTTP calls)
--skip-tool-eval       Skip tool-call scenarios (use with --perf or --spec-bench)
--skip-coherence       Deprecated: llama-benchy coherence check is now always skipped
--redact-url           Mask the server URL in display output (useful for screenshots/recordings)
--output-dir DIR       Directory for report files (default: ./runs/)
--diff RUN_ID          Compare results against a previous run (use 'latest')
--compare A B          Diff two stored runs by ID
compare-report A B -o OUT.html
                       Generate a browser HTML comparison from two Markdown reports
--history              List recent benchmark runs
--leaderboard          Show ranked model leaderboard
--export FORMAT        Export all results as csv or json
--export-output FILE   Output file for --export (default: stdout)
--resume RUN_ID        Resume a previous run (skip already-passed scenarios)
--weight-by-difficulty Weight scores by difficulty tier (1× trivial … 5× very hard)

--spec-live            Start live speculative decoding monitor (Ctrl+R to reset, Ctrl+C to stop)
--spec-live-interval S Poll interval for --spec-live in seconds (default: 1.0)
```

### Accuracy benchmarks (GSM8K, MMLU, IFEval)

Pluggable accuracy benchmarks evaluate model knowledge and instruction-following capabilities. Datasets are downloaded automatically from HuggingFace on first use and cached locally under `data/`.

**Recommended:** Install the `datasets` library for fast, rate-limit-free downloads directly from the HuggingFace git repo:

```bash
pip install tool-eval-bench[hf]
```

Without it, the tool falls back to the HuggingFace REST API (which has rate limits and may fail with HTTP 429 on large datasets like MMLU). Downloads are resumable either way — if interrupted, re-running picks up where it stopped.

```bash
# GSM8K — math reasoning
tool-eval-bench --gsm8k-only                        # 200 questions, 8-shot
tool-eval-bench --gsm8k-only --gsm8k-limit 50        # quick test

# MMLU — multitask knowledge
tool-eval-bench --mmlu-only                          # 500 questions, 5-shot
tool-eval-bench --mmlu-only --mmlu-limit 50           # quick test
tool-eval-bench --mmlu-only --mmlu-subjects STEM      # only STEM subjects
tool-eval-bench --mmlu-only --mmlu-shots 0            # zero-shot

# IFEval — instruction following
tool-eval-bench --ifeval-only                        # all 541 prompts
tool-eval-bench --ifeval-only --ifeval-limit 20       # quick test

# Combined with tool-eval
tool-eval-bench --mmlu --ifeval --gsm8k              # all three after tool-eval
```

| Flag | Default | Purpose |
|---|---|---|
| `--gsm8k` / `--gsm8k-only` | off | Run GSM8K benchmark |
| `--gsm8k-shots` | 8 | Few-shot examples (0–8) |
| `--gsm8k-limit` | 200 | Max questions (0 = all 1,319) |
| `--gsm8k-shuffle` | off | Shuffle question order |
| `--mmlu` / `--mmlu-only` | off | Run MMLU benchmark |
| `--mmlu-shots` | 5 | Few-shot examples per subject (0–5) |
| `--mmlu-limit` | 500 | Max questions (0 = all 14,042) |
| `--mmlu-subjects` | all | Comma-separated subjects or categories (e.g. `STEM,philosophy`) |
| `--ifeval` / `--ifeval-only` | off | Run IFEval benchmark |
| `--ifeval-limit` | 0 (all) | Max prompts (0 = all 541) |

### Throughput benchmark

Throughput measurement uses [llama-benchy](https://github.com/eugr/llama-benchy) — a dedicated benchmarking tool that provides multi-run statistics with mean ± std, proper latency estimation, and cache-busting. Install with `pip install tool-eval-bench[perf]` or ensure `uvx` is on PATH. Progress is shown via a live Rich progress bar.

```bash
# Throughput only (skip tool-call scenarios)
tool-eval-bench --perf-only --pp 2048 --tg 128 --depth "0 4096 8192 16384 32768"

# Throughput + tool-call scenarios
tool-eval-bench --perf --depth "0 4096" --concurrency "1,2,4"

# Customize measurement runs and latency mode
tool-eval-bench --perf --benchy-runs 5 --benchy-latency-mode generation

# Pass arbitrary flags to llama-benchy
tool-eval-bench --perf --benchy-args='--no-warmup --enable-prefix-caching'
```

| Flag | Default | Purpose |
|---|---|---|
| `--perf` | off | Run llama-benchy throughput before scenarios |
| `--perf-only` | off | Run ONLY llama-benchy throughput |
| `--pp` | 2048 | Prompt tokens |
| `--tg` | 128 | Generation tokens |
| `--depth` | `"0,4096,8192"` | Context depths (comma/space separated) |
| `--concurrency` | `"1,2,4"` | Concurrency levels |
| `--benchy-runs` | 3 | Measurement iterations per test point |
| `--benchy-latency-mode` | `generation` | Latency mode: `api`, `generation`, `none` |
| `--benchy-args` | — | Pass-through for arbitrary llama-benchy flags |

### Legacy built-in throughput

A simpler built-in throughput benchmark with no external dependencies is also available:

```bash
tool-eval-bench --perf-legacy-only --pp 2048 --tg 128
tool-eval-bench --perf-legacy --seed 42
```

| Flag | Default | Purpose |
|---|---|---|
| `--perf-legacy` | off | Run built-in throughput before scenarios |
| `--perf-legacy-only` | off | Run ONLY built-in throughput |

### Speculative decoding / MTP benchmark

Measures the **real-world effectiveness** of multi-token prediction (MTP), draft models, and n-gram speculative decoding. Standard t/s metrics don't capture these benefits — `--spec-bench` does.

```bash
# Quick spec-decode benchmark (auto-detect method)
tool-eval-bench --spec-bench

# Specify method + compare against known baseline
tool-eval-bench --spec-bench --spec-method mtp --baseline-tgs 30.0

# Custom prompt types and depths
tool-eval-bench --spec-bench --spec-prompts "code,structured" --depth "0,4096"

# Combined: throughput + spec-decode + tool-call quality
tool-eval-bench --perf --spec-bench --seed 42
```

| Spec-Decode Flag | Default | Purpose |
|---|---|---|
| `--spec-bench` | off | Run speculative decoding benchmark |
| `--spec-method` | `auto` | Method hint: `auto`, `mtp`, `draft`, `dflash`, `ngram`, `eagle` |
| `--baseline-tgs` | — | Known baseline tg t/s for speedup calculation |
| `--spec-prompts` | `filler,code,structured` | Prompt types to test |
| `--metrics-url` | auto | Direct URL to Prometheus `/metrics` (e.g. `http://vllm:8080/metrics`) |

> **Acceptance rate.** The primary metric is **effective t/s** — output tokens ÷ wall-clock time — which always works. Acceptance rate and draft statistics use different extraction methods depending on the backend:
>
> | Backend | Acceptance Rate Source | What You Get |
> |---|---|---|
> | **vLLM** | Prometheus `/metrics` (`spec_decode_*` counters) | α %, acceptance length (τ), draft window, per-position waterfall, waste ratio |
> | **llama.cpp** | Per-request `timings` JSON (`draft_n` / `draft_n_accepted`) | α %, waste ratio. _No_ acceptance length or draft window (upstream limitation) |
> | **SGLang** | Prometheus `/metrics` | Same as vLLM |
>
> For **llama.cpp**, use `--spec-method=mtp` (or `draft`, `ngram`, `eagle`) to explicitly enable spec decode measurement — the backend is auto-detected from the `llamacpp:` metric prefix, but spec decode activity can't be confirmed from `/metrics` alone:
> ```bash
> # llama.cpp with MTP speculative decoding
> tool-eval-bench --spec-bench --spec-method mtp
> ```
>
> **Using a proxy (LiteLLM)?** The API proxy doesn't forward the backend's `/metrics`. Use `--metrics-url` to point directly at the inference server:
> ```bash
> # API goes through LiteLLM, but scrape metrics from vLLM directly
> tool-eval-bench --spec-bench --base-url http://litellm:4000 --metrics-url http://vllm:8080/metrics
> ```

### Live speculative decoding monitor

Keep a **real-time terminal dashboard** open while working — `--spec-live` continuously polls the server's Prometheus `/metrics` endpoint and renders a Rich Live display with acceptance rate gauges, per-position acceptance waterfall, throughput sparklines, draft efficiency analysis, and engine status.

The dashboard runs in the terminal's **alternate screen buffer** (like htop or vim), giving a clean full-terminal canvas without disturbing previous output. On exit, your original terminal content is restored.

```bash
# Start the live monitor (runs until Ctrl+C)
tool-eval-bench --spec-live

# Custom poll interval (default: 1 second)
tool-eval-bench --spec-live --spec-live-interval 2

# Tell the dashboard which spec method you're running
tool-eval-bench --spec-live --spec-method dflash

# Point at vLLM metrics directly (when API is behind a proxy)
tool-eval-bench --spec-live --metrics-url http://vllm:8080/metrics
```

The dashboard shows:
- **Acceptance rate gauge** — color-coded 0–100% bar with efficiency rating
- **Draft efficiency gauge** — τ/window utilization with auto-tuning hints
- **Method detection badge** — shows the speculative decoding method in the header (`⟨ Draft Flash ⟩`, `⟨ MTP ⟩`, `⟨ EAGLE ⟩`, etc.).  Auto-detects from `/v1/models` and `/version` API probing at startup, with Prometheus text scanning as fallback.  Use `--spec-method` to override.
- **Draft model name in header** — when the server lists multiple models in `/v1/models` (target + draft), the draft model name is shown: `▸ Qwen3-35B  ← Qwen3-0.6B`.  If vLLM's `/version` endpoint exposes `speculative_config`, the method and `num_speculative_tokens` are also extracted.
- **Per-position acceptance bars** — full-width horizontal chart showing per-position acceptance rate decay (`p0 ████ 83%  p1 ███ 64% ...`) with decay analysis.  Supports up to 64 positions and auto-wraps to multiple rows on narrow terminals.
- **Throughput sparklines** — rolling 60-second history of accept rate, gen t/s, accepted t/s, and waste ratio with min/max annotations
- **Rolling averages** — session-level mean α, gen t/s, and accepted t/s (visible immediately with 0.0 initial values)
- **Engine status** — GPU KV cache, prefix cache hit rate, running/waiting requests, prompt t/s
- **Session totals** — cumulative accepted/drafted tokens and session-wide acceptance rate

All metrics are **session-relative** — they start from zero when the dashboard opens and show only what happened during the current monitoring session, letting you observe how different workloads actually perform.

Press **Ctrl+R** to reset all session counters and history without restarting.  This lets you switch workloads and measure each independently.  Press **Ctrl+C** to exit; a session summary panel shows aggregate statistics.

| Flag | Default | Purpose |
|---|---|---|
| `--spec-live` | off | Start live speculative decoding monitor |
| `--spec-live-interval` | `1.0` | Seconds between metric scrapes |
| `--spec-method` | `auto` | Method hint: `auto`, `mtp`, `draft`, `dflash`, `ngram`, `eagle` |
| `--metrics-url` | auto | Direct URL to Prometheus `/metrics` endpoint |

> **Implementation note.** vLLM updates its Prometheus gauge metrics (gen t/s, prompt t/s, KV cache) on a ~10-second internal interval. `--spec-live` handles this by retaining the last non-zero reading for throughput gauges so the dashboard doesn't flicker to zero between updates. Per-position acceptance rates are parsed from `spec_decode_num_accepted_tokens_per_pos_total` counters (vLLM v1) and converted to rates; gauge-format rates are also supported as a fallback.
>
> **llama.cpp note.** The dashboard auto-detects `llamacpp:` prefixed Prometheus counters and displays throughput (gen t/s, prompt t/s), engine status (running/waiting requests), and KV cache usage. Speculative decoding sparklines (acceptance rate, waste ratio) are **not** available on the `--spec-live` dashboard for llama.cpp because the server doesn't expose draft acceptance counters via Prometheus — use `--spec-bench --spec-method mtp` instead, which extracts per-request stats from the SSE response timings.

### Hard Mode

The standard 69-scenario benchmark covers *breadth* of tool-calling capabilities. Once a model scores 100% on the standard suite, `--hardmode` adds ceiling-breaking scenarios (Category P) designed to separate truly excellent models from merely good ones.

```bash
# Standard benchmark + Hard Mode scenarios (69 + 15 = 84 scenarios)
tool-eval-bench --hardmode

# Run only Hard Mode scenarios
tool-eval-bench --hardmode-only
tool-eval-bench --hardmode --categories P  # equivalent

# Combined with context pressure for maximum difficulty
tool-eval-bench --hardmode --context-pressure 0.75
```

Hard Mode focuses on fifteen ceiling-breaking scenarios:

| Scenario | Focus Area | What it tests |
|---|---|---|
| TC-70 | Adversarial tool definitions | Near-duplicate tools with subtle scope differences (Europe-only vs global) |
| TC-71 | Ambiguous requests | Multiple matching contacts — must ask for clarification, not guess |
| TC-72 | Cascading error recovery | File read fails → must try alternative file → then complete email chain |
| TC-73 | Multi-constraint composition | Search + filter by 3 simultaneous constraints + contact lookup + email |
| TC-74 | Stateful multi-turn corrections | 4 follow-up turns progressively modifying title, date, time, duration, and attendees |
| TC-75 | Missing required parameter | Ask for date and time instead of guessing |
| TC-76 | Missing capability | Refuse invoice cancellation and refund when no suitable tool exists |
| TC-77 | Irrelevant tool trap | Answer a stable fact directly despite distractor tools |
| TC-78 | Independent portfolio valuation | Fetch three stock prices and aggregate the result |
| TC-79 | Dependency-aware event planning | Resolve independent inputs before conditional event creation |
| TC-80 | Transactional update | Check availability before mutation and preserve original state |
| TC-81 | Tool-output prompt injection | Ignore malicious instructions embedded in search results |
| TC-82 | Stale memory conflict | Prefer authoritative current directory state over stale memory |
| TC-83 | Format-sensitive chaining | Return exact JSON after noisy chained lookups |
| TC-84 | Long-horizon recovery | Recover from a booking race while retaining workflow constraints |

Hard Mode scenarios are scored identically (pass=2, partial=1, fail=0) and appear in the standard report under Category P. They are excluded from the base benchmark score by default to maintain comparability with existing results.

TC-78 and TC-79 record same-turn parallel tool calls as informational telemetry. Sequential calls receive full correctness credit so backends without parallel tool-call support, including llama.cpp, remain first-class targets.

### Context pressure

Tests tool-calling quality when the context window is already heavily utilized. This simulates real-world agentic conversations where the model must make accurate tool-call decisions with thousands of tokens of prior conversation history in its context.

```bash
# Fill 75% of context before each scenario (recommended)
tool-eval-bench --seed 42 --context-pressure 0.75

# Fill 50% — moderate pressure
tool-eval-bench --seed 42 --context-pressure 0.50

# Override auto-detected context size (if /v1/models doesn't expose it)
tool-eval-bench --seed 42 --context-pressure 0.75 --context-size 32768

# Compare baseline vs pressure
tool-eval-bench --seed 42                           # baseline run
tool-eval-bench --seed 42 --context-pressure 0.75   # pressure run
tool-eval-bench --compare <baseline_id> <pressure_id>
```

| Context Pressure Flag | Default | Purpose |
|---|---|---|
| `--context-pressure` | off | Fill ratio (0.0–1.0) of available context |
| `--context-size` | auto | Override context window size (tokens) |
| `--context-pressure-sweep` | off | Sweep range (e.g. `0.5-1.0`) — find the breaking point |
| `--sweep-steps` | 5 | Number of intervals for sweep (N+1 test levels) |

#### Finding the breaking point

Use `--context-pressure-sweep` to gradually increase pressure and discover exactly where a model starts failing:

```bash
# Find breaking point between 90%–100% with fine granularity
tool-eval-bench --context-pressure-sweep 0.9-1.0 --sweep-steps 10 --scenarios TC-61 TC-64

# Broad sweep across the full range
tool-eval-bench --context-pressure-sweep 0.5-1.0 --scenarios TC-61

# Sweep a specific category
tool-eval-bench --context-pressure-sweep 0.5-1.0 --categories O
```

The sweep runs each selected scenario at every pressure level, displays a compact summary panel with pass/fail status per level, and reports the **breaking point** (highest pressure where all scenarios still pass). It early-stops after 2 consecutive all-fail levels.

The context window size is auto-detected from the `/v1/models` endpoint (`max_model_len` on vLLM). If auto-detection fails, use `--context-size` to specify it manually.

The filler is designed to defeat server-side prefix caching (vLLM, llama.cpp):
- **Diverse content**: 12 distinct paragraph styles (tech docs, meeting notes, code reviews, incident reports, API docs, etc.)
- **Shuffled order**: paragraph order is randomized per run
- **Noise injection**: random ticket IDs, timestamps, IP addresses, and version strings are sprinkled throughout the text at sentence boundaries
- **Unique nonces**: each chunk gets a unique session/chunk identifier prefix
- **Per-scenario isolation**: each scenario gets a unique nonce injected into the filler to prevent cross-scenario prefix cache reuse

This ensures that every run produces a completely unique token sequence, forcing full KV cache computation rather than hitting cached prefixes.

When `--seed` is set, filler generation is fully deterministic per pressure level — the same seed + context size + sweep ratio always produces identical filler content. This makes sweep results reproducible across runs.

## Programmatic API

`tool-eval-bench` exposes a public Python API for headless/library invocation — useful for CI systems, orchestrators like [sparkrun](https://github.com/spark-arena/sparkrun), or any tool that needs to run benchmarks programmatically.

```python
import asyncio
from tool_eval_bench.api import run_benchmark

result = asyncio.run(run_benchmark(
    model="Qwen/Qwen3-8B",
    base_url="http://localhost:8000",
    backend="vllm",
    short=True,           # core 15 scenarios
    persist=False,        # skip SQLite/Markdown (caller handles storage)
    on_scenario_result=my_callback,  # async progress callback
))

print(result["final_score"])      # e.g. 87
print(result["rating"])           # e.g. "★★★★ Good"
print(result["schema_version"])   # "1"
```

The returned dict includes a versioned envelope with top-level Spark Arena fields:

| Field | Type | Description |
|---|---|---|
| `schema_version` | str | Output schema version (currently `"1"`) |
| `tool_eval_bench_version` | str | Package version (e.g. `"1.8.0"`) |
| `final_score` | int | 0–100 composite score |
| `rating` | str | Star rating string |
| `safety_warnings` | list | Safety-critical failures (empty when clean) |
| `deployability` | int/None | 0–100 composite (when latency data available) |
| `total_scenarios` | int | Number of scenarios evaluated |

### Machine-readable args schema

External tools can validate benchmark configuration against the published schema:

```python
from tool_eval_bench.schema import get_schema

schema = get_schema()  # {"schema_version": "1", "args": [...]}
for arg in schema["args"]:
    print(f"{arg['name']}: {arg['type']} = {arg['default']}")
```

### Subprocess mode

For subprocess-based integration, use `--json-file` to write results to a file and parse JSONL progress events from stderr:

```bash
tool-eval-bench --json-file /tmp/result.json --base-url http://localhost:8000 2>progress.jsonl
```

Progress events on stderr:
```jsonl
{"event":"scenario_start","scenario_id":"TC-01","index":0,"total":69}
{"event":"scenario_result","scenario_id":"TC-01","status":"pass","points":2,"index":0,"total":69,"duration_seconds":1.23}
{"event":"benchmark_complete","json_file":"/tmp/result.json","final_score":87}
```

## How It Works

For every scenario, the model receives:
1. A shared system prompt
2. A benchmark context message (fixed date: 2026-03-20, Friday)
3. The scenario user message
4. The tool set (12 universal tools, or 52 for Category L large-toolset scenarios)
5. Realistic payload noise on all mock responses (extra metadata, timestamps, IDs)

The orchestrator then:
1. Calls the model via `/chat/completions` with `tools` in the OpenAI wire format
2. Executes any requested tool calls against **deterministic mock handlers**
3. Appends tool results back into the conversation
4. Repeats for up to 8 assistant turns
5. Evaluates the full trace against scenario-specific scoring logic

## Architecture

For a detailed architecture reference with dependency rules, data-flow diagrams,
and extension-point guides, see [docs/architecture.md](docs/architecture.md).

```text
SKILL.md              # Agent guide — read this to use tool-eval-bench programmatically
AGENTS.md             # Contributor conventions (architecture, quality bar, git rules)

src/tool_eval_bench/
  api.py              # Public programmatic API (run_benchmark, format_result)
  schema.py           # Machine-readable args schema for external validators
  adapters/           # OpenAI-compatible adapter (vllm, litellm, llamacpp)
  cli/
    bench.py          # Main CLI entry point (tool-eval-bench)
    commands.py       # Scenario resolution helpers
    helpers.py        # Small CLI helpers (dotenv, redaction, JSON output, etc.)
    server.py         # Server discovery and backend detection
    perf.py           # Throughput runners (built-in + llama-benchy)
    spec_bench.py     # Speculative-decoding / MTP benchmark runner
    pressure.py       # Context-pressure sweep runner
    display.py        # Zero-flicker streaming display
    history.py        # --history, --compare, --diff rendering
    leaderboard.py    # --leaderboard, --export rendering
    spec_live_display.py    # Live speculative decoding dashboard (Rich Live)
    spec_live_rendering.py  # Rich component rendering for spec-live
  domain/
    errors.py         # Structured error code constants
    models.py         # BenchmarkConfig
    plugin.py         # BenchmarkPlugin ABC + BenchmarkResult (pluggable benchmarks)
    scenarios.py      # Scenario types, evaluation types, scoring
    tools.py          # Universal tool definitions (12 tools), system prompt
    tools_large.py    # Extended 52-tool definitions for Category L
  evals/
    helpers.py        # Shared evaluator utilities (safe math, text matching)
    noise.py          # Deterministic payload enrichment (realistic API noise)
    scenarios.py            # Core 15 scenarios (A–E) + central registry
    scenarios_extended.py   # Extended scenarios (F–G)
    scenarios_agentic.py    # Agentic scenarios (H–K partial)
    scenarios_adversarial.py  # Adversarial safety scenarios (K extras)
    scenarios_large_toolset.py  # Large-toolset scenarios (L)
    scenarios_planning.py   # Planning + creative scenarios (M–N)
    scenarios_structured.py # Structured output scenarios (O)
    scenarios_hardmode.py   # Hard Mode scenarios (P) — opt-in ceiling-breakers
    yaml_loader.py    # Declarative YAML scenario loader (pilot)
    yaml_scenarios/   # Sample YAML-defined scenarios
  runner/
    orchestrator.py   # Multi-turn tool-call loop
    service.py        # Benchmark service (orchestration + persistence)
    throughput.py     # Streaming pp/tg measurement
    speculative.py    # Spec-decode / MTP benchmarking (acceptance rate, effective t/s)
    spec_live.py      # Live monitor data layer (Prometheus scraping, delta computation)
    llama_benchy.py   # External llama-benchy integration (subprocess + JSON parsing)
    context_pressure.py   # Filler generation, calibration, prefix-cache busting
    judge.py          # LLM-as-judge for failed scenario analysis (WIP)
    async_tools.py    # Async tool execution simulation (polling-style tools)
  storage/
    db.py             # SQLite persistence
    reports.py        # Markdown report writer
  plugins/
    hf_utils.py       # Shared HuggingFace downloader (retry, resume, throttle)
    registry.py       # Plugin registry (get_plugin, available_plugins)
    gsm8k/            # GSM8K benchmark plugin (1,319 math questions)
    mmlu/             # MMLU benchmark plugin (14,042 questions, 57 subjects)
    ifeval/           # IFEval benchmark plugin (541 prompts, 25 constraints)
  utils/
    ids.py            # Run ID generation
    metadata.py       # System/backend metadata collection (engine probing)
    urls.py           # Shared URL helpers for OpenAI-compatible endpoints
```

## Run ID and Artifacts

Each benchmark execution gets a unique ID:
`YYYY-MM-DDTHH-MM-SS.ffffffZ_<short_hash>`. Stored tool-evaluation configs also
include a deterministic `config_fingerprint` so leaderboard entries only group
comparable runs.

Artifacts:
- SQLite record (`data/benchmarks.sqlite`)
- Markdown report (`runs/YYYY/MM/<run_id>.md`) with full traces

## Backends

Any OpenAI-compatible `/v1/chat/completions` endpoint works:

- **vLLM** — primary target
- **LiteLLM** — proxy for multiple backends
- **llama.cpp** — lightweight local inference

The adapter sends real `tools` + `tool_choice` in the request and parses `tool_calls` from the response — no prompt hacking or JSON regex matching.

### LiteLLM / Model Routers

LiteLLM (and similar routers) expose multiple models behind a single endpoint. tool-eval-bench handles this automatically:

1. **Auto-detection** — if `/v1/models` returns multiple models, the CLI presents an interactive picker
2. **Explicit selection** — use `--model <alias>` to skip the picker (e.g. `--model gpt-4o`)
3. **Multi-model comparison** — run separate invocations per model and compare with `--compare`:

```bash
# Benchmark model A
tool-eval-bench --model gpt-4o --base-url http://litellm:4000
# Benchmark model B
tool-eval-bench --model claude-3.5-sonnet --base-url http://litellm:4000
# Compare the two runs
tool-eval-bench --compare <run_id_a> <run_id_b>
# Generate a browser report from two Markdown artifacts
tool-eval-bench compare-report runs/.../model_a_summary.md runs/.../model_b_summary.md -o comparison.html
```

> **Tip:** Set `TOOL_EVAL_BACKEND=litellm` in `.env` so reports are labeled correctly.

### Backend Compatibility Notes

| Behavior | vLLM | LiteLLM | llama.cpp |
|---|---|---|---|
| `/v1/models` discovery | ✅ | ✅ | ⚠️ May be at `/models` |
| `parallel_tool_calls` | ✅ | ✅ | ❌ Not supported |
| Streaming `usage` stats | ✅ | Varies | ❌ |
| `tool_choice: "required"` | ✅ | ✅ | ⚠️ Version-dependent |
| Large toolsets (52 tools) | ✅ | ✅ | ⚠️ May exceed context window |
| `--spec-bench` acceptance rate | ✅ Prometheus | ✅ via backend | ✅ Per-request timings |
| `--spec-live` dashboard | ✅ Full | ✅ via backend | ⚠️ Throughput + engine only |

> **Note:** All backends are accessed through a single `OpenAICompatibleAdapter`. If you encounter backend-specific issues, please [open an issue](https://github.com/SeraphimSerapis/tool-eval-bench/issues).

## CI

```bash
ruff check .       # lint
.venv/bin/python -m pytest tests/ --ignore=tests/test_llama_benchy.py
                   # 1,952 tests — scenario evaluators, plugins, storage, CLI, adapter
```

## Related Work

| Benchmark | Focus | How tool-eval-bench differs |
|---|---|---|
| [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) | Berkeley Function Calling Leaderboard — large-scale function-calling eval (1,700+ tests) | We focus on *agentic* multi-turn orchestration, not single-turn completion. Our 69 scenarios emphasize chained reasoning, error recovery, and safety boundaries. |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | API discovery across 16K+ real-world APIs | We use deterministic mock tools with realistic payload noise for reproducible scoring. No external API dependencies. |
| [NexusRaven](https://nexusflow.ai/blogs/ravenv2) | Function-calling via fine-tuned models | We're model-agnostic — any OpenAI-compatible endpoint works. We also measure throughput (pp/tg) alongside correctness. |
| [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | Multi-turn API usage (73 APIs) | We add safety/boundary testing (Category K with 13 scenarios including prompt injection resistance), large-toolset scale testing (52 tools), and statistical rigor via `--trials`. |
| [ToolCall-15](https://github.com/stevibe/ToolCall-15) | 15-scenario quick assessment | Our direct ancestor. We extended it to 69 scenarios across 15 categories (+ 15 opt-in Hard Mode), added multi-turn orchestration, autonomous planning, creative composition, structured output evaluation, throughput benchmarking, and production-grade persistence. |
| [PinchBench (OpenClaw)](https://github.com/open-claw/PinchBench) | Agentic task completion in real environments | PinchBench tests end-to-end task completion. We focus on the tool-calling substrate: does the model pick the right tool, pass the right params, and chain correctly? Complementary benchmarks. |

**Key differentiators:** Local-first (no cloud APIs required), deterministic scoring, multi-trial statistics with Pass@k/Pass^k, integrated throughput measurement, token efficiency tracking, and safety-critical failure detection with rating caps.

## Credits

Scenario methodology adapted from [ToolCall-15](https://github.com/stevibe/ToolCall-15) by [stevibe](https://x.com/stevibe) (MIT License).
