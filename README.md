# tool-eval-bench

A **tool-calling quality benchmark** for evaluating LLM tool-use in agentic workflows across open-weight model serving stacks (**vLLM**, **LiteLLM**, **llama.cpp**).

Inspired by [ToolCall-15](https://github.com/stevibe/ToolCall-15), this tool runs **63 deterministic scenarios** through OpenAI-compatible `/chat/completions` endpoints, scores each result as **pass**, **partial**, or **fail**, and produces detailed trace reports. Mock tool responses include realistic payload noise (extra metadata, timestamps, nested objects) to test whether models can extract relevant fields from noisy API responses. It also includes an integrated **throughput benchmark** (llama-bench style) for measuring prefill and token generation speed.

> **Scope.** tool-eval-bench measures *tool-calling quality* — whether a model picks the right tool, passes the right parameters, chains tools correctly, and handles errors and safety boundaries. It is not a full agentic system benchmark (see [Related Work](#related-work) for how it compares to BFCL, PinchBench, and Claw-Eval).

## What It Measures

### Tool-Call Quality (63 scenarios across 14 categories)

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
| **K — Safety & Boundaries** | TC-31 – TC-36, TC-57 – TC-60 | Ambiguity, prompt injection (file/search/system/sleeper), authority escalation, contradictory params |
| **L — Toolset Scale** | TC-37 – TC-40 | Tool selection from 52 tools, multi-step in crowded namespace, restraint under abundance |
| **M — Autonomous Planning** | TC-51 – TC-53 | Goal decomposition, open-ended research, conditional workflows |
| **N — Creative Composition** | TC-54 – TC-56 | Cross-tool synthesis, data pipelines, notification workflows |

### Throughput Performance (optional)

llama-bench style prefill (pp) and token generation (tg) measurement via streaming, with configurable context depth and concurrency sweeps.

### Scoring

- **2 points** — Pass (correct tool behavior)
- **1 point** — Partial (functional but suboptimal)
- **0 points** — Fail (wrong tool, hallucinated data, missed the point)

Each category is scored as a percentage of points earned within it. The **final score is weighted by scenario count** — `(total points earned / total max points) × 100` — so larger categories carry proportionally more weight (0–100).

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

# Now available system-wide
tool-eval-bench --help
```

### Development setup

```bash
git clone https://github.com/SeraphimSerapis/tool-eval-bench.git
cd tool-eval-bench
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Configuration

Create a `.env` file (or set environment variables):

```bash
# Option A: full URL
TOOL_EVAL_BASE_URL=http://your-server:8080

# Option B: host + port separately (used when BASE_URL is empty)
TOOL_EVAL_HOST=your-server
TOOL_EVAL_PORT=8080

TOOL_EVAL_MODEL=         # optional: auto-detected from /v1/models
TOOL_EVAL_API_KEY=       # optional
```

### Run the benchmark

```bash
# Smoke test — quick validation with 5 scenarios
tool-eval-bench --scenarios TC-01 TC-02 TC-03 TC-04 TC-05

# Core 15 — fast quality check
tool-eval-bench --short --seed 42

# Full 63 — the standard benchmark
tool-eval-bench --seed 42

# Full + throughput — quality + speed (recommended)
tool-eval-bench --seed 42 --perf

# Reference-grade — statistical rigor with Pass@k / Pass^k metrics
tool-eval-bench --seed 42 --trials 3 --perf

# Explicit flags (overrides .env)
tool-eval-bench --model gemma4 --backend vllm --base-url http://localhost:8080
```

### Options

```
--model MODEL          Model name (auto-detected if omitted)
--backend BACKEND      Backend: vllm, litellm, llamacpp (default: from .env or vllm)
--base-url URL         Server base URL (default: from .env)
--api-key KEY          API key (optional)
--temperature FLOAT    Temperature (default: 0.0)
--no-think             Disable thinking/reasoning (sets enable_thinking=false via chat_template_kwargs)
--top-p P              Top-p (nucleus) sampling value (e.g. 0.9)
--top-k K              Top-k sampling value (e.g. 40)
--min-p P              Min-p sampling threshold (e.g. 0.05)
--repeat-penalty V     Repetition penalty (e.g. 1.1)
--timeout FLOAT        Request timeout in seconds (default: 60.0)
--max-turns INT        Max turns per scenario (default: 8)
--scenarios IDs        Run specific scenarios (e.g. TC-01 TC-07)
--short                Run only the core 15 scenarios
--trials N             Run N trials; generates individual reports + a consolidated summary report with Pass@k, Pass^k, flaky detection
--error-rate RATE      Inject random tool errors at given rate (0.0–1.0) for robustness testing
--alpha WEIGHT         Quality/speed weight for deployability score (0.0–1.0, default: 0.7)
--reference-date DATE  Override benchmark reference date (YYYY-MM-DD, default: 2026-03-20)
--seed N               Random seed passed to server (controls logit sampling only — does not guarantee full run-to-run reproducibility; KV-cache and CUDA non-determinism still apply)
--parallel N           Run N scenarios concurrently (default: 1). Values >1 may cause server-load timeouts recorded as FAIL — use --parallel 1 for reliable quality scores
--json                 Output raw JSON
--no-live              Disable live progress footer
--no-warmup            Skip server warm-up request
--diff RUN_ID          Compare results against a previous run (use 'latest')
--compare A B          Diff two stored runs by ID
--history              List recent benchmark runs
```

### Throughput benchmark

```bash
# Throughput only (skip tool-call scenarios)
tool-eval-bench --perf-only --pp 2048 --tg 128 --depth "0 4096 8192 16384 32768"

# Throughput + tool-call scenarios
tool-eval-bench --perf --depth "0 4096" --concurrency "1,2,4"
```

| Throughput Flag | Default | Purpose |
|---|---|---|
| `--perf` | off | Run throughput before scenarios |
| `--perf-only` | off | Run ONLY throughput |
| `--pp` | 2048 | Prompt tokens |
| `--tg` | 128 | Generation tokens |
| `--depth` | `"0,4096,8192"` | Context depths (comma/space separated) |
| `--concurrency` | `"1,2,4"` | Concurrency levels |

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
| `--spec-method` | `auto` | Method hint: `auto`, `mtp`, `draft`, `ngram`, `eagle` |
| `--baseline-tgs` | — | Known baseline tg t/s for speedup calculation |
| `--spec-prompts` | `filler,code,structured` | Prompt types to test |

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

```text
src/tool_eval_bench/
  adapters/           # OpenAI-compatible adapter (vllm, litellm, llamacpp)
  cli/
    bench.py          # Main CLI entry point (tool-eval-bench)
    display.py        # Zero-flicker streaming display
  domain/
    models.py         # BenchmarkConfig
    scenarios.py      # Scenario types, evaluation types, scoring
    tools.py          # Universal tool definitions, system prompt
  evals/
    helpers.py        # Shared evaluator utilities (safe math, text matching)
    noise.py          # Deterministic payload enrichment (realistic API noise)
    scenarios.py      # Core 15 scenarios (A–E)
    scenarios_extended.py   # Extended scenarios (F–G)
    scenarios_agentic.py    # Agentic scenarios (H–K)
    scenarios_large_toolset.py  # Large-toolset scenarios (L)
  runner/
    orchestrator.py   # Multi-turn tool-call loop
    service.py        # Benchmark service (orchestration + persistence)
    throughput.py     # Streaming pp/tg measurement
    speculative.py    # Spec-decode / MTP benchmarking (acceptance rate, effective t/s)
  storage/
    db.py             # SQLite persistence
    reports.py        # Markdown report writer
  utils/
    ids.py            # Run ID generation
    metadata.py       # System/backend metadata
    urls.py           # Shared URL helpers for OpenAI-compatible endpoints
```

## Run ID and Artifacts

Each benchmark run gets a unique ID: `YYYY-MM-DDTHH-MM-SSZ_<short_hash>`

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

> **Note:** All backends are accessed through a single `OpenAICompatibleAdapter`. If you encounter backend-specific issues, please [open an issue](https://github.com/SeraphimSerapis/tool-eval-bench/issues).

## CI

```bash
ruff check .       # lint
pytest             # scenario evaluators + storage
```

## Related Work

| Benchmark | Focus | How tool-eval-bench differs |
|---|---|---|
| [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) | Berkeley Function Calling Leaderboard — large-scale function-calling eval (1,700+ tests) | We focus on *agentic* multi-turn orchestration, not single-turn completion. Our 63 scenarios emphasize chained reasoning, error recovery, and safety boundaries. |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | API discovery across 16K+ real-world APIs | We use deterministic mock tools with realistic payload noise for reproducible scoring. No external API dependencies. |
| [NexusRaven](https://nexusflow.ai/blogs/ravenv2) | Function-calling via fine-tuned models | We're model-agnostic — any OpenAI-compatible endpoint works. We also measure throughput (pp/tg) alongside correctness. |
| [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | Multi-turn API usage (73 APIs) | We add safety/boundary testing (Category K with 13 scenarios including prompt injection resistance), large-toolset scale testing (52 tools), and statistical rigor via `--trials`. |
| [ToolCall-15](https://github.com/stevibe/ToolCall-15) | 15-scenario quick assessment | Our direct ancestor. We extended it to 63 scenarios across 14 categories, added multi-turn orchestration, autonomous planning, creative composition, throughput benchmarking, and production-grade persistence. |
| [PinchBench (OpenClaw)](https://github.com/open-claw/PinchBench) | Agentic task completion in real environments | PinchBench tests end-to-end task completion. We focus on the tool-calling substrate: does the model pick the right tool, pass the right params, and chain correctly? Complementary benchmarks. |

**Key differentiators:** Local-first (no cloud APIs required), deterministic scoring, multi-trial statistics with Pass@k/Pass^k, integrated throughput measurement, token efficiency tracking, and safety-critical failure detection with rating caps.

## Credits

Scenario methodology adapted from [ToolCall-15](https://github.com/stevibe/ToolCall-15) by [stevibe](https://x.com/stevibe) (MIT License).
