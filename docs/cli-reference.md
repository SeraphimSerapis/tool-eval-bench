# CLI reference

Every flag, exit code, and JSON field, for scripting `tool-eval-bench` in CI or driving it from an
agent. If you are running the benchmark by hand for the first time, start with the
[README](../README.md) instead.

## What this tool does

`tool-eval-bench` evaluates LLM tool-calling quality using 69 deterministic
scenarios across 16 categories.  It produces a 0–100 score with per-category
breakdowns, safety warnings, and full conversation traces.

## Quick start (zero-config)

If an inference server is running on localhost at a standard port, no
configuration is needed:

```bash
# Auto-discovers server and model, runs core 15 scenarios
tool-eval-bench --short --json

# Full 69 scenarios
tool-eval-bench --json
```

Auto-discovery probes these ports in order:
8000 (vLLM), 8080, 8081, 8082, 30000 (SGLang), 4000 (LiteLLM),
3000, 11434 (Ollama), 5000 (TGI).

## Headless / JSON mode

Always use `--json` for machine-readable output.  In this mode:

- **stdout** contains only the JSON result envelope
- **stderr** contains JSONL progress events (one per line)
- Interactive prompts are skipped (first model is auto-selected)
- Warmup and informational banners are suppressed

```bash
# JSON to stdout
tool-eval-bench --json --short

# JSON to file (keeps stdout clean for logging)
tool-eval-bench --json-file results.json --short
```

## Server readiness check

Use `--probe` to wait for the server to be ready before benchmarking:

```bash
tool-eval-bench --probe --base-url http://localhost:8000
# Exit 0 = ready, exit 1 = not reachable
```

In a script:
```bash
until tool-eval-bench --probe --json 2>/dev/null; do
  sleep 5
done
tool-eval-bench --json --short
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Success |
| 1    | Runtime error or server not ready (--probe) |
| 2    | Connection/HTTP error (server unreachable, bad response) |
| 3    | No models found on server |

## Specifying the server

Priority order (highest wins):
1. `--base-url` CLI flag
2. `TOOL_EVAL_BASE_URL` environment variable
3. `TOOL_EVAL_HOST` + `TOOL_EVAL_PORT` environment variables
4. `.env` file (never overrides existing env vars)
5. Auto-discovery on localhost

```bash
# Explicit URL
tool-eval-bench --base-url http://192.168.1.5:8000 --json --short

# Via environment
export TOOL_EVAL_BASE_URL=http://192.168.1.5:8000
tool-eval-bench --json --short
```

## Key CLI flags

| Flag | Purpose |
|------|---------|
| `--json` | Machine-readable JSON output |
| `--json-file PATH` | Write JSON to file instead of stdout |
| `--short` | Run core 15 scenarios (~2 min) instead of full 69 |
| `--probe` | Check server readiness and exit |
| `--dry-run` | List scenarios that would run (no server needed) |
| `--base-url URL` | Server endpoint |
| `--model NAME` | Model name (auto-detected if omitted) |
| `--backend NAME` | Backend label for reports: `vllm`, `litellm`, `llamacpp` |
| `--seed N` | Random seed for reproducibility |
| `--temperature F` | Sampling temperature (default: 0.0 = greedy) |
| `--timeout F` | Per-request timeout in seconds (default: 60) |
| `--no-think` | Disable thinking/reasoning (critical for Qwen3/DeepSeek) |
| `--no-warmup` | Skip server warm-up request |
| `--hardmode` | Include 19 Hard Mode scenarios (Category P) |
| `--categories A B K` | Run only specific categories (A–P) |
| `--scenarios TC-01 TC-07` | Run specific scenario IDs |
| `--perf` | Also run throughput benchmark |
| `--trials N` | Run N trials for statistical analysis |
| `--resume RUN_ID` | Resume a previous run (skip already-passed scenarios) |
| `--hardmode-only` | Run ONLY Hard Mode scenarios (equivalent to --hardmode --categories P) |
| `--weight-by-difficulty` | Weight scores by difficulty tier (harder scenarios count more) |

## Accuracy benchmarks (pluggable)

In addition to the 69-scenario tool-call benchmark, `tool-eval-bench` supports
accuracy benchmarks via the same OpenAI-compatible endpoints. These only require
`/v1/chat/completions` — no `tools` support needed.

```bash
# GSM8K — math reasoning (1,319 questions, 8-shot)
tool-eval-bench --gsm8k-only --gsm8k-limit 50

# MMLU — multitask knowledge (14,042 questions, 5-shot)
tool-eval-bench --mmlu-only --mmlu-limit 50

# IFEval — instruction following (541 prompts, 25 constraints)
tool-eval-bench --ifeval-only --ifeval-limit 20

# Needle in a haystack — long-context retrieval (no dataset download)
tool-eval-bench --needle-only --needle-lengths 4 --needle-depths 5

# Run all accuracy benchmarks (skip tool-call scenarios)
tool-eval-bench --gsm8k-only --mmlu-only --ifeval-only
```

GSM8K, MMLU, and IFEval datasets are downloaded from HuggingFace on first use
and cached locally. Install `pip install tool-eval-bench[hf]` for rate-limit-free
downloads. The needle benchmark generates its own cases and downloads nothing.

| Flag | Purpose |
|------|---------|
| `--gsm8k-only` | GSM8K math reasoning |
| `--mmlu-only` | MMLU multitask knowledge |
| `--ifeval-only` | IFEval instruction following |
| `--needle-only` | Needle-in-a-haystack retrieval |
| `--gsm8k-limit N` | Limit questions (default: 200) |
| `--mmlu-limit N` | Limit questions (default: 500) |
| `--ifeval-limit N` | Limit prompts (default: all 541) |
| `--needle-lengths N` | Haystack sizes to probe (default: 4) |
| `--needle-depths N` | Needle depths to probe (default: 5) |

See [needle.md](needle.md) for the retrieval grid and how to read it.

## Understanding the JSON output

```jsonc
{
  "schema_version": "1",
  "tool_eval_bench_version": "1.8.0",
  "final_score": 85,           // 0–100, the headline number
  "rating": "★★★★ Good",       // star rating with label
  "safety_warnings": [],       // empty = safe; non-empty = failures in safety scenarios
  "deployability": 78,         // quality × speed composite (if latency data available)
  "responsiveness": 65,        // latency score 0–100
  "total_scenarios": 69,
  "run_id": "2026-05-07T...",
  "config": { ... },
  "scores": {
    "final_score": 85,
    "category_scores": [ ... ],
    "scenario_results": [ ... ]
  }
}
```

## JSONL progress events (stderr)

Each line on stderr is a JSON object:

```jsonc
{"event": "server_discovered", "base_url": "http://localhost:8000", "backend": "vllm", ...}
{"event": "model_auto_selected", "model": "Qwen/Qwen3-8B", ...}
{"event": "scenario_start", "scenario_id": "TC-01", "index": 0, "total": 15}
{"event": "scenario_result", "scenario_id": "TC-01", "status": "pass", "points": 2, ...}
{"event": "benchmark_complete", "json_file": "results.json", "final_score": 85}
{"event": "error", "error": "no_server", "message": "..."}
```

## Error codes (structured)

When `--json` mode emits an error event, the `error` field is one of:

| Code | Exit | Meaning |
|------|------|---------|
| `connection_failed` | 2 | TCP connection refused, DNS failure, or timeout |
| `http_error` | 2 | Server responded with 4xx/5xx status |
| `detection_failed` | 2 | Unexpected exception during server probing |
| `invalid_response` | 2 | Response body is not valid JSON |
| `no_models` | 3 | Server responded but model list is empty |
| `no_server` | 2 | Auto-discovery found no server on localhost |

These constants are defined in `tool_eval_bench.domain.errors` for
programmatic consumers.

## Programmatic API (Python)

```python
import asyncio
from tool_eval_bench.api import run_benchmark

result = asyncio.run(run_benchmark(
    model="Qwen/Qwen3-8B",
    base_url="http://localhost:8000",
    short=True,
    persist=False,  # skip SQLite/Markdown artifacts
))

print(result["final_score"])  # 85
```

## Interpreting results for automated decisions

```python
import json, subprocess

r = subprocess.run(
    ["tool-eval-bench", "--json", "--short"],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print("Benchmark failed")
else:
    data = json.loads(r.stdout)
    score = data["final_score"]
    warnings = data.get("safety_warnings", [])

    if warnings:
        print(f"UNSAFE: {len(warnings)} safety failures")
    elif score >= 75:
        print(f"GOOD: score {score}")
    elif score >= 60:
        print(f"ADEQUATE: score {score}")
    else:
        print(f"POOR: score {score}")
```

## Score tiers

| Score | Rating | Meaning |
|-------|--------|---------|
| 90–100 | ★★★★★ Excellent | Production-ready tool calling |
| 75–89 | ★★★★ Good | Reliable for most agentic tasks |
| 60–74 | ★★★ Adequate | Works but has notable gaps |
| 40–59 | ★★ Weak | Significant tool-calling issues |
| 0–39 | ★ Poor | Not suitable for agentic use |

## Common pitfalls

1. **Don't parse stderr as JSON** — it's JSONL (one object per line), not a single JSON document.
2. **Don't assume model is auto-detected** — if the server has 0 models loaded, exit code is 3.
3. **Use `--short` for fast checks** — the full suite takes 10–20 minutes; core 15 takes ~2 minutes.
4. **Timeout with thinking models** — models like Qwen3 with thinking enabled may need `--timeout 120`.
5. **Warmup is automatic** — the first request primes the server. Use `--no-warmup` only if already warmed.
6. **Use `--dry-run` to preview** — before committing to a long run, check which scenarios would execute.
