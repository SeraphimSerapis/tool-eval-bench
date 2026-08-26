# Programmatic API

`tool-eval-bench` provides two levels of programmatic access:

1. **`tool_eval_bench.api`** — high-level async function (recommended)
2. **`tool_eval_bench.application.service`** — low-level service class (advanced)

## Quick Start (Recommended)

```python
import asyncio
from tool_eval_bench.api import run_benchmark

result = asyncio.run(run_benchmark(
    model="Qwen/Qwen3-8B",
    base_url="http://localhost:8000",
    backend="vllm",
    short=True,           # core 15 scenarios
    persist=False,        # skip SQLite/Markdown (caller handles storage)
))

print(result["final_score"])      # e.g. 87
print(result["rating"])           # e.g. "★★★★ Good"
print(result["schema_version"])   # "1"
```

The convenience re-export also works:

```python
from tool_eval_bench import run_benchmark  # same function
```

## Return Value

`run_benchmark()` returns a versioned JSON-serializable dict:

| Field | Type | Description |
|---|---|---|
| `schema_version` | str | Output schema version (currently `"1"`) |
| `tool_eval_bench_version` | str | Package version, such as `"2.6.0"` or an identifiable development version |
| `final_score` | int | 0–100 composite score |
| `rating` | str | Star rating string |
| `safety_warnings` | list | Safety-critical failures (empty when clean) |
| `deployability` | int/None | 0–100 composite (when latency data available) |
| `responsiveness` | int/None | 0–100 latency score |
| `total_scenarios` | int | Number of scenarios evaluated |
| `run_id` | str | Unique run identifier |
| `config` | dict | Full configuration used |
| `scores` | dict | Detailed per-category and per-scenario scores |
| `metadata` | dict | System/backend metadata |
| `report_path` | str/None | Path to Markdown report (when `persist=True`) |
| `weighted_score` | int/None | 0–100 difficulty-weighted score (when `weight_by_difficulty=True`) |

The top-level `final_score`, `rating`, `safety_warnings`, `deployability`,
and `total_scenarios` fields are promoted from the nested `scores` dict for
easy consumption by leaderboard pipelines and external integrators.

## Parameters

```python
result = asyncio.run(run_benchmark(
    # Required
    model="Qwen/Qwen3-8B",
    base_url="http://localhost:8000",

    # Optional — defaults shown
    backend="vllm",
    api_key=None,
    scenarios=None,       # explicit list, or use short=True/False
    short=False,          # True = core 15, False = standard 69
    temperature=0.0,
    timeout_seconds=60.0,
    max_turns=8,
    seed=None,
    reference_date=None,  # "YYYY-MM-DD"
    concurrency=1,
    error_rate=0.0,
    alpha=0.7,
    extra_params=None,    # e.g. {"chat_template_kwargs": {"enable_thinking": False}}
    weight_by_difficulty=False,  # weight scores by difficulty tier
    wire_format=None,      # auto, openai, or gemini
    on_scenario_start=None,
    on_scenario_result=None,
    persist=True,         # False = skip SQLite + Markdown
    output_dir=None,      # default: ./runs/
))
```

## Persistence Control

By default, `run_benchmark()` persists results to SQLite and generates
Markdown reports. Set `persist=False` to disable all file I/O — useful
when the caller handles its own storage (e.g., sparkrun, CI pipelines):

```python
# No files written — pure in-memory benchmark
result = asyncio.run(run_benchmark(
    model="my-model",
    base_url="http://localhost:8000",
    persist=False,
))
```

## Selecting Scenarios

The Python API does not have a separate `hardmode` boolean. Pass the desired
definitions through `scenarios`. The standard registry contains 69 scenarios,
and `ALL_SCENARIOS_WITH_HARDMODE` contains all 88.

```python
from tool_eval_bench.evals.scenarios import SCENARIOS, ALL_SCENARIOS
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.evals.scenarios_hardmode import HARDMODE_SCENARIOS

# Core 15 (equivalent to --short)
result = asyncio.run(run_benchmark(
    model="my-model", base_url="http://localhost:8000",
    short=True,
))

# All 69 (default)
result = asyncio.run(run_benchmark(
    model="my-model", base_url="http://localhost:8000",
))

# Explicit scenario list
selected = [s for s in ALL_SCENARIOS if s.category.value == "K"]
result = asyncio.run(run_benchmark(
    model="my-model", base_url="http://localhost:8000",
    scenarios=selected,
))

# All 88 including Hard Mode
result = asyncio.run(run_benchmark(
    model="my-model", base_url="http://localhost:8000",
    scenarios=list(ALL_SCENARIOS_WITH_HARDMODE),
))

# Hard Mode only, or a selected Hard Mode subset
result = asyncio.run(run_benchmark(
    model="my-model", base_url="http://localhost:8000",
    scenarios=list(HARDMODE_SCENARIOS),
))
selected = [s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id in {"TC-85", "TC-88"}]
result = asyncio.run(run_benchmark(
    model="my-model", base_url="http://localhost:8000",
    scenarios=selected,
))
```

## Callbacks

Attach async callbacks for real-time progress monitoring:

```python
async def on_start(scenario, idx, total):
    print(f"[{idx + 1}/{total}] Starting {scenario.id}: {scenario.title}")

async def on_result(scenario, result, idx, total):
    print(f"[{idx + 1}/{total}] {scenario.id}: {result.status.value} ({result.points}/2)")

result = asyncio.run(run_benchmark(
    model="my-model",
    base_url="http://localhost:8000",
    on_scenario_start=on_start,
    on_scenario_result=on_result,
))
```

## Accessing Detailed Results

```python
scores = result["scores"]

# Overall
scores["final_score"]   # 0-100
scores["total_points"]  # sum of all scenario points
scores["max_points"]    # maximum possible points
scores["rating"]        # e.g. "★★★★ Good"

# Per-category
for cs in scores["category_scores"]:
    print(f"{cs['label']}: {cs['earned']}/{cs['max']} ({cs['percent']}%)")

# Per-scenario
for sr in scores["scenario_results"]:
    print(f"{sr['scenario_id']}: {sr['status']} — {sr['summary']}")
```

## Error Handling

When the benchmark fails before scenario execution (connection errors,
no models, etc.), the errors use structured codes from
`tool_eval_bench.domain.errors`:

| Code | Meaning |
|------|---------|
| `connection_failed` | Server unreachable |
| `http_error` | HTTP 4xx/5xx response |
| `detection_failed` | Probing exception |
| `invalid_response` | Non-JSON response |
| `no_models` | Empty model list |
| `model_not_available` | Model is listed but fails a pre-flight inference request |
| `no_server` | Auto-discovery found nothing |

## Machine-Readable Args Schema

External tools can validate benchmark configuration:

```python
from tool_eval_bench.schema import get_schema

schema = get_schema()  # {"schema_version": "7", "args": [...]}
for arg in schema["args"]:
    print(f"{arg['name']}: {arg['type']} = {arg['default']}")
```

## Low-Level Service API (Advanced)

For fine-grained control over persistence and storage. `BenchmarkService` is owned by
`tool_eval_bench.application.service`; the older `tool_eval_bench.runner.service` path still
imports but is a compatibility re-export only, so new code should not use it.

```python
import asyncio
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter

# RunRepository supports context manager for automatic cleanup
with RunRepository(db_path="my_results.sqlite") as repo:
    reporter = MarkdownReporter(root="my_reports/")
    service = BenchmarkService(repo=repo, reporter=reporter)

    result = asyncio.run(service.run_benchmark(
        model="my-model-name",
        backend="vllm",
        base_url="http://localhost:8080",
        temperature=0.0,
        timeout_seconds=30.0,
    ))

# Or disable persistence entirely
service = BenchmarkService(repo=None, reporter=None)
```

## Historical Queries

```python
from tool_eval_bench.storage.db import RunRepository

with RunRepository() as repo:
    # List recent runs
    runs = repo.list(limit=10)

    # Get a specific run
    run = repo.get("run_id_here")

    # Get latest run for a model
    latest = repo.get_latest(model="my-model")
```

## Notes

- The `backend` parameter is a **label** for reports. OpenAI-compatible
  backends use the OpenAI adapter; Gemini can use its native adapter when
  `wire_format="gemini"` or the URL selects it.
- The `base_url` should be the server root **without** `/v1`
  (e.g. `http://localhost:8080`). The adapter appends `/v1/chat/completions`
  automatically. If you include `/v1`, it will be detected and not duplicated.
- Set `api_key` if your server requires authentication.
- For thinking models (Qwen3, DeepSeek), pass
  `extra_params={"chat_template_kwargs": {"enable_thinking": False}}` to
  disable thinking — or use the CLI's `--no-think` flag.

## Accuracy Benchmarks (GSM8K, MMLU, IFEval)

The accuracy benchmark plugins are currently **CLI-only** (`--gsm8k-only`,
`--mmlu-only`, `--ifeval-only`).  They do not yet have a Python API equivalent
of `run_benchmark()`.

To run accuracy benchmarks programmatically, use `subprocess`:

```python
import json, subprocess

r = subprocess.run(
    ["tool-eval-bench", "--mmlu-only", "--mmlu-limit", "50", "--json"],
    capture_output=True, text=True,
)
# Note: accuracy benchmark results are currently rendered to the
# terminal and Markdown reports, not to the JSON output envelope.
```

Each plugin implements the `BenchmarkPlugin` ABC from
`tool_eval_bench.domain.plugin` and can be instantiated directly for
advanced usage — see the plugin source code for the `run()` method
signature.
