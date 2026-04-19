# Programmatic API

`tool-eval-bench` is primarily a CLI tool, but the service layer can be used
directly from Python code. This is useful for CI/CD integration, custom
evaluation pipelines, and comparison harnesses.

## Quick Start

```python
import asyncio
from tool_eval_bench.runner.service import BenchmarkService
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter

# Optional: override storage locations (defaults: ./data/ and ./runs/)
repo = RunRepository(db_path="my_results.sqlite")
reporter = MarkdownReporter(root="my_reports/")
service = BenchmarkService(repo=repo, reporter=reporter)

result = asyncio.run(service.run_benchmark(
    model="my-model-name",
    backend="vllm",
    base_url="http://localhost:8080/v1",
    temperature=0.0,
    timeout_seconds=30.0,
))

# result is a dict with keys: status, run_id, scores, report_path
print(f"Score: {result['scores']['final_score']} / 100")
print(f"Rating: {result['scores']['rating']}")
print(f"Report: {result['report_path']}")
```

## Selecting Scenarios

```python
from tool_eval_bench.evals.scenarios import SCENARIOS, ALL_SCENARIOS

# Run only the core 15 scenarios (equivalent to --short)
result = asyncio.run(service.run_benchmark(
    model="my-model",
    backend="vllm",
    base_url="http://localhost:8080/v1",
    scenarios=SCENARIOS,  # core 15 only
))

# Run all 69 scenarios (default)
result = asyncio.run(service.run_benchmark(
    model="my-model",
    backend="vllm",
    base_url="http://localhost:8080/v1",
    scenarios=ALL_SCENARIOS,
))

# Run specific scenarios by ID
selected = [s for s in ALL_SCENARIOS if s.id in ("TC-01", "TC-34", "TC-37")]
result = asyncio.run(service.run_benchmark(
    model="my-model",
    backend="vllm",
    base_url="http://localhost:8080/v1",
    scenarios=selected,
))
```

## Callbacks

You can attach async callbacks to monitor progress:

```python
async def on_start(scenario, idx, total):
    print(f"[{idx + 1}/{total}] Starting {scenario.id}: {scenario.title}")

async def on_result(scenario, result, idx, total):
    print(f"[{idx + 1}/{total}] {scenario.id}: {result.status.value} ({result.points}/2)")

result = asyncio.run(service.run_benchmark(
    model="my-model",
    backend="vllm",
    base_url="http://localhost:8080/v1",
    on_scenario_start=on_start,
    on_scenario_result=on_result,
))
```

## Accessing Results

The return dict contains the full scoring breakdown:

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

## Historical Queries

```python
# List recent runs
runs = repo.list(limit=10)

# Get a specific run
run = repo.get("run_id_here")

# Get latest run for a model
latest = repo.get_latest(model="my-model")
```

## Notes

- The `backend` parameter is a **label** for reports — all backends use the
  same OpenAI-compatible HTTP adapter internally.
- The `base_url` should include `/v1` (e.g. `http://localhost:8080/v1`).
- Set `api_key` if your server requires authentication.
- Call `repo.close()` when done to release the SQLite connection.
