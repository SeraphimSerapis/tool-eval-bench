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
| `deployability` | int/None | `alpha × final_score + (1 − alpha) × responsiveness`; `None` without latency data. See [methodology](methodology.md#responsiveness-and-deployability) |
| `responsiveness` | int/None | 0–100 from median turn latency, logistic curve centred on 3 s; `None` without latency data |
| `total_scenarios` | int/None | Number of scenarios with a result, including infrastructure failures excluded from the score; `None` when `scores` has no `scenario_results` |
| `run_id` | str | Unique run identifier |
| `config` | dict | Full configuration used |
| `scores` | dict | Detailed per-category and per-scenario scores |
| `metadata` | dict | Run context: environment, run parameters, backend label, and engine facts |
| `report_path` | str/None | Path to Markdown report (when `persist=True`) |
| `weighted_score` | int/None | 0–100 difficulty-weighted score (when `weight_by_difficulty=True`) |

The `metadata` dictionary is the run context that `tool-eval-bench --json` records,
with the same keys for the same server. It carries the local environment
(`tool_version`, `git_sha`, `hostname`, `platform_info`, `python_version`), the run
parameters (`model`, `backend`, redacted `base_url`, `temperature`, `max_turns`,
`timeout_seconds`, `seed`, `scenario_selector`, `trials`, `parallel`, `error_rate`,
`thinking_enabled`, `max_tokens`, and `extra_params` or `system_prompt` when set),
and the best-effort engine facts used in reports and comparison fingerprints
(`engine_name`, `engine_version`, `server_model_id`, `server_model_root`,
`max_model_len`, `quantization`, `slot_count`, `spec_decoding`). Keys whose value
is unknown are omitted. `slot_count` records the llama.cpp server slot count from
`/props.total_slots`. It is request concurrency capacity, not a physical GPU count.
`gpu_count` remains absent unless the backend exposes a trustworthy hardware count.
If the context cannot be built, the run still completes and `metadata` falls back
to a smaller dictionary of host facts and the `/v1/models` probe.

Earlier releases always recorded that smaller dictionary for API runs. Its keys map as follows:
`host` is now `hostname`, `platform` is `platform_info`, `config.model`,
`config.backend` and `config.base_url` are top-level `model`, `backend` and `base_url`,
and the `backend_probe` fields are top-level `server_model_id`, `server_model_root`
and `max_model_len`. `pid` is no longer recorded.

The top-level `final_score`, `rating`, `safety_warnings`, `deployability`,
`weighted_score`, and `total_scenarios` fields are promoted from the nested
`scores` dict for easy consumption by leaderboard pipelines and external
integrators.

An optional answer audit appears as `decision_audit` within each audited
`scores.scenario_results` item. It records the exact input and question,
`check_id`, `question_sha256`, evidence scope, judge identity, probabilities,
latency, and disagreement with the scenario's deterministic check. It never
changes official scores, safety warnings, or `config_fingerprint`. Unconfigured
runs retain the original JSON shape.
`on_scenario_audit` receives updates when an actual judge request starts and
finishes, or when a previously performed audit is reused during resume. It is
not called for unaudited scenarios, empty evidence, or oversized inputs that
never reach the judge. The callback receives `(scenario, result, phase)` with
phase `started`, `completed`, or `reused`; audit data is in
`result.decision_audit`. A failed request still emits a completed update with
status `unavailable`. Callback errors cannot change a judgment or official score.
See [answer audits](decision-models.md#answer-audits).

## Parameters

```python
result = asyncio.run(run_benchmark(
    # Required
    model="Qwen/Qwen3-8B",
    base_url="http://localhost:8000",

    # Optional — defaults shown
    backend="unknown",    # detected from the server; any other label is kept
    api_key=None,
    scenarios=None,       # explicit list, or use short=True/False
    short=False,          # True = core 15, False = standard 69
    temperature=0.0,
    timeout_seconds=120.0,
    max_turns=8,
    seed=None,
    reference_date=None,  # "YYYY-MM-DD"
    concurrency=1,
    error_rate=0.0,
    alpha=0.7,            # quality weight in deployability; speed gets 1 − alpha
    extra_params=None,    # e.g. {"chat_template_kwargs": {"enable_thinking": False}}
    weight_by_difficulty=False,  # weight scores by difficulty tier
    wire_format=None,      # auto, openai, gemini, or anthropic
    extra_headers=None,    # e.g. {"User-Agent": "my-agent/1.0"}
    session_header=None,   # e.g. "x-opencode-session"; one id per scenario
    system_prompt=None,    # replaces the built-in prompt; date line still appended
    decision_judge=None,           # "recommended" (default with a judge URL) or "all"
    decision_judge_base_url=None,  # separate /v1/systemone endpoint for answer audits
    decision_judge_model=None,     # required if the judge URL is set
    decision_judge_api_key=None,   # judge-only key; no environment fallback
    on_scenario_start=None,
    on_scenario_result=None,
    on_scenario_audit=None,  # async (scenario, result, phase): started/completed/reused
    persist=True,         # False = skip SQLite + Markdown
    output_dir=None,      # default: ./runs/
    probe_engine=True,    # False = no detection or engine metadata requests
))
```

`max_turns` must be at least 1, and `error_rate` and `alpha` must be between 0
and 1 inclusive. An out-of-range or NaN value raises `ValueError` before any
request is sent.

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
and `ALL_SCENARIOS_WITH_HARDMODE` contains all 97.

```python
from tool_eval_bench.evals.scenarios import SCENARIOS, ALL_SCENARIOS
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.evals.scenarios import HARDMODE_SCENARIOS

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

schema = get_schema()  # {"schema_version": "8", "args": [...]}
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

- The `backend` parameter is a **label** for reports. Left at its `unknown`
  default, it is detected the way the CLI detects it: a hosted Gemini or Anthropic
  endpoint is labelled by its wire format without any probe request, and any other
  server is identified from its `/metrics` namespace, vLLM's `/version`, the identity
  it declares, or llama.cpp's `/props`. A server that does not identify itself stays
  `unknown`, and a failed probe never fails the run. Any other value, such as
  `halogen` behind a proxy that hides `/metrics`, is kept as given and skips detection.
  Engine metadata is still read for an explicit label, as the CLI does.
- `probe_engine=False` sends no detection or engine metadata requests, like the CLI's
  `--no-probe-engine`. A hosted endpoint is still named by its wire format; any other
  label stays as passed, and `metadata` carries no engine facts.
- The backend label and engine facts are part of the comparison fingerprint. Runs that
  recorded `unknown` before detection existed do not share a fingerprint with new runs
  against the same server.
- The label does not choose the request format. OpenAI-compatible backends use the
  OpenAI adapter; Gemini can use its native adapter when `wire_format="gemini"` or the
  URL selects it.
- The `base_url` should be the server root **without** `/v1`
  (e.g. `http://localhost:8080`). The adapter appends `/v1/chat/completions`
  automatically. If you include `/v1`, it will be detected and not duplicated.
- Set `api_key` if your server requires authentication.
- `extra_headers` ride on every request. `session_header` names a header that
  carries a per-conversation id; each scenario sends one id for all its turns.
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
# Accuracy benchmarks have no JSON result envelope, so stdout stays empty.
# The results are in the Markdown report and SQLite; the run_saved event on
# stderr names them.
saved = next(
    event
    for event in map(json.loads, r.stderr.splitlines())
    if event["event"] == "run_saved"
)
print(saved["run_id"], saved["report_path"])
```

Each plugin implements the `BenchmarkPlugin` ABC from
`tool_eval_bench.domain.plugin` and can be instantiated directly for
advanced usage — see the plugin source code for the `run()` method
signature.
