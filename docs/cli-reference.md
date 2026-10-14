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
3000, 11434 (Ollama), 5000 (TGI). The first port whose `/v1/models` (or `/models`
when that is a 404) answers HTTP 200 with a JSON model list wins. A port that
answers with an HTML page or any other body is skipped.

## Headless / JSON mode

Always use `--json` for machine-readable output.  In this mode:

- **stdout** contains only the JSON result envelope
- **stderr** contains JSONL progress events (one per line); warnings arrive as `log` events
- Interactive prompts are skipped (first model is auto-selected)
- Warmup, banners, tables, and progress bars are suppressed in every mode. llama-benchy
  still runs its own warm-up, so `--perf` measures the same way with or without `--json`

Modes that have no result envelope (`--perf-only`, `--spec-bench`,
`--context-pressure-sweep`, and the accuracy plugins such as `--gsm8k-only`)
write nothing to stdout. Their results go to the Markdown report and SQLite as
usual, and a `run_saved` event on stderr gives the run ID and report path.
`--perf` combined with `--skip-tool-eval`, `--context-pressure-sweep`, or a
plugin-only run saves its throughput sweep the same way, as a separate `perf` run,
before the other mode starts.

`--spec-live` and `--decision-live` are interactive monitors and exit 2 when
combined with `--json`. Invalid arguments exit 2 with an `invalid_arguments`
error event. The one exception to the stderr rule is an error argparse raises
while parsing, such as an unknown flag or a missing or malformed value: it
keeps argparse's usage text, because `--json` is not known yet.

`--history`, `--leaderboard`, `--compare`, and `card` print tables or write
files and exit 2 with `invalid_arguments` under `--json`. Use `--export json` for machine-readable
run data.

When a scored run fails after it starts, stdout (or the `--json-file` path)
still gets an envelope. The score fields are null and `error` carries the
redacted message:

```jsonc
{
  "schema_version": "1",
  "tool_eval_bench_version": "...",
  "final_score": null,
  "rating": null,
  "safety_warnings": [],
  "deployability": null,
  "responsiveness": null,
  "total_scenarios": null,
  "error": "..."
}
```

stderr gets a `run_failed` error event with the same message, and no
`benchmark_complete` event follows. The exit code is 1. Check `error` before
anything else: its `safety_warnings: []` means no scores exist, not that the
model is safe. A Ctrl+C before the scenarios start emits `run_failed` with the
message `interrupted` and exits 1.

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
# Exit 0 = ready, 1 = not reachable, 2 = answers, but not with a JSON model listing
```

Exit 2 (`invalid_response`, also the `error_code` of the `--json` `probe_result`
event) means something such as a proxy or a web app answers at that URL with a
redirect, an HTML page, or another body that is not a JSON object. Waiting will
not fix it, so stop polling.

In a script:
```bash
until tool-eval-bench --probe --json 2>/dev/null; do
  test $? -eq 2 && exit 2  # not an inference server; check the URL
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

Console and `--json` runs use the same codes. Invalid arguments and a tripped
`--fail-on-safety` gate also exit 2.

## Specifying the server

Priority order (highest wins):
1. `--base-url` CLI flag
2. `TOOL_EVAL_<PROVIDER>_BASE_URL`, when `--provider` or `TOOL_EVAL_PROVIDER` names one
3. `TOOL_EVAL_BASE_URL` environment variable
4. `TOOL_EVAL_HOST` + `TOOL_EVAL_PORT` environment variables
5. `.env` file (never overrides existing env vars)
6. Auto-discovery on localhost

The CLI searches for the nearest `.env` starting in the current working
folder and walking up through its parents. If none is found, it uses the
previous lookup starting at the installed CLI module's folder. It loads only
the selected file, without overriding exported environment variables.

```bash
# Explicit URL
tool-eval-bench --base-url http://192.168.1.5:8000 --json --short

# Via environment
export TOOL_EVAL_BASE_URL=http://192.168.1.5:8000
tool-eval-bench --json --short

# Named provider: TOOL_EVAL_OPENAI_BASE_URL / _API_KEY / _MODEL
tool-eval-bench --provider openai --json --short
```

A provider owns its API key and model: `TOOL_EVAL_API_KEY` and `TOOL_EVAL_MODEL`
are not consulted when one is selected, and a provider with no `_BASE_URL` is an
error rather than a fall-through to auto-discovery. `--api-key` and `--model`
still override. `gemini`, `openai`, and `anthropic` set the backend label and
skip engine probing; any other name leaves detection alone.

A provider also owns its extra headers: `TOOL_EVAL_<NAME>_HEADERS` and
`TOOL_EVAL_<NAME>_SESSION_HEADER` replace the generic `TOOL_EVAL_HEADERS` and
`TOOL_EVAL_SESSION_HEADER` when one is selected. `--header` adds to whichever
set applies and wins on a name clash, compared case-insensitively;
`--session-header` overrides. Every request carries
`User-Agent: tool-eval-bench/<version>` unless a header replaces it.

## Key CLI flags

| Flag | Purpose |
|------|---------|
| `--json` | Machine-readable JSON output |
| `--json-file PATH` | Write JSON to file instead of stdout |
| `--include-held-out` | Keep held-out pack titles, summaries, and traces in `--json`, `--json-file`, and stderr events. They are withheld by default, as in reports |
| `--short` | Run core 15 scenarios (~2 min) instead of full 69 |
| `--probe` | Check server readiness and exit |
| `--dry-run` | List scenarios that would run (no server needed). Reports 0 when the invocation runs no tool-call scenarios, such as `--perf-only` |
| `--base-url URL` | Server endpoint |
| `--redact-url` | Mask the server URL in console output, including `--probe`, pre-flight errors, the auto-discovered localhost URL, and the `spec-live` metrics endpoint, for screenshots and recordings. Reports, stored runs, and `--json` error output are always redacted |
| `--model NAME` | Model name (auto-detected if omitted) |
| `--backend NAME` | Backend label for reports: `vllm`, `litellm`, `llamacpp`, `sglang`, `gemini`, `openai`, `anthropic`, `ninfer`, `tensorfold`, `halogen`, `strata`, `tabbyapi`, `unknown`. Auto-detected when omitted; otherwise `unknown`. |
| `--provider NAME` | Read the endpoint from `TOOL_EVAL_<NAME>_*` env vars |
| `--header NAME=VALUE` | Extra request header, repeatable (env: `TOOL_EVAL_HEADERS`, `;`-separated) |
| `--session-header NAME` | Header that carries a per-conversation id, e.g. `x-opencode-session` (env: `TOOL_EVAL_SESSION_HEADER`) |
| `--seed N` | Random seed for reproducibility |
| `--temperature F` | Sampling temperature (default: 0.0 = greedy) |
| `--timeout F` | Per-request timeout in seconds (default: 120) |
| `--no-think` | Disable thinking/reasoning (critical for Qwen3/DeepSeek) |
| `--no-warmup` | Skip the tool-eval-bench warm-up request. llama-benchy (`--perf`, `--perf-only`) always runs its own warm-up |
| `--hardmode` | Include 28 Hard Mode scenarios (Category P) |
| `--categories A B K` | Run only specific categories (A–P). P needs `--hardmode`; a filter that matches no scenario is a usage error (exit 2) |
| `--scenarios TC-01 TC-07` | Run specific scenario IDs |
| `--perf` | Also run throughput benchmark |
| `--trials N` | Run N trials for statistical analysis. With `--resume`, trial 1 finishes the resumed run and trials 2..N run the full protocol as new runs |
| `--max-turns N` | Max turns per scenario (default: 8, minimum 1) |
| `--error-rate RATE` | Inject random tool errors at this rate, 0 to 1, for robustness testing. An injected call returns a simulated 429, 500, or 503 without running the mock tool, so it changes no scenario state. When the model retries that tool in a later turn, scenarios that count calls exactly do not count the retry as a duplicate, and graders inspect the retry instead of the failed attempt |
| `--fail-on-safety` | Exit 2 when a safety-critical scenario fails. With `--trials`, every trial still runs and an unsafe trial fails the gate |
| `--diff RUN_ID` | Print a per-scenario comparison with a previous run after the run. `latest` means the newest completed tool-call run before this one. Ignored with `--json` |
| `--resume RUN_ID` | Resume a previous run (skip already-passed scenarios). The run must exist and be unfinished; that is checked before any server work. A run ID that starts with `-` is passed as `--resume=-ID` or `resume -- -ID` |
| `--hardmode-only` | Run ONLY Hard Mode scenarios (equivalent to --hardmode --categories P) |
| `--weight-by-difficulty` | Weight scores by difficulty tier (harder scenarios count more) |
| `--system-prompt TEXT` | Replace the built-in system prompt with TEXT for every scenario |
| `--system-prompt-file PATH` | Read the system prompt override from a UTF-8 file, max 32 KiB (exclusive with `--system-prompt`) |
| `--decision-judge [SET]` | Answer-audit set: `recommended` (11 scenarios; the bare flag, and the default when only the connection flags are given) or `all` (17). Needs a judge URL from the flag or env |
| `--decision-judge-base-url URL` | Independent `/v1/systemone` endpoint for answer auditing; official scores stay unchanged (env: `TOOL_EVAL_DECISION_JUDGE_BASE_URL`) |
| `--decision-judge-model NAME` | Judge model, auto-detected from the judge's `/v1/models` when omitted (env: `TOOL_EVAL_DECISION_JUDGE_MODEL`); authentication uses `TOOL_EVAL_DECISION_JUDGE_API_KEY`, never the benchmark's key |
| `--alpha F` | Quality weight in the deployability composite, 0 to 1 (default: 0.7; speed gets 1 − alpha) |

## Benchmark modes

These run alongside the tool-call scenarios, or on their own with `--skip-tool-eval`
(`--perf-only` for throughput). Method and metric details live in
[benchmarks](benchmarks.md), [speculative-decoding](speculative-decoding.md), and
[context-pressure](context-pressure.md).

| Flag | Purpose |
|------|---------|
| `--perf` | Throughput benchmark (llama-benchy style pp/tg sweep) |
| `--perf-only` | Throughput only, skip the scenarios |
| `--perf-preset NAME` | Throughput grid: `short` (default) or `long`, which adds a single-stream depth sweep to 131072 tokens within the served context window. See [benchmarks](benchmarks.md#presets) |
| `--pp N`, `--tg N` | Prompt and generation tokens per request (default: 2048, 128) |
| `--depth LIST` | Context depths, comma separated (default: the preset's, `0,4096,8192` for `short`). Under `long` it sets the depth pass |
| `--concurrency LIST` | Concurrency levels for `--perf` (default: `1,2,4`). Under `long` it sets the depth-0 pass |
| `--benchy-runs N` | Measurement runs per throughput test point (default: 3) |
| `--spec-bench` | Speculative-decoding benchmark: effective t/s, acceptance rate, τ, per-position acceptance |
| `--spec-method M` | Method hint when the server does not identify it (`--help` lists the choices) |
| `--spec-prompts LIST` | Prompt types (default: `filler,code,structured`); labels from `--spec-prompt-file` are accepted |
| `--spec-prompt-file PATH` | Your own workload: one prompt per line, or JSON lines with `prompt` and an optional `label` |
| `--spec-runs N` | Measurements per depth × prompt cell, pooled into one row (default: 3) |
| `--baseline-tgs F` | Known no-spec tg t/s for a measured speedup; without it a ceiling is derived from verify steps/s |
| `--metrics-url URL` | Prometheus `/metrics` when the API sits behind a proxy |
| `--spec-live` | Live dashboard of speculative-decoding metrics (also the `spec-live` subcommand) |
| `--spec-live-interval S` | Seconds between scrapes (default: 1.0) |
| `--context-pressure R` | Fill the context to ratio R (0–1) before each scenario |
| `--context-pressure-sweep A-B` | Sweep pressure from A to B in `--sweep-steps` levels (default: 5) |
| `--context-size N` | Override the detected context window (`/v1/models`, TensorFold `/health`, or llama.cpp or Strata `/props`) |

`--temperature` applies to spec-bench requests as well; greedy is the default and the report
records the value, since acceptance falls as sampling temperature rises.

### Combining modes

An invocation runs its modes in a fixed order: throughput, spec-bench, the
context-pressure sweep, the accuracy plugins, then the tool-call scenarios.
Combinations where a requested mode would never run exit 2 with
`invalid_arguments` instead of quietly skipping it:

| Rejected combination | Why | Instead |
|----------------------|-----|---------|
| `--spec-live` or `--decision-live` with any other benchmark or monitor | A live monitor runs until Ctrl+C | Run the monitor on its own |
| `--perf-only` with `--spec-bench`, `--context-pressure-sweep`, or a plugin | `--perf-only` stops after throughput | `--perf` |
| `--context-pressure-sweep` with `--spec-bench` or a plugin | One of the two would not run | Separate invocations |
| A `--<plugin>-only` flag with a different plugin's flag | `-only` runs that plugin alone | Plain plugin flags plus `--skip-tool-eval` |
| `--resume` with `--context-pressure-sweep` or a mode that runs no tool-call scenarios | Nothing would be resumed | Resume with a scenario run |

`--perf` with `--perf-only`, and `--gsm8k` with `--gsm8k-only`, are accepted:
they ask for the same thing twice. `--spec-bench` with a plugin and
`--skip-tool-eval` runs spec-bench, then the plugin. `--probe` and the storage
commands ignore benchmark mode flags, but conflicting pairs are still rejected.

### Auditing answers

Supply `--decision-judge-base-url` to audit the selected scenarios' answers with
the `recommended` set. Add `--decision-judge all` for the larger set. With
`TOOL_EVAL_DECISION_JUDGE_BASE_URL` set, `--decision-judge` alone is enough; the
variable never turns audits on by itself. The judge model comes from
`--decision-judge-model`, then `TOOL_EVAL_DECISION_JUDGE_MODEL`, then the judge's
`/v1/models`. A judge serving several models opens the model picker, and stops
a `--json` run with `invalid_arguments`. Scenarios outside the set, and
held-out packs, remain unaudited. The flags apply to `run`, `resume`, and flat
scenario invocations.
Keep the same judge flags, including the set, when resuming an audited run.
Results include a separate `decision_audit`; no audit changes points, safety
warnings, or the run's comparison fingerprint.
See [answer audits](decision-models.md#answer-audits) for input limits, abstention,
credential isolation, and persisted evidence.

### Overriding the system prompt

`--system-prompt` and `--system-prompt-file` replace the built-in persona for every
scenario in the run. Both forms are equivalent: surrounding whitespace and a leading
byte-order mark are stripped, so the same words read from a file and passed inline
produce the same run.

The benchmark reference-date line is appended after the override and remains
authoritative — an override that asserts its own "today" does not move the date that
relative-time scenarios are graded against.

An override is recorded in the run config and folded into `config_fingerprint`, so runs
with different prompts are not compared with each other, and `--resume` refuses to
continue a run whose prompt differs — including a run recorded before this option
existed, which is treated as having used the built-in prompt. A run without either flag
persists and fingerprints exactly as it did before the option existed.

The prompt must be valid UTF-8, non-blank, and at most 32 KiB, because it is stored
verbatim in the run config and metadata. Reports mark only *that* a custom prompt was
used; they never reproduce it. The flags are accepted but ignored, with a warning, by
invocations that run no tool-call scenarios (`--perf-only`, `--spec-bench` alone, a
`plugin` run, or `--skip-tool-eval`).

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

# Decision models — llama.cpp /v1/systemone (no dataset download, no chat preflight)
tool-eval-bench --decision-bench-only --base-url http://host:8084

# Live canary monitor for a decision model (runs until Ctrl+C)
tool-eval-bench decision-live --base-url http://host:8084

# Chained onto a full sweep, like --perf
tool-eval-bench --hardmode --seed 42 --perf --needle

# Run all accuracy benchmarks (skip tool-call scenarios)
tool-eval-bench --gsm8k --mmlu --ifeval --skip-tool-eval
```

GSM8K, MMLU, and IFEval datasets are downloaded from HuggingFace on first use
and cached locally. Install the `[hf]` extra for rate-limit-free downloads
(`pip install -e '.[hf]'` in a checkout, or the `uv tool install` form in
[benchmarks.md](benchmarks.md)). The needle benchmark generates its own cases and downloads nothing.

| Flag | Purpose |
|------|---------|
| `--gsm8k-only` | GSM8K math reasoning |
| `--mmlu-only` | MMLU multitask knowledge |
| `--ifeval-only` | IFEval instruction following |
| `--needle-only` | Needle-in-a-haystack retrieval |
| `--decision-bench-only` | Decision-model scoring on `/v1/systemone` |
| `--gsm8k-limit N` | Limit questions (default: 200) |
| `--mmlu-limit N` | Limit questions, sampled proportionally across subjects (default: 500) |
| `--ifeval-limit N` | Limit prompts (default: all 541) |
| `--needle-lengths N` | Haystack sizes to probe (default: 4) |
| `--needle-depths N` | Needle depths to probe (default: 5) |

See [needle.md](needle.md) for the retrieval grid and how to read it, and
[decision-models.md](decision-models.md) for the decision-model report.

## Web UI (`serve`)

`tool-eval-bench serve` starts a local browser UI over the same SQLite history
the CLI writes. It needs the `[web]` extra (Starlette and Uvicorn).

```bash
tool-eval-bench serve                    # http://localhost:8765, or the next free port
tool-eval-bench serve --serve-port 9090
```

| Flag | Purpose |
|------|---------|
| `--serve-host HOST` | Bind address (default: `127.0.0.1`; `0.0.0.0` listens on every interface) |
| `--serve-port PORT` | Bind port (default: `8765`, or the next free port up to 8774). An explicit port that is busy is an error rather than a silent move; `0` asks the OS for any free port |
| `--output-dir DIR` | Where runs launched from the browser write their Markdown reports |
| `--redact-url` | Keep the probe from echoing endpoint URLs back to the browser |

The UI shows run history with full traces, cohort leaderboards with CSV and JSON
export, and two-run comparisons with McNemar's test. It also launches runs,
streams them live over Server-Sent Events, and stops or resumes them. CLI runs
can be watched too: the UI follows their SQLite checkpoints.

**Sign-in.** On startup the server prints a link ending in `?token=...`. The token
changes on every start. Opening the link sets a session cookie
(`HttpOnly`, `SameSite=Strict`), and every request except static assets must
carry it, or `Authorization: Bearer <token>` for scripts. State-changing
requests must also come from the UI's own origin and send `application/json`.
This is what stops another website you visit from using the server: the
server holds `TOOL_EVAL_API_KEY` and can start runs that spend it.

**Binding beyond loopback.** Anyone with the link can read stored traces and
start runs with this machine's keys, over plain HTTP. Prefer the default bind
address and SSH port forwarding (`ssh -L 8765:localhost:8765 host`).

**Stop and resume.** The UI stops only runs it launched; stop a CLI run with
Ctrl+C in its terminal. It resumes only runs marked `interrupted`, using the
conditions stored with the run. Because the stored URL is redacted, enter the
endpoint again (or set `TOOL_EVAL_BASE_URL`); a different endpoint is refused.
Runs that used context pressure, a decision judge, scenario variants, or a
held-out pack need options only the CLI has, so resume those with
`tool-eval-bench resume RUN_ID` and the original flags.

## Results card

`card` renders stored runs as one shareable SVG, with optional PNG. It reads
SQLite only and needs no server, like `history` and `export`. Every run one
invocation stores is tagged with the same session, so a card normally shows one
invocation:

```bash
tool-eval-bench --seed 42 --perf --perf-preset long --needle --gsm8k --mmlu --ifeval
tool-eval-bench card --png --hardware "RTX 5090 32GB"
```

| Option | Purpose |
|--------|---------|
| `SESSION` | Session to render, as a full id or a unique prefix of at least 6 characters, in any case (default: `latest`, the newest session holding a run the card can show) |
| `--run RUN_ID` | Build the card from this run instead of a session, repeatable. A repeated id counts once. Cannot be combined with `SESSION`, and works for runs stored before sessions existed |
| `--model NAME` | With `latest`, use the newest session of this model. Matches the stored model name or its last path segment, case-sensitively, so `Qwen3-32B` matches `Qwen/Qwen3-32B`. Ignored, with a warning, alongside `SESSION` or `--run` |
| `--hardware TEXT` | Hardware line on the card. Nothing is detected, so it is shown only when given |
| `--layout portrait\|landscape` | Card layout (default: `portrait`). Landscape suits a link preview |
| `-o`, `--output PATH` | Card path. A `.png` path writes PNG and puts the SVG beside it. A directory gets the default filename inside it. Another image or document suffix, such as `.jpg` or `.pdf`, exits 2. Any other name gets `.svg` appended, so `-o Qwen3.5-27B` writes `Qwen3.5-27B.svg` |
| `--png` | Also write a PNG beside the SVG |
| `--scale N` | PNG pixel scale relative to the SVG, from 0.25 to 8 (default: 2) |
| `--output-dir DIR` | Root of the default path (default: `./runs`) |

The default path is `<output-dir>/cards/<model>-<YYYYMMDD>-<fingerprint>.svg`:
the model name lowercased with other punctuation turned into dashes, so
`Qwen/Qwen3-32B-AWQ` becomes `qwen-qwen3-32b-awq`, then the date of the newest
run on the card, and the first
8 characters of the configuration fingerprint, so two configurations of one
model on one day do not overwrite each other. The date comes from the stored
run, never from today. When an SVG is rendered again without `--png`, `card`
warns if an older PNG with the same name sits beside it.

The card uses the newest completed run of each kind in the selection: the
tool-call run, the throughput sweep (a `--perf-only` row or the samples a scored
`--perf` run carries), needle, GSM8K, MMLU, and IFEval. A sweep at concurrency
above 1 stores the batch total, so the card shows that total. The per-request
rate is the measured one stored with the sweep; for older rows without it, the
card divides the total by the concurrency and marks the value as estimated.
Failed grid cells and a sweep that stopped before its plan finished are marked
on the card. Throughput rows saved before per-cell results were stored hold
only a sample count and cannot be drawn. Missing sections render as not
measured, and the command prints the flags that would fill them, naming each
missing accuracy benchmark.

A session run with `--trials N` shows the mean of its trials: the score and
category percentages are averaged, the pass, partial, and fail counts are the
mean per trial rounded so they still add up to the scenario count, and the
rating is recomputed from the mean score. It is safety-capped if any trial
was. The card compares the selected scenarios with today's suite (public, public
plus Hard Mode, or Hard Mode alone) to mark a subset, and counts scenarios
excluded after an infrastructure failure as selected. A scenario pack, or a run
from an older registry such as the 88-scenario Hard Mode runs, gets no subset
mark because it cannot be compared.

Accuracy and needle scores count only answered items: an item or needle cell
whose request errored, such as a context overflow, is shown as an error rather
than scored as a miss. A limited run says how much of the set it used, for
example `200 of 1,319 q`.

PNG export needs the `card` extra:
`uv tool install 'tool-eval-bench[card] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'`,
or the same requirement with `pip install`. Without it, a PNG request still
writes the SVG, prints the install command, then exits 1. A PNG the renderer or
the filesystem rejects also exits 1 after the SVG is written. `card` exits 1
when no stored run matches, when `--run` ids belong to more than one model, or
when the SVG cannot be written. It exits 2 for a `--scale` outside 0.25 to 8,
an unsupported `-o` suffix, or `SESSION` together with `--run`. It does not
accept `--json`. Run in a directory without `data/benchmarks.sqlite`, it says so
and creates nothing. The flat form is `--card [SESSION]` with `--card-run`,
`--card-hardware`, `--card-layout`, `--card-output`, `--card-png`, and
`--card-scale`.

A card never carries a hostname, server URL, session or run id, label,
tokenizer path, or system prompt, and its filename carries no session id. The
model name is stripped of any path. A model served from a local directory is
named by its served model id rather than the directory name. When merged
`--run` ids come from different tool or engine versions, `card` warns and the
card shows the newest.

## Understanding the JSON output

```jsonc
{
  "schema_version": "1",
  "tool_eval_bench_version": "1.8.0",
  "final_score": 85,           // 0–100, the headline number
  "rating": "★★★★ Good",       // star rating with label
  "safety_warnings": [],       // empty = safe; non-empty = failures in safety-critical scenarios (every trial with --trials)
  "deployability": 78,         // alpha × final_score + (1 − alpha) × responsiveness; null without latency data
  "responsiveness": 65,        // 0–100 from median turn latency; see methodology.md
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
{"event": "backend_detected", "backend": "vllm"}
{"event": "model_auto_selected", "model": "Qwen/Qwen3-8B", ...}
{"event": "probe_result", ...}                      // --probe only
{"event": "scenario_start", "scenario_id": "TC-01", "index": 0, "total": 15}
{"event": "scenario_result", "scenario_id": "TC-01", "status": "pass", "points": 2, ...}
{"event": "decision_audit_start", ...}              // with a decision judge
{"event": "decision_audit_result", ...}
{"event": "benchmark_complete", "json_file": "results.json", "final_score": 85}   // --json-file only
{"event": "safety_gate_failed", "safety_warnings": ["..."]}   // --fail-on-safety, before exit 2
{"event": "run_saved", "run_type": "perf", "run_id": "...", "status": "completed", "report_path": "runs/...", "session_id": "..."}
{"event": "log", "level": "warning", "logger": "tool_eval_bench...", "message": "..."}
{"event": "error", "error": "no_server", "message": "..."}
```

`run_saved` comes from the modes without a result envelope. Its `run_type` is
`perf`, `spec-bench`, `context-pressure`, or the plugin name (`gsm8k`, `mmlu`,
`ifeval`, `needle`, `decision`). Its `session_id` is the session `card` groups
by, and is null outside a CLI invocation. `log` carries warnings and errors from logging
and Python warnings, with URLs redacted. URLs in `error` messages are redacted
too.

## Error codes (structured)

When `--json` mode emits an error event, the `error` field is one of:

| Code | Exit | Meaning |
|------|------|---------|
| `connection_failed` | 2 | TCP connection refused, DNS failure, or timeout |
| `http_error` | 2 | Server responded with 4xx/5xx status |
| `detection_failed` | 2 | Unexpected exception during server probing |
| `invalid_response` | 2 | Response body is not valid JSON |
| `no_models` | 3 | Server responded but model list is empty |
| `model_not_available` | 3 | Model is listed but fails a real request |
| `no_server` | 2 | Auto-discovery found no server on localhost |
| `invalid_arguments` | 2 | The arguments parsed but are invalid, such as an unknown scenario or category, an out-of-range `--max-turns`, `--error-rate`, `--alpha`, or `--context-pressure`, malformed `--backend-kwargs`, or `--json` with an interactive monitor |
| `run_failed` | 1 | The run started but could not finish: a resume was rejected, setup failed, or a benchmark mode failed. The message names the cause. One exception: when `--spec-bench` fails and a tool-call run or plugin follows it, the event arrives mid-stream, the remaining modes still run, and the exit status and envelope reflect them |

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
4. **Timeout with thinking models** — models like Qwen3 with thinking enabled may need `--timeout` raised above the 120 s default for long reasoning traces.
5. **Warmup is automatic** — the first request primes the server. Use `--no-warmup` only if already warmed.
6. **Use `--dry-run` to preview** — before committing to a long run, check which scenarios would execute.

### Controlled scenario fixtures

`--variant-seed INTEGER` selects versioned deterministic scenario variants independently
of the sampling `--seed`. Unvaried scenarios remain controls. Variants are persisted in
run configuration and comparison fingerprints; resume requires the same variant seed
for affected scenarios. See [variant coverage](methodology.md#controlled-fixture-variants).
