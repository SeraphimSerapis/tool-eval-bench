# Troubleshooting

Failure modes that come up often, and what each one actually means. For the full flag list and
JSON output shape, see [cli-reference.md](cli-reference.md).

## Start by isolating the endpoint

`probe` checks readiness without spending a benchmark run:

```bash
tool-eval-bench probe
tool-eval-bench probe --base-url http://localhost:8000
tool-eval-bench probe --check-tools --base-url http://localhost:8000
tool-eval-bench probe --check-tools --check-template --tokenizer /path/to/tokenizer.json --base-url http://localhost:8000
```

Without `--check-tools` or `--check-template`, `probe` checks only the model
listing and sends no chat-completion requests. Add `--check-tools` to send one
tool-calling request and verify structured `tool_calls` parsing (printing
server flag advice if raw markup leaks into `content`). Add `--check-template`
to run four paired chat-template fidelity and special-token leak checks
(`tool_replay` and `multi_turn`, each paired with a single-turn twin); pass
`--tokenizer` to override local HuggingFace cache lookup for special tokens.

Its exit code tells you where the problem is:

| Exit | Meaning | Usual cause |
|---|---|---|
| 0 | Ready | Endpoint reached, and any requested `--check-tools` / `--check-template` checks passed |
| 1 | Server not ready or probe check failed | Nothing listening, model still loading, HTTP error on model listing, or `--check-tools` / `--check-template` failed |
| 2 | Invalid response, no server found, or invalid arguments | Endpoint answered with non-JSON (`invalid_response`, such as an HTML page or proxy redirect), auto-discovery found nothing on localhost, or flags are invalid |

A scored run exits 2 for connection and HTTP errors and 3 when the server lists no models.

## The server is running but nothing is discovered

Auto-discovery scans common localhost ports for a JSON model-list response on
`/v1/models`, falling back to `/models` on HTTP 404 (implemented in
`cli/server.py`, covering llama.cpp builds that expose `/models`). It finds
vLLM, llama.cpp, SGLang, LiteLLM, Ollama, and TGI on their defaults, and
nothing else. Pass `--base-url` explicitly for a non-standard port, a remote
host, or anything behind a proxy.

## Exit 3, or the wrong model gets benchmarked

The model is auto-detected from the server's model list. When the server has several models loaded,
name the one you mean:

```bash
tool-eval-bench run --model Qwen/Qwen3-8B --base-url http://localhost:8000
```

## The run stops at the pre-flight gate

Some endpoints have provider-specific startup behavior that the strict gate reads as not ready.
When you know the endpoint is good, skip it:

```bash
tool-eval-bench run --no-preflight
```

## Timeouts on thinking models

A timed-out scenario shows as `⏱  TIMEOUT` with `–/2` points, not as a failure. It is excluded from
both the numerator and the denominator, so the run says nothing about the model on that scenario.
Check `completion_rate` before comparing the score to anything.

The run prints what to change, computed from the slowest turn it actually measured. Take that
suggestion first.

Reasoning models routinely exceed the 120-second default. Raise it:

```bash
tool-eval-bench run --timeout 600
```

Or shorten what the model generates, which fixes the cause rather than the symptom:

```bash
tool-eval-bench run --no-think                                    # reasoning off entirely
tool-eval-bench run --backend-kwargs '{"reasoning_effort": "low"}'  # if your server honours it
tool-eval-bench run --backend-kwargs '{"max_tokens": 1024}'         # hard cap on generation
```

`--no-think` changes both the latency and the tool-calling behavior you are measuring, so it is a
diagnostic rather than a setting to keep. TC-88 scores only visible values and records
`reasoning_transport` separately (`observed`, `unconfirmed`, or `unavailable`), so disabling
reasoning does not automatically cap TC-88, though models that rely on thinking may still miss the
visible constraints.

### What the timeout bounds

Every turn of a scenario is streamed, so the read timeout measures the gap between tokens, not the
length of the generation. A model that thinks for six minutes while still emitting tokens stays
alive; an endpoint that goes silent for `--timeout` seconds is treated as hung and the scenario is
excluded as an infrastructure failure. Raise `--timeout` only when a healthy endpoint pauses that
long between tokens, which is rare outside heavily loaded shared servers.

### A turn that hit the token ceiling

Each turn is sent with `max_tokens` 16384 when thinking is enabled and 4096 under `--no-think`;
the value is printed under Run Context. On OpenAI-compatible reasoning endpoints the thinking
tokens count against that ceiling. When a turn ends on `finish_reason=length` with no visible
answer and no tool call, the scenario stops there, the trace carries a `truncated=` line, the
evaluation note says how much reasoning was cut, and the failure kind is `reasoning_truncated`.
The same applies when the ceiling cuts a tool call off mid-arguments, so its arguments are not
valid JSON: the partial call stays in the trace but never runs. A call whose arguments finished
before the ceiling runs normally.
That result still scores, because the model did not answer, but the tag separates "ran out of room
to think" from a wrong answer. `--backend-kwargs '{"max_tokens": 32768}'` raises the ceiling.

### A turn that came back empty

A turn can also end normally, with `finish_reason=stop`, and still carry no visible text and no
tool call. Reasoning models do this when they think through the next step and then stop without
writing it. That reply closes the phase like any other, so the scenario is graded on it, but the
trace carries an `empty_turn=` line with the finish reason and the reasoning length, the evaluation
note repeats it, and a failing result gets the failure kind `empty_turn`. A passing result keeps the
trace line and no failure kind. Many `empty_turn` results on one model usually point at the chat
template or reasoning parser dropping content, so check those before reading it as a judgement
failure.

### Raw tool markup leaked into `content` (`unparsed_tool_call`)

When a server runs without its chat-template tool parser, the model emits native
tool-call tags or JSON call objects inside `message.content` while `tool_calls`
stays empty. The orchestrator detects raw JSON calls and parser-family markers
including Hermes/Qwen (`<tool_call>`, `<|tool_call|>`), Qwen3-Coder
(`<function=...>`), Mistral (`[TOOL_CALLS]`), Llama 3 (`<|python_tag|>`),
DeepSeek (`<|tool_calls_begin|>`), and Gemma 4 (`<|tool_call>`, `<tool_call|>`,
`<channel|>`, `<turn|>`, `call:<name>{...}`), records
`failure_kind="unparsed_tool_call"`, and suggests the matching server flag, such
as `--enable-auto-tool-choice --tool-call-parser gemma4` on vLLM,
`--tool-call-parser gemma4` on SGLang, or `--jinja` on llama.cpp.

Unlike transport errors, `unparsed_tool_call` results stay in the score
denominator as 0 points so a missing server parser cannot shrink the graded set.
Run `tool-eval-bench probe --check-tools --check-template` after updating your
server flags to verify clean tool parsing before re-running the suite.

### Degenerate repetition loops (`degenerate_loop`)

When a model gets trapped repeating a 20-word phrase at least 3 times, an
identical character run (120+ ordinary/whitespace characters, or 200+ rule/border
characters), or a 2 to 4 character pattern 40+ times consecutively until hitting
`max_tokens`, the orchestrator classifies the failure as `degenerate_loop`
(`domain/loop_detection.py`) rather than `reasoning_truncated`. Raising
`max_tokens` will not fix a degenerate loop; it only burns more tokens. Check
quantization, sampling parameters (`--temperature`, `--repeat-penalty`,
`--top-p`, `--min-p`), and the server's chat template instead.

### Answer trapped in thinking (`answer_in_thinking`)

A reasoning model can write its intended answer inside `reasoning` while
returning empty visible `content`, or hit `finish_reason=length` with reasoning
but no visible answer. The orchestrator counts these turns in
`answer_in_thinking_turns` and `answer_in_thinking_scenarios` and records an
`answer_in_thinking` diagnostic on the scenario. Frequent occurrences point to
a reasoning parser mismatch, a chat template that fails to close the thinking
block, or a token budget too small for the model's reasoning style.

### Score shows as `Withheld`

When every scenario in a run is excluded by infrastructure failures (`timeout`,
`connection_error`, or `server_error`), `max_points` is 0. Rather than reporting
a misleading 0 or 100, `tool-eval-bench` records `final_score: null` and
`rating: "Withheld"`, excludes the run from leaderboard rankings, and emits a
comparability warning. Inspect the per-scenario failure kinds and fix the
underlying server reachability, timeout, or 429/5xx issue before re-running.

## Rate limits against a hosted endpoint

Hosted APIs return 429s under sustained load. The runner already shares backoff across scenarios
and paces requests adaptively rather than retrying blindly, so the usual fix is patience rather
than configuration.

What matters is how the result is reported. A scenario lost to a persistent 429 is dropped from
both the numerator and the denominator instead of being scored zero, because it measures the
serving environment rather than the model. Check `completion_rate` before comparing that run to
anything else.

## A score moved and you do not know why

Check both `config_fingerprint` and the leaderboard cohort. Per-model
`config_fingerprint` covers evaluation and operational flags, exact code
identity (`tool_version` and `git_sha`), and discovered deployment facts
(engine, quantization, context window, slots, speculative decoding). Repeat
runs of one model collapse into a single leaderboard row only when the full
`config_fingerprint` matches, and `history` shows both runs' fingerprints.

Cross-model leaderboard cohorts use rubric versioning: modern runs group on
the shared evaluation config (excluding model identity, deployment facts,
`concurrency`, `timeout_seconds`, and `alpha`) plus `eval_rubric_version`,
`locale_version`, and `noise_version` from `domain/rubric.py`. Unrelated CLI or
UI commits stay in the same cohort until a rubric version bumps; older rows
without `eval_rubric_version` fall back to `tool_version` plus `git_sha`.
Deployment facts stay out of cohorts on purpose because different models on one
box often run at different quantizations or context windows; compare the engine
columns in `export` yourself.

Conditions that affect comparability without splitting a cohort appear in
`comparability_warnings` (in `leaderboard`, `compare`, and the Web Compare UI),
including dirty git trees (`-dirty`), completion rates below 100%, withheld
scores, subset limits, unprobed serving engines, `response_format` fallbacks,
and cross-run rubric or tool version mismatches.

Also check `completion_rate` on both runs. A run graded on 60 of 69 scenarios is
not comparable to one graded on all 69, however similar the headline scores look.

## Backend-specific behavior

Some capabilities are not uniform across serving stacks, and the benchmark degrades rather than
failing when one is missing:

| Behavior | vLLM | SGLang | LiteLLM | llama.cpp |
|---|---|---|---|---|
| `/v1/models` discovery | Yes | Yes | Yes | May be at `/models` |
| `parallel_tool_calls` | Yes | Yes | Yes | Not supported |
| Streaming `usage` stats | Yes | Varies | Varies | No |
| `tool_choice: "required"` | Yes | Yes | Yes | Version-dependent |
| Large toolsets (52 tools) | Yes | Yes | Yes | May exceed the context window |

Category L runs 52 tools at once, so on llama.cpp it can exceed the context window on smaller
builds. That shows up as failures concentrated in one category rather than a uniformly lower score.

## Parsing the JSON output in CI

`--json` writes JSONL to stderr, one object per line, not a single document. Parse it line by line,
or use `--json-file PATH` to get the final result as one file.

To fail a CI job on safety-critical failures specifically, `--fail-on-safety` exits 2. Under `--json`
it also writes a `safety_gate_failed` event to stderr listing the warnings. With `--trials`, every
trial runs first and the gate checks the warnings from all of them.

## `card` writes the SVG but no PNG

PNG export uses the optional `resvg-py` renderer. Without it, `card --png` (or
`-o card.png`) still writes the SVG, prints the install command, and exits 1.
Install the `card` extra the same way you installed the tool:

```bash
uv tool install 'tool-eval-bench[card] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'
```

The SVG alone is enough for most uses: browsers, GitHub READMEs, and slide
tools display it directly. If `card` reports that no benchmark runs are
stored, run it from the directory that holds `data/benchmarks.sqlite`.

## Still stuck

Backend-specific issues are worth reporting, since the adapter layer is where most of them belong:
[open an issue](https://github.com/SeraphimSerapis/tool-eval-bench/issues).
