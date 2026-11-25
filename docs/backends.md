# Backends

OpenAI-compatible backends must expose `/v1/chat/completions` and support the
`tools` and `tool_choice` request fields the tool-call scenarios use. The
accuracy benchmarks (GSM8K, MMLU, IFEval, needle) need only chat completions.
The decision-model benchmark needs llama.cpp's `/v1/systemone` instead, and
needs no chat endpoint; see [decision-models.md](decision-models.md).

- **vLLM** — primary target
- **SGLang** — OpenAI-compatible model server
- **LiteLLM** — proxy for multiple backends
- **llama.cpp** — lightweight local inference
- **Strata** — OpenAI-compatible model server, identified by its declared health service or build information
- **TabbyAPI** — OpenAI-compatible server, identified by its model owner or `/.well-known/serviceinfo`
- **NInfer** — OpenAI-compatible inference engine, detected via `/v1/models`
- **TandemLLM** — OpenAI-compatible engine, identified by its model owner or `Server` header
- **TensorFold** serves MLX and CUDA models through the existing OpenAI adapter;
  detected by `owned_by: "tensorfold"` or the `tensorfold:` metrics namespace
- **Flare** serves models via [Hyperflare](https://github.com/NVIDIA/hyperflare-inference) through
  the existing OpenAI adapter, and natively serves `/v1/messages` with `--format anthropic`;
  detected by the `flare_` metrics namespace or the `flare` metadata object on `/v1/models`
- **Splash** serves GGUF models on Apple silicon ([incoai/splash](https://github.com/incoai/splash))
  through the existing OpenAI adapter, and also serves `/v1/messages`; detected by the `splash_`
  metrics namespace, `owned_by: "splash"` on `/v1/models`, or the `Server: Splash` header
- **Gemini** — supported through its native API as well as its OpenAI-compatible
  endpoint; the native wire format is detected from the URL, and `--format` pins
  it manually
- **Anthropic** — supported through the native Messages API (`/v1/messages`),
  which also covers gateways that front other models with it, such as OpenCode
  Zen; detected from the URL, or pinned with `--format anthropic`

## Backend identification

Detection runs in a fixed order and stops at the first answer. The engine facts
it relies on, such as namespaces, their precedence and the names servers declare,
live in one profile per engine in `src/tool_eval_bench/domain/engines.py`:

1. A native Prometheus namespace on `/metrics` (`halogen:`, `tensorfold:`, `strata:`, `flare_`,
   `splash_`, `vllm:`, `sglang:` or `sglang_`, then `llamacpp:`).
2. vLLM's `/version`.
3. The identity a server declares about itself: `owned_by` or a `flare` metadata object in `/v1/models`, an
   identifying `Server` header, `service` in `/health`, `software.name` in
   `/.well-known/serviceinfo`, or the product name at the start of `build_info` in
   `/props`. Names must match exactly, ignoring case: `tabbyAPI` identifies TabbyAPI
   and `tabby` identifies nothing.
4. llama.cpp's characteristic `/props` or `/health` fields: build information or
   number, or generation settings together with a slot count.

Step 3 comes before step 4 because Strata and TabbyAPI both serve a
llama-server-shaped `/props`. llama.cpp itself is identified in step 3 too, through
`owned_by: "llamacpp"` or `Server: llama.cpp` on current builds. A `/props` response
that names another server is never read as llama.cpp. Generic JSON from `/health`
or `/props` does not identify an engine. Ports locate a server, not its type.

Step 3 asks `/v1/models`, `/health`, `/.well-known/serviceinfo` and `/props` in that
order, and the first declaration wins. A server whose `owned_by` is `llamacpp` but
whose `/props` names Strata is labelled llama.cpp, with no llama.cpp build metadata,
because that `/props` is not a llama.cpp build.

An explicit `--backend` skips detection and is trusted, the same way
`--backend halogen` is: `--backend strata` or `--backend tabbyapi` against llama.cpp
reports that engine, and `--backend llamacpp` against a Strata that names itself
records no llama.cpp engine metadata.

Probes send `--api-key` as a bearer token, only to URLs on the `--base-url` origin,
and do not follow redirects. A missing key, a 401, or a malformed body skips that
endpoint rather than failing the run. Each probe waits at most 5 s in total, even for a
body that arrives a byte at a time. A refused connection
ends the remaining probes at once, and so do two timeouts in a row with no answer
between them, so a server that accepts connections but never replies costs at most
about 20 s of probing, 10 s for detection and 10 s for engine metadata. A single slow
endpoint, such as llama.cpp's `/metrics` while it is decoding, only skips itself.

Halogen Flash 0.16.2 [documents](https://github.com/peonist-ai/halogen-flash-server)
both `halogen:` and llama.cpp-compatible `llamacpp:` metrics. Its native namespace
wins regardless of scrape order. It uses the existing OpenAI-compatible adapter;
`--backend halogen` pins the reporting label when metrics are hidden. Version,
context size, and slot count are not inferred from these metric names.

When detection is inconclusive or disabled, the CLI records `unknown` rather than
assuming vLLM. The public Python API runs the same detection when `backend` is left
at its `unknown` default, and `probe_engine=False` disables it like `--no-probe-engine`.
`--backend`, `TOOL_EVAL_BACKEND`, and provider labels still override CLI detection, and
an explicit `backend=` overrides it in the API.
An unknown label does not change the request format; `--format` selects that
independently. `openai` identifies the hosted OpenAI API, not every compatible server.

New labels and discovered metadata affect comparison fingerprints for new runs.
Historical records are not rewritten. Unknown metadata is not proof that two
deployments match; keep known engine labels explicit when a proxy hides identity.

## How the adapter talks to a server

The adapter sends real `tools` and `tool_choice` in the request and parses
`tool_calls` out of the response. There is no prompt hacking and no JSON regex
matching.

It accepts SSE `data:` fields with or without the optional space, and parses a
normal JSON 200 response when an endpoint ignores `stream=true`. It defaults to
the widely supported `max_tokens` field; if an endpoint rejects that field and
asks for `max_completion_tokens`, the adapter retries once and remembers the
choice for that endpoint and model. This capability check is response-driven
rather than tied to provider or model names. SSE error events are recorded as
failures, not empty answers. Partial tool calls from a failed stream are discarded.

## Compatibility notes

| Behavior | vLLM | SGLang | LiteLLM | llama.cpp |
|---|---|---|---|---|
| `/v1/models` discovery | ✅ | ✅ | ✅ | ⚠️ May be at `/models` |
| `parallel_tool_calls` | ✅ | ✅ | ✅ | ❌ Not supported |
| Streaming `usage` stats | ✅ | Varies | Varies | ❌ |
| `tool_choice: "required"` | ✅ | ✅ | ✅ | ⚠️ Version-dependent |
| Large toolsets (52 tools) | ✅ | ✅ | ✅ | ⚠️ May exceed context window |
| `--spec-bench` acceptance rate | ✅ Per-request response metrics when enabled, else Prometheus | ⚠️ Live gauges are not request-local | ✅ when backend metrics are reachable | ✅ Counters or per-request timings |
| `--spec-live` dashboard | ✅ Counters | ✅ Gauges | ✅ when backend metrics are separately reachable | ✅ Counters on current builds; engine-only fallback |

OpenAI-compatible backends use `OpenAICompatibleAdapter`; native Gemini and the
Anthropic Messages API each use their own adapter. If you hit a backend-specific
issue, please
[open an issue](https://github.com/SeraphimSerapis/tool-eval-bench/issues).

## llama.cpp

The metadata probe reads llama-server's `/props`, sending `--api-key` when one is
given. It records `build_info` as the engine version, `total_slots` as the slot
count, and `default_generation_settings.n_ctx` as the context window. That
`n_ctx` is the per-slot context, so it is the longest single request the server
accepts, the same limit vLLM reports as `max_model_len`. Context pressure, the
pressure sweep, and the needle benchmark size themselves from it, so llama.cpp
needs `--context-size` only when the probe is off (`--no-probe-engine`) or the
server reports no usable `n_ctx`.

Quantization still comes from the model name when the name identifies a specific
type, such as `Q8_0` or `UD-Q4_K_XL`. The GGUF file type cannot make that
distinction: Unsloth's `UD-Q4_K_XL` and bartowski's `Q4_K_L` files both declare
`Q4_K_M`. When the name identifies none, as with an alias such as `gemma4` or a
filename that only says `GGUF`, the probe fills in `model_ftype`, the file type
the server loaded. llama-server reports it from build b9927 (July 2026). Its
names are normalized to the same labels: `Q4_K - Medium` becomes `Q4_K_M`,
`IQ2_XXS - 2.0625 bpw` becomes `IQ2_XXS`, and `F16` becomes `FP16`. A file type
marked `(guessed)`, an unknown one, or an older build without the field leaves
quantization unknown unless the name supplies it. A `/props` that names another
server, such as Strata, contributes nothing to the llama.cpp probe.

## Strata

```bash
tool-eval-bench run --short --backend strata --base-url http://127.0.0.1:8080 --seed 42
# Drop --short to run the standard suite. The /v1 form of the base URL also works.
```

[Strata](https://github.com/Niko1221/Strata) uses the existing OpenAI-compatible
adapter, including streaming tool calls. Automatic detection recognizes
`service: "strata"` in `/health` or `build_info: "Strata <version>"` in `/props`.
Strata adds `build_info` only when it knows its version, so `/health` identifies
builds that report none. Its llama.cpp-compatible generation settings alone do not
establish Strata identity. Strata serves Prometheus text from `/metrics` only when
asked, through `Accept: text/plain` or `?format=prometheus`, and JSON otherwise.
Every `/metrics` request the benchmark makes asks for text; a build that predates
the text format still answers with JSON, which is not treated as Prometheus data.
The text format repeats Strata's metrics under vLLM's `vllm:` names, so detection
checks its `strata:` namespace first and never labels it vLLM. Strata exports its
spec-decode counters even when it has drafted nothing, so spec decoding counts as
active only once Strata has offered draft tokens.

The metadata probe sends bearer authentication to `/props`, records the declared
engine version, effective context window and slot count, and leaves absent GPU
and speculative-decoding metadata unknown. When `/props` has no context window,
the probe reads `max_context` from `/health`. That window is Strata's
per-request limit, so context pressure, the pressure sweep, and the needle
benchmark size themselves from it and need `--context-size` only when the probe
is off (`--no-probe-engine`) or the engine has not finished starting. Quantization inferred from a model
name is a heuristic; a custom alias does not prove which checkpoint is loaded.
Tool-choice and structured-output capabilities remain response-probed, not assumed
from the backend label. Use `--no-think` when a comparison deliberately disables
reasoning, and keep that setting identical across models.

## TabbyAPI

```bash
tool-eval-bench run --short --base-url http://127.0.0.1:5000/v1 --api-key "$TABBY_API_KEY" --seed 42
# Pin the reporting label when a proxy hides identification endpoints:
tool-eval-bench run --backend tabbyapi --base-url http://127.0.0.1:5000/v1 --seed 42
```

[TabbyAPI](https://github.com/theroyallab/tabbyAPI) uses the existing
OpenAI-compatible adapter. Automatic detection recognizes `owned_by: "tabbyAPI"` in
`/v1/models` or `software.name: "TabbyAPI"` in `/.well-known/serviceinfo`. The
serviceinfo document needs no API key, so a keyed server is still identified when
`--api-key` is missing. Its llama-server-style `/props` is not read as llama.cpp.

`/props` answers once a model is loaded, with a key unless authentication is
disabled. The probe then reads the context window and slot count from it, which
TabbyAPI fills from the model's `max_seq_len` and `max_batch_size`. Otherwise both
stay unknown. TabbyAPI
reports no version for itself or for its model backend, so the engine version stays
empty rather than guessed. Quantization comes from the model name, for example
`EXL3` in `Qwen3-8B-exl3-4.0bpw`, and is a heuristic.

The recorded server model ID comes from `/v1/model`, which describes the model
loaded when the run starts and has the same key and loaded-model requirements as
`/props`. The `/v1/models` entry whose ID matches `--model` is only a fallback: with an admin key, or with
authentication disabled, that list is the whole model directory, and dummy model
aliases come first when they are enabled. With TabbyAPI's inline model loading
enabled, a request for a different model loads that model mid-run, and the recorded
ID and capacity still describe the one loaded at the start.

## TandemLLM

```bash
# The plain engine, without the conductor; use it for benchmark runs:
tool-eval-bench run --short --base-url http://127.0.0.1:8000/v1 --model TandemLLM --header X-Tandem=raw --seed 42
# The default path, with the conductor in front of the model, for comparison:
tool-eval-bench run --short --base-url http://127.0.0.1:8000/v1 --model TandemLLM --seed 42
# Pin the reporting label when a proxy hides identification endpoints:
tool-eval-bench run --backend tandemllm --base-url http://127.0.0.1:8000/v1 --model TandemLLM --header X-Tandem=raw --seed 42
```

[TandemLLM](https://github.com/0xBakeer/TandemLLM) serves one model, `TandemLLM`,
through the existing OpenAI-compatible adapter. Automatic detection recognizes
`owned_by: "tandemllm"` in `/v1/models` or a `Server` header that starts with
`TandemLLM`, for example `TandemLLM Python/3.11.16`. Relay backends configured on the
server appear in the same list under their own owners, such as `deepseek`, so pass
`--model TandemLLM` to skip the model prompt. The probe records the engine version
from `version` in `/health`, which answers without a key, for example
`0.11.4+exl3-6.0`. The model listing declares no top-level context window, so context
pressure and the needle benchmark need `--context-size`.

By default a [conductor](https://github.com/0xBakeer/TandemLLM/blob/main/docs/server.md#tandemllm-and-the-conductor)
runs in front of the model. It classifies each turn, adds its own notes to the
prompt, such as writing templates and memory notes, and sets the thinking effort.
That shapes answers and can turn a tool task into a written draft. The
`X-Tandem: raw` header, sent with `--header X-Tandem=raw`, serves the plain path:
no turn class, no notes, no effort change. Use it for benchmark runs. Run the
default path, without the header, only when the conductor itself is under test; the
two runs then differ in the conductor alone, and its decision journal on the server
shows what it added to each turn.

Thinking is on by default and the model decides its length. `--no-think` sends
`enable_thinking: false`, which TandemLLM honours as thinking off. An effort word,
`--backend-kwargs '{"reasoning_effort": "low"}'` from `low` to `xhigh`, caps the
reasoning, and a token count in `{"reasoning": {"max_tokens": N}}` wins over a word.
The accepted fields and their precedence are listed under
[Thinking budgets from the client](https://github.com/0xBakeer/TandemLLM/blob/main/docs/server.md#thinking-budgets-from-the-client).

The first request of each scenario pays the full prefill of the tool prompt, about
4,400 tokens; on a DGX Spark the first token arrives after about 7 s, and that turn's
latency counts toward responsiveness. Later turns of the same scenario reuse the
prefix.

`/metrics` needs a bearer token and uses TandemLLM's own metric names, none of which
the speculative-decoding detection reads, so spec-bench and spec-live report no
speculative-decoding counters.

## TensorFold

```bash
tool-eval-bench run --short --base-url http://127.0.0.1:8080/v1 --seed 42
# Pin the reporting label when a proxy hides identification endpoints:
tool-eval-bench run --backend tensorfold --base-url http://127.0.0.1:8080/v1 --seed 42
```

Both model discovery and engine probing recognize TensorFold without relying
on a port or Python HTTP server header. CUDA health supplies the effective
context window and, under concurrent serving, the maximum stream count.
The maximum MLX batch width is not interpreted as a server slot count.
The discovered context window participates in the comparison fingerprint.

Current upstream model listings do not expose a version, original checkpoint
behind an alias, checkpoint revision, or deployment drafting configuration.
Those remain unknown. Quantization inferred from checkpoint names remains a
heuristic; aliases cannot recover it. Use the original model ID where possible
and record deployment details with `--label`. Labels annotate reports but do
not change comparison cohorts. Do not treat otherwise matching fingerprints
as proof that unknown deployment settings match.

CUDA context-pressure and needle runs can discover their window through
`/health`. MLX currently requires `--context-size`. If health is inaccessible,
pass the effective serving window explicitly, not the architecture maximum.

Structured-output requests require TensorFold's grammar extra on the server:
`pip install 'tensorfold[grammar]'`. Tool-choice capabilities are still probed,
not assumed from the engine name. Protocol fixtures cover MLX and CUDA; live
model-family qualification is not implied by these tests.

Spec-bench reads request-local draft counts from MLX's `speculative` object or
CUDA's `tensorfold` statistics before falling back to Prometheus deltas.
Spec-live reads the `tensorfold:` counter aliases and engine metrics once,
without adding their duplicate native counters. Missing speculative-step
counts leave acceptance length and draft-window estimates unavailable;
engine `rounds` are not substituted. The `mtp_*` names do not identify the
proposer. Pass `--spec-method` when you know the configured method.

For a speedup comparison, measure an otherwise identical serial run using
`draft: false` where the engine supports it, then supply its measured generation
rate through `--baseline-tgs`. Keep checkpoint, backend, prompt, seed, and
sampling settings fixed. At nonzero temperature, independent repeats require
different seeds; without a seed TensorFold derives one from the prompt.

## Flare

```bash
# OpenAI-compatible endpoint (default format; http://127.0.0.1:8000 also works):
tool-eval-bench run --short --base-url http://127.0.0.1:8000/v1 --seed 42
# Native Anthropic Messages API on the same server:
tool-eval-bench run --short --base-url http://127.0.0.1:8000 --format anthropic --seed 42
# Pin the reporting label when a proxy hides identification endpoints (`hyperflare` is an alias):
tool-eval-bench run --backend flare --base-url http://127.0.0.1:8000/v1 --seed 42
```

[Flare](https://github.com/NVIDIA/hyperflare-inference) (Hyperflare) listens on
`http://127.0.0.1:8000` by default and runs through the existing OpenAI-compatible
adapter (`--format openai`, the default). The same server also exposes the native
Anthropic Messages API at `/v1/messages`, selectable with `--format anthropic`.

Automatic detection checks `/metrics` first for the `flare_` Prometheus namespace.
Flare serves `/metrics` unauthenticated as Prometheus text 0.0.4 by default. When
`/metrics` is hidden or proxied, detection falls back to the `flare` metadata object
on `/v1/models` cards; `owned_by: "local"` alone is never treated as Flare. Pass
`--backend flare` (or its alias `--backend hyperflare`) to pin the reporting label
explicitly.

The metadata probe reads `max_model_len` (Flare's effective per-request prompt plus
output token limit) and `flare.max_concurrency` (recorded as `slot_count`) from
`/v1/models`. Flare does not expose a server version over HTTP, so `engine_version`
stays unknown. Quantization inferred from the model name remains a heuristic; use
`--label` to record deployment details such as core build or quantization. Because
`max_model_len` is the single-request context limit, context pressure, the pressure
sweep, and the needle benchmark pick it up directly from `/v1/models` (and also fall
back to the value recorded by the engine probe).

For `--spec-bench` and `spec-live`, Flare exposes Prometheus speculative counters
(`flare_accepted_draft_tokens_total`, `flare_verification_draft_tokens_total`, and
`flare_draft_request_rounds_total`) and per-depth draft outcomes
(`flare_speculative_drafts_total`) covering both MTP learned drafters and
shared-history copy drafts. Flare renders those counters even when no drafting takes
place, so speculative decoding counts as active only once draft tokens are offered,
and the proposer method stays `unknown` unless `--spec-method` is passed. `spec-live`
also reads Flare's prefill and generation token counters (`flare_prefill_tokens_total`,
`flare_generated_tokens_total`), KV page utilization (`flare_total_kv_pages` and
`flare_free_kv_pages`), prefix cache lookups and hits (`flare_cache_lookups_total`,
`flare_cache_hits_total`), active and pending request gauges (`flare_active_requests`,
`flare_pending_requests`), and per-position acceptance rates across MTP and copy
drafts.

Practical serving notes:

- **Sampling clamping.** Flare clamps `temperature`, `top_p`, `top_k`, `min_p`, and
  penalty parameters to the loaded core's `sampling_ranges`, or fixes them to
  `sampling_defaults` when a range is null. Passing `--temperature 0` clamps to the
  core's minimum or default rather than forcing greedy decoding when the core
  disallows 0. `seed` remains request-controlled.
- **Prefix caching.** GPU prefix caching (and optional host caching) is always on,
  with no per-request bypass flag or reset endpoint. `llama-benchy` (used by `--perf`
  when installed) draws fresh random slices per request to defeat prefix caching,
  whereas the built-in fallback sweep (`runner/throughput.py`) reuses deterministic
  filler. Use `llama-benchy` for cold-prefill sweeps, or verify cold prefill via
  `usage.prompt_tokens_details.cached_tokens == 0`.
- **Tools and reasoning.** Standard tool calling (`tools`, `tool_choice`,
  `parallel_tool_calls`) and reasoning (`reasoning_content`) are supported natively.
  Flare rejects `strict: true` in tool schemas with HTTP 400.

## Splash

```bash
# OpenAI-compatible endpoint (default port 8000 is already scanned):
tool-eval-bench run --short --base-url http://127.0.0.1:8000 --seed 42
# The Messages API on the same server; --backend keeps the Splash label and probe:
tool-eval-bench run --short --base-url http://127.0.0.1:8000 --backend splash --format anthropic --seed 42
```

[Splash](https://github.com/incoai/splash) listens on `http://127.0.0.1:8000` by
default and runs through the existing OpenAI-compatible adapter. Instances on ports
outside the discovery list need `--base-url`. So does an instance with an API key,
together with `--api-key`: localhost discovery asks `/v1/models` without a key, and Splash refuses
that request when it has one. The same server also serves `/v1/messages`. With
`--format anthropic` and no `--backend`, a run is labelled `anthropic`, as for any
Messages API endpoint, and skips the engine probe; pass `--backend splash` to keep
the Splash label and metadata.

Automatic detection finds the `splash_` Prometheus namespace on `/metrics`. When
`/metrics` is hidden, `owned_by: "splash"` on `/v1/models` or the `Server: Splash`
header identifies it. Splash requires its API key for `/metrics`, `/status`, and
`/v1/models` when one is configured, so pass `--api-key` to such a server.

The metadata probe reads `max_model_len` from `/v1/models`, which is the limit
Splash enforces on one request's prompt plus output, and
`frontend.preparation_capacity` from `/status`, which is the server's
`--max-concurrent-requests` (recorded as `slot_count`). Splash does not report its
version over HTTP (its `Server` header is the bare token `Splash`), so
`engine_version` stays unknown. Quantization inferred from the model name remains
a heuristic.

For `--spec-bench`, each Splash response that drafted carries
`metrics.speculation` (`drafted_tokens`, `accepted_draft_tokens`, and `cycles`),
which gives an exact α, τ, and draft window per request. Spec-bench shows no Steps/s
or speedup ceiling for Splash, because the cycles also cover Splash's first emission,
which comes before the interval Steps/s divides by. A response that drafted
nothing omits that object and `timings.draft_n`. Spec-bench records such a response
as zero drafts when it carries a `timings` object and the server is identified as
Splash, by `/metrics` or by the response's own `Server: Splash` header; a response
without `timings` proves nothing. Splash's Prometheus counters
(`splash_drafted_tokens_total`, `splash_accepted_draft_tokens_total`) appear on
every scrape that carries the engine's status, zeros included, so speculative
decoding counts as active only once drafted tokens are above zero. They carry no
cycle count, so `spec-live` shows α, waste, accepted and drafted t/s, but no τ or
draft window. `spec-live` also reads the prefill and decode token counters
(`splash_prefill_input_tokens_total`, which leaves out prefix-cached prompt
tokens, and `splash_decode_output_tokens_total`) and the scheduler's request
counts by phase for running and waiting requests. It maps no KV cache or prefix
cache gauge for Splash, so those two rows read 0%. Splash names no proposer, so
the method stays `unknown` unless `--spec-method` is passed.

Practical serving notes:

- **Thinking.** `--no-think` sends `chat_template_kwargs.enable_thinking=false`.
  `/status` reports as `reasoning_switch` whether a request can turn the served
  model's reasoning on and off. Of the two instances this support was tested
  against, a Ternary Bonsai 2 model reporting `true` stopped reasoning under
  `--no-think`, and an Inkling 2 Mini model reporting `false` kept reasoning.
- **Prefix caching.** Splash reuses cached prompt prefixes across requests; a
  repeated request reports them in `usage.prompt_tokens_details.cached_tokens`. A
  request shares cached prompts only with requests that send the same
  `cache_salt`. Verify cold prefill in throughput sweeps through `cached_tokens`.
- **Token calibration.** Splash's `/tokenize` reads llama.cpp's `content` field
  and answers HTTP 400 to the `prompt` field the benchmark sends, so throughput
  and spec-bench calibrate the characters-per-token ratio with probe requests.

## LiteLLM and other model routers

LiteLLM and similar routers expose several models behind one endpoint:

1. **Auto-detection** — when `/v1/models` returns multiple models, the CLI shows
   an interactive picker.
2. **Explicit selection** — `--model <alias>` skips the picker.
3. **Multi-model comparison** — run one invocation per model, then compare:

```bash
tool-eval-bench run --model gpt-4o --base-url http://litellm:4000
tool-eval-bench run --model claude-3.5-sonnet --base-url http://litellm:4000
tool-eval-bench compare <run_id_a> <run_id_b>

# Or a browser report from two Markdown artifacts
tool-eval-bench compare --report runs/.../model_a_summary.md runs/.../model_b_summary.md \
  -o comparison.html
```

Set `TOOL_EVAL_BACKEND=litellm` in `.env` so reports carry the right label.

## Hosted Gemini

```bash
tool-eval-bench run --model gemini-3-flash --api-key "$GEMINI_API_KEY" \
  --base-url https://generativelanguage.googleapis.com
```

The native API is detected from the URL. Engine probing (`/metrics`, `/props`,
`/version`) is skipped for hosted APIs, since it would be meaningless and would
put a false backend label on every report.

## Anthropic Messages API

```bash
tool-eval-bench run --model claude-opus-5 --api-key "$ANTHROPIC_API_KEY" \
  --base-url https://api.anthropic.com

# A gateway that serves the Messages API for other models
tool-eval-bench run --model qwen3-coder --api-key "$OPENCODE_API_KEY" \
  --base-url https://opencode.ai/zen/go/v1/messages
```

OpenCode Go routes by conversation and refuses a request without a stable
`x-opencode-session`; it also asks clients to identify themselves. Both are
endpoint configuration, so they sit next to the endpoint:

```bash
TOOL_EVAL_ZEN_BASE_URL=https://opencode.ai/zen/go/v1/messages
TOOL_EVAL_ZEN_API_KEY=...
TOOL_EVAL_ZEN_HEADERS=User-Agent=my-coding-agent/1.0
TOOL_EVAL_ZEN_SESSION_HEADER=x-opencode-session
```

```bash
tool-eval-bench run --provider zen --model union-alpha
```

Each scenario is one conversation: every turn of it sends the same id, and the
next scenario gets a new one. Single-shot requests (pre-flight, warm-up, the
accuracy plugins) each get their own. The same two settings work on any
gateway with the same shape, through `--header` and `--session-header` on the
command line or the generic `TOOL_EVAL_HEADERS` and `TOOL_EVAL_SESSION_HEADER`.

`api.anthropic.com` and any base URL whose path ends in `/messages` select the
native format; a gateway root that serves several formats side by side needs
`--format anthropic`. The adapter sends both `x-api-key` and a bearer token, so
gateways that read either header work without configuration.

What the translation does and does not carry:

- Tool definitions, `tool_choice`, and `parallel_tool_calls=false` map onto
  their Messages API equivalents. A `json_schema` response format becomes
  `output_config.format`, minus the numeric and string constraints the API
  rejects (the evaluators still check them).
- Thinking blocks and their signatures are replayed verbatim on the next
  turn, as the API requires when a turn also carried a tool call. Thinking is
  reported as reasoning; its visibility follows the model's default unless
  `--backend-kwargs` sets `thinking` explicitly. `--no-think` maps onto
  `thinking: {"type": "disabled"}`, which some models reject.
- Current Claude models reject `temperature`, `top_p`, and `top_k` with HTTP
  400. The adapter drops them on that response and remembers the choice for the
  endpoint, so the benchmark's `temperature=0` costs one extra request per run
  rather than failing it.
- Throughput sweeps (`--perf`, `--perf-only`) and engine metrics remain OpenAI-only.
