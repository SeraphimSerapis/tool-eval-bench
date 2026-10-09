# Context pressure

Tool-calling quality often degrades long before the context window is full. These flags pre-fill a configurable share of the window before each scenario, so you can find where a model starts to slip.


Tests tool-calling quality when the context window is already heavily utilized. This simulates real-world agentic conversations where the model must make accurate tool-call decisions with thousands of tokens of prior conversation history in its context.

```bash
# Fill 75% of context before each scenario (recommended)
tool-eval-bench run --seed 42 --context-pressure 0.75

# Fill 50% — moderate pressure
tool-eval-bench run --seed 42 --context-pressure 0.50

# Override the auto-detected context size
tool-eval-bench run --seed 42 --context-pressure 0.75 --context-size 32768

# Compare baseline vs pressure
tool-eval-bench run --seed 42                           # baseline run
tool-eval-bench run --seed 42 --context-pressure 0.75   # pressure run
tool-eval-bench compare <baseline_id> <pressure_id>
```

| Context Pressure Flag | Default | Purpose |
|---|---|---|
| `--context-pressure` | off | Fill ratio (0.0–1.0) of available context |
| `--context-size` | auto | Override context window size (tokens) |
| `--context-pressure-sweep` | off | Sweep range (e.g. `0.5-1.0`) — find the breaking point |
| `--sweep-steps` | 5 | Number of pressure levels to test (minimum 2) |

`tool-eval-bench resume RUN_ID` does not read the pressure settings back from
the stored run, so pass the same `--context-pressure`, and the same
`--context-size` if the original run had one. Resume refuses a different ratio,
an added or dropped `--context-pressure`, and a context size that changes the
fill target. Context drift that leaves the fill target alone, such as a
restarted server reporting a slightly different KV capacity, is accepted.

## Finding the breaking point

Use `--context-pressure-sweep` to gradually increase pressure and discover exactly where a model starts failing:

```bash
# Find breaking point between 90%–100% with fine granularity
tool-eval-bench bench --context-pressure-sweep 0.9-1.0 --sweep-steps 10 --scenarios TC-61 TC-64

# Broad sweep across the full range
tool-eval-bench bench --context-pressure-sweep 0.5-1.0 --scenarios TC-61

# Sweep a specific category
tool-eval-bench bench --context-pressure-sweep 0.5-1.0 --categories O
```

The sweep runs each selected scenario at every pressure level, displays a compact summary panel with pass/fail status per level, and reports the **breaking point** (highest pressure where all scenarios still pass). It early-stops after 2 consecutive all-fail levels.

Infrastructure failures are left out of the score, as they are in scored runs:
timeouts, connection errors, 5xx responses, and TC-45 on an endpoint that does
not enforce `tool_choice='required'`. They do not count against a level's pass
rate, the breaking point, the first degradation, or the all-fail early stop. A
level that fails as a whole, before any scenario is scored, counts the same way.
Each level reports how many scenarios it excluded. A level where every scenario
was excluded shows no pass rate, and two such levels in a row stop the sweep,
because the endpoint rather than the model has stopped answering.

An interrupted sweep (Ctrl-C) is still saved, but the stored run has
`interrupted: true`, records `planned_levels`, and withholds the breaking point,
since the levels it never reached could still have passed. The report says
after how many of the planned levels it stopped.

The sweep refuses to start when the context window cannot hold real pressure:
16,096 tokens are reserved for output and the scenario, and the fill at the top
of the range must be at least one 2,048-token filler chunk. On a smaller window
every level would run nearly unpressured and still report a breaking point. A
single `--context-pressure` run refuses a window that leaves no room for filler
at all.

`--scenario-pack` cannot be combined with `--context-pressure-sweep`. The sweep
report publishes every trace, which would burn a held-out pack. A single
`--context-pressure` run with a pack is a scored run and withholds pack traces
as usual.

The stored sweep config records the effective context size, after
`--context-size` and the KV-capacity cap, and the seed. Both change every
level's filler, so sweeps that differ in either are not grouped together.

The context window size is auto-detected. `--context-size` wins when given.
Otherwise the first of these that answers is used:

1. `/v1/models`: `max_model_len` (vLLM), `context_window`, or `max_tokens`.
2. TensorFold's `/health` `context_length`.
3. llama.cpp's or Strata's `/props` `default_generation_settings.n_ctx`, or
   Strata's `/health` `max_context`, as recorded by the engine probe in the run
   metadata. `--no-probe-engine` skips the probe, and with it this source.

That llama.cpp `n_ctx` is the per-request limit. Without `--kv-unified` it is
already the server's `n_ctx` divided by its `--parallel` slots. With
`--kv-unified`, which llama-server turns on when `--parallel` is left on auto,
every slot reports the whole shared pool, and one request can fill it only while
it runs alone. For high pressure on such a server, keep tool-eval-bench's
`--parallel` at 1 or pass a smaller `--context-size`.

Strata reports the context its engine announced at startup, the same value its
`/metrics` exports as `strata:engine_max_context`. That is also a per-request
limit: Strata refuses a request whose prompt and completion would not fit in it,
and every `--batch` slot reports the same value.

vLLM's KV cache capacity on `/metrics` then caps the result, except for hybrid
attention models. If auto-detection fails, use `--context-size` to specify it
manually.

The filler is designed to defeat server-side prefix caching (vLLM, llama.cpp):
- **Diverse content**: 12 distinct paragraph styles (tech docs, meeting notes, code reviews, incident reports, API docs, etc.)
- **Shuffled order**: paragraph order is randomized per run
- **Noise injection**: random ticket IDs, timestamps, IP addresses, and version strings are sprinkled throughout the text at sentence boundaries
- **Unique nonces**: each chunk gets a unique session/chunk identifier prefix
- **Per-scenario isolation**: each scenario gets a unique nonce injected into the filler to prevent cross-scenario prefix cache reuse

Filler is sized at about 4 characters per token, then calibrated against the
server's `/tokenize` endpoint (vLLM). On a server without a compatible
`/tokenize`, the fill stays at that estimate, which runs above the real count on
common tokenizers. A scored run then stores `fill_tokens_estimated: true` in its
`context_pressure` config, a sweep stores it on each affected level, and the
reports label the fill as estimated.

Unseeded filler uses nonces to reduce prefix-cache reuse. When `--seed` is set,
filler generation is deterministic per pressure level. The same seed, context
size, and sweep ratio can therefore reuse identical filler content, which makes
sweeps reproducible but changes the cache-busting guarantee.
