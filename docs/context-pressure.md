# Context pressure

Tool-calling quality often degrades long before the context window is full. These flags pre-fill a configurable share of the window before each scenario, so you can find where a model starts to slip.


Tests tool-calling quality when the context window is already heavily utilized. This simulates real-world agentic conversations where the model must make accurate tool-call decisions with thousands of tokens of prior conversation history in its context.

```bash
# Fill 75% of context before each scenario (recommended)
tool-eval-bench run --seed 42 --context-pressure 0.75

# Fill 50% — moderate pressure
tool-eval-bench run --seed 42 --context-pressure 0.50

# Override auto-detected context size (if /v1/models doesn't expose it)
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
| `--sweep-steps` | 5 | Number of intervals for sweep (N+1 test levels) |

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

The context window size is auto-detected from model metadata when the backend
exposes a recognized context-length field, including vLLM's `max_model_len`.
If auto-detection fails, use `--context-size` to specify it manually.

The filler is designed to defeat server-side prefix caching (vLLM, llama.cpp):
- **Diverse content**: 12 distinct paragraph styles (tech docs, meeting notes, code reviews, incident reports, API docs, etc.)
- **Shuffled order**: paragraph order is randomized per run
- **Noise injection**: random ticket IDs, timestamps, IP addresses, and version strings are sprinkled throughout the text at sentence boundaries
- **Unique nonces**: each chunk gets a unique session/chunk identifier prefix
- **Per-scenario isolation**: each scenario gets a unique nonce injected into the filler to prevent cross-scenario prefix cache reuse

Unseeded filler uses nonces to reduce prefix-cache reuse. When `--seed` is set,
filler generation is deterministic per pressure level. The same seed, context
size, and sweep ratio can therefore reuse identical filler content, which makes
sweeps reproducible but changes the cache-busting guarantee.
