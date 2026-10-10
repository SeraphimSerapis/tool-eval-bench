# Accuracy and throughput benchmarks

Beyond tool-calling quality, `tool-eval-bench` runs external accuracy benchmarks through the same adapter layer, and measures prefill and generation speed against the same endpoint. Neither needs `tools` support from the server.

## Accuracy benchmarks (GSM8K, MMLU, IFEval)


Pluggable accuracy benchmarks evaluate model knowledge and instruction-following capabilities. Datasets are downloaded automatically from HuggingFace on first use and cached locally under `data/`.

**Recommended:** Install the `datasets` library for fast, rate-limit-free downloads directly from the HuggingFace git repo:

```bash
uv tool install 'tool-eval-bench[hf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'

# or, in a checkout
pip install -e '.[hf]'
```

Without it, the tool falls back to the HuggingFace REST API (which has rate limits and may fail with HTTP 429 on large datasets like MMLU). Downloads are resumable either way — if interrupted, re-running picks up where it stopped.

```bash
# GSM8K — math reasoning
tool-eval-bench plugin gsm8k                         # 200 questions, 8-shot
tool-eval-bench plugin gsm8k --limit 50              # quick test

# MMLU — multitask knowledge
tool-eval-bench plugin mmlu                          # 500 questions, 5-shot
tool-eval-bench plugin mmlu --limit 50               # quick test
tool-eval-bench plugin mmlu --subjects STEM          # only STEM subjects
tool-eval-bench plugin mmlu --shots 0                # zero-shot

# IFEval — instruction following
tool-eval-bench plugin ifeval                        # all 541 prompts
tool-eval-bench plugin ifeval --limit 20             # quick test

# Combined with tool-eval
tool-eval-bench bench --mmlu --ifeval --gsm8k        # all three after tool-eval
```

| Flag | Default | Purpose |
|---|---|---|
| `--gsm8k` / `--gsm8k-only` | off | Run GSM8K benchmark |
| `--gsm8k-shots` | 8 | Few-shot examples (0–8) |
| `--gsm8k-limit` | 200 | Max questions (0 = all 1,319) |
| `--gsm8k-shuffle` | off | Shuffle question order (uses `--seed`; without one, draws a seed and records it as `shuffle_seed`) |
| `--mmlu` / `--mmlu-only` | off | Run MMLU benchmark |
| `--mmlu-shots` | 5 | Few-shot examples per subject (0–5) |
| `--mmlu-limit` | 500 | Max questions, sampled proportionally across subjects (0 = all 14,042) |
| `--mmlu-subjects` | all | Comma-separated subjects or categories (e.g. `STEM,philosophy`); names match in any case, unknown names fail the run |
| `--ifeval` / `--ifeval-only` | off | Run IFEval benchmark |
| `--ifeval-limit` | 0 (all) | Max prompts (0 = all 541) |

How each benchmark extracts and grades answers, how truncated answers and
rejected requests count, and how dataset revisions are pinned is covered in
[methodology.md](methodology.md#accuracy-benchmarks-pluggable).

## Throughput benchmark

Throughput measurement uses [llama-benchy](https://github.com/eugr/llama-benchy) — a dedicated benchmarking tool that provides multi-run statistics with mean ± std, proper latency estimation, and cache-busting. Install with `uv tool install 'tool-eval-bench[perf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'` (or `pip install -e '.[perf]'` in a checkout) or ensure `uvx` is on PATH. Progress is shown via a live Rich progress bar. For authenticated endpoints, the regular `--api-key` value is forwarded to llama-benchy's supported CLI option and redacted from logs. Because llama-benchy 0.4.x does not support environment-based credentials, the key may still be visible to process inspection by other users on the same host while the benchmark is running.

An asterisk on a prefill rate means the first response chunk arrived before the first content token. Older llama-benchy versions can count that early chunk as the end of prefill, so the displayed rate is estimated from end-to-end time to first content token. The numerator follows the benchmark phase: prompt plus depth for a standard run, depth for context load, and prompt for a prefix-cached follow-up. This estimate includes network and queue time; compare it with other estimated rates rather than directly measured prefill rates.

llama-benchy always runs its own warm-up: a few global requests and a discarded first run of every test point, as llama-bench does, unless you pass `--no-warmup` through `--benchy-args`. `--no-warmup` skips only the tool-eval-bench warm-up request.

A test point where any request failed is reported as failed, even when the other requests succeeded. llama-benchy computes a point from the requests that survived, so a c2 row that lost one request would otherwise show one request's throughput as the batch total, and a c1 row would average fewer runs than requested. With `--perf-only`, any failed point makes the run exit 1.

The Tokens column shows the prompt size plus the mean number of tokens the model actually generated. For a `ctx pp{depth}` context-load row, the prompt size is the depth, because that request sends the depth-token context, not the pp-token prompt. A model that stops before `--tg` generates fewer, and Total (ms) is computed from that observed count. Total is time to first content token plus generation time, so it is never below TTFT. With `--enable-prefix-caching`, llama-benchy's progress events do not say which phase a request belongs to, so the context-load and inference rows of a test point both show the mean over the requests of both phases.

With `--enable-prefix-caching`, the context-load row is labelled `ctx pp{depth}`, and tool-eval-bench adds `--extra-body cache_prompt=true` so llama.cpp reuses the cached prefix on the follow-up request. Without it, the always-on `--no-cache` would send `cache_prompt: false` and llama.cpp would prefill the whole prompt again. A `cache_prompt` value passed in `--benchy-args` takes precedence. Progress events do not say which phase a request belonged to, so a failed request in either phase marks both rows of that test point as failed.

`--perf` combined with `--skip-tool-eval`, `--context-pressure-sweep`, `--spec-bench --skip-tool-eval`, or a plugin-only run such as `--gsm8k-only` saves the throughput sweep as its own run, exactly as `--perf-only` would. The run is saved as soon as the sweep finishes, before the other mode starts, so a failure there cannot lose it. A failed throughput cell still exits 1, after the other mode has run.

```bash
# Throughput only (skip tool-call scenarios)
tool-eval-bench bench --perf-only --pp 2048 --tg 128 --depth "0 4096 8192 16384 32768"

# Throughput + tool-call scenarios
tool-eval-bench bench --perf --depth "0 4096" --concurrency "1,2,4"

# Long-context sweep: concurrency at depth 0, then cached-prefix depths to 128K
tool-eval-bench bench --perf-only --perf-preset long

# Customize measurement runs and latency mode
tool-eval-bench bench --perf --benchy-runs 5 --benchy-latency-mode generation

# Pass arbitrary flags to llama-benchy
tool-eval-bench bench --perf --benchy-args='--enable-prefix-caching'

# Override the auto-detected tokenizer
tool-eval-bench bench --perf --tokenizer /models/Qwen3.6/tokenizer.json
```

> **Offline hosts:** llama-benchy always needs a tokenizer to construct prompts, and
> tool-eval-bench runs it in offline mode. The tokenizer is now located automatically:
> the served model id (including the vLLM `root` behind an alias) is matched against
> your HuggingFace cache (`~/.cache/huggingface/hub`, or `HF_HOME`/`HF_HUB_CACHE`),
> against local model directories, and against the llama.cpp `/props.model_path`.
> Pass `--tokenizer /path/to/tokenizer.json` (a file or a directory containing one)
> only to override that, or when nothing is found — the error then lists the
> tokenizers your cache does have. To fetch just the tokenizer on a networked host:
>
> ```bash
> hf download <org>/<model> --include "tokenizer*" "*config.json"
> ```
>
| Flag | Default | Purpose |
|---|---|---|
| `--perf` | off | Run llama-benchy throughput before scenarios |
| `--perf-only` | off | Run ONLY llama-benchy throughput |
| `--perf-preset` | `short` | Grid: `short`, or `long` for a depth sweep to 131072 (see [Presets](#presets)) |
| `--pp` | 2048 | Prompt tokens |
| `--tg` | 128 | Generation tokens |
| `--depth` | preset | Context depths (comma/space separated); `"0,4096,8192"` under `short` |
| `--concurrency` | `"1,2,4"` | Concurrency levels |
| `--benchy-runs` | 3 | Measurement iterations per test point |
| `--benchy-latency-mode` | `generation` | Latency mode: `api`, `generation`, `none` |
| `--benchy-args` | — | Pass-through for arbitrary llama-benchy flags |
| `--tokenizer` | auto | Local tokenizer.json path; overrides HF-cache auto-detection |

### Presets

`--perf-preset` picks the grid. `short`, the default, is the sweep `--perf` has always run: pp 2048, tg 128, depths 0, 4096, and 8192, concurrency 1, 2, and 4, three runs per point. A `short` run keeps the same config fingerprint as a run made before presets existed.

`long` measures what a long agent conversation costs. llama-benchy runs the full cross product of its flags in one invocation, so asking for 128K depth at concurrency 4 would put four 128K requests in flight and exhaust the KV cache. `long` runs two llama-benchy invocations instead and stores them as one run:

1. Concurrency 1, 2, and 4 at depth 0.
2. Concurrency 1 at depths 16384, 32768, 65536, and 131072, with llama-benchy's `--enable-prefix-caching`. Depth 0 at concurrency 1 is not measured again.

Depths where `depth + pp + tg` plus 256 tokens of chat-template headroom would not fit the served context window are skipped, with a note. Depth 0 always runs in the first pass; when `pp + tg` alone does not fit, a warning says even those cells will fail. When the server does not report its window, every depth runs and a warning says the deepest cells may fail. The tokenizer is looked up once for both passes. After the first pass, a rough lower bound on the second pass's wall time is printed, computed from the depth-0 speeds. The second pass is skipped when every cell of the first one failed, and the stored config then records only the first pass.

If llama-benchy stops during the second pass, with an error or on Ctrl+C, the first pass's results are kept and each cell of the second pass counts as failed. Its error names the cell, for example `pp2048 tg128 @ d16384 c1: not measured`. After Ctrl+C the finished passes are saved as their own run and the command exits 1, even if a scored run was meant to follow; the message then says the tool-call scenarios were not run.

Failures are counted in grid cells, one per depth and concurrency level. A cached-prefix cell has two rows, so a failure in either row counts that cell once. Every message and stored count uses the same number:

- A `--perf-only` run, or a `--perf` run saved on its own, stores `scores.failed` as the failed cell count, and `scores.incomplete: true` when a planned pass was skipped, failed, or interrupted. Either one gives the run status failed and exits 1.
- A `--perf` run followed by scenarios stores the same two keys under `scores.throughput`. The scored run's status comes from its scenarios, so throughput never changes it, its score, or its fingerprint. A notice before the scenarios start says how many cells failed and whether the sweep is incomplete.

Under `--json`, these notices arrive on stderr as `log` events, and stdout still holds the single result envelope.

Explicit flags override the preset. Under `long`, `--depth` sets the depth pass and `--concurrency` sets the depth-0 pass; the depth pass always runs at concurrency 1. `--pp`, `--tg`, and `--benchy-runs` apply to both passes. The run config stores `perf_preset` as `short` or `long`, or as `custom` when `--pp`, `--tg`, `--depth`, or `--concurrency` changed the preset's grid. `--benchy-runs` and depths skipped for the context window leave the name alone. A scored run also stores the passes of a multi-pass plan as `perf_passes`, matching the `passes` key of a `--perf-only` run. Neither key is part of the fingerprint, which already covers the depths and concurrency levels the plan expands to.

Each depth in the second pass produces two rows. Every run, the warm-up included, draws fresh corpus text from a random offset, so a run never reuses a prefix cached by an earlier one:

- The `ctx pp{depth}` row is the context-load request: `depth` tokens of system message and a one-character user message. Its TTFT is a cold prefill of `depth` tokens.
- The row after it sends the same system message with the `pp`-token prompt. On a server with prefix caching, its TTFT is the cost of `pp` new tokens on a cached `depth`-token prefix, which is what each turn of a growing conversation pays. Its tg t/s is decode speed at `depth + pp` context. If the server does not cache prefixes, this TTFT stays close to the cold one.

Depth 0 has only the standard row, which is cold.
