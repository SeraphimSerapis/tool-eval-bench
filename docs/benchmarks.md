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
| `--gsm8k-shuffle` | off | Shuffle question order |
| `--mmlu` / `--mmlu-only` | off | Run MMLU benchmark |
| `--mmlu-shots` | 5 | Few-shot examples per subject (0–5) |
| `--mmlu-limit` | 500 | Max questions (0 = all 14,042) |
| `--mmlu-subjects` | all | Comma-separated subjects or categories (e.g. `STEM,philosophy`) |
| `--ifeval` / `--ifeval-only` | off | Run IFEval benchmark |
| `--ifeval-limit` | 0 (all) | Max prompts (0 = all 541) |

## Throughput benchmark

Throughput measurement uses [llama-benchy](https://github.com/eugr/llama-benchy) — a dedicated benchmarking tool that provides multi-run statistics with mean ± std, proper latency estimation, and cache-busting. Install with `uv tool install 'tool-eval-bench[perf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'` (or `pip install -e '.[perf]'` in a checkout) or ensure `uvx` is on PATH. Progress is shown via a live Rich progress bar. For authenticated endpoints, the regular `--api-key` value is forwarded to llama-benchy's supported CLI option and redacted from logs. Because llama-benchy 0.4.x does not support environment-based credentials, the key may still be visible to process inspection by other users on the same host while the benchmark is running.

An asterisk on a prefill rate means the first response chunk arrived before the first content token. Older llama-benchy versions can count that early chunk as the end of prefill, so the displayed rate is estimated from end-to-end time to first content token. The numerator follows the benchmark phase: prompt plus depth for a standard run, depth for context load, and prompt for a prefix-cached follow-up. This estimate includes network and queue time; compare it with other estimated rates rather than directly measured prefill rates.

llama-benchy always runs its own warm-up: a few global requests and a discarded first run of every test point, as llama-bench does. `--no-warmup` skips only the tool-eval-bench warm-up request.

A test point where any request failed is reported as failed, even when the other requests succeeded. llama-benchy computes a point from the requests that survived, so a c2 row that lost one request would otherwise show one request's throughput as the batch total, and a c1 row would average fewer runs than requested. With `--perf-only`, any failed point makes the run exit 1.

The Tokens column shows the prompt size plus the mean number of tokens the model actually generated. A model that stops before `--tg` generates fewer, and Total (ms) is computed from that observed count. Total is time to first content token plus generation time, so it is never below TTFT.

With `--enable-prefix-caching`, the context-load row is labelled `ctx pp{depth}`, and tool-eval-bench adds `--extra-body cache_prompt=true` so llama.cpp reuses the cached prefix on the follow-up request. Without it, the always-on `--no-cache` would send `cache_prompt: false` and llama.cpp would prefill the whole prompt again. A `cache_prompt` value passed in `--benchy-args` takes precedence. Progress events do not say which phase a request belonged to, so a failed request in either phase marks both rows of that test point as failed.

`--perf` combined with `--skip-tool-eval`, `--context-pressure-sweep`, `--spec-bench --skip-tool-eval`, or a plugin-only run such as `--gsm8k-only` saves the throughput sweep as its own run, exactly as `--perf-only` would. The run is saved as soon as the sweep finishes, before the other mode starts, so a failure there cannot lose it. A failed throughput cell still exits 1, after the other mode has run.

```bash
# Throughput only (skip tool-call scenarios)
tool-eval-bench bench --perf-only --pp 2048 --tg 128 --depth "0 4096 8192 16384 32768"

# Throughput + tool-call scenarios
tool-eval-bench bench --perf --depth "0 4096" --concurrency "1,2,4"

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
| `--pp` | 2048 | Prompt tokens |
| `--tg` | 128 | Generation tokens |
| `--depth` | `"0,4096,8192"` | Context depths (comma/space separated) |
| `--concurrency` | `"1,2,4"` | Concurrency levels |
| `--benchy-runs` | 3 | Measurement iterations per test point |
| `--benchy-latency-mode` | `generation` | Latency mode: `api`, `generation`, `none` |
| `--benchy-args` | — | Pass-through for arbitrary llama-benchy flags |
| `--tokenizer` | auto | Local tokenizer.json path; overrides HF-cache auto-detection |
