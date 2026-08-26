# Accuracy and throughput benchmarks

Beyond tool-calling quality, `tool-eval-bench` runs external accuracy benchmarks through the same adapter layer, and measures prefill and generation speed against the same endpoint. Neither needs `tools` support from the server.

## Accuracy benchmarks (GSM8K, MMLU, IFEval)


Pluggable accuracy benchmarks evaluate model knowledge and instruction-following capabilities. Datasets are downloaded automatically from HuggingFace on first use and cached locally under `data/`.

**Recommended:** Install the `datasets` library for fast, rate-limit-free downloads directly from the HuggingFace git repo:

```bash
pip install 'tool-eval-bench[hf]'
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

Throughput measurement uses [llama-benchy](https://github.com/eugr/llama-benchy) — a dedicated benchmarking tool that provides multi-run statistics with mean ± std, proper latency estimation, and cache-busting. Install with `pip install 'tool-eval-bench[perf]'` or ensure `uvx` is on PATH. Progress is shown via a live Rich progress bar. For authenticated endpoints, the regular `--api-key` value is forwarded to llama-benchy's supported CLI option and redacted from logs. Because llama-benchy 0.4.x does not support environment-based credentials, the key may still be visible to process inspection by other users on the same host while the benchmark is running.

```bash
# Throughput only (skip tool-call scenarios)
tool-eval-bench bench --perf-only --pp 2048 --tg 128 --depth "0 4096 8192 16384 32768"

# Throughput + tool-call scenarios
tool-eval-bench bench --perf --depth "0 4096" --concurrency "1,2,4"

# Customize measurement runs and latency mode
tool-eval-bench bench --perf --benchy-runs 5 --benchy-latency-mode generation

# Pass arbitrary flags to llama-benchy
tool-eval-bench bench --perf --benchy-args='--no-warmup --enable-prefix-caching'

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
