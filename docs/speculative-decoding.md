# Speculative decoding and MTP

Measures how much speculative decoding actually buys you on your deployment: acceptance rate, effective tokens per second, and the speedup against a known baseline. Includes a live monitor for watching acceptance behave under real traffic.


Measures the **real-world effectiveness** of multi-token prediction (MTP), draft models, and n-gram speculative decoding. Standard t/s metrics don't capture these benefits — `--spec-bench` does.

```bash
# Quick spec-decode benchmark (auto-detect method)
tool-eval-bench bench --spec-bench

# Specify method + compare against known baseline
tool-eval-bench bench --spec-bench --spec-method mtp --baseline-tgs 30.0

# Custom prompt types and depths
tool-eval-bench bench --spec-bench --spec-prompts "code,structured" --depth "0,4096"

# Combined: throughput + spec-decode + tool-call quality
tool-eval-bench bench --perf --spec-bench --seed 42
```

| Spec-Decode Flag | Default | Purpose |
|---|---|---|
| `--spec-bench` | off | Run speculative decoding benchmark |
| `--spec-method` | `auto` | Method hint for MTP/NEXTN, draft/STANDALONE, DFlash, DSpark, n-gram, EAGLE/EAGLE3, Medusa, MLP speculators, suffix, and custom proposers |
| `--baseline-tgs` | — | Known baseline tg t/s for speedup calculation |
| `--spec-prompts` | `filler,code,structured` | Prompt types to test |
| `--metrics-url` | auto | Direct URL to Prometheus `/metrics` (e.g. `http://vllm:8080/metrics`) |

> **Acceptance rate.** The primary metric is **effective t/s** — output tokens ÷ wall-clock time — which always works. Acceptance rate and draft statistics use different extraction methods depending on the backend:
>
> | Backend | Acceptance Rate Source | What You Get |
> |---|---|---|
> | **vLLM** | Prometheus `/metrics` (`spec_decode_*` counters) | α %, acceptance length (τ), draft window, per-position waterfall, waste ratio |
> | **llama.cpp** | Current Prometheus counters, with per-request `timings` fallback | Full counter metrics on current builds; α % and waste ratio from response timings on older builds |
> | **SGLang** | No request-local counter contract | Effective t/s remains available; use `spec-live` for the server's current acceptance gauges |
>
> For **llama.cpp**, use `--spec-method=mtp` or the matching configured method
> when a build only exposes per-request timings. Current builds also expose
> cumulative speculative counters for `spec-live` when metrics are enabled:
> ```bash
> # llama.cpp with MTP speculative decoding
> tool-eval-bench --spec-bench --spec-method mtp
> ```
>
> **Using a proxy (LiteLLM)?** The API proxy doesn't forward the backend's `/metrics`. Use `--metrics-url` to point directly at the inference server:
> ```bash
> # API goes through LiteLLM, but scrape metrics from vLLM directly
> tool-eval-bench --spec-bench --base-url http://litellm:4000 --metrics-url http://vllm:8080/metrics
> ```
>
> Because `--metrics-url` can name a different host, `--api-key` is only sent to
> it when its scheme, host, and port match `--base-url`. If your metrics endpoint
> needs the same credential and lives elsewhere, expose it without auth or put it
> behind the same origin.

## Live speculative decoding monitor

Keep a **real-time terminal dashboard** open while working. `spec-live`
continuously polls the server's Prometheus `/metrics` endpoint and renders
acceptance, throughput, draft efficiency, and engine status when the selected
backend exports those metric families.

The dashboard runs in the terminal's **alternate screen buffer** (like htop or vim), giving a clean full-terminal canvas without disturbing previous output. On exit, your original terminal content is restored.

```bash
# Start the live monitor (runs until Ctrl+C)
tool-eval-bench spec-live

# Custom poll interval (default: 1 second)
tool-eval-bench spec-live --spec-live-interval 2

# Tell the dashboard which spec method you're running
tool-eval-bench spec-live --spec-method dflash

# Point at vLLM metrics directly (when API is behind a proxy)
tool-eval-bench spec-live --metrics-url http://vllm:8080/metrics
```

The dashboard shows:
- **Acceptance rate gauge** — color-coded 0–100% bar when the backend exposes supported speculative metrics
- **Draft efficiency gauge** — τ/window utilization with tuning hints when both values are available
- **Method badge** — uses explicit configuration or metric labels only. Generic speculative counters report `unknown`; use `--spec-method` to label a known configuration.
- **Draft model name** — shown only when an explicit server configuration identifies it. Multiple `/v1/models` entries are not treated as a drafter relationship.
- **Per-position acceptance bars** — shown for vLLM and current llama.cpp builds when their per-position counters are exported
- **Throughput sparklines** — rolling 60-second history of accept rate, gen t/s, accepted t/s, and waste ratio with min/max annotations
- **Rolling averages** — session-level mean α, gen t/s, and accepted t/s (visible immediately with 0.0 initial values)
- **Engine status** — GPU KV cache, prefix cache hit rate, running/waiting requests, prompt t/s
- **Session totals** — cumulative accepted/drafted tokens and session-wide acceptance rate

Cumulative counter history and token totals are session-relative. Throughput,
KV-cache, queue, and SGLang acceptance gauges show the server's current state.
Prometheus counter series are server-wide, so other clients and concurrent
requests contribute to session deltas. Use an isolated server when you need a
request-local comparison.

Press **Ctrl+R** to reset all session counters and history without restarting.  This lets you switch workloads and measure each independently.  Press **Ctrl+C** to exit; a session summary panel shows aggregate statistics.

| Flag | Default | Purpose |
|---|---|---|
| `--spec-live` | off | Start live speculative decoding monitor |
| `--spec-live-interval` | `1.0` | Seconds between metric scrapes |
| `--spec-method` | `auto` | Method hint. Use `--help` for the current cross-engine choices. |
| `--metrics-url` | auto | Direct URL to Prometheus `/metrics` endpoint |

> **Backend contracts.** vLLM publishes cumulative `spec_decode_*` counters,
> one series per engine; the monitor sums those series. SGLang publishes direct
> `sglang:spec_*` gauges; the monitor selects a stable rank-zero series instead
> of summing replicas. Current llama.cpp builds publish cumulative
> `llamacpp:spec_decode_*` counters and per-position counters. Older llama.cpp
> builds fall back to engine and throughput data when those counters are absent.
> Acceptance length follows each upstream contract and includes the verifier's
> bonus token.
