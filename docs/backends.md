# Backends

OpenAI-compatible backends must expose `/v1/chat/completions` and support the
`tools` and `tool_choice` request fields the tool-call scenarios use. The
accuracy benchmarks (GSM8K, MMLU, IFEval, needle) need only chat completions.

- **vLLM** — primary target
- **SGLang** — OpenAI-compatible model server
- **LiteLLM** — proxy for multiple backends
- **llama.cpp** — lightweight local inference
- **NInfer** — OpenAI-compatible inference engine, detected via `/v1/models`
- **Gemini** — supported through its native API as well as its OpenAI-compatible
  endpoint; the native wire format is detected from the URL, and `--format` pins
  it manually

## How the adapter talks to a server

The adapter sends real `tools` and `tool_choice` in the request and parses
`tool_calls` out of the response. There is no prompt hacking and no JSON regex
matching.

It accepts SSE `data:` fields with or without the optional space, and parses a
normal JSON 200 response when an endpoint ignores `stream=true`. It defaults to
the widely supported `max_tokens` field; if an endpoint rejects that field and
asks for `max_completion_tokens`, the adapter retries once and remembers the
choice for that endpoint and model. This capability check is response-driven
rather than tied to provider or model names.

## Compatibility notes

| Behavior | vLLM | SGLang | LiteLLM | llama.cpp |
|---|---|---|---|---|
| `/v1/models` discovery | ✅ | ✅ | ✅ | ⚠️ May be at `/models` |
| `parallel_tool_calls` | ✅ | ✅ | ✅ | ❌ Not supported |
| Streaming `usage` stats | ✅ | Varies | Varies | ❌ |
| `tool_choice: "required"` | ✅ | ✅ | ✅ | ⚠️ Version-dependent |
| Large toolsets (52 tools) | ✅ | ✅ | ✅ | ⚠️ May exceed context window |
| `--spec-bench` acceptance rate | ✅ Prometheus | ⚠️ Live gauges are not request-local | ✅ when backend metrics are reachable | ✅ Counters or per-request timings |
| `--spec-live` dashboard | ✅ Counters | ✅ Gauges | ✅ when backend metrics are separately reachable | ✅ Counters on current builds; engine-only fallback |

OpenAI-compatible backends use `OpenAICompatibleAdapter`; native Gemini uses its
own adapter. If you hit a backend-specific issue, please
[open an issue](https://github.com/SeraphimSerapis/tool-eval-bench/issues).

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
