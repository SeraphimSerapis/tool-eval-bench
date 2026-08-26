# Adding a backend adapter

An adapter translates between the benchmark's internal request shape and one provider's wire
format. There is exactly one abstract method, so a new adapter is a small file plus two lines of
wiring.

The benchmark ships two: `OpenAICompatibleAdapter` (vLLM, LiteLLM, llama.cpp, SGLang, and Google's
OpenAI compatibility layer) and `GeminiAdapter` (the native `generateContent` API). Add a third
only when a provider's request or tool-call shape genuinely differs. A provider that speaks
`POST /v1/chat/completions` needs no adapter at all, only a `--base-url`.

## The contract

`domain/adapters.py` defines `BackendAdapter` with one abstract method:

```python
async def chat_completion(self, ...) -> ChatCompletionResult
```

It returns a `ChatCompletionResult` carrying the assistant message, any `ProviderToolCall`s, token
counts, and timing. The runner never sees provider JSON, so parsing quirks stop at this boundary.

## The steps

1. **Write the adapter** in `src/tool_eval_bench/adapters/<provider>.py`, subclassing
   `BackendAdapter` and implementing `chat_completion`. Normalize tool calls into
   `ProviderToolCall` and populate `ChatCompletionResult`.

2. **Name the wire format** in `adapters/wire_format.py`: add it to the `WireFormat` literal and to
   `WIRE_FORMATS`, and teach `detect_wire_format` how to recognize the endpoint from its URL.
   Detection is by host and path, never by guessing from the model name.

3. **Dispatch to it** in `adapters/factory.py`, where `build_adapter` maps the resolved format to a
   class. Both the CLI's `--format` flag and auto-detection come through here.

Test against a mocked HTTP transport rather than a live endpoint. `tests/test_adapter.py` shows the
shape: assert on the request body your adapter builds and on how it parses a canned response,
including a malformed tool-call payload.

## What stays out of an adapter

Retries, rate-limit backoff, and connection pooling live in `utils/http_retry.py` and apply to
every adapter. Do not add per-provider retry logic; a second retry loop underneath the shared one
multiplies the wait instead of shortening it.
