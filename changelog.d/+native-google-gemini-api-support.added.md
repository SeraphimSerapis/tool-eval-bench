**Native Google Gemini API support** — pointing `--base-url` at
`https://generativelanguage.googleapis.com` now speaks the native
`:generateContent` API (https://ai.google.dev/api) instead of requiring
Google's OpenAI compatibility layer. The format is detected from the URL —
the compatibility layer lives under `/v1beta/openai` on the same host, so both
keep working — and `--format auto|openai|gemini` pins it when detection is
wrong. `gemini` is now a valid `--backend` label, selected automatically for
hosted endpoints so reports stop claiming "vllm", and engine probing
(`/metrics`, `/props`, `/version`) is skipped where it means nothing.
Translation covers system instructions, function declarations and tool-choice
modes, tool results, streaming SSE, thinking budgets, `usageMetadata` token
counts, and Gemini 3 thought signatures, which round-trip through tool calls
as the API requires. GSM8K / MMLU / IFEval and the context-pressure sweep
follow the same format as the main run.
