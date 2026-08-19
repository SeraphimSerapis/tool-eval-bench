**Warm-up no longer fails on strict endpoints** — warm-up asks the server to
skip chain-of-thought via `chat_template_kwargs`, which vLLM and friends
understand and most hosted APIs ignore. Gemini's OpenAI-compatibility layer
instead rejects the unknown field with HTTP 400, so every run against it
opened with `⚠ Warm-up failed`. Warm-up now retries once without the optional
hints before giving up, and callers can hand it a request built for the
endpoint's own wire format.
