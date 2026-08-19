**Output-token compatibility across OpenAI-style endpoints** — requests now
default to `max_tokens` for vLLM, LiteLLM, llama.cpp, and existing compatible
servers, then retry once with `max_completion_tokens` only when a 400/422
response explicitly requests that field. The learned choice is cached per
endpoint and model for benchmark, plugin, judge, and throughput requests;
preflight and warm-up use the same response-driven fallback. Explicit
`max_completion_tokens` backend parameters now suppress the legacy default
instead of sending both fields.
