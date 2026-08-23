**Streaming and measurement compatibility.** Adapters accept normal JSON
responses to streaming requests and both legal SSE data-field forms. OpenAI
and Gemini streams start TTFT on reasoning, content, or tool output, and
OpenAI streams request usage without replacing explicit `stream_options`.
Measurement runners use an injected domain port that preserves raw arrival
timing while the HTTP adapter owns endpoint routing and authentication. Strict
endpoints can reject optional token-ID fields, and speculative and live-counter
labels remain truthful.
