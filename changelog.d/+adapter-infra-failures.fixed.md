**Server failures no longer score as model failures.** Several endpoint failures used to land in the model's quality score:

- An HTTP 429 or 503 whose JSON body carries a string `error`, as Hugging Face TGI and many gateways send, crashed the retry loop. The request was never retried and the scenario was scored as a model crash. It is now retried, and a persistent failure is a server error excluded from scoring.
- A native Gemini stream that failed after HTTP 200 with an `{"error": ...}` chunk returned an empty answer. It is now a transport error, and partial output is discarded.
- llama.cpp builds from before September 2025 report a mid-stream failure in an `error:` SSE field rather than `data:`. That field was ignored and the empty answer graded. It is now a transport error.
- A mid-stream Anthropic `rate_limit_error` was graded as the model's final answer after a tool call. Every wire format now treats a mid-stream 429, like the other retryable statuses, as infrastructure.
