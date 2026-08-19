**DeepSeek thinking tool-loop history** — OpenAI-compatible responses already
parsed `reasoning_content`, but the orchestrator dropped it while rebuilding
assistant messages for the next request. Every assistant message in a user
turn that called tools now preserves the exact field as DeepSeek requires.
Ordinary no-tool turns no longer replay it; doing so caused HTTP 500 errors
on follow-up-heavy TC-46, TC-47, and TC-50 with DeepSeek V4 Flash via both
vLLM and the hosted API.
