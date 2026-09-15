`--provider NAME` (or `TOOL_EVAL_PROVIDER`) reads the endpoint from
`TOOL_EVAL_<NAME>_BASE_URL`, `_API_KEY`, and `_MODEL`, so one `.env` can hold Gemini, OpenAI,
Anthropic, and a local box side by side and an A/B run is a flag change. `openai` and `anthropic`
join `gemini` as hosted backend labels that skip engine probing. `.env.example` documents the vendor
endpoints, including the known gaps in Anthropic's OpenAI-compatible layer.
