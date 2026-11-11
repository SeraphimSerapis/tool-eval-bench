**Unparsed tool-call diagnostics and server flag hints.** When an endpoint runs without its tool
parser enabled, models emit native tool markup such as `<tool_call>`, `<function=...>`,
`[TOOL_CALLS]`, `<|python_tag|>`, `<|tool_calls_begin|>`, or raw JSON call objects inside
`content` while `tool_calls` stays empty. Failing scenarios now record
`failure_kind="unparsed_tool_call"` and an `unparsed_tool_call=` trace line instead of a generic
`missing_step`, while keeping full denominator points so a broken parser cannot inflate scores.
Live output prints a one-time warning on the first affected scenario, and final panels, Markdown
reports, and tool probes name the server flags that fix it, such as
`--enable-auto-tool-choice --tool-call-parser <parser>` on vLLM, `--tool-call-parser <parser>` on
SGLang, and `--jinja` on llama.cpp.
