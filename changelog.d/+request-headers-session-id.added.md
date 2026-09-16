Added `--header NAME=VALUE` (repeatable) and `--session-header NAME`, with `TOOL_EVAL_HEADERS`,
`TOOL_EVAL_SESSION_HEADER`, and the provider-scoped `TOOL_EVAL_<NAME>_HEADERS` and
`TOOL_EVAL_<NAME>_SESSION_HEADER`, for gateways that need a header the wire format does not
define. A session header carries one id per scenario across all of its turns, and a fresh id per
single-shot request, which is what OpenCode Go's `x-opencode-session` asks for. Every request
now identifies itself as `tool-eval-bench/<version>` unless a header replaces it. The public
API takes the same options as `extra_headers` and `session_header`.
