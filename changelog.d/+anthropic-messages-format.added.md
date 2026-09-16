Added a native adapter for the Anthropic Messages API (`/v1/messages`), selected automatically
for `api.anthropic.com` and for any base URL ending in `/messages`, such as OpenCode Zen's
gateway, or pinned with `--format anthropic`. Tool calls, tool results, `tool_choice`,
`json_schema` response formats, and thinking blocks with their signatures all translate in both
directions, so every scenario runs unchanged. Current Claude models reject `temperature`; the
adapter, pre-flight check, and warm-up drop it on that response and remember the choice. HTTP 529
now counts as a retryable overload status.
