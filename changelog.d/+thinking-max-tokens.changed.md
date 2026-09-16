Each turn is sent with `max_tokens` 16384 when thinking is enabled and 4096
under `--no-think`, instead of a fixed 4096. DeepSeek V4.1 Flash lost three
scenarios to turns where roughly 4,500 tokens of reasoning hit the old cap
before any answer. An explicit `max_tokens` or `max_completion_tokens` in
`--backend-kwargs` still wins. The value used is printed under Run Context so
runs made under the old ceiling are not compared silently.
