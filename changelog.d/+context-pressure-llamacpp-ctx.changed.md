**Context pressure detects llama.cpp's context window.** `--context-pressure`,
`--context-pressure-sweep`, and `--needle` no longer ask for `--context-size` against llama.cpp.
When `/v1/models` declares no window, they use the `/props` `n_ctx` that the run metadata already
records. An explicit `--context-size` still wins, and other backends detect their window exactly
as before.
