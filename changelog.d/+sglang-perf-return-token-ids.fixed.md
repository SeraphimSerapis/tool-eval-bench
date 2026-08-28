`--perf` against SGLang failed with "no usable throughput metrics". SGLang now rejects a streaming
`/v1/chat/completions` request that carries `return_token_ids` (sgl-project/sglang#30917), which
llama-benchy sends on every generation request, so every sample came back empty. On an SGLang
endpoint the field is now switched off through `--extra-body`, and llama-benchy counts tokens from
the stream's `usage` block instead. `--benchy-args` still wins if it sets the field itself. The
failure also quotes the server's response now, rather than dropping it with the rest of
llama-benchy's non-JSON stdout.
