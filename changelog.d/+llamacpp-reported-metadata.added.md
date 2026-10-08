**llama.cpp context window and quantization.** Run metadata for llama.cpp now records the
context window from `/props`. When the model name does not identify a specific quantization, as
with an alias such as `gemma4` or a filename that only says `GGUF`, it also records the type the
server reports for the loaded GGUF file, for example `Q4_0` or `FP16`. A name that does identify
one, such as `UD-Q4_K_XL`, keeps its label. Both values feed `config_fingerprint`, so llama.cpp
runs that gain them start a new leaderboard cohort rather than grouping with earlier runs of the
same setup. Other backends record exactly what they did before.
