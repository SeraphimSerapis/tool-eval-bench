**Run metadata describes the model under test.** On a server that lists several models in
`/v1/models`, such as llama-swap, LiteLLM, Ollama or vLLM with LoRA modules, the server model
ID, root, context length and the quantization guessed from them came from the first entry, which
could be a different model. They now come from the entry whose ID matches `--model`. A listing
with a single entry is still used when its ID differs, as llama.cpp's file-path IDs do. With
several entries and no match, those fields stay empty instead of describing another model.
Context pressure sizes its fill by the same rule. These fields feed `config_fingerprint`, so
affected runs start a new cohort, and the cohort no longer changes when the server reorders its
listing.
