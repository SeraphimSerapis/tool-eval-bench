**Strata backend support** — `--backend strata` now uses the existing OpenAI-compatible
adapter. Automatic detection recognizes Strata's declared `/health` service and
`/props` build identity instead of labeling its compatibility endpoints as llama.cpp.
Authenticated probes preserve engine version, effective context and slot count;
native GGUF quantization names such as Q2_0, Q8_0 and IQ2_XS appear in run metadata.
