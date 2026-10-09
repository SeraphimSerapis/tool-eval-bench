**Strata `/metrics` is now read.** Strata answers `/metrics` with JSON unless the request asks for
Prometheus text, and every scrape sent `Accept: */*`. Spec-decode acceptance counters and the
`--spec-live` load and counter panels came back empty, and backend detection never saw the
`strata:` namespace. Every `/metrics` request now sends
`Accept: text/plain; version=0.0.4, */*;q=0.1`. vLLM, SGLang, LiteLLM, llama.cpp, TensorFold, and
NInfer return the same text as before. Strata exports its spec-decode counters even when it has
drafted nothing, so detection reports spec decoding as active only after Strata has offered draft
tokens, and names the method MTP.
