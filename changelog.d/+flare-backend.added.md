**Flare backend**: added support for [Flare](https://github.com/NVIDIA/hyperflare-inference)
(`--backend flare`, alias `hyperflare`) over the OpenAI-compatible adapter and `/v1/messages` with
`--format anthropic`. Automatic detection recognizes the `flare_` Prometheus namespace on `/metrics`
and falls back to the `flare` metadata object on `/v1/models`. Engine probing records
`max_model_len` (used automatically by context pressure and needle runs) and `flare.max_concurrency`
as `slot_count`, while `--spec-bench` and `spec-live` read Flare's speculative decoding, per-depth
draft, KV page, prefix cache, and request load metrics.
