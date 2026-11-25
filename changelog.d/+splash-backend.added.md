**Splash backend**: added support for [Splash](https://github.com/incoai/splash) (`--backend splash`)
over the OpenAI-compatible adapter. Automatic detection recognizes the `splash_` Prometheus namespace
on `/metrics`, `owned_by: "splash"` on `/v1/models`, and the `Server: Splash` header. Engine probing
records `max_model_len` (Splash's single-request limit, used by context pressure and needle runs)
and `/status` `frontend.preparation_capacity` as `slot_count`. `--spec-bench` reads the
`metrics.speculation` draft counts and cycles that Splash adds to each response that drafted, which
give exact per-request acceptance rate and acceptance length, and `spec-live` reads Splash's draft,
token, and scheduler counters.
