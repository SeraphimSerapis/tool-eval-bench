**Reasoning-model preflight and warm-up compatibility** — hosted endpoints
that report a small probe's output-token exhaustion as HTTP 400/422 now count
as successfully serving and warming the model. Warm-up also uses the
benchmark's configured temperature and backend parameters instead of an
independent `temperature: 0.0` request, preventing false startup failures on
models that only support their default sampling configuration.
