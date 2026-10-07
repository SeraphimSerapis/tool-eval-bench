**Removed the import cycle between throughput and speculative benchmarks.**
Speculative-decoding detection and its Prometheus counter helpers now live in an
independent runner module. Existing imports from `runner.speculative` remain
supported; benchmark behavior and metrics are unchanged.
