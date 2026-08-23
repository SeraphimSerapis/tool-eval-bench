**Reports, API, and containers.** Markdown reports contain hostile trace
fences and escaped table text, persisted endpoint URLs omit hosts and query
credentials, and `run_benchmark()` forwards difficulty weighting. Docker
builds install the tracked `uv.lock` with `uv sync --locked`, retain source
version provenance without shipping Git metadata, and run as a non-root user.
Compose requires the host UID and GID for writable report and database mounts.
The runtime, builder, and uv images are digest-pinned.
