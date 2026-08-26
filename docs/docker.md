# Running with Docker

The image bundles the CLI and its dependencies, so you can benchmark a server without installing Python locally. It runs as a non-root user and writes results to bind-mounted `data/` and `runs/` directories on the host.


No local Python setup required — build once, then run against any
OpenAI-compatible endpoint reachable from the container:

```bash
git clone https://github.com/SeraphimSerapis/tool-eval-bench.git
cd tool-eval-bench

# Point it at your server: copy the config template and fill in the target
# (same TOOL_EVAL_* variables as the README's Configuration section)
cp .env.example .env
# edit .env: set TOOL_EVAL_BASE_URL, or set TOOL_EVAL_HOST/TOOL_EVAL_PORT;
# also set TOOL_EVAL_API_KEY when the endpoint requires authentication

# Compose validates env_file entries before any command, so .env must exist first.
# It also requires the host identity for writable bind mounts.
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
docker compose build

# Confirm the image identifies the source commit used for the build
docker compose run --rm tool-eval-bench --version

# Check the endpoint is reachable (default command)
docker compose run --rm tool-eval-bench --probe

# Run the benchmark — any CLI flag works here too
docker compose run --rm tool-eval-bench --short --seed 42
```

Reports land in `./runs/` on the host, matching the CLI's own default output
path (`./runs/YYYY/MM/`). SQLite history lands in `./data/`. The Compose file
mounts both directories, so reports, traces, history, and leaderboard data
survive `--rm` cleaning up the container.

The image runs as an unprivileged `tool-eval` user. Compose requires
`LOCAL_UID` and `LOCAL_GID`, set them with `id -u` and `id -g` above, so
bind-mounted outputs retain host ownership. Ensure `runs/` and `data/` are
writable by that user. The container never runs a root ownership-fixing
entrypoint. This is deliberate. Compose fails before starting if the variables
are absent instead of silently using an incorrect host identity.

Docker builds launched from a linked Git worktree need an explicit source
version because its `.git` file points outside the build context:

```bash
docker build \
  --build-arg BUILD_VERSION="0.0.0+g$(git rev-parse --short HEAD)" \
  -t tool-eval-bench:local .
```

That version remains commit-identifiable. A normal checkout derives the fuller
setuptools-scm version from its in-context `.git` directory automatically.

Build with the throughput/HF-dataset extras via `docker compose build --build-arg EXTRAS=perf,hf`.
