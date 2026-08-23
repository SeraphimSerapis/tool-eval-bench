FROM ghcr.io/astral-sh/uv:0.10.8@sha256:88234bc9e09c2b2f6d176a3daf411419eb0370d450a08129257410de9cfafd2a AS uv

FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a AS build

WORKDIR /build

# setuptools-scm reads this stage's Git metadata. It never reaches the final
# image, but makes every installed package version identify its source commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY .git ./.git

# Build with `--build-arg EXTRAS=perf,hf` to bundle the throughput/HF-dataset extras.
# The static path is intentional: console-script shebangs remain valid after the
# virtual environment is copied into the final image.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ARG EXTRAS=""
ARG BUILD_VERSION=""
RUN extra_args=""; \
    for extra in $(printf '%s' "$EXTRAS" | tr ',' ' '); do \
        case "$extra" in \
            perf|hf) extra_args="$extra_args --extra $extra" ;; \
            '') ;; \
            *) echo "unsupported EXTRAS value: $extra" >&2; exit 2 ;; \
        esac; \
    done; \
    if [ -n "$BUILD_VERSION" ]; then \
        export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TOOL_EVAL_BENCH="$BUILD_VERSION"; \
    elif [ -f .git ]; then \
        echo "A linked worktree needs --build-arg BUILD_VERSION=0.0.0+g\$(git rev-parse --short HEAD)" >&2; \
        exit 2; \
    fi; \
    uv sync --locked --no-dev --no-editable $extra_args

FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

WORKDIR /app

# git is needed by tool-eval-bench's `compare` git helper; kept minimal otherwise.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The image itself runs as this unprivileged identity. Compose defaults to the
# caller's UID/GID instead, allowing existing host-owned bind mounts to stay
# host-owned. It deliberately never fixes mount ownership at container start.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --gid "$APP_GID" tool-eval \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --home-dir /app --no-create-home \
        --shell /usr/sbin/nologin tool-eval \
    && install -d --owner=tool-eval --group=tool-eval /app/data /app/runs
ENV HOME=/tmp
USER tool-eval
VOLUME ["/app/data", "/app/runs"]

ENTRYPOINT ["tool-eval-bench"]
CMD ["--help"]
