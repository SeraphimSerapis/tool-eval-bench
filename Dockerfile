FROM python:3.12-slim

WORKDIR /app

# git is needed by tool-eval-bench's `compare` git helper; kept minimal otherwise.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Build with `--build-arg EXTRAS=perf,hf` to bundle the throughput/HF-dataset extras.
ARG EXTRAS=""
RUN pip install --no-cache-dir .$( [ -n "$EXTRAS" ] && echo "[$EXTRAS]" )

RUN mkdir -p /app/runs
VOLUME ["/app/runs"]

ENTRYPOINT ["tool-eval-bench"]
CMD ["--help"]
