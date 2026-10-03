"""Integration with llama-benchy for external performance benchmarking.

llama-benchy (https://github.com/eugr/llama-benchy) provides llama-bench style
pp/tg measurement for any OpenAI-compatible endpoint.  This module invokes it
as an external subprocess — either via ``uvx`` (zero-install) or via a locally
installed ``llama-benchy`` binary — and parses the JSON output into our
:class:`ThroughputSample` dataclass so results feed into the same display,
reports, and SQLite persistence as the built-in throughput benchmark.

Usage from the CLI::

    tool-eval-bench --perf          # run llama-benchy then scenarios
    tool-eval-bench --perf-only     # run llama-benchy only

The module never imports llama-benchy at the Python level; it communicates
exclusively through JSON I/O, keeping it a soft/optional dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tool_eval_bench.runner.throughput import ThroughputSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class LlamaBenchyResult:
    """Parsed results from a llama-benchy JSON report."""

    version: str = ""
    timestamp: str = ""
    latency_mode: str = ""
    latency_ms: float = 0.0
    model: str = ""
    samples: list[ThroughputSample] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def _find_llama_benchy() -> str | None:
    """Find the llama-benchy executable.

    Preference order:
      1. ``llama-benchy`` on PATH (pip/pipx/uv install)
      2. ``uvx`` available (zero-install via PyPI)

    Returns the command prefix as a string, or None if unavailable.
    """
    if shutil.which("llama-benchy"):
        return "llama-benchy"
    if shutil.which("uvx"):
        return "uvx llama-benchy"
    return None


def is_available() -> bool:
    """Check whether llama-benchy can be invoked."""
    return _find_llama_benchy() is not None


def _strip_url_credentials(url: str) -> str:
    """Return *url* with userinfo, query, and fragment removed.

    A base URL carries credentials by two routes that ``--api-key`` redaction
    does not cover: ``https://user:password@host`` and a query parameter such
    as ``?api_key=…``. Both are dropped.

    Scheme, host, port, and path survive, because the reason to log the command
    at all is to show which server was addressed. That is a deliberate contrast
    with :func:`~tool_eval_bench.utils.urls.redact_url`, which masks the whole
    authority for URLs that get *persisted* into run records.
    """
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        # A malformed authority (bad port, unbalanced brackets). Withhold the
        # whole value rather than guess which part was the credential.
        return "<unparsable-url>"
    if not host:
        return url
    # urlsplit strips the brackets from an IPv6 literal; put them back or the
    # rebuilt netloc runs the address and the port together.
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _redact_argument(arg: str) -> str:
    """Strip credentials from an argument that is, or carries, an http(s) URL."""
    if arg.startswith(("http://", "https://")):
        return _strip_url_credentials(arg)
    flag, separator, value = arg.partition("=")
    if separator and value.startswith(("http://", "https://")):
        return f"{flag}={_strip_url_credentials(value)}"
    return arg


def _redact_command(cmd: list[str]) -> str:
    """Render a command for logging without exposing credentials.

    Covers both channels an endpoint credential can travel by: an explicit
    ``--api-key`` value, and anything embedded in a URL argument.
    """
    redacted: list[str] = []
    redact_next = False

    for arg in cmd:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
        elif arg == "--api-key":
            redacted.append(arg)
            redact_next = True
        elif arg.startswith("--api-key="):
            redacted.append("--api-key=<redacted>")
        else:
            redacted.append(_redact_argument(arg))

    return shlex.join(redacted)


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------


def _extra_args_set_return_token_ids(extra_args: list[str] | None) -> bool:
    """Report whether pass-through args already decide ``return_token_ids``.

    ``--benchy-args`` is the escape hatch for exactly this kind of payload
    tweak, so a value set there wins over the SGLang default below.
    """
    return any("return_token_ids" in arg for arg in extra_args or ())


def _failure_hint(output_lines: list[str]) -> str:
    """Summarise why every benchmark request failed, from llama-benchy's output."""
    errors = [line for line in output_lines if line.startswith("HTTP ")]
    if not errors:
        return "Check endpoint authentication and llama-benchy output."
    # One shared cause per run, so the distinct set is short; cap it anyway.
    unique = list(dict.fromkeys(errors))[:3]
    detail = "\n".join(f"  {line}" for line in unique)
    return f"The server answered with:\n{detail}"


def _build_command(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    tokenizer: str | None = None,
    pp: list[int] | None = None,
    tg: list[int] | None = None,
    depths: list[int] | None = None,
    concurrency_levels: list[int] | None = None,
    runs: int = 3,
    latency_mode: str = "generation",
    no_cache: bool = True,
    skip_coherence: bool = False,
    skip_warmup: bool = False,
    backend: str | None = None,
    output_file: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the llama-benchy command line.

    Returns the full argument list suitable for ``asyncio.create_subprocess_exec``.
    """
    prefix = _find_llama_benchy()
    if prefix is None:
        raise RuntimeError(
            "llama-benchy is not available. Install it via:\n"
            "  pip install llama-benchy\n"
            "  # or ensure 'uvx' is on PATH for zero-install usage"
        )

    # Split prefix into parts (handles "uvx llama-benchy")
    cmd = prefix.split()

    # Normalise base URL — llama-benchy wants the base WITHOUT /v1
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    cmd.extend(["--base-url", url])
    cmd.extend(["--model", model])

    # llama-benchy 0.4.x only reads endpoint credentials from --api-key.
    # Respect an explicit pass-through value so existing --benchy-args
    # workarounds do not produce duplicate options.
    extra_args_have_api_key = bool(
        extra_args and any(arg == "--api-key" or arg.startswith("--api-key=") for arg in extra_args)
    )
    if api_key and not extra_args_have_api_key:
        cmd.extend(["--api-key", api_key])

    # Tokenizer: when the API model name is an alias (e.g. "Qwen3.6-35B")
    # but the real HF model ID is different (e.g. "Qwen/Qwen3.6-35B-A3B-FP8"),
    # pass --tokenizer so llama-benchy can find the HF tokenizer.
    if tokenizer and tokenizer != model:
        cmd.extend(["--tokenizer", tokenizer])

    # Prompt / generation sizes
    # llama-benchy uses nargs='+' (space-separated values after the flag),
    # e.g. --pp 1024 2048   NOT  --pp 1024 --pp 2048
    cmd.extend(["--pp", *(str(v) for v in (pp or [2048]))])
    cmd.extend(["--tg", *(str(v) for v in (tg or [128]))])
    cmd.extend(["--depth", *(str(v) for v in (depths or [0]))])
    cmd.extend(["--concurrency", *(str(v) for v in (concurrency_levels or [1]))])

    cmd.extend(["--runs", str(runs)])
    cmd.extend(["--latency-mode", latency_mode])

    if no_cache:
        cmd.append("--no-cache")
    if skip_coherence:
        cmd.append("--skip-coherence")
    if skip_warmup:
        cmd.append("--no-warmup")

    # Always disable adapt-prompt: tool-eval-bench has its own tokenizer
    # calibration, and llama-benchy's adapt-prompt forces extra warmup probes
    # that add latency without benefit.  Also reduces the tokenizer's role
    # to prompt construction only, where the gpt2 fallback is acceptable.
    cmd.append("--no-adapt-prompt")

    # SGLang answers a streaming request that carries ``return_token_ids`` with
    # a 400 (sgl-project/sglang#30917 turned the field from ignored-unknown into
    # recognized-but-unsupported-under-streaming).  llama-benchy sends it on
    # every generation request, so without this every sample comes back empty.
    # ``--extra-body`` is merged into the payload after the defaults, so this
    # switches the field off; llama-benchy then counts tokens from the stream's
    # ``usage`` block, which SGLang does send.
    if (backend or "").lower() == "sglang" and not _extra_args_set_return_token_ids(extra_args):
        cmd.extend(["--extra-body", "return_token_ids=false"])

    # JSON output
    cmd.extend(["--format", "json"])
    if output_file:
        cmd.extend(["--save-result", output_file])

    # Structured progress events (JSONL on stdout).  Requires llama-benchy >=0.4.0,
    # which is the current minimum.  If the user already specified --emit-progress
    # in extra_args, respect their choice.
    if not (extra_args and "--emit-progress" in extra_args):
        cmd.extend(["--emit-progress", "-"])

    # Pass-through extra args
    if extra_args:
        cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# Parse JSON output → ThroughputSample
# ---------------------------------------------------------------------------

# llama-benchy <0.4.2 (eugr/llama-benchy#33) stamps first_response_ts on any
# choices chunk. A role-only chunk can arrive before prefill, so est_ppt
# measures time to that chunk instead of the first content token. ttfr records
# the first response and e2e_ttft records the first content or reasoning token.
# Equal timings mean the response was content-first, even if latency clamped
# est_ppt to zero.
#
# The priming chunk's residual grows with depth on servers that tokenize
# before flushing it. The live mlx-serve row took 10.87 ms at depth 16384
# and escaped a fixed 10 ms ceiling. Bound this residual using the tokens
# the current phase prefilled.
_ROLE_CHUNK_PREPROCESS_MS_PER_TOKEN = 0.005
_ROLE_CHUNK_MAX_EST_PPT_MS = 10.0  # the floor of the bound, for depth-0 rows
# A pre-content chunk can also arrive after real prefill. Keep the cap and
# ratio so its mere presence does not force a rewrite.
_ROLE_CHUNK_MIN_E2E_RATIO = 10.0


def _role_chunk_prefill_inflated(
    concurrency: int,
    ttfr_ms: float,
    est_ppt_ms: float,
    e2e_ttft_ms: float,
    *,
    prefill_tokens: int = 0,
    pp_tps: float = 0.0,
) -> bool:
    """Return whether a single-stream row has the role-chunk timing signature.

    Concurrent ``pp_throughput`` is llama-benchy's batch formula: total prompt
    tokens divided by ``max(first_token_ts) - min(start_ts)``. ``first_token_ts``
    is the first content or reasoning chunk, so the role-only chunk does not
    enter it. ``pp_req_throughput`` can still be inflated; the sample displays
    the batch total for concurrency above 1 and leaves that row alone.

    A zero est_ppt is estimated only when the first response preceded the
    first content token. A content-first response can also have zero est_ppt
    when the latency probe exceeds ttfr.
    """
    if concurrency != 1 or not 0 < ttfr_ms < e2e_ttft_ms:
        return False
    if est_ppt_ms <= 0:
        return pp_tps <= 0
    cap = max(
        _ROLE_CHUNK_MAX_EST_PPT_MS,
        prefill_tokens * _ROLE_CHUNK_PREPROCESS_MS_PER_TOKEN,
    )
    if est_ppt_ms >= cap:
        return False
    return (e2e_ttft_ms / est_ppt_ms) >= _ROLE_CHUNK_MIN_E2E_RATIO


def _labeled_prefill_tokens(
    is_ctx_prefill: bool,
    depth: int,
    pp_tokens: int,
    *,
    prefix_caching_enabled: bool = False,
) -> int:
    """Return the token count this row's prefill rate should be divided by.

    llama-benchy defines ``context_size`` as the prefix, ``prompt_size`` as the
    prompt, and ``is_context_prefill_phase`` as the context-load step. That
    step's numerator is the depth (``ctx_pp @ d{context_size}``); this tool
    labels it ``pp{depth}``. Every other row is labeled ``pp{prompt_size}``.

    Match llama-benchy's numerator for each phase: depth for context load,
    prompt for its prefix-cached follow-up, and depth plus prompt for a
    standard run. ``--enable-prefix-caching`` is accepted through
    ``--benchy-args`` even when this tool passes ``--no-cache``.
    """
    if is_ctx_prefill:
        return depth
    if prefix_caching_enabled and depth > 0:
        return pp_tokens
    return depth + pp_tokens


def _parse_benchmark_entry(
    entry: dict[str, Any], *, prefix_caching_enabled: bool = False
) -> ThroughputSample:
    """Convert a single llama-benchy benchmark entry to a ThroughputSample."""
    concurrency = entry.get("concurrency", 1)
    depth = entry.get("context_size", 0)
    pp_tokens = entry.get("prompt_size", 0)
    tg_tokens = entry.get("response_size", 0)
    is_ctx_prefill = entry.get("is_context_prefill_phase", False)

    # Extract mean values from stat objects
    pp_tps = _stat_mean(entry.get("pp_throughput", {}))
    tg_tps = _stat_mean(entry.get("tg_throughput", {}))
    pp_req_tps = _stat_mean(entry.get("pp_req_throughput", {}))
    tg_req_tps = _stat_mean(entry.get("tg_req_throughput", {}))

    ttfr_ms = _stat_mean(entry.get("ttfr", {}))
    est_ppt_ms = _stat_mean(entry.get("est_ppt", {}))
    e2e_ttft_ms = _stat_mean(entry.get("e2e_ttft", {}))

    pp_estimated = False
    prefill_tokens = _labeled_prefill_tokens(
        is_ctx_prefill, depth, pp_tokens, prefix_caching_enabled=prefix_caching_enabled
    )
    if _role_chunk_prefill_inflated(
        concurrency,
        ttfr_ms,
        est_ppt_ms,
        e2e_ttft_ms,
        prefill_tokens=prefill_tokens,
        pp_tps=pp_tps,
    ):
        if prefill_tokens > 0:
            honest_pp_tps = prefill_tokens / (e2e_ttft_ms / 1000.0)
            pp_tps = honest_pp_tps
            pp_req_tps = honest_pp_tps
            est_ppt_ms = e2e_ttft_ms
            pp_estimated = True

    # Concurrent rows display the batch total, which is what llama-benchy's
    # table shows. Single-stream req and total are the same value.
    if concurrency > 1:
        display_pp = pp_tps
        display_tg = tg_tps
    else:
        display_pp = pp_req_tps if pp_req_tps > 0 else pp_tps
        display_tg = tg_req_tps if tg_req_tps > 0 else tg_tps

    # Estimate total time from est_ppt + generation time
    gen_time_ms = (tg_tokens / tg_req_tps * 1000) if tg_req_tps > 0 else 0
    total_ms = est_ppt_ms + gen_time_ms if est_ppt_ms > 0 else 0

    # If this is a context prefill phase, override pp label
    req_pp = depth if is_ctx_prefill else pp_tokens

    return ThroughputSample(
        pp_tokens=pp_tokens,
        tg_tokens=tg_tokens,
        depth=depth,
        concurrency=concurrency,
        ttft_ms=e2e_ttft_ms,
        total_ms=total_ms,
        pp_tps=display_pp,
        tg_tps=display_tg,
        requested_pp=req_pp,
        requested_depth=depth,
        calibration_confidence="llama-benchy",
        pp_estimated=pp_estimated,
    )


def _stat_mean(stat: dict[str, Any]) -> float:
    """Extract the mean value from a llama-benchy stat object."""
    if isinstance(stat, dict):
        return float(stat.get("mean", 0))
    return 0.0


def _sample_cell_key(sample: ThroughputSample) -> tuple[int, int, int, int]:
    """Return the matrix coordinates shared by results and progress events."""
    return (sample.pp_tokens, sample.tg_tokens, sample.depth, sample.concurrency)


def _progress_cell_key(event: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return one progress event's matrix coordinates."""
    return (
        int(event.get("prompt_size", 0)),
        int(event.get("response_size", 0)),
        int(event.get("context_size", 0)),
        int(event.get("concurrency", 1)),
    )


def _invalid_sample_error(sample: ThroughputSample, errors: list[str]) -> str | None:
    """Describe an all-zero cell, using llama-benchy's request error when present."""
    if any(value > 0 for value in (sample.pp_tps, sample.tg_tps, sample.ttft_ms, sample.total_ms)):
        return None
    detail = "; ".join(dict.fromkeys(errors)) or "no usable throughput metrics"
    return (
        f"pp{sample.label_pp} tg{sample.tg_tokens} @ d{sample.label_depth} "
        f"c{sample.concurrency}: {detail}"
    )


def parse_json_output(data: dict[str, Any]) -> LlamaBenchyResult:
    """Parse a complete llama-benchy JSON output into a LlamaBenchyResult."""
    result = LlamaBenchyResult(
        version=data.get("version", ""),
        timestamp=data.get("timestamp", ""),
        latency_mode=data.get("latency_mode", ""),
        latency_ms=data.get("latency_ms", 0.0),
        model=data.get("model", ""),
        raw_json=data,
    )

    prefix_caching_enabled = data.get("prefix_caching_enabled") is True
    for entry in data.get("benchmarks", []):
        sample = _parse_benchmark_entry(entry, prefix_caching_enabled=prefix_caching_enabled)
        result.samples.append(sample)

    return result


# ---------------------------------------------------------------------------
# Run llama-benchy
# ---------------------------------------------------------------------------


async def run_llama_benchy(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    tokenizer: str | None = None,
    pp: list[int] | None = None,
    tg: list[int] | None = None,
    depths: list[int] | None = None,
    concurrency_levels: list[int] | None = None,
    runs: int = 3,
    latency_mode: str = "generation",
    no_cache: bool = True,
    skip_coherence: bool = False,
    skip_warmup: bool = False,
    backend: str | None = None,
    extra_args: list[str] | None = None,
    on_output: Callable[[str], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> LlamaBenchyResult:
    """Run llama-benchy as a subprocess and parse the results.

    Parameters
    ----------
    on_output : callable, optional
        Called with each line of stderr (regular llama-benchy output) for
        real-time display.  Signature: ``(line: str) -> None``
    on_progress : callable, optional
        Called with each structured progress event (JSONL) from llama-benchy's
        ``--emit-progress`` stream.  Each event is a dict with a ``"type"`` key
        (e.g. ``"request_start"``, ``"request_end"``, ``"bench_complete"``).
        Signature: ``(event: dict[str, Any]) -> None``

    Returns
    -------
    LlamaBenchyResult
        Parsed results with ThroughputSample objects.

    Raises
    ------
    RuntimeError
        If llama-benchy is not available or the subprocess fails.
    FileNotFoundError
        If llama-benchy is not installed and uvx is not available.
    """
    # Write JSON results to a temp file.  Progress JSONL goes to stdout when
    # --emit-progress - is set; human-readable logs go to stderr.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        output_file = f.name

    try:
        cmd = _build_command(
            base_url,
            model,
            api_key=api_key,
            tokenizer=tokenizer,
            pp=pp,
            tg=tg,
            depths=depths,
            concurrency_levels=concurrency_levels,
            runs=runs,
            latency_mode=latency_mode,
            no_cache=no_cache,
            skip_coherence=skip_coherence,
            skip_warmup=skip_warmup,
            backend=backend,
            output_file=output_file,
            extra_args=extra_args,
        )

        logger.info("Running llama-benchy: %s", _redact_command(cmd))

        # Suppress noisy warnings from transformers/HF Hub in the subprocess:
        # - "PyTorch was not found" (only tokenizers are needed)
        # - "You are sending unauthenticated requests to the HF Hub"
        # PYTHONUNBUFFERED forces line-by-line streaming instead of buffering
        # all output until exit (Python buffers stdout when writing to a pipe).
        env = {**os.environ}
        env["PYTHONUNBUFFERED"] = "1"
        env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        # Force offline mode: tool-eval-bench only needs throughput measurement,
        # not model/tokenizer downloads.  This prevents transformers from
        # loading large files for the model path (the OOM root cause in #14).
        # The gpt2 fallback tokenizer (used by llama-benchy for prompt
        # construction) is either cached locally or fails with a clear error.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # With --emit-progress -, llama-benchy writes JSONL progress events to
        # stdout and regular output to stderr.  Read both concurrently to avoid
        # pipe-buffer deadlock.
        output_lines: list[str] = []
        request_cells: dict[int, tuple[int, int, int, int]] = {}
        cell_errors: dict[tuple[int, int, int, int], list[str]] = {}
        # Known noisy lines from transformers/HF Hub we suppress from display
        _SUPPRESS = (
            "PyTorch was not found",
            "Models won't be available",
            "unauthenticated requests to the HF Hub",
        )

        async def _read_stderr() -> None:
            if proc.stderr is None:
                return
            async for raw_line in proc.stderr:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(line)
                if on_output and not any(s in line for s in _SUPPRESS):
                    on_output(line)

        async def _read_stdout() -> None:
            if proc.stdout is None:
                return
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # llama-benchy prints per-request failures (``HTTP 400: …``)
                    # to stdout, alongside the progress JSONL.  Keep them: they
                    # are the only explanation available when a run finishes
                    # with every metric empty.
                    if line:
                        output_lines.append(line)
                    continue
                event_type = event.get("type")
                request_id = event.get("request_id")
                if event_type == "request_start" and request_id is not None:
                    request_cells[int(request_id)] = _progress_cell_key(event)
                elif event_type == "request_end" and request_id is not None:
                    cell = request_cells.pop(int(request_id), None)
                    error = event.get("error")
                    if cell is not None and isinstance(error, str) and error:
                        cell_errors.setdefault(cell, []).append(error)
                if on_progress:
                    on_progress(event)

        await asyncio.gather(_read_stderr(), _read_stdout())

        returncode = await proc.wait()

        if returncode != 0:
            # Detect OOM kill (Linux SIGKILL = signal 9 → exit code -9 or 137)
            _OOM_MARKERS = (
                "MemoryError",
                "Killed",
                "out of memory",
                "Cannot allocate memory",
            )
            tail = output_lines[-30:] if output_lines else []
            is_oom = returncode in (-9, 137) or any(
                marker in line for line in tail for marker in _OOM_MARKERS
            )
            if is_oom:
                raise RuntimeError(
                    "llama-benchy was killed — likely out of memory (OOM).\n"
                    "This can happen when the HuggingFace transformers library "
                    "loads tokenizer data for large models, consuming excessive "
                    "RAM in the subprocess."
                )

            # Detect offline tokenizer failure: llama-benchy always needs a
            # tokenizer to construct benchmark prompts, and in offline mode
            # (HF_HUB_OFFLINE=1, set above) neither the model tokenizer nor
            # the gpt2 fallback can be fetched when the HuggingFace cache is
            # empty — transformers raises this OSError.
            tail = output_lines[-30:] if output_lines else []
            is_offline_tokenizer = any(
                "couldn't find them in the cached files" in line for line in tail
            ) or any("Error loading tokenizer" in line for line in tail)
            if is_offline_tokenizer:
                from tool_eval_bench.utils.tokenizers import (
                    format_candidates,
                    iter_cached_repos,
                )

                cached = format_candidates(sorted(iter_cached_repos()))
                cache_hint = (
                    f"\nTokenizers found in your HuggingFace cache:\n{cached}\n"
                    "Pass one of these with --tokenizer if it matches the served model.\n"
                    if cached
                    else ""
                )
                raise RuntimeError(
                    "llama-benchy could not load a tokenizer.\n"
                    "llama-benchy needs a tokenizer to construct prompts, and this "
                    "host is running in offline mode with no tokenizer in the "
                    "HuggingFace cache.\n"
                    f"{cache_hint}"
                    "Fixes:\n"
                    "  - Pass --tokenizer /path/to/tokenizer.json (or a directory "
                    "containing tokenizer.json) to use a local tokenizer.\n"
                    "  - Or fetch just the tokenizer once with network access:\n"
                    '      hf download <org>/<model> --include "tokenizer*" "*config.json"'
                )
            output_text = "\n".join(output_lines[-20:])  # last 20 lines
            raise RuntimeError(f"llama-benchy exited with code {returncode}:\n{output_text}")

        # Parse the JSON output file
        output_path = Path(output_file)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                "llama-benchy did not produce JSON output. "
                "Check that the server is running and reachable."
            )

        raw_data = json.loads(output_path.read_text(encoding="utf-8"))
        result = parse_json_output(raw_data)
        for sample in result.samples:
            sample.error = _invalid_sample_error(
                sample,
                cell_errors.get(_sample_cell_key(sample), []),
            )
        if not result.samples:
            raise RuntimeError(
                "llama-benchy produced no benchmark samples. "
                "Check endpoint authentication and llama-benchy output."
            )
        if not any(sample.pp_tps > 0 or sample.tg_tps > 0 for sample in result.samples):
            raise RuntimeError(
                "llama-benchy produced no usable throughput metrics — "
                "every benchmark request failed.\n"
                f"{_failure_hint(output_lines)}"
            )
        return result

    finally:
        # Clean up temp file
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            logger.debug("Failed to clean up temp file %s", output_file)
