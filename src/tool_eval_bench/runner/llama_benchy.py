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
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

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
    skip_coherence: bool = True,
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

    if api_key:
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

    # JSON output
    cmd.extend(["--format", "json"])
    if output_file:
        cmd.extend(["--save-result", output_file])

    # Pass-through extra args
    if extra_args:
        cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# Parse JSON output → ThroughputSample
# ---------------------------------------------------------------------------

def _parse_benchmark_entry(entry: dict[str, Any]) -> ThroughputSample:
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

    _stat_mean(entry.get("ttfr", {}))  # ttfr available but we use e2e_ttft
    est_ppt_ms = _stat_mean(entry.get("est_ppt", {}))
    e2e_ttft_ms = _stat_mean(entry.get("e2e_ttft", {}))

    # For concurrent runs, use per-request throughput for the sample's
    # pp_tps/tg_tps (total throughput is in the aggregated fields).
    # For single-stream, req and total are the same.
    if concurrency > 1:
        # Use total throughput for display (matches llama-benchy table)
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
    )


def _stat_mean(stat: dict[str, Any]) -> float:
    """Extract the mean value from a llama-benchy stat object."""
    if isinstance(stat, dict):
        return float(stat.get("mean", 0))
    return 0.0


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

    for entry in data.get("benchmarks", []):
        sample = _parse_benchmark_entry(entry)
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
    skip_coherence: bool = True,
    extra_args: list[str] | None = None,
    on_output: Any | None = None,
) -> LlamaBenchyResult:
    """Run llama-benchy as a subprocess and parse the results.

    Parameters
    ----------
    on_output : callable, optional
        Called with each line of stdout for real-time progress display.
        Signature: ``(line: str) -> None``

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
    # Write JSON output to a temp file for reliable parsing
    # (stdout may contain progress/debug output mixed with JSON)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        output_file = f.name

    try:
        cmd = _build_command(
            base_url, model,
            api_key=api_key,
            tokenizer=tokenizer,
            pp=pp, tg=tg,
            depths=depths,
            concurrency_levels=concurrency_levels,
            runs=runs,
            latency_mode=latency_mode,
            no_cache=no_cache,
            skip_coherence=skip_coherence,
            output_file=output_file,
            extra_args=extra_args,
        )

        logger.info("Running llama-benchy: %s", " ".join(cmd))

        # Suppress noisy warnings from transformers/HF Hub in the subprocess:
        # - "PyTorch was not found" (only tokenizers are needed)
        # - "You are sending unauthenticated requests to the HF Hub"
        # PYTHONUNBUFFERED forces line-by-line streaming instead of buffering
        # all output until exit (Python buffers stdout when writing to a pipe).
        env = {**os.environ}
        env["PYTHONUNBUFFERED"] = "1"
        env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        # Stream stdout for real-time display
        output_lines: list[str] = []
        # Known noisy lines from transformers/HF Hub we suppress from display
        _SUPPRESS = (
            "PyTorch was not found",
            "Models won't be available",
            "unauthenticated requests to the HF Hub",
        )
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line)
            if on_output and not any(s in line for s in _SUPPRESS):
                on_output(line)

        returncode = await proc.wait()

        if returncode != 0:
            output_text = "\n".join(output_lines[-20:])  # last 20 lines
            raise RuntimeError(
                f"llama-benchy exited with code {returncode}:\n{output_text}"
            )

        # Parse the JSON output file
        output_path = Path(output_file)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                "llama-benchy did not produce JSON output. "
                "Check that the server is running and reachable."
            )

        raw_data = json.loads(output_path.read_text(encoding="utf-8"))
        return parse_json_output(raw_data)

    finally:
        # Clean up temp file
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Synchronous convenience wrapper
# ---------------------------------------------------------------------------

def run_llama_benchy_sync(
    base_url: str,
    model: str,
    **kwargs: Any,
) -> LlamaBenchyResult:
    """Synchronous wrapper around :func:`run_llama_benchy`."""
    return asyncio.run(run_llama_benchy(base_url, model, **kwargs))
