"""Server warm-up and streaming throughput measurement.

Provides llama-bench style pp/tg measurement using SSE streaming:
- warmup(): Prime the server with a small request before benchmarking.
- measure_single(): One streaming request → TTFT + pp t/s + tg t/s.
- run_throughput_matrix(): Sweep over depth × concurrency.

Uses the server's /tokenize endpoint (vLLM) for exact token counts,
falling back to calibration-based estimation for other backends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from tool_eval_bench.domain.measurement import MeasurementClient, MeasurementClientFactory
from tool_eval_bench.utils.openai_compat import (
    max_tokens_retry_payload,
    output_token_limit_reached,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filler text for prompt padding
# ---------------------------------------------------------------------------

_FILLER_PARAGRAPH = (
    "The server processed each request methodically, allocating memory for the "
    "key-value cache before any token generation could begin. During the prefill "
    "phase, every input token was evaluated in parallel, building the internal "
    "representation that would guide subsequent predictions. The attention "
    "mechanism compared each position against all previous positions, computing "
    "weighted scores that determined which context was most relevant for the "
    "next output. As the sequence grew longer, the computational cost scaled "
    "quadratically with the number of tokens, making efficient batching and "
    "memory management essential for maintaining acceptable throughput under "
    "production workloads. The scheduling algorithm balanced latency-sensitive "
    "interactive requests against throughput-optimized batch completions. "
)

# Default fallback — gets overridden by calibration
_DEFAULT_CHARS_PER_TOKEN = 4.0


@dataclass
class TokenizerConfig:
    """Calibration state for prompt building.

    Encapsulates per-model tokenizer data so the module is reentrant
    (safe for multi-model comparison without recalibration bugs).
    """

    chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN
    has_tokenize_endpoint: bool = False
    _filler_pool: str = ""
    _filler_pool_chars: int = 0
    # Confidence level for token count accuracy:
    #   "tokenize"  — exact counts from /tokenize endpoint (best)
    #   "probe"     — ratio estimated from a real request
    #   "heuristic" — 4 chars/token fallback (may be off by 20-40% for non-English text)
    calibration_confidence: str = "heuristic"
    # OpenAI-compatible endpoints disagree on the output-token field name.
    output_token_field: str = "max_tokens"
    # Per-run flag: ensures MTP detection is logged once per calibration
    # context instead of using module-level mutable state.
    mtp_warned: bool = False
    # ``return_token_ids`` is a useful vLLM/SGLang extension, but strict
    # OpenAI-compatible servers reject unknown request fields with 400/422.
    # ``None`` means capability is unknown and the field is probed once.
    supports_return_token_ids: bool | None = None

    def get_filler_pool(self, min_chars: int) -> str:
        """Return a cached filler text pool of at least min_chars length."""
        if self._filler_pool_chars >= min_chars:
            return self._filler_pool
        reps = max(1, min_chars // len(_FILLER_PARAGRAPH) + 1)
        self._filler_pool = _FILLER_PARAGRAPH * reps
        self._filler_pool_chars = len(self._filler_pool)
        return self._filler_pool


# ---------------------------------------------------------------------------
# Exact prompt building via /tokenize
# ---------------------------------------------------------------------------


async def _tokenize_text(
    client: MeasurementClient,
    base_url: str,
    model: str,
    text: str,
    api_key: str | None,
) -> int | None:
    """Count tokens via server's /tokenize endpoint. Returns None if unavailable."""
    try:
        measurement = client
        resp = await measurement.tokenize(model=model, text=text)
        if resp.status_code == 200:
            data = resp.json()
            count = data.get("count") or len(data.get("tokens", []))
            return count if count > 0 else None
    except Exception:
        logger.debug("/tokenize unavailable — expected for non-vLLM backends")
    return None


async def _build_exact_prompt(
    client: MeasurementClient,
    base_url: str,
    model: str,
    target_tokens: int,
    api_key: str | None,
    tok_cfg: TokenizerConfig,
    role: str = "user",
) -> str:
    """Build a text that tokenizes to exactly target_tokens.

    Uses binary search with /tokenize to converge on the exact length.
    Falls back to heuristic if /tokenize is unavailable.
    """
    if not tok_cfg.has_tokenize_endpoint:
        return _build_filler_heuristic(target_tokens, tok_cfg)

    # Start with a rough estimate
    target_chars = int(target_tokens * tok_cfg.chars_per_token)

    # Get cached filler pool (at least 2× target to allow binary search)
    pool = tok_cfg.get_filler_pool(target_chars * 2)

    # Binary search for the right character count
    lo, hi = 0, len(pool)
    best_text = pool[:target_chars]

    for _ in range(12):  # max 12 iterations for convergence
        mid = (lo + hi) // 2
        candidate = pool[:mid]
        count = await _tokenize_text(client, base_url, model, candidate, api_key)
        if count is None:
            return _build_filler_heuristic(target_tokens, tok_cfg)

        if count == target_tokens:
            return candidate
        elif count < target_tokens:
            lo = mid + 1
            best_text = candidate  # closest-from-below
        else:
            hi = mid - 1
            # Don't update best_text — this candidate overshot

    return best_text


def _build_filler_heuristic(target_tokens: int, tok_cfg: TokenizerConfig | None = None) -> str:
    """Build a text string of approximately target_tokens tokens (heuristic)."""
    cpt = tok_cfg.chars_per_token if tok_cfg else _DEFAULT_CHARS_PER_TOKEN
    target_chars = int(target_tokens * cpt)
    if tok_cfg:
        pool = tok_cfg.get_filler_pool(target_chars)
    else:
        reps = max(1, target_chars // len(_FILLER_PARAGRAPH) + 1)
        pool = _FILLER_PARAGRAPH * reps
    return pool[:target_chars]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


async def calibrate(
    client: MeasurementClient,
    base_url: str,
    model: str,
    api_key: str | None = None,
    tok_cfg: TokenizerConfig | None = None,
) -> TokenizerConfig:
    """Calibrate prompt building for accurate token counts.

    1. Try /tokenize endpoint for exact counts (preferred).
    2. Fall back to probe request + usage.prompt_tokens for ratio estimation.

    Returns a TokenizerConfig with the calibrated state.
    """
    cfg = tok_cfg or TokenizerConfig()

    # Try /tokenize first
    probe_text = _FILLER_PARAGRAPH * 3
    token_count = await _tokenize_text(client, base_url, model, probe_text, api_key)
    if token_count and token_count > 10:
        cfg.has_tokenize_endpoint = True
        cfg.calibration_confidence = "tokenize"
        cfg.chars_per_token = len(probe_text) / token_count
        logger.info(
            "Calibrated via /tokenize: %.2f chars/token (%d chars → %d tokens)",
            cfg.chars_per_token,
            len(probe_text),
            token_count,
        )
        return cfg

    # Fallback: send a real request and read usage.prompt_tokens
    logger.info("/tokenize not available, falling back to probe request calibration")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": probe_text}],
        cfg.output_token_field: 1,
        "temperature": 0.0,
    }
    try:
        measurement = client
        resp = await measurement.completion(payload)
        retry_payload = max_tokens_retry_payload(payload, resp.status_code, resp.text)
        if retry_payload is not None:
            cfg.output_token_field = "max_completion_tokens"
            resp = await measurement.completion(retry_payload)
        resp.raise_for_status()
        data = resp.json()
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        if prompt_tokens > 10:
            content_tokens = prompt_tokens - 4  # subtract chat template overhead
            cfg.chars_per_token = len(probe_text) / content_tokens
            cfg.calibration_confidence = "probe"
            logger.info(
                "Calibrated via probe: %.2f chars/token (%d chars → %d pt)",
                cfg.chars_per_token,
                len(probe_text),
                prompt_tokens,
            )
            return cfg
    except Exception as exc:
        logger.warning("Calibration failed: %s — using default", exc)

    logger.warning(
        "Token calibration fell back to heuristic (%.1f chars/token). "
        "pp t/s figures may be inaccurate by 20-40%% for non-English models.",
        cfg.chars_per_token,
    )
    return cfg


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThroughputSample:
    """Result of a single streaming throughput measurement."""

    pp_tokens: int = 0  # prompt tokens (from server usage)
    tg_tokens: int = 0  # generated tokens (counted from stream)
    depth: int = 0  # context depth (system-message tokens)
    concurrency: int = 1  # concurrent requests
    ttft_ms: float = 0.0  # time to first content token (ms)
    total_ms: float = 0.0  # total request time (ms)
    pp_tps: float = 0.0  # prefill tokens/sec
    tg_tps: float = 0.0  # token generation tokens/sec
    error: str | None = None  # error message if failed
    requested_pp: int = 0  # user-requested pp (for clean labels)
    requested_depth: int = 0  # user-requested depth (for clean labels)
    # Calibration confidence — propagated from TokenizerConfig so reports can warn
    # when token counts are heuristic estimates rather than exact measurements.
    calibration_confidence: str = "heuristic"  # "tokenize" | "probe" | "heuristic"
    # Per-token timestamps for accurate peak t/s calculation.
    # Each entry is the (real or interpolated) perf_counter time a token arrived.
    # Populated by _stream_one when token_ids are available in SSE chunks.
    token_timestamps: list[float] = field(default_factory=list)
    # Whether multi-token chunks were detected (MTP / speculative decoding).
    mtp_chunks_detected: bool = False
    # Per-request speculative decoding stats from llama.cpp timings object.
    # llama.cpp doesn't expose spec decode counters in Prometheus /metrics;
    # instead it embeds draft_n/draft_n_accepted in the response timings.
    draft_n: int | None = None  # tokens drafted (llama.cpp timings)
    draft_n_accepted: int | None = None  # tokens accepted (llama.cpp timings)

    @property
    def effective_tg_tps(self) -> float:
        """Output tokens ÷ wall-clock generation time.

        Unlike ``tg_tps`` (measured from stream inter-token timing), this
        metric captures the full benefit of speculative decoding and MTP
        since it measures what the user actually experiences.  For standard
        autoregressive decoding the two values should be similar; for
        spec-decode-enabled servers ``effective_tg_tps`` can be significantly
        higher.
        """
        if self.total_ms > 0 and self.tg_tokens > 0:
            gen_ms = self.total_ms - self.ttft_ms if self.ttft_ms > 0 else self.total_ms
            if gen_ms > 0:
                return self.tg_tokens / (gen_ms / 1000)
        return 0.0

    @property
    def peak_tg_tps(self) -> float:
        """Peak token generation t/s over any 1-second sliding window.

        Requires ``token_timestamps`` to be populated.  Returns 0.0 if
        fewer than 2 timestamps are available.
        """
        if len(self.token_timestamps) < 2:
            return 0.0
        ts = sorted(self.token_timestamps)
        total_dur = ts[-1] - ts[0]
        if total_dur <= 0:
            return 0.0
        # If the entire generation fits in < 1 second, use actual duration
        if total_dur < 1.0:
            return len(ts) / total_dur
        max_tokens = 0
        start_idx = 0
        for end_idx in range(len(ts)):
            while start_idx < end_idx and ts[start_idx] <= ts[end_idx] - 1.0:
                start_idx += 1
            count = end_idx - start_idx + 1
            if count > max_tokens:
                max_tokens = count
        return float(max_tokens)  # tokens in best 1-second window

    @property
    def label_pp(self) -> int:
        """PP value for display labels: requested value if set, else actual."""
        return self.requested_pp if self.requested_pp else self.pp_tokens

    @property
    def label_depth(self) -> int:
        """Depth value for display labels: requested value if set, else actual."""
        return self.requested_depth if self.requested_depth else self.depth


@dataclass
class ThroughputMatrixResult:
    """Full matrix sweep result."""

    model: str = ""
    backend: str = ""
    base_url: str = ""
    warmup_ms: float = 0.0
    latency_ms: float = 0.0  # network/server latency estimate
    samples: list[ThroughputSample] = field(default_factory=list)
    # Set to True if speculative decoding was detected on the server during
    # this run. When True, the CLI will suggest running --spec-bench.
    spec_decoding_detected: bool = False
    spec_decoding_method: str = ""  # e.g. "mtp", "ngram", "eagle", or ""


# Callback type for throughput sample events
OnThroughputSample = Callable[[ThroughputSample, int, int], Awaitable[None]]
"""(sample, index, total) → called after each throughput measurement."""


# ---------------------------------------------------------------------------
# Latency estimation
# ---------------------------------------------------------------------------


async def estimate_latency(
    client: MeasurementClient,
    base_url: str,
    api_key: str | None = None,
    rounds: int = 3,
) -> float:
    """Estimate network latency by calling /v1/models repeatedly.

    Returns median latency in milliseconds.
    """
    measurement = client
    times: list[float] = []

    for _ in range(rounds):
        t0 = time.perf_counter()
        try:
            resp = await measurement.models()
            resp.raise_for_status()
        except Exception:
            logger.debug("Latency probe round failed, skipping")
            continue  # latency probe failure — skip this round
        times.append((time.perf_counter() - t0) * 1000)

    if not times:
        return 0.0
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


WARMUP_MAX_TOKENS = 4

# Thinking/reasoning models (Qwen3.x, QwQ, Gemini 3, …) can spend a long time in
# an internal chain-of-thought phase before emitting visible tokens.  Warm-up
# only needs to prime the KV cache, so it asks for thinking to be turned off.
# vLLM and friends read ``chat_template_kwargs``; the Gemini adapter maps the
# same key onto ``generationConfig.thinkingConfig``.
WARMUP_EXTRA_PARAMS: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

# Hints that speed warm-up up but are not part of any endpoint's contract.
# Strict endpoints (notably Gemini's OpenAI-compatibility layer) reject unknown
# request fields outright with HTTP 400 rather than ignoring them.
_OPTIONAL_WARMUP_KEYS = ("chat_template_kwargs", "extra_body")
_REJECTION_STATUS_CODES = (400, 422)


def _without_optional_hints(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Strip speed hints from a warm-up payload, or return None if there are none."""
    stripped = {k: v for k, v in payload.items() if k not in _OPTIONAL_WARMUP_KEYS}
    config = payload.get("generationConfig")
    if isinstance(config, dict) and "thinkingConfig" in config:
        stripped["generationConfig"] = {k: v for k, v in config.items() if k != "thinkingConfig"}
    return stripped if stripped != payload else None


async def warmup(
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 120.0,
    *,
    client_factory: MeasurementClientFactory,
    request: tuple[str, dict[str, Any], dict[str, str]] | None = None,
    tok_cfg: TokenizerConfig | None = None,
) -> float:
    """Send a trivial completion to prime the server.

    Returns elapsed time in milliseconds.

    ``request`` overrides the (url, payload, headers) triple, so callers that
    know the endpoint's wire format can supply a non-OpenAI request.  If the
    endpoint rejects the payload outright, warm-up retries once without the
    optional speed hints before giving up.
    """
    if request is None:
        completion_url = None
        completion_headers = None
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello."},
            ],
            "max_tokens": WARMUP_MAX_TOKENS,
            "temperature": 0.0,
            **WARMUP_EXTRA_PARAMS,
        }
    else:
        completion_url, payload, completion_headers = request

    t0 = time.perf_counter()
    async with client_factory(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        completion_url=completion_url,
        completion_headers=completion_headers,
    ) as client:
        measurement = client
        for _ in range(3):
            resp = await measurement.completion(payload)
            if output_token_limit_reached(resp.status_code, resp.text):
                logger.debug("Warm-up reached the model's output-token limit")
                break
            if resp.status_code not in _REJECTION_STATUS_CODES:
                break
            retry_payload = max_tokens_retry_payload(payload, resp.status_code, resp.text)
            if retry_payload is not None:
                if tok_cfg is not None:
                    tok_cfg.output_token_field = "max_completion_tokens"
                logger.debug("Warm-up rejected max_tokens; retrying with max_completion_tokens")
            else:
                retry_payload = _without_optional_hints(payload)
                if retry_payload is None:
                    break
                logger.debug(
                    "Warm-up rejected with HTTP %s; retrying without optional hints",
                    resp.status_code,
                )
            payload = retry_payload
        if not output_token_limit_reached(resp.status_code, resp.text):
            resp.raise_for_status()
    return (time.perf_counter() - t0) * 1000
    # Note: warmup intentionally uses its own client since it runs
    # before the throughput sweep and has a much longer timeout.


# ---------------------------------------------------------------------------
# MTP-aware token counting helpers
# ---------------------------------------------------------------------------


def _count_chunk_tokens(
    choices: list[dict[str, Any]],
    content: str,
) -> int:
    """Count actual tokens in a single SSE chunk.

    Strategy (mirrors llama-benchy's approach):
    1. If the server includes ``token_ids`` in the choice, use its length
       for an exact count.  vLLM/SGLang send this when ``return_token_ids``
       is in the request.
    2. Fall back to 1 token per chunk (standard autoregressive streaming).

    Returns the number of tokens in this chunk (≥ 1).
    """
    if choices:
        token_ids = choices[0].get("token_ids")
        if token_ids and isinstance(token_ids, list):
            return len(token_ids)
    return 1


def _sse_payload(line: str) -> str | None:
    """Return the payload from an SSE data line, with or without a space."""
    if not line.startswith("data:"):
        return None
    payload = line[5:]
    return payload[1:] if payload.startswith(" ") else payload


async def _completion_lines(response: Any) -> Any:
    """Yield SSE lines, or normalize a non-streaming JSON response once."""
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        yield f"data: {json.dumps(response.json())}"
        return
    async for line in response.aiter_lines():
        yield line


# ---------------------------------------------------------------------------
# Single streaming measurement
# ---------------------------------------------------------------------------


async def _stream_one(
    client: MeasurementClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tg: int,
    api_key: str | None,
    tok_cfg: TokenizerConfig | None = None,
    _include_token_ids: bool | None = None,
) -> ThroughputSample:
    """Execute a single streaming request and collect timing data.

    Timing strategy:
    - **first_usable_time**: timestamp of the first SSE delta carrying generated
      content, reasoning, or a tool call.  HTTP headers can arrive before model
      output and are intentionally not used as TTFT.
    - **first_content_time / last_content_time**: timestamps of the first and
      last SSE chunks that carry ``delta.content``.  When the server streams
      token-by-token, ``last - first`` gives accurate inter-token generation
      time.
    - **end_of_stream_time**: timestamp of the ``[DONE]`` sentinel (or the
      last data line).  Used as a fallback generation window when all content
      arrives in a single chunk (``first == last``).

    Multi-token prediction (MTP) handling:
    - Reads ``token_ids`` from each SSE chunk to determine how many tokens
      arrived per event (1 for standard AR, 2-4+ for MTP/DFlash).
    - Interpolates per-token timestamps within multi-token chunks so that
      ``tg_tps`` and ``peak_tg_tps`` reflect the true generation speed.

    The generation speed (``tg_tps``) prefers inter-content-chunk timing, then
    falls back to ``end_of_stream - first_usable_time``, ensuring a non-zero
    measurement even when the server flushes everything in one burst.
    """
    include_token_ids = (
        _include_token_ids
        if _include_token_ids is not None
        else tok_cfg is None or tok_cfg.supports_return_token_ids is not False
    )
    payload = {
        "model": model,
        "messages": messages,
        (tok_cfg.output_token_field if tok_cfg else "max_tokens"): tg,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if include_token_ids:
        # Request token IDs for accurate MTP counting.  This is a vLLM/SGLang
        # extension, so strict OpenAI-compatible servers may reject it.
        payload["return_token_ids"] = True

    sample = ThroughputSample()
    t0 = time.perf_counter()
    first_usable_time: float | None = None
    first_content_time: float | None = None
    last_content_time: float = t0
    end_of_stream_time: float = t0
    stream_token_count = 0
    server_completion_tokens = 0
    prompt_tokens = 0
    token_timestamps: list[float] = []
    mtp_detected = False

    measurement = client
    try:
        async with measurement.stream_completion(payload) as response:
            response.raise_for_status()

            async for raw_line in _completion_lines(response):
                data_payload = _sse_payload(raw_line)
                if data_payload is None:
                    continue
                data_str = data_payload.strip()
                if data_str == "[DONE]":
                    end_of_stream_time = time.perf_counter()
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check for usage in final chunk
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    server_completion_tokens = usage.get("completion_tokens", 0)

                # llama.cpp embeds speculative decoding stats in timings
                timings = chunk.get("timings")
                if timings and isinstance(timings, dict):
                    dn = timings.get("draft_n")
                    dna = timings.get("draft_n_accepted")
                    if dn is not None:
                        sample.draft_n = int(dn)
                    if dna is not None:
                        sample.draft_n_accepted = int(dna)

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                # Some OpenAI-compatible endpoints accept ``stream=true``
                # but return a normal JSON completion. Treat its ``message``
                # as one generated event instead of scoring zero tokens.
                delta = choices[0].get("delta") or choices[0].get("message", {})
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                tool_delta = delta.get("tool_calls")
                generated_text = content or reasoning
                if first_usable_time is None and (generated_text or tool_delta):
                    first_usable_time = time.perf_counter()
                if generated_text:
                    now = time.perf_counter()
                    if first_content_time is None:
                        first_content_time = now
                    last_content_time = now

                    # Count actual tokens in this chunk (MTP-aware)
                    chunk_tokens = _count_chunk_tokens(choices, generated_text)
                    stream_token_count += chunk_tokens

                    if chunk_tokens > 1:
                        mtp_detected = True

                    # Build per-token timestamps.
                    # Single-token chunks get the chunk arrival time directly.
                    # Multi-token chunks (MTP) get interpolated timestamps
                    # spread evenly across the time window since the last
                    # recorded timestamp.
                    if chunk_tokens == 1:
                        token_timestamps.append(now)
                    else:
                        prev_ts = (
                            token_timestamps[-1] if token_timestamps else (first_content_time or t0)
                        )
                        time_window = now - prev_ts
                        for i in range(chunk_tokens):
                            ts = prev_ts + (time_window * (i + 1) / chunk_tokens)
                            token_timestamps.append(ts)

            # If [DONE] was never received (unusual), use last timestamp
            if end_of_stream_time == t0:
                end_of_stream_time = time.perf_counter()

    except Exception as exc:
        # The measurement port deliberately does not expose a concrete HTTP
        # exception type. Adapters may surface native errors that carry a
        # response with the standard status/body facts below.
        error_response: Any = getattr(exc, "response", None)
        if error_response is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return ThroughputSample(error=str(exc), total_ms=elapsed)
        await error_response.aread()
        retry_payload = max_tokens_retry_payload(
            payload, error_response.status_code, error_response.text
        )
        if retry_payload is not None and tok_cfg is not None:
            tok_cfg.output_token_field = "max_completion_tokens"
            return await _stream_one(client, base_url, model, messages, tg, api_key, tok_cfg)
        # ``return_token_ids`` is not part of the OpenAI API contract.  Probe
        # it once, then remember the endpoint cannot accept it for this run.
        # Retry the same request without the extension on strict 4xx responses,
        # even when the server's error body does not name the rejected field.
        if include_token_ids and error_response.status_code in _REJECTION_STATUS_CODES:
            if tok_cfg is not None:
                tok_cfg.supports_return_token_ids = False
            return await _stream_one(
                client,
                base_url,
                model,
                messages,
                tg,
                api_key,
                tok_cfg,
                _include_token_ids=False,
            )
        elapsed = (time.perf_counter() - t0) * 1000
        return ThroughputSample(error=str(exc), total_ms=elapsed)
    total_ms = (time.perf_counter() - t0) * 1000

    # Log MTP detection once per run (via tok_cfg to avoid module-level state)
    if mtp_detected and tok_cfg is not None and not tok_cfg.mtp_warned:
        tok_cfg.mtp_warned = True
        logger.info(
            "Multi-token prediction (MTP) detected: SSE chunks contain "
            "multiple token_ids. Token counts and timing are MTP-aware."
        )

    # TTFT must end at the first generated content, reasoning, or tool delta.
    # HTTP headers can arrive before the server flushes any model output, so
    # measuring them understates prefill latency on buffered or delayed streams.
    if first_usable_time is not None:
        ttft_ms = (first_usable_time - t0) * 1000
    else:
        ttft_ms = total_ms

    # Use server-reported count when available, fall back to stream-counted
    # tokens (now MTP-aware — counts actual token_ids per chunk).
    generated_tokens = (
        server_completion_tokens if server_completion_tokens > 0 else stream_token_count
    )

    # Generation time: prefer inter-content-chunk timing (most precise when
    # the server streams one token per SSE event).  When MTP is active, we
    # use the interpolated token timestamps for a more accurate window.
    gen_ms = 0.0
    if len(token_timestamps) >= 2:
        gen_ms = (token_timestamps[-1] - token_timestamps[0]) * 1000
    elif first_content_time is not None and last_content_time > first_content_time:
        gen_ms = (last_content_time - first_content_time) * 1000

    # Fallback: use the window from first usable output to end of
    # stream.  This captures generation time even when all content arrives in
    # a single SSE chunk.
    if gen_ms <= 0 and generated_tokens > 0:
        ref_time = first_usable_time or first_content_time or t0
        gen_ms = (end_of_stream_time - ref_time) * 1000

    sample.pp_tokens = prompt_tokens
    sample.tg_tokens = generated_tokens
    sample.ttft_ms = ttft_ms
    sample.total_ms = total_ms
    sample.pp_tps = (prompt_tokens / (ttft_ms / 1000)) if ttft_ms > 0 and prompt_tokens > 0 else 0
    sample.token_timestamps = token_timestamps
    sample.mtp_chunks_detected = mtp_detected

    # Compute tg_tps from the best available generation timing
    if gen_ms > 0 and generated_tokens > 1:
        sample.tg_tps = (generated_tokens - 1) / (gen_ms / 1000)
    elif gen_ms > 0 and generated_tokens > 0:
        sample.tg_tps = generated_tokens / (gen_ms / 1000)
    return sample


async def _build_messages(
    client: MeasurementClient,
    base_url: str,
    model: str,
    pp: int,
    depth: int,
    api_key: str | None,
    tok_cfg: TokenizerConfig,
) -> list[dict[str, Any]]:
    """Build the messages list with exact token counts where possible."""
    messages: list[dict[str, Any]] = []

    if depth > 0:
        if tok_cfg.has_tokenize_endpoint:
            system_text = await _build_exact_prompt(
                client,
                base_url,
                model,
                depth,
                api_key,
                tok_cfg,
                "system",
            )
        else:
            system_text = _build_filler_heuristic(depth, tok_cfg)
        messages.append({"role": "system", "content": system_text})
    else:
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant. Continue the text provided by the user.",
            }
        )

    if tok_cfg.has_tokenize_endpoint:
        user_text = await _build_exact_prompt(
            client,
            base_url,
            model,
            pp,
            api_key,
            tok_cfg,
            "user",
        )
    else:
        user_text = _build_filler_heuristic(pp, tok_cfg)
    messages.append({"role": "user", "content": user_text})

    return messages


async def measure_single(
    client: MeasurementClient,
    base_url: str,
    model: str,
    *,
    pp: int = 2048,
    tg: int = 128,
    depth: int = 0,
    api_key: str | None = None,
    tok_cfg: TokenizerConfig | None = None,
) -> ThroughputSample:
    """Measure throughput for a single configuration point."""
    tok_cfg = tok_cfg or TokenizerConfig()
    messages = await _build_messages(client, base_url, model, pp, depth, api_key, tok_cfg)
    sample = await _stream_one(client, base_url, model, messages, tg, api_key, tok_cfg)
    sample.depth = depth
    sample.concurrency = 1
    sample.requested_pp = pp
    sample.requested_depth = depth
    return sample


async def measure_concurrent(
    client: MeasurementClient,
    base_url: str,
    model: str,
    *,
    pp: int = 2048,
    tg: int = 128,
    depth: int = 0,
    concurrency: int = 1,
    api_key: str | None = None,
    tok_cfg: TokenizerConfig | None = None,
) -> ThroughputSample:
    """Measure aggregate throughput with N concurrent requests."""
    tok_cfg = tok_cfg or TokenizerConfig()
    if concurrency <= 1:
        return await measure_single(
            client,
            base_url,
            model,
            pp=pp,
            tg=tg,
            depth=depth,
            api_key=api_key,
            tok_cfg=tok_cfg,
        )

    # Build shared messages (exact token count)
    messages = await _build_messages(client, base_url, model, pp, depth, api_key, tok_cfg)

    # Launch N requests in parallel
    t0 = time.perf_counter()
    tasks = [
        _stream_one(client, base_url, model, messages, tg, api_key, tok_cfg)
        for _ in range(concurrency)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    # Convert any bare exceptions to error samples
    results: list[ThroughputSample] = [
        r
        if isinstance(r, ThroughputSample)
        else ThroughputSample(error=str(r), concurrency=concurrency)
        for r in raw_results
    ]
    total_wall_ms = (time.perf_counter() - t0) * 1000

    # Aggregate
    errors: list[ThroughputSample] = [r for r in results if r.error]
    successes: list[ThroughputSample] = [r for r in results if not r.error]

    if not successes:
        error_msg = "; ".join(e.error or "?" for e in errors[:3])
        return ThroughputSample(error=error_msg, concurrency=concurrency)

    # Average per-request metrics
    avg_ttft = sum(s.ttft_ms for s in successes) / len(successes)
    total_tg_tokens = sum(s.tg_tokens for s in successes)
    avg_pp_tokens = sum(s.pp_tokens for s in successes) / len(successes)

    # Aggregate throughput: total tokens / wall clock time
    total_tg_tps = (total_tg_tokens / (total_wall_ms / 1000)) if total_wall_ms > 0 else 0

    return ThroughputSample(
        pp_tokens=int(avg_pp_tokens),
        tg_tokens=total_tg_tokens,
        depth=depth,
        concurrency=concurrency,
        ttft_ms=avg_ttft,
        total_ms=total_wall_ms,
        pp_tps=sum(s.pp_tps for s in successes) / len(successes),
        tg_tps=total_tg_tps,  # aggregate throughput
        requested_pp=pp,
        requested_depth=depth,
    )


# ---------------------------------------------------------------------------
# Matrix sweep
# ---------------------------------------------------------------------------


async def run_throughput_matrix(
    base_url: str,
    model: str,
    *,
    pp: int = 2048,
    tg: int = 128,
    depths: list[int] | None = None,
    concurrency_levels: list[int] | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    client_factory: MeasurementClientFactory,
    on_sample: OnThroughputSample | None = None,
) -> ThroughputMatrixResult:
    """Run the full depth × concurrency sweep.

    Sweep order: all concurrency levels at depth 0, then depth 1, etc.
    Within each depth: c1, c2, c4... (ascending concurrency).

    Calls `on_sample(sample, idx, total)` after each measurement.

    Also probes /metrics for speculative decoding counters. If spec decode is
    active, ThroughputMatrixResult.spec_decoding_detected is set to True and
    the CLI will suggest running --spec-bench for acceptance rate metrics.
    """
    depths = depths or [0, 4096, 8192]
    concurrency_levels = concurrency_levels or [1, 2, 4]

    # Warm up first and retain any endpoint capability learned from its response.
    tok_cfg = TokenizerConfig()
    warmup_ms = await warmup(
        base_url, model, api_key, client_factory=client_factory, tok_cfg=tok_cfg
    )

    # Shared client for all throughput measurements
    async with client_factory(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_connections=50,
        max_keepalive_connections=20,
    ) as client:
        # Calibrate tokenizer ratio + detect /tokenize
        tok_cfg = await calibrate(client, base_url, model, api_key, tok_cfg)
        method = "/tokenize" if tok_cfg.has_tokenize_endpoint else "probe"
        logger.info(
            "Using %.2f chars/token (%s) for prompt building", tok_cfg.chars_per_token, method
        )

        # Probe for speculative decoding (async, non-blocking — best effort)
        spec_detected = False
        spec_method_name = ""
        try:
            from tool_eval_bench.runner.speculative import detect_spec_decoding

            spec_info = await detect_spec_decoding(client, base_url, api_key)
            if spec_info.active:
                spec_detected = True
                spec_method_name = spec_info.method
                logger.warning(
                    "Speculative decoding detected on server (method: %s). "
                    "Standard tg t/s under-reports real throughput for spec-decode models. "
                    "Run with --spec-bench to measure acceptance rate and effective t/s.",
                    spec_method_name or "auto",
                )
            elif spec_info.has_per_request_timings:
                # llama.cpp detected — spec decode may be active but we can't
                # confirm from /metrics alone. Hint the user.
                logger.info(
                    "llama.cpp backend detected. If speculative decoding is enabled, "
                    "run with --spec-bench --spec-method=mtp to measure acceptance rate "
                    "and effective t/s via per-request timings.",
                )
        except Exception as exc:
            logger.debug("Spec decode detection during throughput: %s", exc)

        # Estimate latency
        latency_ms = await estimate_latency(client, base_url, api_key)

        result = ThroughputMatrixResult(
            model=model,
            base_url=base_url,
            warmup_ms=warmup_ms,
            latency_ms=latency_ms,
            spec_decoding_detected=spec_detected,
            spec_decoding_method=spec_method_name,
        )

        # Sweep: concurrency ascending within each depth
        combos = [(d, c) for d in depths for c in sorted(concurrency_levels)]
        total = len(combos)

        for idx, (depth, conc) in enumerate(combos):
            sample = await measure_concurrent(
                client,
                base_url,
                model,
                pp=pp,
                tg=tg,
                depth=depth,
                concurrency=conc,
                api_key=api_key,
                tok_cfg=tok_cfg,
            )
            sample.calibration_confidence = tok_cfg.calibration_confidence
            result.samples.append(sample)

            if on_sample:
                await on_sample(sample, idx, total)

    return result
