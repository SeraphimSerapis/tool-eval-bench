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

import httpx

from tool_eval_bench.utils.urls import chat_completions_url as _chat_url
from tool_eval_bench.utils.urls import models_url as models_url_fn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filler text for prompt padding
# ---------------------------------------------------------------------------

_FILLER_PARAGRAPH = (
    "In my younger and more vulnerable years my father gave me some advice "
    "that I have been turning over in my mind ever since. Whenever you feel "
    "like criticizing anyone, he told me, just remember that all the people "
    "in this world have not had the advantages that you have had. He did not "
    "say any more, but we have always been unusually communicative in a "
    "reserved way, and I understood that he meant a great deal more than that. "
    "In consequence I am inclined to reserve all judgements, a habit that has "
    "opened up many curious natures to me and also made me the victim of not "
    "a few veteran bores. The abnormal mind is quick to detect and attach "
    "itself to this quality when it appears in a normal person, and so it "
    "came about that in college I was unjustly accused of being a politician, "
    "because I was privy to the secret griefs of wild, unknown men. "
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

    def get_filler_pool(self, min_chars: int) -> str:
        """Return a cached filler text pool of at least min_chars length."""
        if self._filler_pool_chars >= min_chars:
            return self._filler_pool
        reps = max(1, min_chars // len(_FILLER_PARAGRAPH) + 1)
        self._filler_pool = _FILLER_PARAGRAPH * reps
        self._filler_pool_chars = len(self._filler_pool)
        return self._filler_pool


def _tokenize_url(base_url: str) -> str:
    """vLLM exposes /tokenize at the root, not under /v1."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return f"{b}/tokenize"


def _headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ---------------------------------------------------------------------------
# Exact prompt building via /tokenize
# ---------------------------------------------------------------------------

async def _tokenize_text(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    text: str,
    api_key: str | None,
) -> int | None:
    """Count tokens via server's /tokenize endpoint. Returns None if unavailable."""
    try:
        resp = await client.post(
            _tokenize_url(base_url),
            json={"model": model, "prompt": text},
            headers=_headers(api_key),
        )
        if resp.status_code == 200:
            data = resp.json()
            count = data.get("count") or len(data.get("tokens", []))
            return count if count > 0 else None
    except Exception:
        pass
    return None


async def _build_exact_prompt(
    client: httpx.AsyncClient,
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
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    api_key: str | None = None,
) -> TokenizerConfig:
    """Calibrate prompt building for accurate token counts.

    1. Try /tokenize endpoint for exact counts (preferred).
    2. Fall back to probe request + usage.prompt_tokens for ratio estimation.

    Returns a TokenizerConfig with the calibrated state.
    """
    cfg = TokenizerConfig()

    # Try /tokenize first
    probe_text = _FILLER_PARAGRAPH * 3
    token_count = await _tokenize_text(client, base_url, model, probe_text, api_key)
    if token_count and token_count > 10:
        cfg.has_tokenize_endpoint = True
        cfg.calibration_confidence = "tokenize"
        cfg.chars_per_token = len(probe_text) / token_count
        logger.info(
            "Calibrated via /tokenize: %.2f chars/token (%d chars → %d tokens)",
            cfg.chars_per_token, len(probe_text), token_count,
        )
        return cfg

    # Fallback: send a real request and read usage.prompt_tokens
    logger.info("/tokenize not available, falling back to probe request calibration")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": probe_text}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    try:
        resp = await client.post(
            _chat_url(base_url), json=payload, headers=_headers(api_key),
        )
        resp.raise_for_status()
        data = resp.json()
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        if prompt_tokens > 10:
            content_tokens = prompt_tokens - 4  # subtract chat template overhead
            cfg.chars_per_token = len(probe_text) / content_tokens
            cfg.calibration_confidence = "probe"
            logger.info(
                "Calibrated via probe: %.2f chars/token (%d chars → %d pt)",
                cfg.chars_per_token, len(probe_text), prompt_tokens,
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
    pp_tokens: int = 0              # prompt tokens (from server usage)
    tg_tokens: int = 0              # generated tokens (counted from stream)
    depth: int = 0                  # context depth (system-message tokens)
    concurrency: int = 1            # concurrent requests
    ttft_ms: float = 0.0            # time to first content token (ms)
    total_ms: float = 0.0           # total request time (ms)
    pp_tps: float = 0.0             # prefill tokens/sec
    tg_tps: float = 0.0             # token generation tokens/sec
    error: str | None = None        # error message if failed
    requested_pp: int = 0           # user-requested pp (for clean labels)
    requested_depth: int = 0        # user-requested depth (for clean labels)
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
    draft_n: int | None = None             # tokens drafted (llama.cpp timings)
    draft_n_accepted: int | None = None    # tokens accepted (llama.cpp timings)

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
    latency_ms: float = 0.0        # network/server latency estimate
    samples: list[ThroughputSample] = field(default_factory=list)
    # Set to True if speculative decoding was detected on the server during
    # this run. When True, the CLI will suggest running --spec-bench.
    spec_decoding_detected: bool = False
    spec_decoding_method: str = ""   # e.g. "mtp", "ngram", "eagle", or ""


# Callback type for throughput sample events
OnThroughputSample = Callable[[ThroughputSample, int, int], Awaitable[None]]
"""(sample, index, total) → called after each throughput measurement."""


# ---------------------------------------------------------------------------
# Latency estimation
# ---------------------------------------------------------------------------

async def estimate_latency(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None = None,
    rounds: int = 3,
) -> float:
    """Estimate network latency by calling /v1/models repeatedly.

    Returns median latency in milliseconds.
    """
    url = models_url_fn(base_url)
    hdrs = _headers(api_key)
    times: list[float] = []

    for _ in range(rounds):
        t0 = time.perf_counter()
        try:
            resp = await client.get(url, headers=hdrs)
            resp.raise_for_status()
        except Exception:
            continue
        times.append((time.perf_counter() - t0) * 1000)

    if not times:
        return 0.0
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

async def warmup(
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> float:
    """Send a trivial completion to prime the server.

    Returns elapsed time in milliseconds.

    Thinking/reasoning models (Qwen3.x, QwQ, etc.) may spend significant
    time in an internal chain-of-thought phase before emitting visible
    tokens.  Since warm-up only needs to prime the KV cache, we disable
    thinking via ``chat_template_kwargs`` and ``extra_body`` to keep the
    request fast and avoid read-timeout failures.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        # Disable thinking/reasoning to avoid long chain-of-thought delays.
        # Supported by vLLM (chat_template_kwargs) and other backends (extra_body).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            _chat_url(base_url), json=payload, headers=_headers(api_key),
        )
        resp.raise_for_status()
    return (time.perf_counter() - t0) * 1000
    # Note: warmup intentionally uses its own client since it runs
    # before the throughput sweep and has a much longer timeout.


# ---------------------------------------------------------------------------
# MTP-aware token counting helpers
# ---------------------------------------------------------------------------

_mtp_warned = False


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


# ---------------------------------------------------------------------------
# Single streaming measurement
# ---------------------------------------------------------------------------

async def _stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tg: int,
    api_key: str | None,
) -> ThroughputSample:
    """Execute a single streaming request and collect timing data.

    Timing strategy:
    - **first_byte_time**: captured right after HTTP headers arrive
      (``response.raise_for_status()``).  This marks the end of prefill /
      beginning of generation — i.e. true TTFT.
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
    falls back to ``end_of_stream - first_byte`` minus TTFT, ensuring a
    non-zero measurement even when the server flushes everything in one burst.
    """
    global _mtp_warned

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": tg,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Request token IDs for accurate MTP counting.  Ignored by servers
        # that don't support it — harmless extra field.
        "return_token_ids": True,
    }

    sample = ThroughputSample()
    t0 = time.perf_counter()
    first_byte_time: float | None = None
    first_content_time: float | None = None
    last_content_time: float = t0
    end_of_stream_time: float = t0
    stream_token_count = 0
    server_completion_tokens = 0
    prompt_tokens = 0
    token_timestamps: list[float] = []
    mtp_detected = False

    try:
        async with client.stream(
            "POST", _chat_url(base_url),
            json=payload, headers=_headers(api_key),
        ) as response:
            response.raise_for_status()
            # HTTP headers have arrived — server has finished prefill
            first_byte_time = time.perf_counter()

            async for raw_line in response.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:].strip()
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

                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if first_content_time is None:
                        first_content_time = now
                    last_content_time = now

                    # Count actual tokens in this chunk (MTP-aware)
                    chunk_tokens = _count_chunk_tokens(choices, content)
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
                        prev_ts = token_timestamps[-1] if token_timestamps else (first_content_time or t0)
                        time_window = now - prev_ts
                        for i in range(chunk_tokens):
                            ts = prev_ts + (time_window * (i + 1) / chunk_tokens)
                            token_timestamps.append(ts)

            # If [DONE] was never received (unusual), use last timestamp
            if end_of_stream_time == t0:
                end_of_stream_time = time.perf_counter()

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return ThroughputSample(error=str(exc), total_ms=elapsed)

    total_ms = (time.perf_counter() - t0) * 1000

    # Log MTP detection once
    if mtp_detected and not _mtp_warned:
        _mtp_warned = True
        logger.info(
            "Multi-token prediction (MTP) detected: SSE chunks contain "
            "multiple token_ids. Token counts and timing are MTP-aware."
        )

    # TTFT: prefer HTTP first-byte (most accurate prefill measurement),
    # fall back to first content chunk timestamp.
    if first_byte_time is not None:
        ttft_ms = (first_byte_time - t0) * 1000
    elif first_content_time is not None:
        ttft_ms = (first_content_time - t0) * 1000
    else:
        ttft_ms = total_ms

    # Use server-reported count when available, fall back to stream-counted
    # tokens (now MTP-aware — counts actual token_ids per chunk).
    generated_tokens = server_completion_tokens if server_completion_tokens > 0 else stream_token_count

    # Generation time: prefer inter-content-chunk timing (most precise when
    # the server streams one token per SSE event).  When MTP is active, we
    # use the interpolated token timestamps for a more accurate window.
    gen_ms = 0.0
    if len(token_timestamps) >= 2:
        gen_ms = (token_timestamps[-1] - token_timestamps[0]) * 1000
    elif first_content_time is not None and last_content_time > first_content_time:
        gen_ms = (last_content_time - first_content_time) * 1000

    # Fallback: use the window from first-byte (or first-content) to end of
    # stream.  This captures generation time even when all content arrives in
    # a single SSE chunk.
    if gen_ms <= 0 and generated_tokens > 0:
        ref_time = first_byte_time or first_content_time or t0
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
    client: httpx.AsyncClient,
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
                client, base_url, model, depth, api_key, tok_cfg, "system",
            )
        else:
            system_text = _build_filler_heuristic(depth, tok_cfg)
        messages.append({"role": "system", "content": system_text})
    else:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Continue the text provided by the user.",
        })

    if tok_cfg.has_tokenize_endpoint:
        user_text = await _build_exact_prompt(
            client, base_url, model, pp, api_key, tok_cfg, "user",
        )
    else:
        user_text = _build_filler_heuristic(pp, tok_cfg)
    messages.append({"role": "user", "content": user_text})

    return messages


async def measure_single(
    client: httpx.AsyncClient,
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
    sample = await _stream_one(client, base_url, model, messages, tg, api_key)
    sample.depth = depth
    sample.concurrency = 1
    sample.requested_pp = pp
    sample.requested_depth = depth
    return sample


async def measure_concurrent(
    client: httpx.AsyncClient,
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
            client, base_url, model, pp=pp, tg=tg, depth=depth,
            api_key=api_key, tok_cfg=tok_cfg,
        )

    # Build shared messages (exact token count)
    messages = await _build_messages(client, base_url, model, pp, depth, api_key, tok_cfg)

    # Launch N requests in parallel
    t0 = time.perf_counter()
    tasks = [
        _stream_one(client, base_url, model, messages, tg, api_key)
        for _ in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    total_wall_ms = (time.perf_counter() - t0) * 1000

    # Aggregate
    errors = [r for r in results if r.error]
    successes = [r for r in results if not r.error]

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

    # Warm up first
    warmup_ms = await warmup(base_url, model, api_key)

    # Shared client for all throughput measurements
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    ) as client:
        # Calibrate tokenizer ratio + detect /tokenize
        tok_cfg = await calibrate(client, base_url, model, api_key)
        method = "/tokenize" if tok_cfg.has_tokenize_endpoint else "probe"
        logger.info("Using %.2f chars/token (%s) for prompt building", tok_cfg.chars_per_token, method)

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
        except Exception:
            pass  # Detection is best-effort; never fail a throughput run

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
                client, base_url, model,
                pp=pp, tg=tg, depth=depth, concurrency=conc,
                api_key=api_key, tok_cfg=tok_cfg,
            )
            sample.calibration_confidence = tok_cfg.calibration_confidence
            result.samples.append(sample)

            if on_sample:
                await on_sample(sample, idx, total)

    return result
