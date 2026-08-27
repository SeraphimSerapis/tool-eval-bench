"""Context pressure: pre-fill the conversation window with filler turns.

Simulates real-world agentic conversations where the context is already
heavily utilized before the model needs to make tool-calling decisions.

Usage::

    pressure = await prepare_context_pressure(
        base_url, model, api_key, ratio=0.75,
        client_factory=client_factory,
    )
    messages = build_pressure_messages(pressure)
    # Prepend these messages before the real scenario messages.

``client_factory`` is supplied by the application composition layer.

Auto-detection strategy for context size:
  1. ``/v1/models`` → ``max_model_len`` (vLLM)
  2. ``/v1/models`` → ``context_window`` or ``max_tokens`` (LiteLLM / others)
  3. Fall back to ``--context-size`` CLI override (required if auto-detect fails)
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from tool_eval_bench.domain.filler import (
    CHARS_PER_TOKEN_ESTIMATE,
    FILLER_PARAGRAPHS,
    build_filler_text,
)
from tool_eval_bench.domain.measurement import MeasurementClientFactory
from tool_eval_bench.domain.models import ChatMessage

logger = logging.getLogger(__name__)

# Short assistant acknowledgements for alternating turns
_ASSISTANT_RESPONSES = [
    "Understood. I've reviewed the background context you provided. Please continue.",
    "Thank you for the additional context. I'm ready for your next request.",
    "Got it. I've noted all the details. What would you like me to help with?",
    "I see. I've taken all of that into account. Please go ahead.",
    "Acknowledged. I have the full context now. How can I assist you?",
]

# Conservative defaults
_RESERVED_FOR_OUTPUT = 4096  # max_tokens for generation
_RESERVED_FOR_SCENARIO = 12000  # tool definitions + system prompt + user message +
# multi-turn conversation growth + token estimation
# margin.  The server counts tool schemas against
# the context window — the 52-tool LARGE_TOOLSET
# alone is ~6000 tokens.  The extra margin (~4K)
# absorbs char→token estimation error so that
# ratio=1.0 can still succeed.
_TOKENS_PER_FILLER_CHUNK = 2048


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ContextPressureConfig:
    """Configuration for context pressure pre-filling."""

    ratio: float = 0.75  # Fill this fraction of the available context
    context_size: int | None = None  # Override auto-detected context size
    fill_tokens: int = 0  # Computed: actual tokens to fill
    detected_context: int = 0  # Actual context size used

    def summary(self) -> str:
        """Human-readable summary for display."""
        pct = int(self.ratio * 100)
        fill_k = self.fill_tokens / 1024
        ctx_k = self.detected_context / 1024
        return (
            f"{pct}% of available fill budget "
            f"(~{fill_k:.0f}K tokens prefilled in {ctx_k:.0f}K context)"
        )

    def budget_breakdown(self, *, tool_tokens: int = 0) -> dict[str, int]:
        """Return a consistent token budget breakdown for display/reporting."""
        scenario_budget = self.detected_context - self.fill_tokens - _RESERVED_FOR_OUTPUT
        remaining_headroom = scenario_budget - tool_tokens
        return {
            "fill_tokens": self.fill_tokens,
            "tool_tokens": tool_tokens,
            "output_tokens": _RESERVED_FOR_OUTPUT,
            "scenario_budget_tokens": scenario_budget,
            "remaining_headroom_tokens": remaining_headroom,
        }


# ---------------------------------------------------------------------------
# Context size detection
# ---------------------------------------------------------------------------


# Regex to extract num_gpu_blocks and block_size from vllm:cache_config_info
_CACHE_CONFIG_RE = re.compile(
    r"^vllm:cache_config_info\{[^}]*"
    r'num_gpu_blocks="(\d+)"'
    r"[^}]*?"
    r'block_size="(\d+)"'
    r"[^}]*\}",
    re.MULTILINE,
)

# Fallback: try the reverse label order (block_size before num_gpu_blocks)
_CACHE_CONFIG_RE_ALT = re.compile(
    r"^vllm:cache_config_info\{[^}]*"
    r'block_size="(\d+)"'
    r"[^}]*?"
    r'num_gpu_blocks="(\d+)"'
    r"[^}]*\}",
    re.MULTILINE,
)

# Regex to detect hybrid-attention models via mamba_cache_mode label.
# For hybrid models (e.g. Qwen3.6-35B-A3B), vLLM's hybrid KV cache manager
# maps physical blocks to larger logical token coverage because only a subset
# of layers use standard Transformer KV cache.  In that case,
# num_gpu_blocks × block_size is *physical* block capacity, NOT effective
# max context length.  mamba_cache_mode="none" → standard full-attention;
# any other value (e.g. "align") → hybrid model.
_MAMBA_CACHE_MODE_RE = re.compile(
    r'mamba_cache_mode="([^"]+)"',
)


@dataclass(frozen=True)
class KvCapacityInfo:
    """Result of KV cache capacity detection from vLLM /metrics."""

    capacity: int
    """Physical block capacity in tokens (num_gpu_blocks × block_size)."""
    num_blocks: int
    block_size: int
    is_hybrid: bool
    """True when the model uses hybrid attention (mamba/linear + full).

    For hybrid models, ``capacity`` is NOT the effective max context
    length — vLLM's hybrid KV cache manager maps physical blocks to
    larger logical token coverage.  KV capping should be skipped.
    """


async def detect_kv_capacity(
    base_url: str,
    api_key: str | None = None,
    metrics_url: str | None = None,
    *,
    client_factory: MeasurementClientFactory,
) -> KvCapacityInfo | None:
    """Detect KV cache info from vLLM Prometheus /metrics.

    Parses ``vllm:cache_config_info`` to extract ``num_gpu_blocks``,
    ``block_size``, and ``mamba_cache_mode``.

    For **standard full-attention** models the physical block capacity
    (``num_gpu_blocks × block_size``) equals the effective max context.
    For **hybrid-attention** models (linear/mamba + full attention),
    vLLM's hybrid KV cache manager maps physical blocks to larger
    logical token coverage, so the physical capacity is NOT the
    effective max context.  The ``is_hybrid`` flag indicates this.

    Returns a :class:`KvCapacityInfo`, or ``None`` if detection fails
    (non-vLLM servers, metrics endpoint unavailable, etc.).
    """
    try:
        async with client_factory(base_url=base_url, api_key=api_key, timeout=10.0) as client:
            resp = await client.metrics(metrics_url=metrics_url)
            if resp.status_code != 200:
                logger.debug("KV capacity detection: /metrics returned %d", resp.status_code)
                return None
            text = resp.text
    except Exception as exc:
        logger.debug("KV capacity detection failed: %s", exc)
        return None

    # Try primary label order, then fallback
    match = _CACHE_CONFIG_RE.search(text)
    if match:
        num_blocks = int(match.group(1))
        block_size = int(match.group(2))
    else:
        match = _CACHE_CONFIG_RE_ALT.search(text)
        if match:
            block_size = int(match.group(1))
            num_blocks = int(match.group(2))
        else:
            logger.debug("No vllm:cache_config_info found in /metrics")
            return None

    if num_blocks <= 0 or block_size <= 0:
        return None

    # Detect hybrid-attention models via mamba_cache_mode.
    # match.string is the full /metrics text; search within the matched line.
    is_hybrid = False
    mamba_match = _MAMBA_CACHE_MODE_RE.search(match.group(0))
    if mamba_match:
        mode = mamba_match.group(1)
        is_hybrid = mode not in ("none", "None")
        if is_hybrid:
            logger.info(
                "Hybrid-attention model detected (mamba_cache_mode=%s); "
                "physical block capacity is not the effective max context",
                mode,
            )

    capacity = num_blocks * block_size
    logger.info(
        "Detected KV cache: %d blocks × %d block_size = %d physical token slots%s",
        num_blocks,
        block_size,
        capacity,
        " (hybrid — not capping)" if is_hybrid else "",
    )
    return KvCapacityInfo(
        capacity=capacity,
        num_blocks=num_blocks,
        block_size=block_size,
        is_hybrid=is_hybrid,
    )


async def detect_context_size(
    base_url: str,
    model: str,
    api_key: str | None = None,
    *,
    client_factory: MeasurementClientFactory,
) -> int | None:
    """Auto-detect context window size from /v1/models.

    Tries multiple fields in order of preference:
      - max_model_len (vLLM)
      - context_window (LiteLLM)
      - max_tokens (generic)

    Returns the context size in tokens, or None if detection fails.
    """
    try:
        async with client_factory(base_url=base_url, api_key=api_key, timeout=10.0) as client:
            resp = await client.models()
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.debug("Context size detection failed: %s", exc)
        return None

    model_list = data.get("data", [])
    if not model_list:
        return None

    # Find the matching model entry (or use the first one)
    target = None
    for m in model_list:
        if m.get("id") == model:
            target = m
            break
    if target is None:
        target = model_list[0]

    # Try known fields in order of preference
    for field_name in ("max_model_len", "context_window", "max_tokens"):
        val = target.get(field_name)
        if isinstance(val, int) and val > 0:
            logger.info("Detected context size: %d tokens (via %s)", val, field_name)
            return val

    logger.debug("No context size field found in model metadata")
    return None


# ---------------------------------------------------------------------------
# Server-side tokenization (vLLM /tokenize endpoint)
# ---------------------------------------------------------------------------


async def count_tokens(
    text: str,
    base_url: str,
    model: str,
    api_key: str | None = None,
    *,
    client_factory: MeasurementClientFactory,
) -> int | None:
    """Count tokens using the server's /tokenize endpoint.

    Uses vLLM's ``/tokenize`` endpoint for exact token counts with the
    model's actual tokenizer.  Returns None if the endpoint is
    unavailable (non-vLLM servers), in which case callers should fall
    back to character-based estimation.
    """
    try:
        async with client_factory(base_url=base_url, api_key=api_key, timeout=10.0) as client:
            resp = await client.tokenize(model=model, text=text)
            resp.raise_for_status()
            data = resp.json()
            count = data.get("count")
            if isinstance(count, int) and count >= 0:
                return count
    except Exception as exc:
        logger.debug("Token counting via /tokenize failed: %s", exc)
    return None


async def count_messages_tokens(
    messages: list[ChatMessage],
    base_url: str,
    model: str,
    api_key: str | None = None,
    *,
    client_factory: MeasurementClientFactory,
) -> int | None:
    """Count total tokens in a list of chat messages.

    Concatenates all message content and counts via /tokenize.  Adds a
    small overhead estimate for chat formatting tokens (role tags, etc.).
    """
    if not messages:
        return 0
    # Concatenate all content for a single tokenization call
    all_text = "\n".join(msg.get("content") or "" for msg in messages)
    count = await count_tokens(all_text, base_url, model, api_key, client_factory=client_factory)
    if count is None:
        return None
    # Add ~4 tokens per message for chat template overhead (role, delimiters)
    overhead = len(messages) * 4
    return count + overhead


# ---------------------------------------------------------------------------
# Fill budget calculation
# ---------------------------------------------------------------------------


def compute_fill_budget(
    context_size: int,
    ratio: float,
) -> int:
    """Calculate how many tokens of filler to inject.

    Reserves space for output generation and the actual scenario content.

    The budget is quantised to multiples of `_TOKENS_PER_FILLER_CHUNK` so
    that adjacent sweep levels whose raw token targets would straddle a
    chunk boundary produce the **same** number of filler message pairs.
    Without this, a sweep like 30% → 35% → 41% could alternate between N
    and N+1 filler chunks, creating a deterministic even/odd structural
    pattern in the prompt that manifests as alternating pass/fail results —
    regardless of the model or server (see NVIDIA Forum Issue, May 2026).
    """
    available = context_size - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
    if available <= 0:
        logger.warning(
            "Context size %d is too small for pressure testing "
            "(need at least %d for output + scenario overhead)",
            context_size,
            _RESERVED_FOR_OUTPUT + _RESERVED_FOR_SCENARIO,
        )
        return 0
    fill = int(available * max(0.0, min(1.0, ratio)))
    # Quantise to chunk boundaries so adjacent sweep levels cannot
    # straddle a chunk edge and produce different prompt structures.
    chunk_with_overhead = _TOKENS_PER_FILLER_CHUNK + 20  # chunk + ack
    if fill >= chunk_with_overhead:
        fill = (fill // chunk_with_overhead) * chunk_with_overhead
    return max(0, fill)


# ---------------------------------------------------------------------------
# Filler message builder
# ---------------------------------------------------------------------------


def build_pressure_messages(
    config: ContextPressureConfig,
    *,
    on_chunk: Callable[[int], None] | None = None,
    seed: int | None = None,
) -> list[ChatMessage]:
    """Build alternating user/assistant filler messages.

    Returns a list of messages to prepend before the real scenario messages.
    The structure creates a realistic-looking conversation history::

        [user]      "Here is some background context: <filler_chunk>"
        [assistant]  "Understood. I've reviewed the background context..."
        [user]      "<filler_chunk>"
        [assistant]  "Thank you for the additional context..."
        ... repeat ...

    Each user chunk is ~2048 tokens drawn from diverse source material
    (tech docs, meeting notes, code reviews, etc.) and shuffled per run
    to defeat prefix caching. A unique session nonce is prepended to each
    user message so that no two runs produce identical token prefixes.

    Args:
        config: Pressure configuration with fill_tokens set.
        on_chunk: Optional callback called after each chunk pair with the
            cumulative tokens used so far. Used for progress display.
        seed: Optional RNG seed for deterministic filler generation.
            When provided (e.g. from ``--seed``), the filler paragraph
            order and noise injection are fully reproducible per
            ``(seed, fill_tokens)`` combination.  When ``None``, uses
            ``time.time_ns()`` for a unique-per-call sequence.
    """
    import time

    fill_tokens = config.fill_tokens
    if fill_tokens <= 0:
        return []

    # Shuffle paragraph order per run to defeat cross-run prefix caching.
    # When a seed is provided, derive a deterministic sub-seed that also
    # incorporates fill_tokens so each sweep level is unique yet stable.
    pool_size = len(FILLER_PARAGRAPHS)
    paragraph_order = list(range(pool_size))
    if seed is not None:
        rng = random.Random(seed ^ hash(fill_tokens))
    else:
        # Seeded from OS entropy, not from a clock. `time.time_ns()` advances in
        # ~15.6ms steps on Windows, so two builds inside one tick drew the same
        # seed and produced byte-identical filler — exactly the prefix-cache
        # sharing this noise exists to defeat.
        rng = random.Random()
    rng.shuffle(paragraph_order)

    # Unique session nonce — ensures no two runs share token prefixes.
    # When seeded, derive from the seed so it's reproducible.
    if seed is not None:
        session_nonce = f"{seed:x}-{fill_tokens:x}"
    else:
        session_nonce = f"{time.time_ns():x}"

    messages: list[ChatMessage] = []
    tokens_used = 0
    chunk_idx = 0

    while tokens_used < fill_tokens:
        remaining = fill_tokens - tokens_used
        chunk_size = min(_TOKENS_PER_FILLER_CHUNK, remaining)

        if chunk_size < 50:
            # Too small for a meaningful chunk — stop
            break

        filler_text = build_filler_text(
            chunk_size,
            chunk_idx=chunk_idx,
            paragraph_order=paragraph_order,
            rng=rng,
        )

        # Unique prefix per chunk to bust prefix caching
        nonce_prefix = f"[ref:{session_nonce}-{chunk_idx:04d}] "

        # First chunk gets a framing prefix
        if chunk_idx == 0:
            content = (
                f"{nonce_prefix}Here is some background context for our "
                "conversation that you should keep in mind:\n\n" + filler_text
            )
        else:
            content = nonce_prefix + filler_text

        messages.append({"role": "user", "content": content})
        tokens_used += chunk_size

        # Add a short assistant acknowledgement
        ack = _ASSISTANT_RESPONSES[chunk_idx % len(_ASSISTANT_RESPONSES)]
        messages.append({"role": "assistant", "content": ack})
        # Assistant responses are ~20 tokens — count them against the budget
        tokens_used += 20

        chunk_idx += 1

        if on_chunk:
            on_chunk(tokens_used)

    logger.info(
        "Built %d pressure messages (~%d estimated tokens in %d turn pairs)",
        len(messages),
        tokens_used,
        chunk_idx,
    )
    return messages


async def calibrate_pressure_messages(
    messages: list[ChatMessage],
    target_tokens: int,
    base_url: str,
    model: str,
    api_key: str | None = None,
    *,
    client_factory: MeasurementClientFactory,
    seed: int | None = None,
) -> tuple[list[ChatMessage], int]:
    """Calibrate filler messages to hit the exact token target.

    Uses the server's ``/tokenize`` endpoint to measure actual token
    counts, then trims or extends the last user message content to
    match the target.  Returns ``(calibrated_messages, actual_tokens)``.

    If tokenization is unavailable, returns the messages unchanged with
    the char-based estimate.
    """
    if not messages or target_tokens <= 0:
        return messages, 0

    actual = await count_messages_tokens(
        messages, base_url, model, api_key, client_factory=client_factory
    )
    if actual is None:
        # Tokenizer unavailable — return char-based estimate
        est = sum(len(m.get("content") or "") / CHARS_PER_TOKEN_ESTIMATE for m in messages)
        logger.debug(
            "Tokenizer unavailable, using char estimate: ~%d tokens",
            int(est),
        )
        return messages, int(est)

    delta = actual - target_tokens
    if abs(delta) <= target_tokens * 0.02:
        # Within 2% — close enough
        logger.info(
            "Filler calibration: %d actual tokens vs %d target (%.1f%% off, OK)",
            actual,
            target_tokens,
            abs(delta) / target_tokens * 100,
        )
        return messages, actual

    if delta > 0:
        # Over target — trim characters from the last user message
        # Find the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                content = messages[i].get("content") or ""
                # Estimate chars to remove: delta tokens × chars_per_token
                # Use measured ratio from this run for better accuracy
                total_chars = sum(len(m.get("content") or "") for m in messages)
                measured_cpt = total_chars / actual if actual > 0 else CHARS_PER_TOKEN_ESTIMATE
                chars_to_remove = int(delta * measured_cpt * 1.05)  # slight over-trim
                if chars_to_remove < len(content) - 100:
                    # Trim the content but keep the message — never remove
                    # an entire message pair, as that changes the prompt
                    # structure and can re-introduce alternating pass/fail.
                    messages[i]["content"] = content[:-chars_to_remove]
                else:
                    # Need to remove nearly all content — trim to minimum
                    # viable length rather than removing the pair.
                    messages[i]["content"] = content[:100]
                break

        # Re-measure after trim
        recounted = await count_messages_tokens(
            messages, base_url, model, api_key, client_factory=client_factory
        )
        final = recounted if recounted is not None else actual - delta
        logger.info(
            "Filler calibrated: %d → %d tokens (target %d, %.1f%% accuracy)",
            actual,
            final,
            target_tokens,
            (1 - abs(final - target_tokens) / target_tokens) * 100,
        )
        return messages, final

    # Under target — extend the last user message with more filler
    shortfall = -delta

    if seed is not None:
        cal_rng = random.Random(seed ^ hash(target_tokens) ^ 0xCA1)
    else:
        cal_rng = random.Random()
    extra_text = build_filler_text(
        shortfall,
        chunk_idx=999,
        rng=cal_rng,
    )
    # Find last user message and append
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["content"] = (messages[i].get("content") or "") + "\n\n" + extra_text
            break

    recounted = await count_messages_tokens(
        messages, base_url, model, api_key, client_factory=client_factory
    )
    final = recounted if recounted is not None else actual + shortfall
    logger.info(
        "Filler calibrated: %d → %d tokens (target %d, %.1f%% accuracy)",
        actual,
        final,
        target_tokens,
        (1 - abs(final - target_tokens) / target_tokens) * 100,
    )
    return messages, final


# ---------------------------------------------------------------------------
# High-level: detect + build
# ---------------------------------------------------------------------------


async def prepare_context_pressure(
    base_url: str,
    model: str,
    api_key: str | None,
    ratio: float,
    context_size_override: int | None = None,
    metrics_url: str | None = None,
    *,
    client_factory: MeasurementClientFactory,
) -> ContextPressureConfig:
    """Detect context size and build the pressure config.

    Uses ``min(max_model_len, kv_cache_capacity)`` as the effective context
    size so that pressure targets what the server can actually hold in KV
    cache, not just what the model architecture supports.

    Detection order:
      1. ``--context-size`` override (used as-is, no KV cap applied)
      2. ``max_model_len`` from ``/v1/models`` — capped by KV capacity
         from ``/metrics`` (vLLM) if available

    Returns a fully populated ContextPressureConfig. If auto-detection
    fails and no override is provided, raises ValueError.
    """
    if context_size_override and context_size_override > 0:
        ctx_size: int = context_size_override
        logger.info("Using user-provided context size: %d", ctx_size)
    else:
        detected_context = await detect_context_size(
            base_url, model, api_key, client_factory=client_factory
        )
        if detected_context is None:
            raise ValueError(
                "Could not auto-detect context window size from /v1/models. "
                "Please provide --context-size explicitly "
                "(e.g. --context-size 32768)."
            )

        # Cap by actual KV cache capacity (vLLM: num_gpu_blocks × block_size).
        # max_model_len is the model's architectural limit, but the server
        # may have allocated far less KV cache depending on GPU memory,
        # model size, and gpu_memory_utilization.  Without this cap,
        # --context-pressure 0.9 on a 256K model with 117K KV cache would
        # try to fill 221K tokens — exceeding what the server can handle.
        #
        # EXCEPTION: hybrid-attention models (mamba/linear + full attention).
        # For these, num_gpu_blocks × block_size is the *physical* block
        # capacity, not the effective max context.  vLLM's hybrid KV cache
        # manager maps physical blocks to larger logical token coverage
        # (only a subset of layers need standard Transformer KV cache).
        # If the server starts and advertises max_model_len=X, it has
        # validated it can serve X tokens.  Trust it.
        ctx_size = detected_context
        kv_info = await detect_kv_capacity(
            base_url,
            api_key,
            metrics_url=metrics_url,
            client_factory=client_factory,
        )
        if kv_info is not None and not kv_info.is_hybrid and kv_info.capacity < ctx_size:
            logger.info(
                "Capping context size from %d (max_model_len) to %d "
                "(KV cache capacity: %d blocks × %d)",
                ctx_size,
                kv_info.capacity,
                kv_info.num_blocks,
                kv_info.block_size,
            )
            ctx_size = kv_info.capacity
        elif kv_info is not None and kv_info.is_hybrid:
            logger.info(
                "Hybrid model: trusting max_model_len=%d "
                "(physical block capacity %d is not the effective limit)",
                ctx_size,
                kv_info.capacity,
            )

    fill_tokens = compute_fill_budget(ctx_size, ratio)

    config = ContextPressureConfig(
        ratio=ratio,
        context_size=context_size_override,
        fill_tokens=fill_tokens,
        detected_context=ctx_size,
    )
    return config
