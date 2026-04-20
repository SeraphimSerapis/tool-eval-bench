"""Context pressure: pre-fill the conversation window with filler turns.

Simulates real-world agentic conversations where the context is already
heavily utilized before the model needs to make tool-calling decisions.

Usage::

    pressure = ContextPressureConfig(ratio=0.75, context_size=32768)
    messages = await build_pressure_messages(
        base_url, model, api_key, pressure,
    )
    # Prepend these messages before the real scenario messages.

Auto-detection strategy for context size:
  1. ``/v1/models`` → ``max_model_len`` (vLLM)
  2. ``/v1/models`` → ``context_window`` or ``max_tokens`` (LiteLLM / others)
  3. Fall back to ``--context-size`` CLI override (required if auto-detect fails)
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from tool_eval_bench.utils.urls import models_url as _models_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filler text pool — diverse content to defeat prefix caching and simulate
# realistic conversation history with varied topics
# ---------------------------------------------------------------------------

_FILLER_PARAGRAPHS = [
    # 0: Technical documentation
    (
        "The distributed caching layer uses consistent hashing to partition "
        "keys across nodes. When a node fails, its virtual nodes are reassigned "
        "to the next healthy node in the ring. The replication factor defaults "
        "to 3, meaning each key is stored on three distinct physical nodes. "
        "Write operations require a quorum of 2 acknowledgements before "
        "returning success to the client. Read operations can be configured "
        "for eventual consistency (any single replica) or strong consistency "
        "(quorum read). The cache eviction policy follows an LRU strategy with "
        "a secondary TTL-based expiration. Memory pressure triggers eviction "
        "of the least recently used entries until usage drops below 85%. "
        "The gossip protocol runs every 500ms to propagate membership changes. "
    ),
    # 1: Meeting notes
    (
        "In the Q3 planning meeting, the infrastructure team proposed migrating "
        "the primary database from PostgreSQL 14 to PostgreSQL 16 to take "
        "advantage of improved query parallelism and logical replication "
        "enhancements. The estimated migration window is 4 hours with a "
        "rollback plan that adds another 2 hours. The product team raised "
        "concerns about feature freeze during migration. After discussion, "
        "the team agreed to schedule the migration for the last weekend of "
        "September, with a staged rollout: read replicas first, then the "
        "primary. The monitoring dashboard will track replication lag, "
        "connection pool saturation, and query latency percentiles during "
        "the cutover. Action items were assigned to Sarah for runbook prep "
        "and Marcus for load testing the new connection pooler configuration. "
    ),
    # 2: Code review feedback
    (
        "The pull request introduces a new retry mechanism for the HTTP "
        "client, but there are several concerns. First, the exponential "
        "backoff implementation does not include jitter, which could lead "
        "to thundering herd problems when multiple clients retry "
        "simultaneously. Second, the maximum retry count is hardcoded to 5 "
        "instead of being configurable. Third, the retry logic does not "
        "distinguish between transient errors (429, 503) and permanent "
        "errors (400, 404). The suggested fix is to add a RetryPolicy class "
        "that encapsulates backoff strategy, jitter range, maximum attempts, "
        "and retryable status codes. The circuit breaker integration should "
        "also be considered to prevent cascading failures when the upstream "
        "service is experiencing sustained outages. Tests should cover edge "
        "cases including timeout during retry and concurrent retry storms. "
    ),
    # 3: Data analysis report
    (
        "The analysis of user engagement metrics for March reveals a 12% "
        "increase in daily active users compared to February, driven primarily "
        "by the mobile app redesign launched on March 3rd. Session duration "
        "increased from 4.2 minutes to 5.8 minutes on average. However, the "
        "bounce rate for new users remains elevated at 34%, suggesting the "
        "onboarding flow needs further optimization. The cohort analysis shows "
        "that users who complete the tutorial have a 67% Day-7 retention rate "
        "versus 23% for those who skip it. Revenue per user increased by 8%, "
        "with in-app purchases accounting for 62% of total revenue. The "
        "recommendation is to implement a progressive onboarding experience "
        "that introduces features gradually rather than presenting all options "
        "at once. A/B test results for the simplified checkout flow show a "
        "statistically significant lift of 4.3% in conversion rate. "
    ),
    # 4: System architecture discussion
    (
        "The event-driven architecture uses Apache Kafka as the central "
        "message bus with separate topics for user actions, system events, "
        "and audit logs. Each microservice publishes domain events that other "
        "services consume asynchronously. The order processing service "
        "subscribes to payment-confirmed events and triggers fulfillment "
        "workflows. The notification service listens to multiple topics and "
        "applies user preference filters before dispatching emails, push "
        "notifications, or SMS messages. Schema evolution is managed through "
        "a schema registry with backward compatibility enforcement. Dead "
        "letter queues capture messages that fail processing after three "
        "attempts, and an alert triggers when the DLQ depth exceeds 100 "
        "messages. The consumer group rebalancing strategy uses cooperative "
        "sticky assignment to minimize partition reassignment during scaling. "
    ),
    # 5: Email thread
    (
        "Following up on yesterday's discussion about the API rate limiting "
        "changes: after reviewing the access logs from the past 30 days, "
        "the top 10 API consumers account for 78% of all requests. The "
        "proposed tiered rate limiting would set free-tier users to 100 "
        "requests per minute, standard-tier to 1000 RPM, and enterprise "
        "to 10000 RPM. We need to ensure the rate limiter uses a sliding "
        "window algorithm rather than fixed windows to prevent burst "
        "traffic at window boundaries. The implementation should return "
        "proper 429 status codes with Retry-After headers indicating the "
        "reset time. The client SDKs will need updates to handle rate "
        "limit responses gracefully with automatic retry logic. Please "
        "review the RFC document attached and provide feedback by Friday. "
        "The deployment is tentatively scheduled for the first week of May. "
    ),
    # 6: System monitoring log
    (
        "Alert investigation summary for incident INC-4821: The production "
        "cluster experienced elevated p99 latency from 14:23 to 15:47 UTC "
        "on March 18th. Root cause analysis identified a memory leak in the "
        "connection pool manager introduced in version 2.14.3. The leak "
        "caused gradual memory growth of approximately 50MB per hour, "
        "triggering garbage collection pauses that blocked request processing "
        "for 200-400ms intervals. The fix involved switching from manual "
        "connection lifecycle management to a pooled connection factory with "
        "configurable idle timeout and maximum lifetime settings. The patch "
        "was deployed as version 2.14.4 and confirmed stable with memory "
        "usage plateauing at 1.2GB under normal load. Post-incident review "
        "recommended adding memory growth rate alerts and connection pool "
        "utilization dashboards to the monitoring stack. "
    ),
    # 7: API documentation
    (
        "The REST API endpoint POST /v2/analyses accepts a JSON body with "
        "required fields: dataset_id (string, UUID format), analysis_type "
        "(enum: regression, classification, clustering, anomaly_detection), "
        "and parameters (object, type-specific). Optional fields include "
        "name (string, max 255 chars), description (string, max 2000 chars), "
        "priority (integer, 1-10, default 5), and callback_url (string, "
        "valid HTTPS URL for webhook notification on completion). The response "
        "returns 202 Accepted with the analysis_id and estimated completion "
        "time. Status can be polled via GET /v2/analyses/{analysis_id} which "
        "returns the current state (queued, running, completed, failed) along "
        "with progress percentage and partial results when available. Rate "
        "limiting applies: 10 concurrent analyses per API key for standard "
        "tier. Exceeding this limit returns 429 Too Many Requests. "
    ),
    # 8: Research notes
    (
        "The transformer architecture with rotary position embeddings shows "
        "improved length generalization compared to absolute positional "
        "encodings. In our experiments, models trained on sequences up to "
        "4096 tokens could extrapolate to 16384 tokens with only a 3% "
        "degradation in perplexity when using NTK-aware interpolation. The "
        "key insight is that RoPE encodes relative position information in "
        "the attention computation rather than adding absolute position "
        "information to the input embeddings. This allows the model to "
        "recognize distance-based patterns regardless of absolute position. "
        "Flash attention v2 reduces memory usage from O(n^2) to O(n) while "
        "maintaining exact attention computation, enabling training on longer "
        "sequences within the same memory budget. The combination of these "
        "techniques allows efficient processing of documents up to 128K "
        "tokens on hardware with 80GB of GPU memory. "
    ),
    # 9: Project status update
    (
        "Sprint 14 retrospective highlights: The team completed 34 of 38 "
        "story points, with 4 points carrying over to Sprint 15 due to an "
        "unexpected dependency on the authentication service migration. The "
        "frontend team delivered the new dashboard components ahead of "
        "schedule, including the real-time metrics visualization that uses "
        "WebSocket connections for live data streaming. The backend team "
        "resolved the batch processing bottleneck by parallelizing the ETL "
        "pipeline, reducing processing time from 45 minutes to 12 minutes "
        "for the standard daily ingestion. Three critical bugs were fixed: "
        "the timezone conversion issue affecting users in UTC+13 zones, the "
        "race condition in the session manager causing intermittent logouts, "
        "and the CSV export failing silently for datasets exceeding 100K "
        "rows. Technical debt items addressed include upgrading the ORM "
        "library and adding structured logging to the payment service. "
    ),
    # 10: Security review
    (
        "The security audit of the authentication subsystem identified "
        "several areas requiring attention. The password hashing uses bcrypt "
        "with a cost factor of 10, which should be increased to 12 given "
        "current hardware capabilities. The JWT token expiration is set to "
        "24 hours, which exceeds the recommended maximum of 1 hour for "
        "access tokens. Refresh tokens should be stored server-side with "
        "rotation on each use and a maximum lifetime of 30 days. The CORS "
        "policy currently allows wildcard origins in the staging environment, "
        "which should be restricted to specific domains. The API does not "
        "implement request signing for webhook callbacks, leaving it "
        "vulnerable to replay attacks. Rate limiting on the login endpoint "
        "allows 20 attempts per minute, but should implement progressive "
        "delays after 5 failed attempts to mitigate credential stuffing. "
        "The session management should be updated to invalidate all active "
        "sessions when a user changes their password. "
    ),
    # 11: Database migration plan
    (
        "The schema migration from v3 to v4 introduces several breaking "
        "changes that require careful coordination. The users table gains "
        "two new columns: mfa_enabled (boolean, default false) and "
        "last_password_change (timestamp with timezone). The orders table "
        "is being partitioned by created_at using PostgreSQL declarative "
        "partitioning with monthly partitions. Historical data older than "
        "24 months will be moved to cold storage partitions on cheaper "
        "storage. The products table foreign key to categories is changing "
        "from a single category_id to a many-to-many relationship through "
        "a new product_categories junction table. Existing category "
        "assignments will be migrated automatically. The estimated data "
        "migration time for 47M order records is 3.5 hours based on "
        "staging environment benchmarks. The rollback script preserves "
        "the original schema and data, adding approximately 15 minutes "
        "to the rollback window. Flyway migration scripts are versioned "
        "and tested against a snapshot of production data. "
    ),
]

# Short assistant acknowledgements for alternating turns
_ASSISTANT_RESPONSES = [
    "Understood. I've reviewed the background context you provided. Please continue.",
    "Thank you for the additional context. I'm ready for your next request.",
    "Got it. I've noted all the details. What would you like me to help with?",
    "I see. I've taken all of that into account. Please go ahead.",
    "Acknowledged. I have the full context now. How can I assist you?",
]

# Conservative defaults
_RESERVED_FOR_OUTPUT = 4096     # max_tokens for generation
_RESERVED_FOR_SCENARIO = 12000  # tool definitions + system prompt + user message +
                                 # multi-turn conversation growth + token estimation
                                 # margin.  The server counts tool schemas against
                                 # the context window — the 52-tool LARGE_TOOLSET
                                 # alone is ~6000 tokens.  The extra margin (~4K)
                                 # absorbs char→token estimation error so that
                                 # ratio=1.0 can still succeed.
_CHARS_PER_TOKEN_ESTIMATE = 4.0
_TOKENS_PER_FILLER_CHUNK = 2048


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ContextPressureConfig:
    """Configuration for context pressure pre-filling."""
    ratio: float = 0.75           # Fill this fraction of the available context
    context_size: int | None = None  # Override auto-detected context size
    fill_tokens: int = 0          # Computed: actual tokens to fill
    detected_context: int = 0     # Actual context size used

    def summary(self) -> str:
        """Human-readable summary for display."""
        pct = int(self.ratio * 100)
        fill_k = self.fill_tokens / 1000
        ctx_k = self.detected_context / 1000
        return f"{pct}% (~{fill_k:.0f}K tokens prefilled of {ctx_k:.0f}K context)"


# ---------------------------------------------------------------------------
# Context size detection
# ---------------------------------------------------------------------------

def _headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def detect_context_size(
    base_url: str,
    model: str,
    api_key: str | None = None,
) -> int | None:
    """Auto-detect context window size from /v1/models.

    Tries multiple fields in order of preference:
      - max_model_len (vLLM)
      - context_window (LiteLLM)
      - max_tokens (generic)

    Returns the context size in tokens, or None if detection fails.
    """
    url = _models_url(base_url)
    hdrs = _headers(api_key)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=hdrs)
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
# Fill budget calculation
# ---------------------------------------------------------------------------

def compute_fill_budget(
    context_size: int,
    ratio: float,
) -> int:
    """Calculate how many tokens of filler to inject.

    Reserves space for output generation and the actual scenario content.
    """
    available = context_size - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
    if available <= 0:
        logger.warning(
            "Context size %d is too small for pressure testing "
            "(need at least %d for output + scenario overhead)",
            context_size, _RESERVED_FOR_OUTPUT + _RESERVED_FOR_SCENARIO,
        )
        return 0
    fill = int(available * max(0.0, min(1.0, ratio)))
    return max(0, fill)


# ---------------------------------------------------------------------------
# Filler message builder
# ---------------------------------------------------------------------------

def _inject_noise(text: str, rng: random.Random) -> str:
    """Inject random noise tokens throughout the text to defeat prefix caching.

    Sprinkles random numbers, IDs, timestamps, and references at sentence
    boundaries so the tokenized result is unique across runs.
    """
    noise_generators = [
        lambda: f"(ref #{rng.randint(10000, 99999)})",
        lambda: f"[ticket SRE-{rng.randint(1000, 9999)}]",
        lambda: f"({rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.randint(2023, 2026)})",
        lambda: f"[v{rng.randint(1, 9)}.{rng.randint(0, 99)}.{rng.randint(0, 9)}]",
        lambda: f"(node {rng.randint(1, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)})",
        lambda: f"[{rng.choice(['WARN', 'INFO', 'DEBUG', 'TRACE'])} {rng.randint(100, 999)}ms]",
        lambda: f"(batch {rng.randint(1, 500)}/{rng.randint(500, 1000)})",
        lambda: f"[id:{rng.randint(100000, 999999):x}]",
    ]

    sentences = text.split(". ")
    result: list[str] = []
    for i, sentence in enumerate(sentences):
        result.append(sentence)
        # Inject noise after roughly every 3rd sentence
        if i % 3 == 2 and i < len(sentences) - 1:
            noise = rng.choice(noise_generators)()
            result.append(f" {noise}")
    return ". ".join(result)


def _build_filler_text(
    target_tokens: int,
    chunk_idx: int = 0,
    paragraph_order: list[int] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Build a block of filler text of approximately target_tokens tokens.

    Each chunk_idx selects a different starting paragraph from the pool,
    cycling through diverse content to defeat prefix caching. Random noise
    is injected throughout to ensure unique token sequences per run.

    Args:
        target_tokens: Number of tokens to generate.
        chunk_idx: Index of this chunk (determines paragraph rotation).
        paragraph_order: Shuffled indices into _FILLER_PARAGRAPHS.
            If None, uses sequential order.
        rng: Random number generator for noise injection.
    """
    target_chars = int(target_tokens * _CHARS_PER_TOKEN_ESTIMATE)
    pool_size = len(_FILLER_PARAGRAPHS)
    order = paragraph_order or list(range(pool_size))

    # Build a pool by cycling through paragraphs starting at chunk_idx offset
    parts: list[str] = []
    chars_so_far = 0
    pos = chunk_idx % pool_size
    while chars_so_far < target_chars:
        para = _FILLER_PARAGRAPHS[order[pos % pool_size]]
        parts.append(para)
        chars_so_far += len(para)
        pos += 1
    pool = "".join(parts)
    text = pool[:target_chars]
    # Inject random noise to defeat prefix caching
    if rng:
        text = _inject_noise(text, rng)
    return text


def build_pressure_messages(
    config: ContextPressureConfig,
    *,
    on_chunk: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
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
    """
    import time

    fill_tokens = config.fill_tokens
    if fill_tokens <= 0:
        return []

    # Shuffle paragraph order per run to defeat cross-run prefix caching
    pool_size = len(_FILLER_PARAGRAPHS)
    paragraph_order = list(range(pool_size))
    rng = random.Random(time.time_ns())
    rng.shuffle(paragraph_order)

    # Unique session nonce — ensures no two runs share token prefixes
    session_nonce = f"{time.time_ns():x}"

    messages: list[dict[str, Any]] = []
    tokens_used = 0
    chunk_idx = 0

    while tokens_used < fill_tokens:
        remaining = fill_tokens - tokens_used
        chunk_size = min(_TOKENS_PER_FILLER_CHUNK, remaining)

        if chunk_size < 50:
            # Too small for a meaningful chunk — stop
            break

        filler_text = _build_filler_text(
            chunk_size, chunk_idx=chunk_idx,
            paragraph_order=paragraph_order, rng=rng,
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
        "Built %d pressure messages (~%d tokens in %d turn pairs)",
        len(messages), tokens_used, chunk_idx,
    )
    return messages


# ---------------------------------------------------------------------------
# High-level: detect + build
# ---------------------------------------------------------------------------

async def prepare_context_pressure(
    base_url: str,
    model: str,
    api_key: str | None,
    ratio: float,
    context_size_override: int | None = None,
) -> ContextPressureConfig:
    """Detect context size and build the pressure config.

    Returns a fully populated ContextPressureConfig. If auto-detection
    fails and no override is provided, raises ValueError.
    """
    if context_size_override and context_size_override > 0:
        ctx_size = context_size_override
        logger.info("Using user-provided context size: %d", ctx_size)
    else:
        ctx_size = await detect_context_size(base_url, model, api_key)
        if ctx_size is None:
            raise ValueError(
                "Could not auto-detect context window size from /v1/models. "
                "Please provide --context-size explicitly "
                "(e.g. --context-size 32768)."
            )

    fill_tokens = compute_fill_budget(ctx_size, ratio)

    config = ContextPressureConfig(
        ratio=ratio,
        context_size=context_size_override,
        fill_tokens=fill_tokens,
        detected_context=ctx_size,
    )
    return config
