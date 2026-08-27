"""Filler prose corpus shared by context pressure and needle retrieval.

Both benchmarks need a large block of realistic, topically varied text that no
model has memorised and no prefix cache can serve twice: context pressure fills
a conversation window with it, and the needle benchmark buries a fact inside it.
The corpus lives here because ``plugins`` and ``runner`` may both reach
``domain``, and may not reach each other.

Noise injection is what keeps two runs from sharing a token prefix. Without it
a server's prefix cache answers from the first run's KV blocks, and the
benchmark measures the cache rather than the model.
"""

from __future__ import annotations

import random

# Approximate characters per token. Used to size a text block before the
# server's real tokenizer is consulted; callers that need exact counts
# calibrate against ``/tokenize`` afterwards.
CHARS_PER_TOKEN_ESTIMATE = 4.0

# ---------------------------------------------------------------------------
# Filler text pool — diverse content to defeat prefix caching and simulate
# realistic conversation history with varied topics
# ---------------------------------------------------------------------------

FILLER_PARAGRAPHS = [
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


def inject_noise(text: str, rng: random.Random) -> str:
    """Inject random noise tokens throughout the text to defeat prefix caching.

    Sprinkles random numbers, IDs, timestamps, and references at sentence
    boundaries so the tokenized result is unique across runs.
    """
    noise_generators = [
        lambda: f"(ref #{rng.randint(10000, 99999)})",
        lambda: f"[ticket SRE-{rng.randint(1000, 9999)}]",
        lambda: f"({rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.randint(2023, 2026)})",
        lambda: f"[v{rng.randint(1, 9)}.{rng.randint(0, 99)}.{rng.randint(0, 9)}]",
        lambda: (
            f"(node {rng.randint(1, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)})"
        ),
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


def build_filler_text(
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
        paragraph_order: Shuffled indices into FILLER_PARAGRAPHS.
            If None, uses sequential order.
        rng: Random number generator for noise injection.
    """
    target_chars = int(target_tokens * CHARS_PER_TOKEN_ESTIMATE)
    pool_size = len(FILLER_PARAGRAPHS)
    order = paragraph_order or list(range(pool_size))

    # Build a pool by cycling through paragraphs starting at chunk_idx offset
    parts: list[str] = []
    chars_so_far = 0
    pos = chunk_idx % pool_size
    while chars_so_far < target_chars:
        para = FILLER_PARAGRAPHS[order[pos % pool_size]]
        parts.append(para)
        chars_so_far += len(para)
        pos += 1
    pool = "".join(parts)
    text = pool[:target_chars]
    # Inject random noise to defeat prefix caching
    if rng:
        text = inject_noise(text, rng)
    return text


def build_haystack_text(target_tokens: int, *, seed: int | None = None) -> str:
    """Build one contiguous block of filler prose of roughly *target_tokens*.

    Callers that need a single document rather than a conversation — the
    needle-in-a-haystack benchmark, for instance — get the same shuffled,
    noise-injected source material that :func:`build_pressure_messages` uses,
    without the alternating user/assistant framing.

    A *seed* makes the paragraph order and the injected noise reproducible, so
    the same ``(seed, target_tokens)`` pair always yields the same haystack.
    """
    if target_tokens <= 0:
        return ""
    rng = random.Random(seed) if seed is not None else random.Random()
    paragraph_order = list(range(len(FILLER_PARAGRAPHS)))
    rng.shuffle(paragraph_order)
    return build_filler_text(
        target_tokens,
        chunk_idx=0,
        paragraph_order=paragraph_order,
        rng=rng,
    )
