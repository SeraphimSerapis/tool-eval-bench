"""Logic behind the live decision-model monitor.

Everything here is display-free so it can be tested without a terminal.  The
monitor sends one canary item at a time to the model, scores it against its gold
label, and folds the outcome into rolling statistics.  A separate scrape of the
server's Prometheus endpoint adds load figures, which show traffic from other
clients too.
"""

from __future__ import annotations

import random
import re
import time
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import httpx

from tool_eval_bench.domain.decision import DecisionBackend
from tool_eval_bench.plugins.decision.dataset import QUESTION_NAME, DecisionItem, build_items
from tool_eval_bench.plugins.decision.evaluator import score_item
from tool_eval_bench.plugins.decision.metrics import (
    HIGH_CONFIDENCE,
    expected_calibration_error,
    mean,
    percentile,
    reliability_bins,
)

# Probes in the rolling window, and points kept for each trend line.
ROLLING_WINDOW = 100
TREND_LEN = 60
HISTOGRAM_BINS = 10
# A fixed shuffle keeps one canary cycle from running a whole category in a row.
_CANARY_SEED = 1729


@dataclass(frozen=True)
class ProbeEvent:
    """One scored canary request."""

    item_id: str
    category: str
    state: str
    gold: str
    predicted: str
    correct: bool
    confidence: float
    brier: float
    distribution: Mapping[str, float]
    is_score_scale: bool
    latency_ms: float
    input_tokens: int
    at: float

    @property
    def confident_mistake(self) -> bool:
        return not self.correct and self.confidence >= HIGH_CONFIDENCE


def canary_items() -> Iterator[DecisionItem]:
    """Cycle the base items forever in a fixed, mixed order."""
    items = build_items(variants=False)
    random.Random(_CANARY_SEED).shuffle(items)
    while True:
        yield from items


async def probe_once(
    adapter: DecisionBackend,
    item: DecisionItem,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
) -> ProbeEvent:
    """Send one item and score the answer.  Raises on transport or parse failure."""
    result = await adapter.decide(
        model=model,
        state=item.state,
        questions={QUESTION_NAME: item.question},
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        base_url=base_url,
    )
    score = score_item(item, result.answers[QUESTION_NAME])
    return ProbeEvent(
        item_id=item.id,
        category=item.category,
        state=item.state,
        gold=score.gold,
        predicted=score.predicted,
        correct=score.correct,
        confidence=score.confidence,
        brier=score.brier,
        distribution=score.distribution,
        is_score_scale=type(item.question).__name__ == "ScoreQuestion",
        latency_ms=result.elapsed_ms,
        input_tokens=result.input_tokens,
        at=time.time(),
    )


@dataclass
class LiveStats:
    """Session and rolling statistics over the probes seen so far."""

    window: int = ROLLING_WINDOW
    events: deque[ProbeEvent] = field(init=False)
    total: int = 0
    correct: int = 0
    errors: int = 0
    confident_mistakes: int = 0
    categories: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    accuracy_trend: deque[float] = field(default_factory=lambda: deque(maxlen=TREND_LEN))
    ece_trend: deque[float] = field(default_factory=lambda: deque(maxlen=TREND_LEN))
    latency_trend: deque[float] = field(default_factory=lambda: deque(maxlen=TREND_LEN))
    last_event: ProbeEvent | None = None
    last_miss: ProbeEvent | None = None
    last_error: str | None = None
    started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.events = deque(maxlen=self.window)

    def reset(self) -> None:
        """Forget the session, as Ctrl+R does in the speculative-decoding monitor."""
        self.__init__(window=self.window)  # type: ignore[misc]

    def record(self, event: ProbeEvent) -> None:
        self.events.append(event)
        self.last_event = event
        self.total += 1
        stats = self.categories[event.category]
        stats[1] += 1
        if event.correct:
            self.correct += 1
            stats[0] += 1
        else:
            self.last_miss = event
        if event.confident_mistake:
            self.confident_mistakes += 1
        self.accuracy_trend.append(self.rolling_accuracy)
        self.ece_trend.append(self.rolling_ece)
        self.latency_trend.append(event.latency_ms)

    def record_error(self, message: str) -> None:
        self.errors += 1
        self.last_error = message

    @property
    def rolling_accuracy(self) -> float:
        return sum(e.correct for e in self.events) / len(self.events) if self.events else 0.0

    @property
    def rolling_ece(self) -> float:
        bins = reliability_bins(
            [e.confidence for e in self.events], [e.correct for e in self.events]
        )
        return expected_calibration_error(bins)

    @property
    def rolling_confidence(self) -> float:
        return mean([e.confidence for e in self.events])

    @property
    def session_accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def category_accuracy(self) -> dict[str, float]:
        return {c: ok / n for c, (ok, n) in self.categories.items() if n}

    def confidence_histogram(self) -> list[int]:
        """Probe counts per confidence bin over the rolling window."""
        counts = [0] * HISTOGRAM_BINS
        for e in self.events:
            counts[min(int(e.confidence * HISTOGRAM_BINS), HISTOGRAM_BINS - 1)] += 1
        return counts

    def latency_percentiles(self) -> tuple[float, float]:
        values = [e.latency_ms for e in self.events]
        return percentile(values, 50), percentile(values, 95)


# ---------------------------------------------------------------------------
# Server load
# ---------------------------------------------------------------------------

_LOAD_METRICS = {
    "prompt_tokens": "llamacpp:prompt_tokens_total",
    "decodes": "llamacpp:n_decode_total",
    "processing": "llamacpp:requests_processing",
    "deferred": "llamacpp:requests_deferred",
    "busy_slots": "llamacpp:n_busy_slots_per_decode",
}


@dataclass(frozen=True)
class ServerLoad:
    """One scrape of the llama.cpp counters that matter to a decision model."""

    at: float
    prompt_tokens: float
    decodes: float
    processing: float
    deferred: float
    busy_slots: float


def parse_server_load(text: str, *, at: float | None = None) -> ServerLoad | None:
    """Parse a Prometheus scrape, or return None when it is not llama.cpp's."""
    values: dict[str, float] = {}
    for key, name in _LOAD_METRICS.items():
        match = re.search(rf"^{re.escape(name)}[ \t]+(\S+)", text, re.MULTILINE)
        if match:
            try:
                values[key] = float(match.group(1))
            except ValueError:
                continue
    if "decodes" not in values:
        return None
    return ServerLoad(
        at=time.time() if at is None else at,
        prompt_tokens=values.get("prompt_tokens", 0.0),
        decodes=values["decodes"],
        processing=values.get("processing", 0.0),
        deferred=values.get("deferred", 0.0),
        busy_slots=values.get("busy_slots", 0.0),
    )


async def scrape_server_load(
    client: httpx.AsyncClient, url: str, headers: Mapping[str, str]
) -> ServerLoad | None:
    """Fetch ``url`` and parse it, returning None on any failure.

    A monitor should keep running when the metrics endpoint is off or
    unreachable, so a failed scrape is a missing panel, not an error.
    """
    try:
        response = await client.get(url, headers=dict(headers), timeout=5.0)
        if response.status_code != 200:
            return None
        return parse_server_load(response.text)
    except httpx.HTTPError:
        return None


@dataclass
class LoadTrend:
    """Rates derived from successive scrapes."""

    previous: ServerLoad | None = None
    requests_per_s: deque[float] = field(default_factory=lambda: deque(maxlen=TREND_LEN))
    input_tokens_per_s: deque[float] = field(default_factory=lambda: deque(maxlen=TREND_LEN))
    latest: ServerLoad | None = None

    def update(self, load: ServerLoad) -> None:
        prev, self.previous, self.latest = self.previous, load, load
        if prev is None or load.at <= prev.at:
            return
        elapsed = load.at - prev.at
        # A restarted server resets its counters; a negative step is not a rate.
        self.requests_per_s.append(max(0.0, load.decodes - prev.decodes) / elapsed)
        self.input_tokens_per_s.append(max(0.0, load.prompt_tokens - prev.prompt_tokens) / elapsed)
