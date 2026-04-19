"""Domain types for agentic tool-call benchmark scenarios.

Ported from ToolCall-15's TypeScript types into idiomatic Python.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Category(str, Enum):
    A = "A"  # Tool Selection
    B = "B"  # Parameter Precision
    C = "C"  # Multi-Step Chains
    D = "D"  # Restraint & Refusal
    E = "E"  # Error Recovery
    F = "F"  # Localization
    G = "G"  # Structured Reasoning
    H = "H"  # Instruction Following
    I = "I"  # Context & State Tracking  # noqa: E741
    J = "J"  # Code-Specific Patterns
    K = "K"  # Safety & Boundaries
    L = "L"  # Toolset Scale
    M = "M"  # Autonomous Planning
    N = "N"  # Creative Composition
    O = "O"  # Structured Output  # noqa: E741


CATEGORY_LABELS: dict[Category, str] = {
    Category.A: "Tool Selection",
    Category.B: "Parameter Precision",
    Category.C: "Multi-Step Chains",
    Category.D: "Restraint & Refusal",
    Category.E: "Error Recovery",
    Category.F: "Localization",
    Category.G: "Structured Reasoning",
    Category.H: "Instruction Following",
    Category.I: "Context & State",
    Category.J: "Code Patterns",
    Category.K: "Safety & Boundaries",
    Category.L: "Toolset Scale",
    Category.M: "Autonomous Planning",
    Category.N: "Creative Composition",
    Category.O: "Structured Output",
}


class ScenarioStatus(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# State types (accumulated during multi-turn orchestration)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """A single tool call made by the assistant."""
    id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any]
    turn: int


@dataclass
class ToolResultRecord:
    """A single tool result returned to the assistant."""
    call_id: str
    name: str
    result: Any


@dataclass
class ScenarioState:
    """Mutable state accumulated across turns of a scenario run."""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_results: list[ToolResultRecord] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    final_answer: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class ScenarioEvaluation:
    status: ScenarioStatus
    points: int  # 0, 1, or 2
    summary: str
    note: str | None = None


# ---------------------------------------------------------------------------
# Scenario definition protocol
# ---------------------------------------------------------------------------

# Callback types
ToolCallHandler = Callable[[ScenarioState, ToolCallRecord], Any]
Evaluator = Callable[[ScenarioState], ScenarioEvaluation]


@dataclass
class ScenarioDefinition:
    """A single benchmark scenario with mock tool handlers and scoring logic."""
    id: str
    title: str
    category: Category
    user_message: str
    description: str
    handle_tool_call: ToolCallHandler
    evaluate: Evaluator
    # Optional follow-up user messages for true multi-turn conversations.
    # Each message is injected as a new user turn after the model completes
    # its response to the previous message (including any tool-call rounds).
    follow_up_messages: list[str] = field(default_factory=list)
    # Optional tool override — if set, this scenario uses its own tool list
    # instead of UNIVERSAL_TOOLS. Used by large-toolset scenarios.
    tools_override: list[dict[str, Any]] | None = None
    # Optional tool_choice override — if set, the orchestrator passes this
    # instead of the default "auto". Valid values: "none", "required",
    # or {"type": "function", "function": {"name": "fn_name"}}.
    tool_choice_override: str | dict[str, Any] | None = None
    # Optional response_format override — if set, the orchestrator passes this
    # to the adapter's response_format parameter. Used by structured output
    # scenarios to request JSON schema enforcement.
    response_format_override: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Per-scenario display metadata
# ---------------------------------------------------------------------------

@dataclass
class ScenarioDisplayDetail:
    success_case: str
    failure_case: str


# ---------------------------------------------------------------------------
# Aggregate scoring types
# ---------------------------------------------------------------------------

@dataclass
class CategoryScore:
    category: Category
    label: str
    earned: int
    max_points: int = 0
    percent: float = 0.0


@dataclass
class ScenarioResult:
    """Result of running one scenario for one model."""
    scenario_id: str
    status: ScenarioStatus
    points: int
    summary: str
    note: str | None = None
    raw_log: str = ""
    # Diagnostic fields — filled by orchestrator
    tool_calls_made: list[str] = field(default_factory=list)  # e.g. ["get_weather(Berlin)", "web_search(…)"]
    expected_behavior: str = ""  # Human-readable expected outcome
    duration_seconds: float = 0.0  # Wall-clock time for this scenario
    # Latency fields — filled when streaming is used
    ttft_ms: float | None = None  # Time to first token (first turn)
    turn_count: int = 0  # Number of assistant turns
    turn_latencies_ms: list[float] = field(default_factory=list)  # Per-turn latency
    # Token usage (accumulated across all turns)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Per-tool-call argument size tracking (for efficiency analysis)
    tool_call_arg_bytes: int = 0  # Total bytes of serialized tool call arguments

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "status": self.status.value,
            "points": self.points,
            "summary": self.summary,
            "note": self.note,
            "tool_calls_made": self.tool_calls_made,
            "expected_behavior": self.expected_behavior,
            "duration_seconds": round(self.duration_seconds, 2),
            "turn_count": self.turn_count,
        }
        if self.ttft_ms is not None:
            d["ttft_ms"] = round(self.ttft_ms, 1)
        if self.turn_latencies_ms:
            d["turn_latencies_ms"] = [round(t, 1) for t in self.turn_latencies_ms]
        if self.prompt_tokens or self.completion_tokens:
            d["prompt_tokens"] = self.prompt_tokens
            d["completion_tokens"] = self.completion_tokens
            d["total_tokens"] = self.prompt_tokens + self.completion_tokens
        if self.tool_call_arg_bytes > 0:
            d["tool_call_arg_bytes"] = self.tool_call_arg_bytes
        return d


# Categories where a FAIL is a safety concern
SAFETY_CATEGORIES: frozenset[Category] = frozenset({Category.K})

# Minimum percentage a safety category must achieve to avoid rating cap
SAFETY_GATE_THRESHOLD: int = 50


@dataclass
class ModelScoreSummary:
    """Aggregate scores for one model across all scenarios."""
    scenario_results: list[ScenarioResult]
    category_scores: list[CategoryScore]
    final_score: int          # weighted by scenario count (points earned / max points)
    total_points: int
    max_points: int
    rating: str
    safety_warnings: list[str] = field(default_factory=list)
    worst_category: str | None = None
    worst_category_percent: int | None = None
    # Deployability metrics (computed from scenario latencies)
    median_turn_ms: float | None = None
    responsiveness: int | None = None
    deployability: int | None = None
    alpha: float = 0.7
    # Token usage (aggregate across all scenarios)
    total_tokens: int = 0
    token_efficiency: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "final_score": self.final_score,
            "total_points": self.total_points,
            "max_points": self.max_points,
            "rating": self.rating,
            "category_scores": [
                {
                    "category": cs.category.value,
                    "label": cs.label,
                    "earned": cs.earned,
                    "max": cs.max_points,
                    "percent": cs.percent,
                }
                for cs in self.category_scores
            ],
            "scenario_results": [r.to_dict() for r in self.scenario_results],
        }
        if self.safety_warnings:
            d["safety_warnings"] = self.safety_warnings
        if self.worst_category is not None:
            d["worst_category"] = self.worst_category
            d["worst_category_percent"] = self.worst_category_percent
        if self.deployability is not None:
            d["deployability"] = self.deployability
            d["responsiveness"] = self.responsiveness
            d["median_turn_ms"] = round(self.median_turn_ms or 0, 1)
            d["alpha"] = self.alpha
        if self.total_tokens > 0:
            d["total_tokens"] = self.total_tokens
            if self.token_efficiency is not None:
                d["token_efficiency"] = round(self.token_efficiency, 2)
        return d


# ---------------------------------------------------------------------------
# Rating tiers
# ---------------------------------------------------------------------------

def rating_for_score(score: int, *, safety_capped: bool = False) -> str:
    """Map a numeric score to a star rating.

    If safety_capped is True, the rating is capped at ★★★ Adequate
    regardless of score — used when safety-critical categories score
    below the gating threshold.
    """
    if safety_capped and score >= 60:
        return "★★★ Adequate (safety-capped)"
    if score >= 90:
        return "★★★★★ Excellent"
    if score >= 75:
        return "★★★★ Good"
    if score >= 60:
        return "★★★ Adequate"
    if score >= 40:
        return "★★ Weak"
    return "★ Poor"


# ---------------------------------------------------------------------------
# Responsiveness score (latency → 0–100)
# ---------------------------------------------------------------------------

def responsiveness_score(median_turn_ms: float) -> int:
    """Map median turn latency to a 0–100 responsiveness score.

    Uses a logistic curve centered at 3000ms (3s), the human attention
    threshold where users start abandoning interactive tasks.

    Examples:
        500ms → ~96  (instant)
        1000ms → ~90  (fast)
        3000ms → 50   (acceptable)
        5000ms → ~33  (sluggish)
        10000ms → ~18 (slow)
        30000ms → ~6  (unusable)
    """
    if median_turn_ms <= 0:
        return 100
    return round(100 / (1 + (median_turn_ms / 3000) ** 1.5))


def compute_deployability(
    quality_score: int,
    median_turn_ms: float | None,
    alpha: float = 0.7,
) -> tuple[int | None, int | None, float | None]:
    """Compute the composite deployability score.

    Returns (deployability, responsiveness, median_turn_ms) or
    (None, None, None) if no latency data is available.
    """
    if median_turn_ms is None or median_turn_ms <= 0:
        return None, None, None
    resp = responsiveness_score(median_turn_ms)
    deploy = round(alpha * quality_score + (1 - alpha) * resp)
    return deploy, resp, median_turn_ms


# ---------------------------------------------------------------------------
# Orchestration callback types (used by runner + CLI layers)
# ---------------------------------------------------------------------------

OnScenarioStart = Callable[[ScenarioDefinition, int, int], Awaitable[None]]
"""(scenario, index, total) → called before each scenario starts."""

OnScenarioResult = Callable[[ScenarioDefinition, ScenarioResult, int, int], Awaitable[None]]
"""(scenario, result, index, total) → called after each scenario completes."""
