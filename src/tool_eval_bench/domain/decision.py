"""Domain types and port for decision models.

A decision model answers by scoring options the caller supplies, in a single
forward pass, instead of generating text.  llama.cpp serves them at
``/v1/systemone``.  There is no sampling, so a request is deterministic and
produces no completion tokens.

Three question types exist:

``choice``
    Pick one of several named options.  The answer carries a probability per
    option.
``score``
    Place the state on an ordered scale.  The answer carries a probability per
    level and the expected level.
``noul``
    A yes/no question.  The answer is the probability of "yes".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS


class DecisionUnsupportedError(RuntimeError):
    """The endpoint does not serve decision requests at all.

    Raised when the server answers 404, 405, or 501, so a run aborts once
    instead of scoring every item as a model failure.
    """


class DecisionRequestError(ValueError):
    """The server rejected a decision request as malformed (HTTP 400)."""


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick one option.  ``options`` maps each option name to its description."""

    instructions: str
    options: Mapping[str, str]

    def to_wire(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.options)}


@dataclass(frozen=True)
class ScoreQuestion:
    """Place the state on an ordered scale.  ``levels[i]`` describes level ``i``."""

    instructions: str
    levels: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


@dataclass(frozen=True)
class YesNoQuestion:
    """A yes/no question (the wire type is ``noul``).

    ``criteria`` optionally describes what ``true`` and ``false`` mean.  It is
    part of the input when present, so it is sent unchanged.
    """

    instructions: str
    criteria: Mapping[str, str] | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            wire["criteria"] = dict(self.criteria)
        return wire


DecisionQuestion = ChoiceQuestion | ScoreQuestion | YesNoQuestion

# What a request judges: free text, or a JSON object such as a ticket with the
# account behind it.  An object is sent as JSON, not flattened into text.
DecisionState = str | Mapping[str, Any]


def question_from_wire(raw: Mapping[str, Any]) -> DecisionQuestion:
    """Build a typed question from its wire form, the inverse of ``to_wire``.

    Raises ``ValueError`` for an unknown type or a field of the wrong shape, so
    a malformed dataset row fails before it reaches a server.
    """
    kind = raw.get("type")
    instructions = raw.get("instructions")
    criteria = raw.get("criteria")
    if not isinstance(instructions, str):
        raise ValueError(f"{kind!r} question has no string 'instructions'")
    if kind == "choice" and _is_str_map(criteria):
        return ChoiceQuestion(instructions, dict(criteria))
    if kind == "score" and isinstance(criteria, list) and all(isinstance(c, str) for c in criteria):
        return ScoreQuestion(instructions, tuple(criteria))
    if kind == "noul" and (criteria is None or _is_str_map(criteria)):
        return YesNoQuestion(instructions, None if criteria is None else dict(criteria))
    raise ValueError(f"unsupported or malformed {kind!r} question")


def _is_str_map(value: object) -> TypeGuard[Mapping[str, str]]:
    return isinstance(value, Mapping) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    # The server's own certainty figure.  It is not the top probability, so
    # calibration is computed from ``probabilities`` instead.
    confidence: float | None = None


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: Mapping[str, float]
    confidence: float | None = None


@dataclass(frozen=True)
class YesNoAnswer:
    probability_yes: float


DecisionAnswer = ChoiceAnswer | ScoreAnswer | YesNoAnswer


@dataclass
class DecisionResult:
    """Answers keyed by question name, plus request cost."""

    answers: dict[str, DecisionAnswer]
    input_tokens: int = 0
    elapsed_ms: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)


class DecisionBackend(ABC):
    """Port implemented by adapters whose server speaks the decision wire format."""

    @abstractmethod
    async def decide(
        self,
        *,
        model: str,
        state: DecisionState,
        questions: Mapping[str, DecisionQuestion],
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        api_key: str | None = None,
        base_url: str = "",
    ) -> DecisionResult:
        """Score ``questions`` against ``state`` in one request."""
        ...


def parse_answers(
    payload: Mapping[str, Any], questions: Mapping[str, DecisionQuestion]
) -> dict[str, DecisionAnswer]:
    """Parse a response body into typed answers, one per requested question.

    Raises ``ValueError`` when an answer is missing or does not have the shape
    its question type promises.  A server that silently drops a question is a
    failure to report, not a zero to score.
    """
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise ValueError("decision response has no 'answers' object")
    parsed: dict[str, DecisionAnswer] = {}
    for name, question in questions.items():
        raw = raw_answers.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"decision response is missing an answer for {name!r}")
        try:
            parsed[name] = _parse_one(raw, question)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"malformed {type(question).__name__} answer for {name!r}") from exc
    return parsed


def _parse_one(raw: Mapping[str, Any], question: DecisionQuestion) -> DecisionAnswer:
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=str(raw["choice"]),
            probabilities={str(k): float(v) for k, v in raw["probabilities"].items()},
            confidence=_optional_float(raw.get("confidence")),
        )
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer(
            score=float(raw["score"]),
            probabilities={str(k): float(v) for k, v in raw["probabilities"].items()},
            confidence=_optional_float(raw.get("confidence")),
        )
    return YesNoAnswer(probability_yes=float(raw["noul"]))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
