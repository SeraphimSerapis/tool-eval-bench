"""Score one decision answer against its gold label."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tool_eval_bench.domain.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)
from tool_eval_bench.plugins.decision.dataset import DecisionItem

# Keeps log loss finite when a model gives the gold answer probability 0.
_PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class ItemScore:
    """The outcome for one answered item.

    ``distribution`` maps each answer (option name, score level, or yes/no) to
    its probability.  ``confidence`` is the probability of the predicted answer,
    which is what calibration compares against correctness.
    """

    predicted: str
    gold: str
    correct: bool
    confidence: float
    p_gold: float
    brier: float
    log_loss: float
    distribution: dict[str, float]
    # Score questions only: distance of the expected level from the gold level.
    level_error: float | None = None


def score_item(item: DecisionItem, answer: DecisionAnswer) -> ItemScore:
    """Grade ``answer`` against ``item.gold``.

    Raises ``ValueError`` when the answer type does not match the question type.
    """
    question = item.question
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        return _grade(
            dict(answer.probabilities),
            predicted=answer.choice,
            gold=str(item.gold),
        )
    if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        distribution = dict(answer.probabilities)
        predicted = max(distribution, key=lambda k: distribution[k])
        graded = _grade(distribution, predicted=predicted, gold=str(item.gold))
        return replace(graded, level_error=abs(answer.score - int(item.gold)))
    if isinstance(question, YesNoQuestion) and isinstance(answer, YesNoAnswer):
        distribution = {"yes": answer.probability_yes, "no": 1.0 - answer.probability_yes}
        predicted = "yes" if answer.probability_yes >= 0.5 else "no"
        return _grade(distribution, predicted=predicted, gold="yes" if item.gold else "no")
    raise ValueError(f"{type(answer).__name__} does not answer a {type(question).__name__}")


def _grade(distribution: dict[str, float], *, predicted: str, gold: str) -> ItemScore:
    p_gold = distribution.get(gold, 0.0)
    return ItemScore(
        predicted=predicted,
        gold=gold,
        correct=predicted == gold,
        confidence=distribution.get(predicted, 0.0),
        p_gold=p_gold,
        # Multi-class Brier: squared distance from the one-hot gold vector, so
        # 0 is perfect and 2 is maximally wrong and confident.
        brier=sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in distribution.items()),
        log_loss=-math.log(max(p_gold, _PROBABILITY_FLOOR)),
        distribution=distribution,
    )
