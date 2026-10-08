"""Score one decision answer against its soft gold distribution."""

from __future__ import annotations

from dataclasses import dataclass

from tool_eval_bench.domain.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)
from tool_eval_bench.plugins.decision.metrics import argmax, brier_score, kl_divergence
from tool_eval_bench.plugins.decision.typed_decisions import GoldAnswer


@dataclass(frozen=True)
class SoftScore:
    """The outcome for one answered question whose gold is a distribution.

    ``distribution`` is the prediction over the question's answers, in the
    question's order.  ``confidence`` is its top probability.
    """

    predicted: str
    gold: str
    correct: bool
    confidence: float
    kl: float
    brier: float
    distribution: dict[str, float]


def answer_distribution(question: DecisionQuestion, answer: DecisionAnswer) -> dict[str, float]:
    """The answer's probability for each of the question's answers.

    Yes/no answers are keyed ``true`` and ``false``, as typed-decisions gold is.
    An answer the server left out has probability 0.  Raises ``ValueError``
    when the answer type does not match the question type.
    """
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        return {k: answer.probabilities.get(k, 0.0) for k in question.options}
    if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        levels = [str(i) for i in range(len(question.levels))]
        return {k: answer.probabilities.get(k, 0.0) for k in levels}
    if isinstance(question, YesNoQuestion) and isinstance(answer, YesNoAnswer):
        return {"true": answer.probability_yes, "false": 1.0 - answer.probability_yes}
    raise ValueError(f"{type(answer).__name__} does not answer a {type(question).__name__}")


def score_against_gold(
    question: DecisionQuestion, answer: DecisionAnswer, gold: GoldAnswer
) -> SoftScore:
    """Grade ``answer`` against a soft gold.

    Correct means the prediction's most likely answer is the gold label.
    KL and Brier compare the whole prediction with the gold distribution.
    """
    distribution = answer_distribution(question, answer)
    predicted = argmax(distribution)
    return SoftScore(
        predicted=predicted,
        gold=gold.label,
        correct=predicted == gold.label,
        confidence=distribution[predicted],
        kl=kl_divergence(gold.probabilities, distribution),
        brier=brier_score(gold.probabilities, distribution),
        distribution=distribution,
    )
