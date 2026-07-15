"""MMLU prompt builder — 5-shot per-subject prompting."""

from __future__ import annotations

from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.plugins.mmlu.dataset import MMLUItem

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant taking a multiple choice exam. "
    "For each question, respond with ONLY the letter of the correct answer "
    "(A, B, C, or D). Do not include any explanation."
)

ANSWER_LETTERS = "ABCD"


def _format_question(item: MMLUItem, include_answer: bool = False) -> str:
    """Format a single question with labeled choices."""
    lines = [item.question, ""]
    for i, choice in enumerate(item.choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    if include_answer:
        lines.append(f"\nAnswer: {item.answer_letter}")
    return "\n".join(lines)


def _subject_display_name(subject: str) -> str:
    """Convert 'abstract_algebra' → 'Abstract Algebra'."""
    return subject.replace("_", " ").title()


def build_messages(
    question: MMLUItem,
    few_shot_examples: list[MMLUItem] | None = None,
    n_shots: int = 5,
) -> list[ChatMessage]:
    """Build chat messages for an MMLU question.

    Parameters
    ----------
    question
        The question to answer.
    few_shot_examples
        Examples from the same subject (dev split). If provided,
        up to *n_shots* are used as exemplars.
    n_shots
        Number of few-shot examples to include.
    """
    subject_name = _subject_display_name(question.subject)
    system = f"{SYSTEM_PROMPT}\n\nThe following are multiple choice questions about {subject_name}."

    messages: list[ChatMessage] = [{"role": "system", "content": system}]

    # Add few-shot examples
    if few_shot_examples and n_shots > 0:
        for ex in few_shot_examples[:n_shots]:
            messages.append(
                {
                    "role": "user",
                    "content": _format_question(ex),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": ex.answer_letter,
                }
            )

    # Add the actual question
    messages.append(
        {
            "role": "user",
            "content": _format_question(question),
        }
    )

    return messages
