"""GSM8K prompt templates — system prompt and few-shot chain-of-thought examples.

The 8-shot examples are drawn from the GSM8K training set, following the
original paper's methodology (Cobbe et al., 2021).  Each example includes
a step-by-step solution ending with ``#### N``.
"""

from __future__ import annotations

from tool_eval_bench.domain.models import ChatMessage

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves grade-school math word problems. "
    "Show your work step-by-step, then give the final numeric answer on its "
    "own line in the format: #### <number>"
)

# 8 canonical few-shot chain-of-thought examples from the GSM8K training set
FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": (
            "There are 15 trees in the grove. Grove workers will plant trees "
            "in the grove today. After they are done, there will be 21 trees. "
            "How many trees did the grove workers plant today?"
        ),
        "answer": (
            "There are 15 trees originally. Then there were 21 trees after "
            "some more were planted. So there must have been 21 - 15 = 6.\n"
            "#### 6"
        ),
    },
    {
        "question": (
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot?"
        ),
        "answer": ("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"),
    },
    {
        "question": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, "
            "how many pieces do they have left in total?"
        ),
        "answer": (
            "Originally, Leah had 32 chocolates. Her sister had 42. "
            "So in total they had 32 + 42 = 74. After eating 35, they "
            "had 74 - 35 = 39.\n"
            "#### 39"
        ),
    },
    {
        "question": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
            "has 12 lollipops. How many lollipops did Jason give to Denny?"
        ),
        "answer": (
            "Jason started with 20 lollipops. Then he had 12 after giving "
            "some to Denny. So he gave Denny 20 - 12 = 8.\n"
            "#### 8"
        ),
    },
    {
        "question": (
            "Shawn has five toys. For Christmas, he got two toys each from "
            "his mom and dad. How many toys does he have now?"
        ),
        "answer": (
            "Shawn started with 5 toys. If he got 2 toys each from mom and "
            "dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9.\n"
            "#### 9"
        ),
    },
    {
        "question": (
            "There were nine computers in the server room. Five more computers "
            "were installed each day, from Monday to Thursday. How many "
            "computers are now in the server room?"
        ),
        "answer": (
            "There were originally 9 computers. For each of 4 days, 5 more "
            "computers were added. So 5 * 4 = 20 computers were added. "
            "9 + 20 = 29.\n"
            "#### 29"
        ),
    },
    {
        "question": (
            "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. "
            "On Wednesday, he lost 2 more. How many golf balls did he have "
            "at the end of Wednesday?"
        ),
        "answer": (
            "Michael started with 58 golf balls. After losing 23 on Tuesday, "
            "he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33.\n"
            "#### 33"
        ),
    },
    {
        "question": (
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
        ),
        "answer": (
            "Olivia had 23 dollars. 5 bagels for 3 dollars each is "
            "5 * 3 = 15 dollars. 23 - 15 = 8.\n"
            "#### 8"
        ),
    },
]


def build_messages(
    question: str,
    *,
    n_shots: int = 8,
) -> list[ChatMessage]:
    """Build the message list for a GSM8K question.

    Parameters
    ----------
    question : str
        The math word problem.
    n_shots : int
        Number of few-shot examples (0 = zero-shot, max 8).
    """
    messages: list[ChatMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add few-shot examples as user/assistant pairs
    examples = FEW_SHOT_EXAMPLES[:n_shots]
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})

    # The actual question
    messages.append({"role": "user", "content": question})

    return messages
