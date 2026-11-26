"""Refusal and clarification intent classifiers for assistant responses."""

from __future__ import annotations

import re

from tool_eval_bench.evals.helpers._text import (
    _PROSE_TRANSLATION,
    normalize,
    strip_markdown_emphasis,
)

# Possession denial tolerates a short adverb or hedge between the denial and
# "have": "I don't currently have access" (issue #160) is the same denial as
# "I don't have access". Bounded to discourse adverbs and hedge verbs so the
# unrelated intent idiom "don't want to have" still does not match, and the
# object after "have" must be a capability/inbox noun so a state claim like
# "I don't currently have any spam left" is still not a refusal.
_DENIAL_FILLERS = (
    r"(?:\s+(?:currently|really|actually|directly|presently|necessarily|exactly|"
    r"anymore|even|simply|legitimately|right\s+now|at\s+present|"
    r"at\s+the\s+moment|at\s+this\s+time|seem\s+to|appear\s+to)){0,2}"
)
_DENIAL_OBJECT = (
    r"(?:\s+\w+){0,2}\s+(?:access|any\s+tool|tool|tooling|toolset|permission|"
    r"capabilit|abilit|mailbox|inbox|anything|way|function|method)"
)
_POSSESSION_DENIAL_RE = re.compile(
    r"\b(?:don't|do\s+not|didn't|did\s+not)\b"
    + _DENIAL_FILLERS
    + r"\s+have(?:(?:\s+got\b)?"
    + _DENIAL_OBJECT
    + ")"
)
_POSSESSION_GOT_RE = re.compile(
    r"\b(?:haven't|hasn't|hadn't|have\s+not|has\s+not|had\s+not)\b"
    + _DENIAL_FILLERS
    + r"\s+got\b"
    + _DENIAL_OBJECT
)
_HAVE_NO_CAPABILITY_RE = re.compile(
    r"\b(?:have\s+no|lack(?:\s+the|\s+any)?|there(?:'s|\s+is)\s+no)"
    r"(?:\s+[\w-]+){0,2}\s+"
    r"(?:access|tools?|tooling|toolset|permissions?|capabilit\w*|abilit\w*|mailbox|inbox|way|functions?|methods?|means)\b"
)


def contains_refusal(answer: str) -> bool:
    """Check if the answer contains a refusal phrase.

    Covers common LLM refusal patterns across model families:
    - Explicit inability ("I cannot", "I'm unable to")
    - Apology-based ("I'm sorry", "unfortunately")
    - Boundary-stating ("no tool", "not supported")
    - Capability framing ("beyond my", "outside my", "that's not something I can")
    - Polite declines ("I'm afraid", "I don't have the ability")
    - Phi / Command-R / Qwen3 / Gemma patterns
    """
    low = normalize(strip_markdown_emphasis(answer))
    if (
        _POSSESSION_DENIAL_RE.search(low)
        or _POSSESSION_GOT_RE.search(low)
        or _HAVE_NO_CAPABILITY_RE.search(low)
    ):
        return True
    return any(
        phrase in low
        for phrase in (
            # Explicit inability
            "cannot",
            "can't",
            "can not",
            "couldn't",
            "could not",
            "not able",
            "wasn't able",
            "was not able",
            "weren't able",
            "were not able",
            "unable to",
            "won't",
            "will not",
            "i refuse",
            "refuse to",
            # Possession-denial
            "do not have",
            "don't have",
            "do not possess",
            "don't possess",
            "don't have access",
            "do not have access",
            "don't have the ability",
            "do not have the ability",
            "don't have the capability",
            "lack the",
            # Apology-based
            "unfortunately",
            "i'm sorry",
            "i am sorry",
            "i apologize",
            "i'm afraid",
            "i am afraid",
            "regrettably",
            # Boundary-stating
            "not possible",
            "no way to",
            "there is no way",
            "not supported",
            "aren't supported",
            "isn't supported",
            "not available",
            "aren't available",
            "no tool",
            "no capability",
            "no such tool",
            "no function",
            # Capability framing (Qwen / Mistral style)
            "beyond my",
            "outside my",
            "outside what the tools",
            "outside what my tools",
            "out of scope",
            "beyond the scope",
            "outside the scope",
            "that's not something i",
            "that is not something i",
            "isn't something i",
            "is not something i",
            "isn't something these tools",
            "is not something these tools",
            "not something these tools",
            "isn't part of",
            "is not part of",
            "isn't within",
            "is not within",
            # Toolkit limitation
            "no appropriate tool",
            "no suitable tool",
            "doesn't exist",
            "does not exist",
            "isn't available",
            "is not available",
            "not in my toolset",
            "not in the available",
            "no matching tool",
            # Command-R / Phi patterns
            "i must decline",
            "i need to decline",
            "not something i'm able",
            "not something i am able",
            "i'm not equipped",
            "i am not equipped",
            "falls outside",
            "goes beyond",
            "not within my capabilities",
            "exceeds my",
            # Qwen3 / Gemma patterns
            "i lack the",
            "this exceeds",
            "i'm not designed",
            "i am not designed",
            "not designed to",
            "i don't support",
            "i do not support",
            "no mechanism",
            "no means to",
            "i must refrain",
            "i should not",
            # Refusal-with-alternative
            "instead, i can",
            "however, i can",
            "what i can do",
            "alternatively",
        )
    )


# A closing offer after acting ("Done! Would you like me to do anything else?")
# asks nothing about the task, so it is removed before the phrase checks. Only
# offers with no task content count, and only when they open a sentence:
# "search for something else, like a different file name?" and "or would you
# like me to pick someone else?" are real questions about the task.
_SENTENCE_START = r"(?:^|(?<=[.!?\n]))\s*"
_GENERIC_OFFER = re.compile(
    _SENTENCE_START + r"(?:"
    r"(?:would you like me to|do you want me to|want me to|shall i|should i|can i|may i)\s+"
    r"(?:(?:help|assist)(?:\s+you)?(?:\s+with)?|do)\s+(?:anything|something)\s+(?:else|more)"
    r"|is there anything else (?:i can|you need|you'd like|you would like)"
    r"|(?:please\s+)?let me know if (?:you need|you want|there's|there is) anything else"
    r"|anything else\s*\?"
    r")[^.!?\n]*[.!?]*"
)
# "which" is also a relative pronoun ("Jordan Lee, which was the only match"),
# so it only counts when it opens a question, in interrogative word order,
# before a list of alternatives ("Which Jordan: Park, Lee, or Kim"), or as the
# object of a request.
_WHICH_QUESTION = re.compile(
    _SENTENCE_START + r"(?:(?:so|and|ok|okay|then|also|got it|sure)\W+)?which\b[^.!?\n]*\?"
    r"|\bwhich\s+(?:\w+\s+){0,3}?(?:do|did|would|should|shall|can|could|will|are|is)\s+"
    r"(?:you|i|we)\b"
    r"|\bwhich\s+\w+(?:\s+\w+)?\s*:\s*\w+(?:\s*,\s*\w+)*,?\s+or\b"
    r"|\b(?:know(?:ing)?|tell me|confirm|specify|clarify|sure|unclear|mean|indicate)\s+which\b"
)


def asks_for_clarification(answer: str) -> bool:
    """Check if the answer asks for clarification.

    Covers common LLM patterns for requesting more information,
    including direct questions, conditional qualifications, and
    politeness-prefixed requests.
    """
    low = _GENERIC_OFFER.sub(" ", answer.lower().translate(_PROSE_TRANSLATION))
    direct_question = re.search(
        r"\b(?:what\s+(?:time|date|day|should|topic|would you like|do you want|are you looking for)"
        r"|when\s+(?:is|are|should|would|could|can|do|does|will)"
        r"|who\s+(?:do you want|would you like|is the|are you|to send)"
        r"|search\s+for\s+what\b"
        r"|what(?:'s|\s+is)\s+the\s+(?:search\s+)?(?:query|full\s+name|exact\s+name|file\s*name|title))\b",
        low,
    )
    if _WHICH_QUESTION.search(low):
        return True
    return direct_question is not None or any(
        word in low
        for word in (
            # Direct question forms
            "clarify",
            "could you",
            "can you tell me",
            "please tell me",
            "what is the",
            "who should",
            "where should",
            "did you mean",
            "do you mean",
            # Request forms
            "please provide",
            "please specify",
            "please confirm",
            "please let me know",
            "let me know who",
            "need more information",
            "more details",
            "can you clarify",
            "would you mind",
            "help me understand",
            # Polite conditional (Qwen / Gemma style)
            "could you specify",
            "could you provide",
            "would you like me to",
            "would you prefer",
            "i'd need to know",
            "i would need to know",
            # Confirmation-seeking
            "is that correct",
            "am i right",
            "just to confirm",
            "to make sure",
            "before i proceed",
            "before proceeding",
            # Ambiguity flagging (Mistral style)
            "ambiguous",
            "multiple options",
            "not sure which",
            "unclear which",
            "several possibilities",
            # Command-R / Phi additional patterns
            "can you elaborate",
            "what do you mean",
            "could you be more specific",
            "specify which",
            "to clarify",
            "for clarity",
            "i want to make sure",
            "i need to confirm",
            "a few options",
            "a couple of options",
            "which one",
            "what kind of",
        )
    )
