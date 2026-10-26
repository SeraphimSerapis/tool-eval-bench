"""Lexical similarity between tool definitions.

The similarity tiers are defined by the catalog's structure. This module is
the independent check: a TF-IDF cosine over the words a model reads in a tool
definition (name, description, and parameter names). It needs no model and no
dependency, so the measured axis is identical on every machine.

Lexical overlap undercounts paraphrase. That is acceptable here because the
measurement confirms the tiers rather than defines them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")

# Function words carry no signal about what a tool does, and leaving them in
# makes every pair of descriptions look a little alike.
_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it its of on or that the this to "
    "with without which when where who will can one all any only per than then".split()
)


def tool_tokens(tool: Mapping[str, Any]) -> list[str]:
    """The words a model reads in an OpenAI-format tool definition."""
    function = tool.get("function", tool)
    parts = [str(function.get("name", "")).replace("_", " "), str(function.get("description", ""))]
    properties = (function.get("parameters") or {}).get("properties") or {}
    parts.extend(str(name).replace("_", " ") for name in properties)
    words = _WORD_RE.findall(" ".join(parts).lower())
    return [word for word in words if word not in _STOPWORDS]


class TfidfSpace:
    """TF-IDF vectors with inverse document frequency fitted on a fixed corpus.

    Smoothed IDF, ``ln((1 + n) / (1 + df)) + 1``, gives a word the corpus has
    never seen the highest weight instead of a division by zero, so tools from
    outside the corpus (Category L's set, for one) can be measured in the same
    space.
    """

    def __init__(self, documents: Iterable[Sequence[str]]) -> None:
        df: Counter[str] = Counter()
        n = 0
        for doc in documents:
            n += 1
            df.update(set(doc))
        self._n = n
        self._df = df

    def idf(self, word: str) -> float:
        """Smoothed inverse document frequency of *word*."""
        return math.log((1 + self._n) / (1 + self._df.get(word, 0))) + 1

    def vector(self, tokens: Sequence[str]) -> dict[str, float]:
        """Unit-length TF-IDF vector for *tokens*."""
        counts = Counter(tokens)
        weights = {word: count * self.idf(word) for word, count in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0:
            return {}
        return {word: w / norm for word, w in weights.items()}

    @staticmethod
    def cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        """Cosine of two unit vectors."""
        if len(a) > len(b):
            a, b = b, a
        return sum(weight * b.get(word, 0.0) for word, weight in a.items())


class ToolSimilarity:
    """Pairwise similarity between tools, cached by name."""

    def __init__(self, corpus: Iterable[Mapping[str, Any]]) -> None:
        corpus = list(corpus)
        self.space = TfidfSpace(tool_tokens(tool) for tool in corpus)
        self._vectors: dict[str, dict[str, float]] = {}
        for tool in corpus:
            self.vector_of(tool)

    def vector_of(self, tool: Mapping[str, Any]) -> dict[str, float]:
        """The cached vector for *tool*, keyed by its function name."""
        name = str(tool.get("function", tool).get("name", ""))
        vector = self._vectors.get(name)
        if vector is None:
            vector = self.space.vector(tool_tokens(tool))
            self._vectors[name] = vector
        return vector

    def between(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        """Cosine similarity of two tool definitions."""
        return self.space.cosine(self.vector_of(a), self.vector_of(b))


def nearest_neighbour(
    similarity: ToolSimilarity, target: Mapping[str, Any], others: Iterable[Mapping[str, Any]]
) -> tuple[str, float] | None:
    """The tool in *others* most similar to *target*, with its score."""
    best: tuple[str, float] | None = None
    for other in others:
        score = similarity.between(target, other)
        if best is None or score > best[1]:
            best = (str(other.get("function", other).get("name", "")), score)
    return best
