"""IFEval constraint checkers — 25 programmatic instruction-following checks.

Each checker takes a response string and kwargs dict, returns bool.
The instruction IDs follow the format ``category:check_name``.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CHECKERS: dict[str, Any] = {}


def register(instruction_id: str):
    """Decorator to register a checker function."""

    def decorator(fn):
        _CHECKERS[instruction_id] = fn
        return fn

    return decorator


def check_instruction(
    instruction_id: str,
    response: str,
    kwargs: dict[str, Any],
) -> bool:
    """Run the checker for a given instruction ID.

    Returns ``True`` if the constraint is satisfied, ``False`` otherwise.
    Raises ``KeyError`` if the instruction ID is unknown.
    """
    checker = _CHECKERS.get(instruction_id)
    if checker is None:
        raise KeyError(f"Unknown instruction ID: {instruction_id!r}")
    return checker(response, kwargs)


def available_checkers() -> list[str]:
    """Return all registered instruction IDs."""
    return sorted(_CHECKERS)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def _count_paragraphs(text: str) -> int:
    paragraphs = text.split("\n\n")
    return len([p for p in paragraphs if p.strip()])


def _relation_check(actual: int, expected: int, relation: str) -> bool:
    """Compare actual vs expected using a relation string."""
    rel = relation.lower().strip()
    if rel in ("at least", "atleast"):
        return actual >= expected
    if rel in ("at most", "atmost"):
        return actual <= expected
    if rel in ("exactly", "exact"):
        return actual == expected
    if rel in ("less than",):
        return actual < expected
    if rel in ("more than", "greater than"):
        return actual > expected
    # Default: at least
    return actual >= expected


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------


@register("length_constraints:number_words")
def check_number_words(response: str, kwargs: dict) -> bool:
    num_words = kwargs.get("num_words")
    relation = kwargs.get("relation", "at least")
    if num_words is None:
        return True
    return _relation_check(_count_words(response), num_words, relation)


@register("length_constraints:number_sentences")
def check_number_sentences(response: str, kwargs: dict) -> bool:
    num_sentences = kwargs.get("num_sentences")
    relation = kwargs.get("relation", "at least")
    if num_sentences is None:
        return True
    return _relation_check(_count_sentences(response), num_sentences, relation)


@register("length_constraints:number_paragraphs")
def check_number_paragraphs(response: str, kwargs: dict) -> bool:
    num_paragraphs = kwargs.get("num_paragraphs")
    # IFEval's paragraph prompts use an exact count unless a relation is
    # explicitly supplied.  Treating the default as "at least" lets an
    # answer with extra sections pass prompts that say "exactly N".
    relation = kwargs.get("relation") or "exactly"
    if num_paragraphs is None:
        return True
    return _relation_check(_count_paragraphs(response), num_paragraphs, relation)


@register("length_constraints:nth_paragraph_first_word")
def check_nth_paragraph_first_word(response: str, kwargs: dict) -> bool:
    nth = kwargs.get("nth_paragraph")
    first_word = kwargs.get("first_word")
    if nth is None or first_word is None:
        return True
    paragraphs = [p for p in response.split("\n\n") if p.strip()]
    if nth > len(paragraphs) or nth < 1:
        return False
    words = paragraphs[nth - 1].strip().split()
    return bool(words) and words[0].lower() == first_word.lower()


# ---------------------------------------------------------------------------
# Keyword constraints
# ---------------------------------------------------------------------------


@register("keywords:existence")
def check_keywords_existence(response: str, kwargs: dict) -> bool:
    keywords = kwargs.get("keywords")
    if not keywords:
        return True
    lower = response.lower()
    return all(kw.lower() in lower for kw in keywords)


@register("keywords:frequency")
def check_keywords_frequency(response: str, kwargs: dict) -> bool:
    keyword = kwargs.get("keyword")
    frequency = kwargs.get("frequency")
    relation = kwargs.get("relation", "at least")
    if keyword is None or frequency is None:
        return True
    count = response.lower().count(keyword.lower())
    return _relation_check(count, frequency, relation)


@register("keywords:forbidden_words")
def check_forbidden_words(response: str, kwargs: dict) -> bool:
    forbidden = kwargs.get("forbidden_words")
    if not forbidden:
        return True
    lower = response.lower()
    return not any(w.lower() in lower for w in forbidden)


@register("keywords:letter_frequency")
def check_letter_frequency(response: str, kwargs: dict) -> bool:
    letter = kwargs.get("letter")
    let_frequency = kwargs.get("let_frequency")
    let_relation = kwargs.get("let_relation", "at least")
    if letter is None or let_frequency is None:
        return True
    count = response.lower().count(letter.lower())
    return _relation_check(count, let_frequency, let_relation)


# ---------------------------------------------------------------------------
# Format constraints
# ---------------------------------------------------------------------------


@register("detectable_format:number_highlighted_sections")
def check_highlighted_sections(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_highlights")
    if num is None:
        return True
    # Count *highlighted* sections (markdown bold/italic with *)
    matches = re.findall(r"\*[^*\n]+\*", response)
    return len(matches) >= num


@register("detectable_format:number_bullet_lists")
def check_bullet_lists(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_bullets")
    if num is None:
        return True
    bullets = re.findall(r"^\s*[-*+•]\s+", response, re.MULTILINE)
    # The dataset does not carry a relation for this instruction.  Its
    # prompts consistently ask for exactly N Markdown bullets, so extra
    # bullets are a violation rather than harmless surplus.
    relation = kwargs.get("relation") or "exactly"
    return _relation_check(len(bullets), num, relation)


@register("detectable_format:number_placeholders")
def check_placeholders(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_placeholders")
    if num is None:
        return True
    placeholders = re.findall(r"\[.+?\]", response)
    return len(placeholders) >= num


@register("detectable_content:number_placeholders")
def check_content_placeholders(response: str, kwargs: dict) -> bool:
    # Same logic as detectable_format version
    return check_placeholders(response, kwargs)


@register("detectable_format:json_format")
def check_json_format(response: str, kwargs: dict) -> bool:
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last ``` lines
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


@register("detectable_format:title")
def check_title(response: str, kwargs: dict) -> bool:
    """Response should have a title — a line at the start that looks like a heading."""
    lines = response.strip().split("\n")
    if not lines:
        return False
    first = lines[0].strip()
    # Markdown heading or a short line without ending period
    if first.startswith("#"):
        return True
    return bool(first) and not first.endswith(".") and len(first.split()) <= 15


@register("detectable_format:multiple_sections")
def check_multiple_sections(response: str, kwargs: dict) -> bool:
    num_sections = kwargs.get("num_sections")
    section_splitter = kwargs.get("section_spliter")  # Note: typo is in the dataset
    if num_sections is None:
        return True
    if section_splitter:
        sections = response.split(section_splitter)
    else:
        # Default: split on markdown headings
        sections = re.split(r"\n#{1,6}\s+", response)
    non_empty = [s for s in sections if s.strip()]
    return len(non_empty) >= num_sections


@register("detectable_format:constrained_response")
def check_constrained_response(response: str, kwargs: dict) -> bool:
    """Check that a response selects one of the options named by the prompt.

    The cached IFEval rows leave the options in ``prompt`` rather than in the
    instruction kwargs.  A length-only check therefore accepts arbitrary
    answers such as ``"banana"``.  Callers that have a structured contract
    may provide ``allowed_responses`` (or ``options``/``choices``) directly;
    otherwise we extract the quoted or line-separated options from the prompt.
    """

    allowed = _allowed_responses(kwargs)
    if not allowed:
        return False

    response_text = response.strip()
    if not response_text:
        return False

    folded_response = response_text.casefold()
    matched = {option.casefold() for option in allowed if option.casefold() in folded_response}
    # A response that repeats the complete list is not a constrained choice.
    return len(matched) == 1


def _allowed_responses(kwargs: dict[str, Any]) -> list[str]:
    """Return normalized constrained-response options from kwargs or prompt."""

    for key in ("allowed_responses", "allowed_answers", "options", "choices"):
        values = kwargs.get(key)
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (list, tuple, set)):
            structured_options = [
                _normalize_option(value) for value in values if isinstance(value, str)
            ]
            structured_options = [option for option in structured_options if option]
            if structured_options:
                return list(dict.fromkeys(structured_options))

    prompt = kwargs.get("prompt") or kwargs.get("prompt_text")
    if not isinstance(prompt, str):
        return []

    # The local IFEval snapshot uses the same constrained answer family in
    # both quoted and line-separated forms.  Keep this extraction generic
    # enough for future rows without treating arbitrary quoted source text as
    # an answer option.
    cue = re.search(
        r"(?:one of the following|choose from the following|following options|following phrases)",
        prompt,
        re.IGNORECASE,
    )
    candidate_text = prompt[cue.end() :] if cue else prompt
    options: list[str] = []

    # Handle straight quotes, curly quotes, and the malformed opening quote
    # found in a few cached prompts (``”option\",``).
    quoted_pattern = re.compile(
        r'"([^"\n]+)"|“([^”\n]+)”|”([^"\n]+)"|\'([^\'\n]+)\'',
        re.IGNORECASE,
    )
    for match in quoted_pattern.finditer(candidate_text):
        option = _normalize_option(next(group for group in match.groups() if group is not None))
        if option:
            options.append(option)

    # A number of rows put each permitted phrase on its own line without
    # quotes.  Restrict these to the common explicit-answer wording so that
    # the question itself is not mistaken for an option.
    for line in candidate_text.splitlines():
        option = _normalize_option(line)
        if re.match(r"my\s+answer\s+is\b", option, re.IGNORECASE):
            options.append(option)

    # Also support the compact ``My answer is yes/no/maybe`` contract even
    # when the prompt uses prose rather than a recognizable cue.
    options.extend(
        _normalize_option(match.group(0))
        for match in re.finditer(r"my\s+answer\s+is\s+(?:yes|no|maybe)\.?", prompt, re.IGNORECASE)
    )
    return list(dict.fromkeys(option for option in options if option))


def _normalize_option(option: str) -> str:
    """Normalize punctuation used to enumerate an allowed response."""

    return re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", option).strip(" \t\"'“”(),")


# ---------------------------------------------------------------------------
# Punctuation constraints
# ---------------------------------------------------------------------------


@register("punctuation:no_comma")
def check_no_comma(response: str, kwargs: dict) -> bool:
    return "," not in response


# ---------------------------------------------------------------------------
# Start/end constraints
# ---------------------------------------------------------------------------


@register("startend:end_checker")
def check_end_phrase(response: str, kwargs: dict) -> bool:
    end_phrase = kwargs.get("end_phrase")
    if not end_phrase:
        return True
    return response.strip().endswith(end_phrase)


@register("startend:quotation")
def check_quotation(response: str, kwargs: dict) -> bool:
    text = response.strip()
    return (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
        or (text.startswith("\u201c") and text.endswith("\u201d"))
    )


# ---------------------------------------------------------------------------
# Case constraints
# ---------------------------------------------------------------------------


@register("change_case:english_uppercase")
def check_uppercase(response: str, kwargs: dict) -> bool:
    # Only check alphabetic characters
    alpha = "".join(c for c in response if c.isalpha())
    return alpha == alpha.upper() if alpha else True


@register("change_case:english_lowercase")
def check_lowercase(response: str, kwargs: dict) -> bool:
    alpha = "".join(c for c in response if c.isalpha())
    return alpha == alpha.lower() if alpha else True


@register("change_case:english_capital")
def check_capitalize(response: str, kwargs: dict) -> bool:
    """Every word should be capitalized (title case)."""
    words = response.split()
    return all(w[0].isupper() for w in words if w and w[0].isalpha())


@register("change_case:capital_word_frequency")
def check_capital_word_frequency(response: str, kwargs: dict) -> bool:
    """Check the requested number of all-uppercase words."""

    frequency = kwargs.get("capital_frequency")
    relation = kwargs.get("capital_relation", "at least")
    if frequency is None:
        return True
    capital_words = re.findall(r"\b[A-Z]+\b", response)
    return _relation_check(len(capital_words), frequency, relation)


# ---------------------------------------------------------------------------
# Combination / misc constraints
# ---------------------------------------------------------------------------


@register("combination:repeat_prompt")
def check_repeat_prompt(response: str, kwargs: dict) -> bool:
    prompt = kwargs.get("prompt_to_repeat")
    if not prompt:
        return True
    return prompt in response


@register("combination:two_responses")
def check_two_responses(response: str, kwargs: dict) -> bool:
    """Response should contain two distinct parts separated by specific markers."""
    # Common separators: "******", "---", or section markers
    separators = ["******", "---", "***"]
    for sep in separators:
        parts = response.split(sep)
        if len(parts) >= 2 and all(p.strip() for p in parts[:2]):
            return True
    return False


@register("language:response_language")
def check_response_language(response: str, kwargs: dict) -> bool:
    """Heuristically check that the response uses the requested script.

    Unicode scripts cannot distinguish every language that shares an alphabet
    (for example German and English), but they can still reject a response in
    an unrelated script.  Unknown language codes fail closed instead of being
    treated as passing constraints.
    """

    language = kwargs.get("language")
    if not language:
        return True
    lang = str(language).lower().replace("_", "-")
    aliases = {
        "english": "en",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "finnish": "fi",
        "swahili": "sw",
        "vietnamese": "vi",
        "russian": "ru",
        "bulgarian": "bg",
        "arabic": "ar",
        "persian": "fa",
        "farsi": "fa",
        "urdu": "ur",
        "bengali": "bn",
        "gujarati": "gu",
        "hindi": "hi",
        "marathi": "mr",
        "nepali": "ne",
        "punjabi": "pa",
        "kannada": "kn",
        "tamil": "ta",
        "telugu": "te",
        "thai": "th",
        "korean": "ko",
        "japanese": "ja",
        "chinese": "zh",
    }
    lang = aliases.get(lang, lang)

    letters = [char for char in response if char.isalpha()]
    if not letters:
        return False

    def ratio(predicate: Any) -> bool:
        matching = sum(1 for char in letters if predicate(char))
        return matching > 0 and matching / len(letters) >= 0.5

    def latin(char: str) -> bool:
        return "A" <= char <= "Z" or "a" <= char <= "z"

    def cyrillic(char: str) -> bool:
        return "\u0400" <= char <= "\u04ff"

    def arabic(char: str) -> bool:
        return "\u0600" <= char <= "\u06ff" or "\u0750" <= char <= "\u077f"

    def devanagari(char: str) -> bool:
        return "\u0900" <= char <= "\u097f"

    def gurmukhi(char: str) -> bool:
        return "\u0a00" <= char <= "\u0a7f"

    def gujarati(char: str) -> bool:
        return "\u0a80" <= char <= "\u0aff"

    def bengali(char: str) -> bool:
        return "\u0980" <= char <= "\u09ff"

    def tamil(char: str) -> bool:
        return "\u0b80" <= char <= "\u0bff"

    def telugu(char: str) -> bool:
        return "\u0c00" <= char <= "\u0c7f"

    def kannada(char: str) -> bool:
        return "\u0c80" <= char <= "\u0cff"

    def thai(char: str) -> bool:
        return "\u0e00" <= char <= "\u0e7f"

    def hangul(char: str) -> bool:
        return "\uac00" <= char <= "\ud7af"

    def kana_or_cjk(char: str) -> bool:
        return "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"

    def cjk(char: str) -> bool:
        return "\u3400" <= char <= "\u9fff"

    latin_markers = {
        "de": {"aber", "das", "der", "die", "eine", "ist", "nicht", "und"},
        "fi": {"että", "ja", "joka", "kun", "mutta", "myös", "olla", "on"},
        "it": {"che", "con", "della", "il", "non", "per", "sono", "una"},
        "pt": {"com", "dos", "está", "não", "para", "que", "são", "uma"},
        "sw": {"hii", "katika", "kwa", "lakini", "na", "ni", "wa", "ya"},
        "vi": {"cho", "có", "của", "không", "là", "những", "trong", "và"},
    }
    if lang in latin_markers:
        if not ratio(latin):
            return False
        words = set(re.findall(r"[^\W\d_]+", response.casefold(), re.UNICODE))
        return len(words & latin_markers[lang]) >= 2

    predicates = {
        "en": latin,
        "bg": cyrillic,
        "ru": cyrillic,
        "ar": arabic,
        "fa": arabic,
        "ur": arabic,
        "bn": bengali,
        "gu": gujarati,
        "hi": devanagari,
        "mr": devanagari,
        "ne": devanagari,
        "pa": gurmukhi,
        "kn": kannada,
        "ta": tamil,
        "te": telugu,
        "th": thai,
        "ko": hangul,
        "ja": kana_or_cjk,
        "zh": cjk,
    }
    predicate = predicates.get(lang)
    return predicate is not None and ratio(predicate)


@register("detectable_content:postscript")
def check_postscript(response: str, kwargs: dict) -> bool:
    """Require the requested, exact marker to start the final non-empty line."""

    marker = kwargs.get("postscript_marker") or "P.S."
    if not isinstance(marker, str) or not marker.strip():
        return False
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return False
    return lines[-1].startswith(marker)
