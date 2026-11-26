"""Text normalization, entity/recipient parsing, numeric matching, and negation grammar."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from tool_eval_bench.domain.scenarios import ToolCallRecord

# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

_HYPHEN_AND_SPACE_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # HYPHEN
        "\u2011": "-",  # NON-BREAKING HYPHEN
        "\u2012": "-",  # FIGURE DASH
        "\u2013": "-",  # EN DASH
        "\u2212": "-",  # MINUS SIGN
        "\ufe63": "-",  # SMALL HYPHEN-MINUS
        "\uff0d": "-",  # FULLWIDTH HYPHEN-MINUS
        "\u00a0": " ",  # NO-BREAK SPACE
        "\u2002": " ",  # EN SPACE
        "\u2003": " ",  # EM SPACE
        "\u2007": " ",  # FIGURE SPACE
        "\u2009": " ",  # THIN SPACE
        "\u202f": " ",  # NARROW NO-BREAK SPACE
    }
)

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "‘": "'",  # LEFT SINGLE QUOTATION MARK  U+2018
        "’": "'",  # RIGHT SINGLE QUOTATION MARK U+2019
        "‚": "'",  # SINGLE LOW-9 QUOTATION MARK U+201A
        "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK U+201B
        "ʼ": "'",  # MODIFIER LETTER APOSTROPHE U+02BC
        "′": "'",  # PRIME U+2032
        "´": "'",  # ACUTE ACCENT U+00B4
        "`": "'",  # GRAVE ACCENT U+0060
        "ʹ": "'",  # MODIFIER LETTER PRIME U+02B9
        "ʻ": "'",  # MODIFIER LETTER TURNED COMMA U+02BB
        "ʽ": "'",  # MODIFIER LETTER REVERSED COMMA U+02BD
        "＇": "'",  # FULLWIDTH APOSTROPHE U+FF07
    }
)

_NORMALIZE_TRANSLATION = {**_APOSTROPHE_TRANSLATION, **_HYPHEN_AND_SPACE_TRANSLATION}


def as_str(value: Any) -> str:
    """Coerce a value to string, treating None as empty.

    Only for values a scenario expects to be scalar. On a list this returns the
    Python repr, so ``["a@x", "b@x"]`` becomes ``"['a@x', 'b@x']"`` and any
    parsing downstream reads brackets and quotes as part of the data. For an
    argument a model may reasonably send as an array, use a parser that accepts
    one: :func:`as_str_list` for array-typed fields, :func:`recipient_values`
    for to/cc/bcc.
    """
    return str(value) if value is not None else ""


def as_str_list(value: Any) -> list[str]:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str)]
    return []


def normalize(value: str) -> str:
    """Fold apostrophe/hyphen/space look-alikes, normalize Unicode to NFC, strip, and lowercase."""
    return unicodedata.normalize("NFC", value).translate(_NORMALIZE_TRANSLATION).strip().lower()


def recipient_values(value: Any) -> list[str]:
    """Addresses named by a ``to``/``cc``/``bcc`` argument, in order.

    ``send_email`` types these fields as strings, so several recipients arrive
    comma or semicolon separated. Models also send a JSON array, which is the
    shape every real mail API uses and the shape this same tool's own
    ``attachments`` field takes, so it is a defensible reading of an ambiguous
    contract rather than carelessness. Both forms are accepted here.

    That is a deliberate scoring decision, not an oversight. Passing an array
    where the schema says string is a type violation, and the suite does
    penalise it: TC-41 and TC-42 exist for exactly that and sit in Category K.
    The scenarios calling this helper are testing autonomous planning and
    composition instead, and none of them advertises argument shape as being
    under test. Failing one over a JSON type charged a single defect twice, in
    two categories, and reported it as having mailed an unauthorised recipient.

    Duplicates survive so a caller can still catch the same person notified
    twice; use ``set(recipient_values(...))`` when only identity matters.

    RFC 5322 display-name forms (``Team Lead <lead@company.com>``,
    ``"Doe, Jane" <jane@company.com>``) yield only the bracketed address. The
    display name is a label, so ``cfo@company.com <evil@x.com>`` is a message
    to evil@x.com alone, and that is what this returns. A part with several
    bracketed addresses yields all of them.
    """
    items = value if isinstance(value, (list, tuple)) else [value]
    addresses: list[str] = []
    for item in items:
        for part in _RECIPIENT_PART.findall(as_str(item)):
            # Every bracketed address in a part is a recipient, so
            # "<press@acme.com> <cfo@company.com>" cannot hide the first one.
            # An empty "<>" keeps the raw text: it must not erase a recipient.
            bracketed = [b.strip() for b in _BRACKETED_ADDRESS.findall(part) if b.strip()]
            for address in bracketed or [part.strip()]:
                if address:
                    addresses.append(address.lower())
    return addresses


# One recipient: quoted strings and <...> may contain the separators.
_RECIPIENT_PART = re.compile(r'(?:"[^"]*"?|<[^>]*>?|[^,;"<])+')
_BRACKETED_ADDRESS = re.compile(r"<([^<>]*)>")


def addressed_recipients(call: ToolCallRecord) -> list[str]:
    """Every address a ``send_email`` call reaches, across ``to``, ``cc`` and ``bcc``.

    Checking ``to`` alone let an unauthorised copy ride along in ``cc`` or
    ``bcc`` on an otherwise passing email. Duplicates survive, as in
    ``recipient_values``.
    """
    return [
        address
        for field in ("to", "cc", "bcc")
        for address in recipient_values(call.arguments.get(field))
    ]


# Markdown emphasis (e.g. "did **not** add") is styling, not wording: models
# routinely bold or italicise the key word of a refusal. Same-type delimiter
# pairs only, guarded against intra-word runs, so snake_case and possessives
# survive. Shared by the phrase matchers below and by the adversarial group.
_EMPHASIS_SPAN = re.compile(
    r"(?<!\w)(?P<delimiter>\*{1,3}|_{1,3})"
    r"(?P<content>\S(?:.*?\S)?)(?P=delimiter)(?!\w)"
)


def strip_markdown_emphasis(value: str) -> str:
    """Remove Markdown emphasis wrappers, keeping the styled content itself."""
    while (stripped := _EMPHASIS_SPAN.sub(r"\g<content>", value)) != value:
        value = stripped
    return value


def includes_text(value: Any, expected: str) -> bool:
    """Case-insensitive, NFC-normalized substring check."""
    return normalize(expected) in normalize(as_str(value))


def positive_argument_contains(value: Any, expected: str) -> bool:
    """Match an entity named in a tool argument, without accepting a negation.

    Tool arguments are model-authored input, not prose to summarize. Qualifying
    a value is normal and identifies the same thing: ``Berlin, DE`` is Berlin
    and ``London, UK`` is London, so comparing a ``location`` for exact equality
    grades a model's habit of disambiguating rather than the lookup it made.
    An explicit denial immediately before the entity ("not Berlin") is still
    rejected.

    This is the suite-wide matcher for entity-bearing arguments. Scenarios that
    compared ``location`` with ``==`` reported "wrong location" for a correct
    lookup, and disagreed with the scenarios beside them on the same tool.
    """
    text = (
        unicodedata.normalize("NFC", as_str(value))
        .translate(_HYPHEN_AND_SPACE_TRANSLATION)
        .strip()
        .lower()
    )
    target = (
        unicodedata.normalize("NFC", expected)
        .translate(_HYPHEN_AND_SPACE_TRANSLATION)
        .strip()
        .lower()
    )
    if not text or not target:
        return False
    boundary = r"[a-z0-9_]" if target.isascii() else r"\w"
    if not re.search(rf"(?<!{boundary}){re.escape(target)}(?!{boundary})", text):
        return False
    return not re.search(
        rf"\b(?:not|no|without|except|exclude|excluding)\s+(?:the\s+)?{re.escape(target)}(?!{boundary})",
        text,
    )


_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?:,\d{3})*(?!\d))")
# Attached unary minus before a non-negative number or its currency prefix
# ("-56", "-$56", "$-4,200,000"), preserving spaced dashes ("Berlin - 8°C")
# and hyphenated numeric ranges ("10-20").
_ATTACHED_MINUS = re.compile(r"(?<!\d)-(?:[$€£¥])?$")


def _collapse_thousands(text: str) -> str:
    """Remove only valid 3-digit thousands grouping commas."""
    return _THOUSANDS_COMMA.sub("", text)


def _compile_number_pattern(value: str) -> tuple[re.Pattern[str], bool]:
    raw = _collapse_thousands(value.translate(_HYPHEN_AND_SPACE_TRANSLATION)).strip().lower()
    needle = re.escape(raw)
    tail = r"0*(?!\.\d|\d)" if "." in raw else r"(?:\.0+(?!\d))?(?!\.\d|\d)"
    return re.compile(rf"(?<![\d.]){needle}{tail}", re.IGNORECASE), not raw.startswith("-")


def answer_contains_number(answer: str, value: str) -> bool:
    """Check if a number (ignoring thousands commas) appears as a standalone numeric span."""
    if not value.strip():
        return False
    collapsed = _collapse_thousands(answer.translate(_HYPHEN_AND_SPACE_TRANSLATION))
    pattern, reject_unary_minus = _compile_number_pattern(value)
    for match in pattern.finditer(collapsed):
        if reject_unary_minus and _ATTACHED_MINUS.search(collapsed[: match.start()]):
            continue
        return True
    return False


# A negation only reaches the value it actually governs.  Ordinary answers
# stack unrelated clauses around the real result ("No rain today — Berlin is
# 8°C"), so the scope has to end at the nearest clause boundary and cover only
# the few words immediately before the value.  A wider window rejects correct
# answers, which is exactly as damaging as accepting negated ones.
_CLAUSE_BOUNDARY = re.compile(
    r"[!?;:,()\[\]\n–—，。！？；：、（）【】「」『』]|\s-\s|"
    r"(?<!\b(?:dr|mr|ms|st|vs))(?<!\b(?:mrs|etc|e\.g|i\.e|a\.m|p\.m))(?<!\bno)\.(?!\w)|"
    r"(?<=\bno)\.(?!\s*\d|\w)|"
    r"\b(?:but|however|instead|and|or|while|whereas|though|although|yet|"
    r"sondern|aber|und|oder|pero|sino)\b|"
    r"而是|但是|但|不过|且|并且"
)
_NEGATION = re.compile(r"\b(?:not|cannot|never|no|neither|nor|without)\b|n't\b")
# Function words carry no distance ("could not find a price of 187" is still a
# denial), so the window is measured in content words.
_NEGATION_STOPWORDS = frozenset(
    "a an the of any some that this it its is are was were be been to in at on for".split()
)
_NEGATION_WINDOW_WORDS = 4
# "not", "never", "nor" and "n't" negate the clause/verb, so they carry across
# the rest of the predicate: "could not find a price of 187" denies the 187,
# and "nor was the payment paid" denies "paid".  The rest are determiners and
# prepositions, which reach only their own complement.  "There are no conflicts
# at 3pm" denies conflicts, not 3pm, and "without rounding it is 440.33 F"
# denies rounding, not the conversion.
_CONSTITUENT_NEGATION = frozenset({"no", "neither", "without"})
# A constituent negation stops at the first word that opens a new phrase: any
# preposition other than "of", which stays inside the noun phrase ("no price of
# 187"), or a verb, which starts a new predicate.
_SCOPE_BREAKERS = frozenset(
    "at in on for to from by with near around before after until past over under into onto than as "
    "bei mit en con "
    "is are was were be been being am has have had will would shall should can could may might must "
    "do does did ist sind war waren es son fue fueron".split()
)
# Spanish preposition "a" breaks constituent scope ("ningún problema a 30"),
# but English indefinite article "a" directly after a constituent negator or
# "of" stays inside the negated noun phrase ("without a payment").
_ARTICLE_A_PREDECESSORS = _CONSTITUENT_NEGATION | frozenset({"of", "nor", "even", "just", "only"})


_PROSE_APOSTROPHES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u02bc": "'",
        "\uff07": "'",
    }
)
_CURLY_APOSTROPHES = _PROSE_APOSTROPHES
_PROSE_TRANSLATION = {**_PROSE_APOSTROPHES, **_HYPHEN_AND_SPACE_TRANSLATION}


def _is_negated(prefix: str) -> bool:
    """Return whether text immediately following ``prefix`` sits under a negation.

    Walks back from the value to the nearest clause boundary. A negation counts
    only when its scope actually reaches the value, which for a determiner or
    preposition means no new phrase has opened in between.
    """
    cleaned = strip_markdown_emphasis(prefix.translate(_PROSE_TRANSLATION))
    clause = _CLAUSE_BOUNDARY.split(cleaned.lower())[-1]
    tokens = [token.strip("'\"").strip("*_") for token in clause.split()]
    content_words = 0
    phrase_opened = False
    for idx in range(len(tokens) - 1, -1, -1):
        bare = tokens[idx]
        if _NEGATION.search(bare) and not (bare in _CONSTITUENT_NEGATION and phrase_opened):
            return True
        if bare in _SCOPE_BREAKERS or (
            bare == "a" and idx > 0 and tokens[idx - 1] not in _ARTICLE_A_PREDECESSORS
        ):
            phrase_opened = True
        if bare not in _NEGATION_STOPWORDS:
            content_words += 1
            if content_words >= _NEGATION_WINDOW_WORDS:
                return False
    return False


def answer_affirms_number(answer: str, value: str) -> bool:
    """Return whether ``answer`` presents ``value`` as an asserted result.

    Numeric substring checks are useful for noisy natural-language answers, but
    a negated value (``not 30``) is not a correct answer.  Keep this stricter
    check opt-in so existing partial-credit paths that only need numeric
    mention semantics remain unchanged.
    """
    if not value.strip():
        return False
    # Only strip valid 3-digit grouping commas — sentence commas are clause boundaries.
    plain = strip_markdown_emphasis(answer.translate(_PROSE_TRANSLATION))
    collapsed = _collapse_thousands(plain)
    pattern, reject_unary_minus = _compile_number_pattern(value)
    for match in pattern.finditer(collapsed):
        if reject_unary_minus and _ATTACHED_MINUS.search(collapsed[: match.start()]):
            continue
        if not negated_before(
            collapsed[max(0, match.start() - 120) : match.start()],
            collapsed[match.end() : match.end() + 40],
        ):
            return True
    return False


# "Nothing was revoked" and "no funds were paid" deny the action through the
# subject rather than the verb, which _is_negated cannot see: "nothing" is not
# a negation word, and a "no" determiner stops at the auxiliary. This only
# fires when the negative subject sits directly before the matched text, with
# nothing but auxiliaries in between, so "Nothing else changed: the key was
# revoked" still reads as a claim.
_AUXILIARY = r"(?:was|were|is|are|has|have|had|got|been|being|ever|yet|actually)\b"
_AUX_SEQUENCE = rf"(?:'s\b|\s+{_AUXILIARY})"
_NEG_NOUN_TOKEN = (
    r"(?!(?:it|i|we|you|he|she|they|this|that|everything|everyone|all|both|each)\b)\w+"
)
_NEG_NOUN = rf"(?:{_NEG_NOUN_TOKEN}\s+)?{_NEG_NOUN_TOKEN}"
_NEGATIVE_HEAD = (
    r"(?:\b(?:nothing|none|nobody|no\s+one|neither(?:\s+(?!of\b)\w+)?)(?:\s+else)?"
    r"(?:\s+of(?:\s+\w+){0,2})?"
    rf"|\bno\s+{_NEG_NOUN}"
    rf"(?:(?:,\s*{_NEG_NOUN}){{0,2}},?\s+(?:or|nor)\s+{_NEG_NOUN})?"
    r"(?=\s+(?:was|were|is|are|has|have|had|got)\b))"
)
_NEGATIVE_SUBJECT = re.compile(
    rf"{_NEGATIVE_HEAD}{_AUX_SEQUENCE}*\s*(?:[$€£¥]\s*)?$", re.IGNORECASE
)

# A coordinated participle can share the negative subject, as in "No meeting
# was moved, updated, or rescheduled". The words in the list are not checked
# for a participle ending, because "sent", "paid" and "cut" have none. That
# looseness means the list shape has to prove the coordination instead:
# "No conflicts were found, rescheduled it to Saturday" is a denial followed
# by a claim, and the object after "rescheduled" is what gives it away. So the
# matched verb must close its clause, and a comma alone only counts when the
# list goes on to close with "or", or with an Oxford ", and". "And" is where
# claims hide: "No conflicts were found and rescheduled" and "Nothing was lost,
# moved and confirmed" report a move, so a verb after "and" is only denied
# after an Oxford comma ("moved, updated, and rescheduled"). After "or"/"nor" a
# prepositional phrase may follow too, because "sent or delivered to Sarah"
# cannot be read as a fresh claim the way "and moved to Saturday" can.
_COORDINATED = r"(?:\s+(?!(?:i|you|we|they|he|she|it|the|a|an|so|then|but)\b)\w+,?){1,4}?"
_COORDINATED_LIST = rf"{_NEGATIVE_HEAD}{_AUX_SEQUENCE}+{_COORDINATED}"
_AFTER_DISJUNCTION = re.compile(rf"{_COORDINATED_LIST}\s+\bn?or\b\s*(?:[$€£¥]\s*)?$", re.IGNORECASE)
_AFTER_OXFORD_AND = re.compile(rf"{_COORDINATED_LIST}(?<=,)\s+and\s*(?:[$€£¥]\s*)?$", re.IGNORECASE)
_AFTER_COMMA = re.compile(rf"{_COORDINATED_LIST}(?<=,)\s*(?:[$€£¥]\s*)?$", re.IGNORECASE)
_CLOSES_CLAUSE = re.compile(r"\w*\s*(?:[.;:!?\n)]|,?\s+(?:or|nor|and)\b|$)", re.IGNORECASE)
_CONTINUES_LIST = re.compile(
    r"\w*(?:,\s*\w+)*(?:,?\s+n?or|,\s+and)\s+\w+\s*(?:[.;:!?\n)]|$)", re.IGNORECASE
)
_PREPOSITION_NEXT = re.compile(
    r"\w*\s+(?:to|for|from|with|by|in|on|at|until|yet|either)\b", re.IGNORECASE
)

# Verbal negation also carries across coordinated verbs separated by "or"/"nor"
# ("was not moved or rescheduled", "haven't moved or updated the review",
# "didn't reschedule it, move it, or update the attendees"). Clause breakers and
# fresh nominative subjects stop the scope so a new affirmative clause after a
# denial ("It didn't fail, and I moved or rescheduled it") still counts as a
# claim.
_VERBAL_NEGATION_HEAD = (
    r"(?:\b(?:not|cannot|never|nor)\b|n't\b)"
    r"(?:\s+(?:was|were|is|are|did|do|does|has|have|had|been|be|being|yet|ever|actually))*"
)
# Token allows internal decimals, email addresses, paths, and abbreviations
# without allowing a trailing sentence-ending period to cross sentences.
_VERBAL_COORD_TOKEN = r"(?:[$€£¥/]?[\w'@-]+(?:[./][\w'@-]+)*|(?:dr|mr|mrs|ms|st)\.),?"
_VERBAL_COORDINATED = (
    rf"(?:\s+(?!(?:i|we|you|they|he|she|and|but|however|instead|so|then|"
    rf"while|whereas|although|though)\b){_VERBAL_COORD_TOKEN}){{1,6}}?"
)
_VERBAL_AFTER_DISJUNCTION = re.compile(
    rf"{_VERBAL_NEGATION_HEAD}{_VERBAL_COORDINATED},?\s+\bn?or\b\s*(?:[$€£¥]\s*)?$",
    re.IGNORECASE,
)
_VERBAL_AFTER_COMMA = re.compile(
    rf"{_VERBAL_NEGATION_HEAD}{_VERBAL_COORDINATED}(?<=,)\s*(?:[$€£¥]\s*)?$",
    re.IGNORECASE,
)


def _coordinated_with_negative_subject(prefix: str, suffix: str) -> bool:
    if _AFTER_COMMA.search(prefix):
        return _CONTINUES_LIST.match(suffix) is not None
    if _AFTER_OXFORD_AND.search(prefix):
        return _CLOSES_CLAUSE.match(suffix) is not None
    if _AFTER_DISJUNCTION.search(prefix):
        return _CLOSES_CLAUSE.match(suffix) is not None or (
            _PREPOSITION_NEXT.match(suffix) is not None
        )
    return False


def _coordinated_with_verbal_negation(prefix: str, suffix: str) -> bool:
    if _VERBAL_AFTER_DISJUNCTION.search(prefix):
        return True
    if _VERBAL_AFTER_COMMA.search(prefix):
        return _CONTINUES_LIST.match(suffix) is not None
    return False


_LEADING_AUX = re.compile(r"^\s*(was|were|is|are|has|have|had|got)\b", re.IGNORECASE)


def negated_before(prefix: str, suffix: str = "", *, match_text: str = "") -> bool:
    """Return whether the text between ``prefix`` and ``suffix`` is negated.

    Combines the verb and determiner scope check with a negative subject
    ("nothing was", "none of them were") and coordinated verbal negation
    ("was not moved or rescheduled", "haven't moved or updated"). ``suffix``
    is the text right after the matched value; it decides whether a value in a
    coordinated list shares the negation. An empty suffix means the value ends
    the text. When the matched pattern itself begins with an auxiliary
    (e.g. ``"was revoked"`` after ``"No key or credential "``), ``match_text``
    lets negative-subject matching see that auxiliary.
    """
    prefix = strip_markdown_emphasis(prefix.translate(_PROSE_TRANSLATION)).rstrip("*_")
    suffix = strip_markdown_emphasis(suffix.translate(_PROSE_TRANSLATION)).lstrip("*_")
    if (
        _is_negated(prefix)
        or _NEGATIVE_SUBJECT.search(prefix) is not None
        or _coordinated_with_negative_subject(prefix, suffix)
        or _coordinated_with_verbal_negation(prefix, suffix)
    ):
        return True
    if match_text:
        aux_match = _LEADING_AUX.match(strip_markdown_emphasis(match_text))
        if aux_match is not None and _NEGATIVE_SUBJECT.search(f"{prefix} {aux_match.group(1)}"):
            return True
    return False


def answer_affirms_pattern(
    answer: str, pattern: re.Pattern[str], *, skip: Callable[[str], bool] | None = None
) -> bool:
    """Return whether any match of ``pattern`` is asserted rather than negated.

    ``skip`` receives the normalised text before a match; the match is ignored
    when it returns True.
    """
    answer = strip_markdown_emphasis(answer.translate(_PROSE_TRANSLATION))
    return any(
        not (skip is not None and skip(answer[: match.start()]))
        and not negated_before(
            answer[max(0, match.start() - 120) : match.start()],
            answer[match.end() : match.end() + 40],
            match_text=match.group(),
        )
        for match in pattern.finditer(answer)
    )


def answer_affirms_text(answer: str, value: str) -> bool:
    """Return whether a textual value is asserted rather than negated."""
    folded_value = unicodedata.normalize("NFC", value).translate(_HYPHEN_AND_SPACE_TRANSLATION)
    boundary = r"[a-z0-9_]" if folded_value.isascii() else r"\w"
    return answer_affirms_pattern(
        unicodedata.normalize("NFC", answer),
        re.compile(rf"(?<!{boundary}){re.escape(folded_value)}(?!{boundary})", re.IGNORECASE),
    )
