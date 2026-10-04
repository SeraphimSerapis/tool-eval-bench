"""Built-in labelled items for the decision-model benchmark.

Every item is a short message, one question, and one gold answer.  The items are
written to be unambiguous, because a decision model's probabilities are only
worth calibrating against labels a reader would also agree with.

Routing items additionally produce two variants, built deterministically from
the base item:

``shuffled``
    The same options in reverse order.  A model that reads the options, not
    their position, answers the same.
``opaque``
    The options renamed to meaningless labels, with only the descriptions
    carrying the meaning.  A model that memorised the word "billing" fails
    here; one that follows the criteria passes.

Variants are scored for robustness and never counted in the headline accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tool_eval_bench.domain.decision import (
    ChoiceQuestion,
    DecisionQuestion,
    ScoreQuestion,
    YesNoQuestion,
)

# The wire format keys answers by question name.  Each item asks one question.
QUESTION_NAME = "decision"

CATEGORY_ROUTING = "routing"
CATEGORY_MODERATION = "moderation"
CATEGORY_URGENCY = "urgency"
CATEGORY_YES_NO = "yes-no"
CATEGORY_ABSTAIN = "abstain"
CATEGORIES = (
    CATEGORY_ROUTING,
    CATEGORY_MODERATION,
    CATEGORY_URGENCY,
    CATEGORY_YES_NO,
    CATEGORY_ABSTAIN,
)

VARIANT_BASE = "base"
VARIANT_SHUFFLED = "shuffled"
VARIANT_OPAQUE = "opaque"

# A gold answer is an option name, a score level, or a yes/no verdict.
Gold = str | int | bool


@dataclass(frozen=True)
class DecisionItem:
    id: str
    category: str
    state: str
    question: DecisionQuestion
    gold: Gold
    variant: str = VARIANT_BASE
    # Id of the base item a variant derives from.
    base_id: str | None = None
    # Maps this item's option names back to the base item's, for variants that
    # rename options.  Empty when the names are unchanged.
    to_base: Mapping[str, str] = field(default_factory=dict)


_ROUTING = ChoiceQuestion(
    instructions="Which team should handle this message?",
    options={
        "billing": "payments, charges, refunds, invoices",
        "shipping": "delivery, tracking, lost or damaged parcels",
        "technical": "app bugs, crashes, errors, slow performance",
        "account": "sign-in, password, profile, closing the account",
    },
)

_ROUTING_ITEMS: tuple[tuple[str, str], ...] = (
    (
        "I was charged twice for my March subscription and nobody has replied to my ticket.",
        "billing",
    ),
    ("Please send me an invoice with our company VAT number for last month's payment.", "billing"),
    (
        "My refund for the cancelled order still hasn't shown up on my card after three weeks.",
        "billing",
    ),
    ("Why did the price go up from $9 to $12 without any notice?", "billing"),
    ("Tracking says delivered but there is nothing at my door.", "shipping"),
    ("My package has been stuck in the sorting center since Monday.", "shipping"),
    ("The box arrived crushed and the lamp inside is broken.", "shipping"),
    ("Can you change the delivery address? The courier hasn't picked it up yet.", "shipping"),
    ("The app crashes every time I open the camera screen on my Pixel.", "technical"),
    ("Exporting to PDF fails with error 502 for large reports.", "technical"),
    ("The dashboard takes over a minute to load since yesterday's update.", "technical"),
    ("The sync button spins forever and never finishes.", "technical"),
    ("I forgot my password and the reset email never arrives.", "account"),
    ("Please delete my account and all my personal data.", "account"),
    ("How do I change the email address on my profile?", "account"),
    ("I'm locked out after too many sign-in attempts.", "account"),
    # Harder: negation, typos, and indirect phrasing with no keyword to match.
    ("I'm not asking for money back. The app just crashes whenever I upload a photo.", "technical"),
    ("ordr never came, trackin stuck on same page 2 wks", "shipping"),
    ("There's a charge on my statement from you that I don't recognize.", "billing"),
    ("Someone else has been signing in with my credentials. Please lock it down.", "account"),
    ("Every time I tap Save the whole thing just closes itself.", "technical"),
    (
        "I don't care about the invoice anymore. Where is the parcel? It's been two weeks.",
        "shipping",
    ),
    ("Great product, but please stop charging my old card, I switched banks.", "billing"),
    ("Please merge my two profiles, I accidentally signed up twice.", "account"),
)

_MODERATION = ChoiceQuestion(
    instructions="How should this community message be classified?",
    options={
        "ok": "a normal, on-topic message",
        "spam": "ads, scams, or repeated promotion",
        "abusive": "insults, threats, or harassment",
    },
)

_MODERATION_ITEMS: tuple[tuple[str, str], ...] = (
    ("Thanks for the quick fix, the update works great now.", "ok"),
    ("Does anyone know if the API supports pagination on the /orders endpoint?", "ok"),
    ("I disagree with the roadmap, but I appreciate the transparency.", "ok"),
    ("lol that was the worst meeting ever, but the snacks were fine", "ok"),
    ("CLICK HERE to win a free iPhone!!! Limited time http://bit.example/win", "spam"),
    ("Earn $5000 a week working from home, DM me for the secret method.", "spam"),
    ("Buy cheap watches at best-prices.example, 90% off, order now now now.", "spam"),
    ("You are a complete idiot and nobody wants you on this team.", "abusive"),
    ("Shut up, you worthless clown, or I will find where you live.", "abusive"),
    ("You're pathetic. Quit before you embarrass yourself any further.", "abusive"),
)

_URGENCY = ScoreQuestion(
    instructions="How urgent is this request?",
    levels=("can wait", "this week", "today", "right now"),
)

_URGENCY_ITEMS: tuple[tuple[str, int], ...] = (
    ("It would be nice to have a dark mode at some point.", 0),
    ("Small typo in the footer: 'Contcat us'.", 0),
    ("The monthly report export is missing a column. Our review is next Friday.", 1),
    ("Our SSL certificate expires in six days.", 1),
    ("The client demo is tomorrow morning and the staging login is failing.", 2),
    ("Our nightly backup has failed twice in a row.", 2),
    ("The production database is down and every customer sees errors.", 3),
    ("Payment processing is returning 500 for every order right now.", 3),
    ("There is a ransom note on the file server and files are being encrypted.", 3),
)

_YES_NO_ITEMS: tuple[tuple[str, str, bool], ...] = (
    ("Is the customer angry?", "This is unacceptable, third time you ruined my order!!!", True),
    (
        "Is the customer angry?",
        "I am absolutely livid. Fix this today or I'm going to the press.",
        True,
    ),
    ("Is the customer angry?", "Thanks, the replacement arrived and works fine.", False),
    ("Is the customer angry?", "Quick question about how invoice numbers are assigned.", False),
    (
        "Is the customer asking for a refund?",
        "Please refund the $40, I never used the service.",
        True,
    ),
    ("Is the customer asking for a refund?", "I want my money back for order #4411.", True),
    ("Is the customer asking for a refund?", "Where can I download my invoice?", False),
    ("Is the customer asking for a refund?", "How do I upgrade to the annual plan?", False),
    (
        "Does the message contain personal contact details?",
        "Call me at 415-555-0132 or write to jane.doe@example.com.",
        True,
    ),
    (
        "Does the message contain personal contact details?",
        "My home address is 14 Elm Street, Springfield.",
        True,
    ),
    (
        "Does the message contain personal contact details?",
        "The printer on the third floor is out of toner.",
        False,
    ),
    (
        "Does the message contain personal contact details?",
        "Release notes for version 2.4 are out.",
        False,
    ),
)

_ABSTAIN = ChoiceQuestion(
    instructions="Which team should handle this message? Choose none if no team fits.",
    options={
        "billing": "payments, charges, refunds, invoices",
        "shipping": "delivery, tracking, lost or damaged parcels",
        "technical": "app bugs, crashes, errors, slow performance",
        "none": "the message fits none of the other options",
    },
)

_ABSTAIN_ITEMS: tuple[tuple[str, str], ...] = (
    ("What's the weather like in Lisbon this weekend?", "none"),
    ("Can you recommend a good book on Roman history?", "none"),
    ("Good morning!", "none"),
    ("Do you have any openings for a senior designer?", "none"),
    ("Congratulations on the funding round!", "none"),
    ("I'd like to talk to someone about a partnership.", "none"),
    # Controls: an option does fit, so always answering "none" costs accuracy.
    ("I was billed twice this month.", "billing"),
    ("Where is my parcel? It was due on Tuesday.", "shipping"),
)

# Opaque labels are assigned in a different order from the base options, so a
# model that falls back on "the first option" gains nothing from the rename.
_OPAQUE_LABELS = ("delta", "alpha", "charlie", "bravo")


def _shuffled(item: DecisionItem, question: ChoiceQuestion) -> DecisionItem:
    reversed_options = dict(reversed(list(question.options.items())))
    return DecisionItem(
        id=f"{item.id}~shuffled",
        category=item.category,
        state=item.state,
        question=ChoiceQuestion(question.instructions, reversed_options),
        gold=item.gold,
        variant=VARIANT_SHUFFLED,
        base_id=item.id,
    )


def _opaque(item: DecisionItem, question: ChoiceQuestion) -> DecisionItem:
    renames = dict(zip(question.options, _OPAQUE_LABELS, strict=True))
    return DecisionItem(
        id=f"{item.id}~opaque",
        category=item.category,
        state=item.state,
        question=ChoiceQuestion(
            question.instructions,
            {renames[name]: desc for name, desc in question.options.items()},
        ),
        gold=renames[str(item.gold)],
        variant=VARIANT_OPAQUE,
        base_id=item.id,
        to_base={opaque: name for name, opaque in renames.items()},
    )


def build_items(*, variants: bool = True) -> list[DecisionItem]:
    """Return the base items, followed by routing variants when ``variants``."""
    base: list[DecisionItem] = []
    for i, (state, gold) in enumerate(_ROUTING_ITEMS, 1):
        base.append(DecisionItem(f"route-{i:02d}", CATEGORY_ROUTING, state, _ROUTING, gold))
    for i, (state, gold) in enumerate(_MODERATION_ITEMS, 1):
        base.append(DecisionItem(f"mod-{i:02d}", CATEGORY_MODERATION, state, _MODERATION, gold))
    for i, (state, level) in enumerate(_URGENCY_ITEMS, 1):
        base.append(DecisionItem(f"urg-{i:02d}", CATEGORY_URGENCY, state, _URGENCY, level))
    for i, (instructions, state, verdict) in enumerate(_YES_NO_ITEMS, 1):
        base.append(
            DecisionItem(
                f"yn-{i:02d}", CATEGORY_YES_NO, state, YesNoQuestion(instructions), verdict
            )
        )
    for i, (state, gold) in enumerate(_ABSTAIN_ITEMS, 1):
        base.append(DecisionItem(f"abs-{i:02d}", CATEGORY_ABSTAIN, state, _ABSTAIN, gold))

    if not variants:
        return base
    routing = [item for item in base if item.category == CATEGORY_ROUTING]
    return (
        base
        + [_shuffled(item, _ROUTING) for item in routing]
        + [_opaque(item, _ROUTING) for item in routing]
    )
