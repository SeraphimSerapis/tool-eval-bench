"""TC-62 — 5-Turn Research Chain."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    addressed_recipients as _addressed_recipients,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import counted_calls as _counted_calls
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import forbid_unrequested_side_effects
from tool_eval_bench.evals.helpers import (
    negated_before as _negated_before,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.planning._shared import (
    _call_index,
    _result_has_status,
    _result_matches_if_present,
)

_TC62_FOLLOW_UPS = [
    # Turn 2: drill into detail + correction
    "Wait, I just remembered the Q3 report had a correction. Can you search for and read the latest version?",
    # Turn 3: pivot to competitor
    "OK, now compare that against our competitor. Search for Acme Corp's Q3 performance.",
    # Turn 4: action based on findings
    "Based on this analysis, draft an email to the CFO summarizing our competitive position.",
    # Turn 5: revision and explicit authorization
    "Actually, make the tone more optimistic, add that we expect Q4 to improve, and send it.",
]


def _tc62_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        if "acme" in query:
            return _noise(
                {"results": [{"snippet": "Acme Corp Q3 revenue: $3.8M. Growth rate: 12%."}]},
                "web_search",
            )
        return _noise(
            {"results": [{"snippet": "Company Q3 performance: Revenue up 8% YoY."}]},
            "web_search",
        )
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "q3_latest", "name": "Q3_Report_v2_CORRECTED.xlsx"}]},
            "search_files",
        )
    if call.name == "read_file":
        fid = _as_str(call.arguments.get("file_id", ""))
        if "latest" in fid or "v2" in fid or "correct" in fid.lower():
            return _noise(
                {
                    "content": "Q3 Report (CORRECTED)\nRevenue: $4,150,000\nNote: Previous version showed $4.4M due to accounting error."
                },
                "read_file",
            )
        return _noise(
            {"content": "Q3 Report\nRevenue: $4,400,000\nGrowth: 8% YoY"},
            "read_file",
        )
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "CFO", "email": "cfo@company.com", "role": "CFO"}]},
            "get_contacts",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


_TC62_COMPETITOR_AMOUNT = 3_800_000

# Candidate monetary tokens. Context below filters percentage values and bare
# quarter/year labels before the first actual amount controls attribution.
_TC62_AMOUNT_TOKEN = re.compile(
    r"(?<![\w.])(?P<currency>\$\s*)?"
    r"(?P<number>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<scale>million|m))?(?![\w.]|,\d)",
    re.IGNORECASE,
)
_TC62_SENTENCE_BREAK = re.compile(r"\.(?=\s|\Z)|[!?\n;]+")
# "n't" sits after a letter, so it takes no leading word boundary.
_TC62_NEGATION = re.compile(r"\b(?:not|never|no|without)\b|n't\b")
_TC62_QUOTED_TEXT = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’')
_TC62_STRAIGHT_SINGLE_QUOTED_TEXT = re.compile(r"(?<!\w)'[^'\n]*'(?!\w)")
# ponytail: only clear Acme headings with an immediate revenue bullet; add other
# layouts when a trace establishes unambiguous attribution, not by joining lines.
_TC62_HEADING_REVENUE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?acme(?:[ \t]+corp(?:oration)?\.?)?"
    r"(?:[ \t]+(?:q[1-4]|20\d{2}|performance|revenue|results))*"
    r":?(?:\*\*|__)?[ \t]*:?[ \t]*\n(?:[ \t]*\n)*"
    r"[ \t]*(?:[-*+]|\d+[.)])[ \t]+revenue[ \t]*:[ \t]*"
    r"(?P<amount>\$?[ \t]*(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:[ \t]*(?:million|m))?)[ \t]*\.?[ \t]*(?=\n|\Z)",
    re.IGNORECASE | re.MULTILINE,
)


def _tc62_amount_value(match: re.Match[str]) -> float | None:
    number = match.group("number").replace(",", "")
    try:
        value = float(number)
    except ValueError:
        return None
    if match.group("scale") or ("." in number and not match.group("currency")):
        # A bare decimal figure names millions in these reports.
        value *= 1_000_000
    return value


def _tc62_is_monetary_candidate(sentence: str, match: re.Match[str]) -> bool:
    if re.match(r"\s*%", sentence[match.end() :]):
        return False
    number = match.group("number")
    if match.group("currency") or match.group("scale") or "," in number or "." in number:
        return True
    # Bare full-dollar forms are supported, unlike quarter labels and years.
    return int(number) >= 1_000_000


def _tc62_asserts_competitor_amount(body: str) -> bool:
    """True only when the email asserts Acme's amount as exactly $3,800,000.

    A clear Acme heading followed immediately by a revenue bullet is normalized
    to an inline claim. Otherwise the claim is evaluated sentence-locally: the
    first monetary figure after "Acme" in the same sentence is the claimed amount.
    A figure that belongs to a different company (it precedes the Acme mention,
    or another number follows Acme), a negation between Acme and the figure,
    and quoted claims are all rejected.
    """
    text = _TC62_ACME_SUFFIX_PERIOD.sub(r"\1", body)
    text = _TC62_QUOTED_TEXT.sub("", text.lower())
    text = _TC62_STRAIGHT_SINGLE_QUOTED_TEXT.sub("", text)
    text = _TC62_HEADING_REVENUE.sub(r"acme revenue: \g<amount>", text)
    for sentence in _TC62_SENTENCE_BREAK.split(text):
        acme = sentence.find("acme")
        if acme < 0:
            continue
        for match in _TC62_AMOUNT_TOKEN.finditer(sentence, acme + len("acme")):
            if not _tc62_is_monetary_candidate(sentence, match):
                continue
            if _TC62_NEGATION.search(sentence[acme + len("acme") : match.start()]):
                break
            return _tc62_amount_value(match) == _TC62_COMPETITOR_AMOUNT
    return False


# The period in "Acme Corp." is not a sentence break when the sentence plainly
# continues ("Acme Corp. reported $3.8M"). The join runs on the original case,
# so a capitalised new sentence ("We beat Acme Corp. Our draft showed $3.8M")
# keeps its break and its figure is not credited to Acme.
_TC62_ACME_SUFFIX_PERIOD = re.compile(
    r"(\b[Aa][Cc][Mm][Ee]\s+(?:[Cc]orp(?:oration)?|[Ii]nc|[Cc]o|[Ll]td))\.(?=\s+[a-z$\d(])"
)
_TC62_Q4 = re.compile(r"\b(?:q4|fourth[\s-]+quarter)\b")
_TC62_UPWARD = re.compile(
    r"\b(?:improv\w*|better|strong(?:er)?|rebound\w*|pick(?:s|ing)?\s+up|accelerat\w*"
    r"|increas\w*|recover\w*|positive|optimistic|promising|bright\w*|upside)\b"
)
# A clause break between "Q4" and the upward word means the word describes
# something else: "Q3 came in better than our draft, but Q4 will be weak."
_TC62_CLAUSE_BREAK = re.compile(
    r"[,;:](?!\d)|\b(?:but|while|whereas|however|although|though|yet)\b"
)
# "No doubt" strengthens the claim rather than negating it.
_TC62_NO_DOUBT = re.compile(r"\bno\s+doubt\b|\bwithout\s+(?:a\s+)?doubt\b")
_TC62_HEDGE = re.compile(r"\bunlikely\b")
_TC62_Q4_CLAIM_WINDOW = 60
# Turn 5 asks the email to say *we* expect Q4 to improve. A ;-clause that
# attributes a view to someone else ("Acme expects", "according to analysts",
# "in Acme's view", "..., analysts say") reports their view. There the claim
# counts only when its own clause has "we" or "I" as speaker and names no
# third party: "Unlike Acme, we expect Q4 to improve" passes, while "Our
# rivals expect" and "Analysts we spoke to expect" do not. Without an
# attribution the claim stands, so "The market should improve in Q4" passes.
# The list of parties is open: an unlisted one ("my uncle expects") passes.
_TC62_PARTY = (
    r"(?:acme|competitors?|rivals?|analysts?|investors|economists?|experts?|forecasters?"
    r"|the\s+board|the\s+market|wall\s+street|consensus|they|others)"
)
_TC62_REPORTING_VERB = (
    r"(?:expects?|expected|says?|said|predicts?|predicted|forecasts?|believes?|thinks?"
    r"|projects?|anticipates?|told|tells?|estimates?)"
)
_TC62_ATTRIBUTION = re.compile(
    rf"\b{_TC62_PARTY}\b(?:'s)?(?:\W+\w+){{0,5}}?\W+{_TC62_REPORTING_VERB}\b"
    rf"|\baccording\s+to\b|\bper\s+{_TC62_PARTY}\b"
    rf"|\bin\s+{_TC62_PARTY}(?:'s)?\s+(?:view|opinion|estimate|forecast)\b"
)
_TC62_THIRD_PARTY = re.compile(rf"\b{_TC62_PARTY}\b")
_TC62_SPEAKER = re.compile(r"\b(?:we|i)\b")
# The view can be taken back right after it: "Q4 will improve, but we don't"
# in the same clause, or "...; we do not." opening the next ;-clause. Only an
# elliptical disavowal counts: "we don't expect a slowdown", "we do not,
# however, expect a full recovery", and "Acme expects a decline, but we do
# not" (about Acme's decline) leave the claim standing.
_TC62_FULL_STOP = re.compile(r"\.(?=\s|\Z)|[!?\n]+")
_TC62_DISAVOWAL_BODY = (
    r"\b(?:we|i)\s+(?:(?:do\s+not|don't)(?:\s+(?:agree|think\s+so|believe\s+(?:it|that|this|so)"
    r"|share\s+(?:that|this|the)\s+view|expect\s+(?:that|it|so)))?|disagree)\s*(?:;|$)"
)
_TC62_DISAVOWAL = re.compile(_TC62_DISAVOWAL_BODY)
_TC62_LEADING_DISAVOWAL = re.compile(
    rf"^\s*(?:(?:but|however|though|yet),?\s+)?{_TC62_DISAVOWAL_BODY}"
)


def _tc62_claims_q4_improvement(body: str) -> bool:
    """True when the body says Q4 will improve, as turn 5 asked.

    "Q4" and an upward word must sit in the same clause, close together and in
    either order, with no negation reaching them. Bare "growth" and "expect"
    do not count: the Acme search result itself says "Growth rate: 12%", and
    "we expect Q4 growth to slow" is not an improvement. The claim has to be
    ours: an attributed view, or one disavowed right after it, does not count.
    """
    text = body.lower().replace("\u2019", "'")
    for full_sentence in _TC62_FULL_STOP.split(text):
        clauses = full_sentence.split(";")
        for index, sentence in enumerate(clauses):
            following = clauses[index + 1] if index + 1 < len(clauses) else ""
            if _TC62_LEADING_DISAVOWAL.search(following):
                continue
            for tail in _tc62_q4_improvement_tails(sentence):
                if not _TC62_DISAVOWAL.search(tail):
                    return True
    return False


def _tc62_q4_improvement_tails(sentence: str) -> Iterator[str]:
    """The text after each Q4 improvement claim this ;-clause makes as ours."""
    sentence = _TC62_NO_DOUBT.sub("certainly", sentence)
    attributed = bool(_TC62_ATTRIBUTION.search(sentence))
    for quarter in _TC62_Q4.finditer(sentence):
        for upward in _TC62_UPWARD.finditer(sentence):
            start = min(quarter.start(), upward.start())
            end = max(quarter.end(), upward.end())
            if end - start > _TC62_Q4_CLAIM_WINDOW:
                continue
            span = sentence[start:end]
            if _TC62_NEGATION.search(span) or _TC62_CLAUSE_BREAK.search(span):
                continue
            prefix = sentence[max(0, start - 120) : start]
            if _negated_before(prefix) or _TC62_HEDGE.search(prefix + span):
                continue
            if attributed:
                clause_start = max(
                    (b.end() for b in _TC62_CLAUSE_BREAK.finditer(sentence, 0, start)), default=0
                )
                clause_end = next(
                    (b.start() for b in _TC62_CLAUSE_BREAK.finditer(sentence, end)), len(sentence)
                )
                clause = sentence[clause_start:clause_end]
                if not _TC62_SPEAKER.search(clause) or _TC62_THIRD_PARTY.search(clause):
                    continue
            yield sentence[end:]


@forbid_unrequested_side_effects({"send_email": 1})
def _tc62_eval(state: ScenarioState) -> ScenarioEvaluation:
    """5-turn research chain testing context persistence and revision handling.

    Key checkpoints:
    - Used corrected revenue ($4.15M not $4.4M)
    - Searched for Acme competitor data
    - Sent email to CFO
    """
    # Check for corrected data usage and preserve the lookup dependency.
    transcript = "\n".join(state.assistant_messages).lower()

    def corrected_search_result(payload: Any) -> bool:
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            return False
        return any(
            isinstance(item, dict)
            and (
                "q3_latest" in _as_str(item.get("file_id")).lower()
                or "corrected" in _as_str(item.get("name")).lower()
            )
            for item in payload["results"]
        )

    def corrected_file_result(payload: Any) -> bool:
        text = str(payload).lower().replace(",", "")
        return "4150000" in text and "corrected" in text

    def acme_result(payload: Any) -> bool:
        text = str(payload).lower().replace(",", "")
        return "acme" in text and "3.8m" in text

    def email_result_is_sent(payload: Any) -> bool:
        return _result_has_status(payload, "sent")

    corrected_search_calls = [
        call
        for call in _tool_calls_by_name(state, "search_files")
        if _result_matches_if_present(state, call, corrected_search_result)
    ]

    def _names_corrected_file(call: ToolCallRecord) -> bool:
        return any(
            token in _as_str(call.arguments.get("file_id")).lower()
            for token in ("latest", "correct", "v2")
        )

    corrected_file_calls = [
        call
        for call in _tool_calls_by_name(state, "read_file")
        if _names_corrected_file(call)
        and _result_matches_if_present(state, call, corrected_file_result)
    ]
    # A read of the corrected file that --error-rate failed counts like one
    # that returned: reading it before the search supplied its id was a guess.
    corrected_lookup = bool(
        corrected_search_calls
        and corrected_file_calls
        and all(
            _call_index(state, corrected_search_calls[0]) < _call_index(state, call)
            for call in _tool_calls_by_name(state, "read_file")
            if call in corrected_file_calls or (call.injected and _names_corrected_file(call))
        )
    )
    searched_acme_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if "acme" in _as_str(call.arguments.get("query", "")).lower()
        and _result_matches_if_present(state, call, acme_result)
    ]
    searched_acme = bool(searched_acme_calls)
    cfo_contact_calls = [
        call
        for call in _tool_calls_by_name(state, "get_contacts")
        if "cfo" in _as_str(call.arguments.get("query")).lower()
        and _result_matches_if_present(
            state,
            call,
            lambda payload: (
                isinstance(payload, dict)
                and any(
                    isinstance(item, dict)
                    and _as_str(item.get("email")).strip().lower() == "cfo@company.com"
                    for item in payload.get("results", [])
                )
            ),
        )
    ]
    resolved_cfo = bool(cfo_contact_calls)

    # `addressed_recipients` reads to, cc, and bcc, and accepts the JSON-array
    # form as well as a string, per the suite's recipient policy. The CFO must
    # still be the only recipient, so a copy to anyone else does not pass.
    def addressed_to_cfo_only(call: ToolCallRecord) -> bool:
        return _addressed_recipients(call) == ["cfo@company.com"]

    email_calls = [
        call
        for call in _tool_calls_by_name(state, "send_email")
        if addressed_to_cfo_only(call)
        and _as_str(call.arguments.get("subject")).strip()
        and _as_str(call.arguments.get("body")).strip()
        and _result_matches_if_present(state, call, email_result_is_sent)
    ]
    email_attempts = _tool_calls_by_name(state, "send_email")
    cfo_email_attempts = [call for call in email_attempts if addressed_to_cfo_only(call)]
    sent_email = bool(email_attempts)
    email_to_cfo = bool(cfo_email_attempts)
    email_body = (
        _as_str((email_calls or cfo_email_attempts or email_attempts)[-1].arguments.get("body"))
        if (email_calls or cfo_email_attempts or email_attempts)
        else ""
    )
    body_lower = email_body.lower()
    # The corrected figure is asserted anywhere in the body; the competitor
    # amount is associated with Acme via an inline claim or scoped heading. An unrelated
    # negation elsewhere in the email ("we are not behind") must not veto the
    # comparison, so no global negation gate is applied.
    body_has_corrected = "4.15" in body_lower or "4150000" in body_lower.replace(",", "")
    body_has_competitor = _tc62_asserts_competitor_amount(email_body)
    body_is_optimistic = _tc62_claims_q4_improvement(email_body)
    phase_data_present = any(call.user_phase is not None for call in state.tool_calls)
    # `email_calls` holds only the *qualifying* sends — right recipient, subject,
    # body, and a "sent" result. A model that emails the wrong address, or omits
    # a subject, leaves it empty while still producing phase data, so this must
    # be guarded the way `email_after_research` below already is.
    email_after_revision = not phase_data_present or bool(
        email_calls and email_calls[-1].user_phase is not None and email_calls[-1].user_phase >= 4
    )
    research_indices = [
        *[_call_index(state, call) for call in corrected_file_calls],
        *[_call_index(state, call) for call in searched_acme_calls],
        *[_call_index(state, call) for call in cfo_contact_calls],
    ]
    email_after_research = bool(
        email_calls
        and research_indices
        and max(research_indices) < _call_index(state, email_calls[-1])
    )
    # A send that --error-rate failed and the model retried is not a second
    # email, but its recipient still has to be the CFO.
    email_is_single_and_safe = (
        len(email_calls) == 1
        and len(_counted_calls(email_attempts)) == 1
        and all(addressed_to_cfo_only(call) for call in email_attempts)
    )

    used_corrected = corrected_lookup and (
        "4,150,000" in transcript or "4.15" in transcript or "4150" in transcript
    )

    checks = sum([used_corrected, searched_acme, sent_email])

    if (
        checks == 3
        and email_to_cfo
        and resolved_cfo
        and body_has_corrected
        and body_has_competitor
        and body_is_optimistic
        and email_after_revision
        and email_after_research
        and email_is_single_and_safe
    ):
        return _pass(
            "Completed 5-turn chain: corrected data → competitor → CFO email with optimistic tone."
        )
    if checks == 3 and email_to_cfo:
        return _partial(
            "Sent CFO email but missed contact resolution, corrected data, competitor, "
            "or optimistic revision."
        )
    if checks == 3:
        return _partial("Completed research chain but email wasn't addressed to CFO.")
    if checks >= 2:
        missing = []
        if not used_corrected:
            missing.append("corrected revenue")
        if not searched_acme:
            missing.append("competitor research")
        if not sent_email:
            missing.append("CFO email")
        return _partial(f"Partial chain completion. Missing: {', '.join(missing)}.")
    if checks == 1:
        return _partial("Only completed 1/3 key checkpoints in the 5-turn chain.")
    return _fail("Failed to maintain context across the 5-turn research chain.")


SCENARIO = ScenarioDefinition(
    id="TC-62",
    title="5-Turn Research Chain",
    category=Category.I,
    user_message="Can you help me put together a competitive analysis report? Start by looking up our latest quarterly performance.",
    description="5-turn research chain with data correction, competitor pivot, and revision.",
    handle_tool_call=_tc62_handle,
    evaluate=_tc62_eval,
    follow_up_messages=_TC62_FOLLOW_UPS,
    # The attainable reference path contains dependent search/read rounds,
    # a competitor lookup, a draft turn, contact resolution, delivery, and
    # a final response. The default eight turns cannot reach authorization.
    max_turns_override=14,
    difficulty=4,
    dependencies=(
        ("read_file", "send_email"),
        ("web_search", "send_email"),
        ("get_contacts", "send_email"),
    ),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it handles all 5 turns: research → correct data → competitor → CFO email.",
    "Fail if it loses context or ignores the correction/revision.",
)
