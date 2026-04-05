"""Tests for error-aware evaluator fixes.

These verify that evaluators give correct verdicts when error injection
replaces tool results with 429/500/503 errors, preventing the model from
receiving the data it needs.
"""

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.scenarios import SCENARIOS
from tool_eval_bench.evals.scenarios_extended import EXTENDED_SCENARIOS
from tool_eval_bench.evals.scenarios_agentic import AGENTIC_SCENARIOS


def _find(scenarios, scenario_id):
    """Find a scenario by ID."""
    return next(s for s in scenarios if s.id == scenario_id)


def _make_state(
    tool_calls: list[dict] | None = None,
    tool_results: list[dict] | None = None,
    final_answer: str = "",
) -> ScenarioState:
    state = ScenarioState()
    state.final_answer = final_answer
    state.assistant_messages = [final_answer] if final_answer else []
    if tool_calls:
        for tc in tool_calls:
            state.tool_calls.append(
                ToolCallRecord(
                    id=tc.get("id", f"call_{len(state.tool_calls)}"),
                    name=tc["name"],
                    raw_arguments="{}",
                    arguments=tc.get("arguments", {}),
                    turn=tc.get("turn", 1),
                )
            )
    if tool_results:
        for tr in tool_results:
            state.tool_results.append(
                ToolResultRecord(
                    call_id=tr.get("call_id", f"call_{len(state.tool_results)}"),
                    name=tr["name"],
                    result=tr["result"],
                )
            )
    return state


# ============================================================
# TC-16: German language detection fix
# ============================================================

class TestTC16GermanDetection:
    """TC-16 should detect German even when weather data is error-injected."""

    def test_german_error_response_detected_as_german(self) -> None:
        """Model responds in German after weather tool error — should be pass, not 'English'."""
        scenario = _find(EXTENDED_SCENARIOS, "TC-16")
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "München"}}],
            tool_results=[{"name": "get_weather", "result": {"error": "Rate limit exceeded.", "status": 429}}],
            final_answer=(
                "Es tut mir leid, ich konnte die Wetterdaten für München gerade "
                "nicht abrufen, da der Dienst vorübergehend überlastet ist. "
                "Bitte versuchen Sie es in ein paar Augenblicken noch einmal."
            ),
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS, f"Expected pass, got {result.status}: {result.summary}"

    def test_german_weather_response_still_passes(self) -> None:
        """Normal case: model gets weather data and responds in German — should pass."""
        scenario = _find(EXTENDED_SCENARIOS, "TC-16")
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "München"}}],
            tool_results=[{"name": "get_weather", "result": {"temperature": 14, "condition": "Bewölkt"}}],
            final_answer=(
                "Das Wetter in München ist gerade 14 Grad Celsius, "
                "teilweise bewölkt mit einer Temperatur von 14°C."
            ),
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_english_response_still_partial(self) -> None:
        """Model uses weather tool but responds in English — should be partial."""
        scenario = _find(EXTENDED_SCENARIOS, "TC-16")
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "München"}}],
            tool_results=[{"name": "get_weather", "result": {"temperature": 14}}],
            final_answer="The weather in Munich is 14 degrees and partly cloudy.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_munich_english_spelling_accepted(self) -> None:
        """Model passes 'Munich' instead of 'München' — should still detect tool use."""
        scenario = _find(EXTENDED_SCENARIOS, "TC-16")
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Munich"}}],
            tool_results=[{"name": "get_weather", "result": {"temperature": 14}}],
            final_answer=(
                "Das Wetter in München ist gerade 14 Grad Celsius, "
                "teilweise bewölkt bei einer Temperatur von 14°C."
            ),
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS


# ============================================================
# TC-25: Error-aware conditional evaluation
# ============================================================

class TestTC25ErrorAware:
    """TC-25 should give partial (not fail) when weather tool returns error."""

    def test_weather_error_gives_partial_not_fail(self) -> None:
        """Weather tool errored — model can't evaluate condition — partial, not fail."""
        scenario = _find(AGENTIC_SCENARIOS, "TC-25")
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            tool_results=[{"name": "get_weather", "result": {"error": "Rate limit exceeded.", "status": 429}}],
            final_answer="I'm sorry, I couldn't check the weather. Please try again.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL, f"Expected partial, got {result.status}: {result.summary}"
        assert "error" in result.summary.lower() or "fail" in result.summary.lower()

    def test_weather_error_with_retry_gives_partial(self) -> None:
        """Model retries weather after error — should credit retry attempt."""
        scenario = _find(AGENTIC_SCENARIOS, "TC-25")
        state = _make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Berlin"}, "turn": 1},
                {"name": "get_weather", "arguments": {"location": "Berlin"}, "turn": 2},
            ],
            tool_results=[
                {"name": "get_weather", "result": {"error": "Rate limit exceeded.", "status": 429}},
                {"name": "get_weather", "result": {"error": "Rate limit exceeded.", "status": 429}},
            ],
            final_answer="The weather service is down. Please try later.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL
        assert "retri" in result.summary.lower()

    def test_normal_pass_still_works(self) -> None:
        """Normal case: weather returns 5°C, model sets coat reminder — should pass."""
        scenario = _find(AGENTIC_SCENARIOS, "TC-25")
        state = _make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Berlin"}},
                {"name": "set_reminder", "arguments": {"message": "Bring a coat tomorrow!"}},
            ],
            tool_results=[
                {"name": "get_weather", "result": {"temperature": 5, "condition": "Clear"}},
                {"name": "set_reminder", "result": {"status": "created"}},
            ],
            final_answer="It's 5°C in Berlin. I've set a reminder to bring a coat.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_no_reminder_without_error_still_fails(self) -> None:
        """Weather returned normally but model didn't set reminder — should still fail."""
        scenario = _find(AGENTIC_SCENARIOS, "TC-25")
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            tool_results=[{"name": "get_weather", "result": {"temperature": 5, "condition": "Clear"}}],
            final_answer="It's 5°C in Berlin.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


# ============================================================
# TC-15: Error-aware search evaluation
# ============================================================

class TestTC15ErrorAware:
    """TC-15 should give partial when search tool returns error and model uses knowledge."""

    def test_search_error_with_knowledge_fallback_gives_partial(self) -> None:
        """Search errored, model used training data — partial, not fail."""
        scenario = _find(SCENARIOS, "TC-15")
        state = _make_state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "population of Iceland"}}],
            tool_results=[{"name": "web_search", "result": {"error": "Rate limit exceeded.", "status": 429}}],
            final_answer="Iceland's population is about 375,000. 2% is 7,500.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL, f"Expected partial, got {result.status}: {result.summary}"
        assert "fallback" in result.summary.lower() or "background" in result.summary.lower()

    def test_search_error_no_answer_gives_partial(self) -> None:
        """Search errored, no meaningful calculation — still partial for attempting."""
        scenario = _find(SCENARIOS, "TC-15")
        state = _make_state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "population of Iceland"}}],
            tool_results=[{"name": "web_search", "result": {"error": "Request timed out.", "status": 503}}],
            final_answer="I couldn't retrieve the population data due to a service error.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_normal_pass_still_works(self) -> None:
        """Normal case: search works, calculator uses exact value — should pass."""
        scenario = _find(SCENARIOS, "TC-15")
        state = _make_state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "population of Iceland"}},
                {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
            ],
            tool_results=[
                {"name": "web_search", "result": {"results": [{"snippet": "372,520"}]}},
                {"name": "calculator", "result": {"result": 7450.4}},
            ],
            final_answer="2% of Iceland's population (372,520) is 7,450.4.",
        )
        result = scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS
