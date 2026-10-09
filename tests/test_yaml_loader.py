"""Tests for the declarative YAML scenario loader pilot."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals import yaml_scenarios as yaml_scenarios_pkg
from tool_eval_bench.evals.yaml_loader import _load_yaml_file, load_yaml_scenarios


def _scenarios_dir() -> Path:
    return Path(yaml_scenarios_pkg.__file__).parent


def _record(name: str, arguments: dict | None = None) -> ToolCallRecord:
    return ToolCallRecord(
        id="call_1",
        name=name,
        raw_arguments="{}",
        arguments=arguments or {},
        turn=1,
    )


def _bundled(scenario_id: str):
    """Select a bundled example by id, so adding one does not shuffle the rest."""
    return next(s for s in load_yaml_scenarios(_scenarios_dir()) if s.id == scenario_id)


class TestYamlLoader:
    def test_loads_sample_weather_scenario(self) -> None:
        sc = _bundled("YAML-01")
        assert sc.id == "YAML-01"
        assert sc.title == "Simple weather lookup"
        assert sc.category == Category.A
        assert sc.difficulty == 1
        assert "Berlin" in sc.user_message

    def test_handler_returns_declarative_response(self) -> None:
        sc = _bundled("YAML-01")
        state = ScenarioState()
        record = _record("get_weather", {"location": "Berlin"})
        result = sc.handle_tool_call(state, record)
        assert result["location"] == "Berlin"
        assert result["condition"] == "cloudy"

    def test_handler_returns_generic_fallback_when_no_rule_matches(self) -> None:
        sc = _bundled("YAML-01")
        state = ScenarioState()
        record = _record("get_weather", {"location": "Paris"})
        result = sc.handle_tool_call(state, record)
        assert result == {"result": "ok"}

    def test_evaluator_passes_on_expected_call(self) -> None:
        sc = _bundled("YAML-01")
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))
        evaluation = sc.evaluate(state)
        assert evaluation.status == ScenarioStatus.PASS
        assert evaluation.points == 2

    def test_evaluator_fails_on_wrong_arguments(self) -> None:
        sc = _bundled("YAML-01")
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Paris"}))
        evaluation = sc.evaluate(state)
        assert evaluation.status == ScenarioStatus.FAIL

    def test_evaluator_fails_on_extra_calls(self) -> None:
        sc = _bundled("YAML-01")
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))
        state.tool_calls.append(_record("calculator", {"expression": "1+1"}))
        evaluation = sc.evaluate(state)
        assert evaluation.status == ScenarioStatus.FAIL
        assert "Extra" in evaluation.summary

    def test_restraint_evaluator_passes_when_no_tool_is_called(self, tmp_path: Path) -> None:
        path = tmp_path / "restraint.yaml"
        path.write_text(
            "id: YAML-R\ntitle: Restraint\ncategory: A\nuser_message: Answer directly\n"
            "expected_tool_calls: []\n",
            encoding="utf-8",
        )

        evaluation = _load_yaml_file(path).evaluate(ScenarioState())

        assert evaluation.status == ScenarioStatus.PASS
        assert evaluation.points == 2

    def test_restraint_evaluator_fails_when_any_tool_is_called(self, tmp_path: Path) -> None:
        path = tmp_path / "restraint.yaml"
        path.write_text(
            "id: YAML-R\ntitle: Restraint\ncategory: A\nuser_message: Answer directly\n"
            "expected_tool_calls: []\n",
            encoding="utf-8",
        )
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))

        evaluation = _load_yaml_file(path).evaluate(state)

        assert evaluation.status == ScenarioStatus.FAIL
        assert evaluation.points == 0
        assert "get_weather" in evaluation.summary

    def test_loads_multiple_files_sorted(self) -> None:
        yaml_a = """
id: YAML-A
title: A
category: A
difficulty: 1
user_message: A
expected_tool_calls: []
tool_responses: {}
"""
        yaml_b = """
id: YAML-B
title: B
category: A
difficulty: 1
user_message: B
expected_tool_calls: []
tool_responses: {}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "b.yaml").write_text(yaml_b, encoding="utf-8")
            (root / "a.yaml").write_text(yaml_a, encoding="utf-8")
            scenarios = load_yaml_scenarios(root)
            assert [s.id for s in scenarios] == ["YAML-A", "YAML-B"]

    def test_invalid_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text("not a mapping", encoding="utf-8")
            with pytest.raises(ValueError):
                _load_yaml_file(path)

    def test_missing_id_field_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "noid.yaml"
            path.write_text("title: No ID\ncategory: A\nuser_message: hi\n", encoding="utf-8")
            with pytest.raises(ValueError, match="'id'"):
                _load_yaml_file(path)

    def test_missing_category_field_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nocat.yaml"
            path.write_text("id: X\ntitle: No Cat\nuser_message: hi\n", encoding="utf-8")
            with pytest.raises(ValueError, match="'category'"):
                _load_yaml_file(path)

    @pytest.mark.parametrize("field", ["title", "user_message"])
    def test_other_missing_required_fields_raise_with_path(
        self, tmp_path: Path, field: str
    ) -> None:
        values = {
            "id": "X",
            "title": "Required fields",
            "category": "A",
            "user_message": "hi",
        }
        values.pop(field)
        path = tmp_path / "missing.yaml"
        path.write_text(
            "\n".join(f"{key}: {value}" for key, value in values.items()), encoding="utf-8"
        )

        with pytest.raises(ValueError, match=rf"{field!r}.*{re.escape(str(path))}"):
            _load_yaml_file(path)

    @pytest.mark.parametrize("field", ["id", "title", "category", "user_message"])
    def test_required_fields_must_be_non_empty_strings(self, tmp_path: Path, field: str) -> None:
        path = tmp_path / "invalid.yaml"
        fields = {"id": "X", "title": "Required fields", "category": "A", "user_message": "hi"}
        fields[field] = "[]"
        path.write_text(
            "".join(f"{key}: {value}\n" for key, value in fields.items()), encoding="utf-8"
        )

        with pytest.raises(
            ValueError, match=rf"{field!r}.*non-empty string.*{re.escape(str(path))}"
        ):
            _load_yaml_file(path)

    def test_invalid_category_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "badcat.yaml"
            path.write_text(
                "id: X\ntitle: Bad Cat\ncategory: Z\nuser_message: hi\n", encoding="utf-8"
            )
            with pytest.raises(ValueError, match="Invalid category"):
                _load_yaml_file(path)

    def test_yaml_parse_error_includes_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "syntax.yaml"
            path.write_text("id: X\n  bad: : : indent\n", encoding="utf-8")
            with pytest.raises(ValueError, match="YAML parse error"):
                _load_yaml_file(path)


class TestAnswerContains:
    """``answer_contains`` is the only route to PARTIAL from a YAML scenario."""

    def _scenario(self, tmp_path: Path, body: str):
        path = tmp_path / "answer.yaml"
        path.write_text(body, encoding="utf-8")
        return load_yaml_scenarios(tmp_path)[0]

    BODY = """
id: YAML-A
title: Weather with a stated result
category: A
user_message: What is the weather in Berlin?
expected_tool_calls:
  - tool: get_weather
    arguments:
      location: Berlin
answer_contains:
  - "18"
  - cloudy
"""

    def test_right_calls_and_a_complete_answer_pass(self, tmp_path: Path) -> None:
        sc = self._scenario(tmp_path, self.BODY)
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))
        state.final_answer = "It is 18 degrees and CLOUDY in Berlin."

        evaluation = sc.evaluate(state)

        assert evaluation.status == ScenarioStatus.PASS
        assert evaluation.points == 2

    def test_right_calls_but_a_silent_answer_is_partial(self, tmp_path: Path) -> None:
        sc = self._scenario(tmp_path, self.BODY)
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))
        state.final_answer = "I looked it up."

        evaluation = sc.evaluate(state)

        assert evaluation.status == ScenarioStatus.PARTIAL
        assert evaluation.points == 1
        assert "18" in evaluation.summary and "cloudy" in evaluation.summary

    def test_a_partially_complete_answer_names_only_what_is_missing(self, tmp_path: Path) -> None:
        sc = self._scenario(tmp_path, self.BODY)
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))
        state.final_answer = "It is cloudy."

        evaluation = sc.evaluate(state)

        assert evaluation.status == ScenarioStatus.PARTIAL
        assert "18" in evaluation.summary
        assert "cloudy" not in evaluation.summary.split("states:")[1]

    def test_wrong_tool_calls_still_fail_outright(self, tmp_path: Path) -> None:
        """A perfect answer does not rescue a scenario about tool discipline."""
        sc = self._scenario(tmp_path, self.BODY)
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Paris"}))
        state.final_answer = "It is 18 and cloudy."

        evaluation = sc.evaluate(state)

        assert evaluation.status == ScenarioStatus.FAIL

    def test_restraint_scenarios_are_scored_the_same_way(self, tmp_path: Path) -> None:
        sc = self._scenario(
            tmp_path,
            """
id: YAML-B
title: No tool needed
category: E
user_message: How many minutes are in a day?
expected_tool_calls: []
answer_contains:
  - "1440"
""",
        )
        silent, complete = ScenarioState(), ScenarioState()
        silent.final_answer = "Quite a few."
        complete.final_answer = "1440."

        assert sc.evaluate(silent).status == ScenarioStatus.PARTIAL
        assert sc.evaluate(complete).status == ScenarioStatus.PASS

    def test_omitting_the_field_keeps_the_old_pass_or_fail_behaviour(self, tmp_path: Path) -> None:
        sc = self._scenario(
            tmp_path,
            """
id: YAML-C
title: No answer assertion
category: A
user_message: What is the weather in Berlin?
expected_tool_calls:
  - tool: get_weather
""",
        )
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))

        assert sc.evaluate(state).status == ScenarioStatus.PASS

    @pytest.mark.parametrize("value", ["cloudy", "{a: 1}", "[[nested]]", '["", "x"]'])
    def test_a_field_that_is_not_a_list_of_strings_is_rejected(
        self, tmp_path: Path, value: str
    ) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            f"id: YAML-D\ntitle: t\ncategory: A\nuser_message: m\nanswer_contains: {value}\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="answer_contains"):
            load_yaml_scenarios(tmp_path)


class TestBundledExamples:
    """The shipped examples are the reference a pack author copies."""

    def test_every_bundled_example_loads_and_is_rated(self) -> None:
        scenarios = load_yaml_scenarios(_scenarios_dir())

        assert {s.id for s in scenarios} == {"YAML-01", "YAML-02", "YAML-03"}
        assert all(s.difficulty in {1, 2, 3, 4, 5} for s in scenarios)

    def test_the_chained_example_requires_both_calls_in_order(self) -> None:
        sc = _bundled("YAML-02")
        state = ScenarioState()
        state.tool_calls.append(_record("send_email", {"to": "priya@example.com"}))

        assert sc.evaluate(state).status == ScenarioStatus.FAIL

    def test_the_chained_example_resolves_the_address_from_the_first_call(self) -> None:
        sc = _bundled("YAML-02")

        contact = sc.handle_tool_call(ScenarioState(), _record("get_contacts", {"query": "Priya"}))

        assert contact["email"] == "priya@example.com"

    def test_the_chained_example_passes_a_model_that_uses_the_offered_tools(self) -> None:
        sc = _bundled("YAML-02")
        state = ScenarioState()
        state.tool_calls.append(_record("get_contacts", {"query": "Priya"}))
        state.tool_calls.append(
            _record("send_email", {"to": "priya@example.com", "subject": "Q3 summary"})
        )
        state.final_answer = "Sent the Q3 summary to priya@example.com."

        assert sc.evaluate(state).status == ScenarioStatus.PASS

    def test_the_restraint_example_fails_when_a_tool_is_used(self) -> None:
        sc = _bundled("YAML-03")
        state = ScenarioState()
        state.tool_calls.append(_record("calculator", {"expression": "60*24"}))
        state.final_answer = "1440"

        assert sc.evaluate(state).status == ScenarioStatus.FAIL


def _pack_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pack.yaml"
    path.write_text(
        "id: PACK-01\ntitle: t\ncategory: A\nuser_message: Weather in Berlin?\n" + body,
        encoding="utf-8",
    )
    return path


class TestAmbiguousScalars:
    """YAML 1.1 readings a JSON tool argument can never match are load errors."""

    @pytest.mark.parametrize(
        "value",
        [
            "2026-03-21",  # date
            "2026-03-21T09:30:00Z",  # datetime
            "14:30",  # sexagesimal int 870
            "1:30:00",
            "NO",  # bool False
            "yes",
            "on",
            "Off",
            "01234",  # octal int 668
            "0b101",
            "1_000",
            ".inf",
            ".nan",
            "1e5",  # a string in YAML 1.1, a number in JSON
            "-.5",
            "09",
        ],
    )
    def test_an_unquoted_ambiguous_argument_is_rejected_with_its_location(
        self, tmp_path: Path, value: str
    ) -> None:
        path = _pack_file(
            tmp_path,
            f"expected_tool_calls:\n  - tool: create_calendar_event\n    arguments:\n"
            f"      time: {value}\n",
        )

        with pytest.raises(ValueError) as exc_info:
            _load_yaml_file(path)

        message = str(exc_info.value)
        assert str(path) in message
        assert repr(value) in message
        assert "line 8" in message
        assert "quote it" in message

    def test_an_exponent_without_a_dot_suggests_a_spelling_that_stays_a_number(
        self, tmp_path: Path
    ) -> None:
        path = _pack_file(
            tmp_path,
            "expected_tool_calls:\n  - tool: create_calendar_event\n    arguments:\n"
            "      time: 1e5\n",
        )

        with pytest.raises(ValueError, match=re.escape("1.0e+5")):
            _load_yaml_file(path)

    def test_an_ambiguous_mapping_key_is_rejected(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  get_weather:\n    - match:\n        on: Berlin\n",
        )

        with pytest.raises(ValueError, match="'on'"):
            _load_yaml_file(path)

    def test_an_unquoted_date_in_a_response_is_rejected_rather_than_crashing_the_run(
        self, tmp_path: Path
    ) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  set_reminder:\n    - response:\n        due: 2026-03-21\n",
        )

        with pytest.raises(ValueError, match="timestamp"):
            _load_yaml_file(path)

    def test_quoted_values_reach_the_evaluator_as_the_strings_a_model_sends(
        self, tmp_path: Path
    ) -> None:
        path = _pack_file(
            tmp_path,
            "expected_tool_calls:\n  - tool: create_calendar_event\n    arguments:\n"
            '      date: "2026-03-21"\n      time: "14:30"\n      country: "NO"\n'
            "      zip: '01234'\n",
        )
        sc = _load_yaml_file(path)
        state = ScenarioState()
        state.tool_calls.append(
            _record(
                "create_calendar_event",
                {"date": "2026-03-21", "time": "14:30", "country": "NO", "zip": "01234"},
            )
        )

        assert sc.evaluate(state).status == ScenarioStatus.PASS

    @pytest.mark.parametrize(
        ("raw", "loaded"),
        [
            ("18", 18),
            ("-3", -3),
            ("0", 0),
            ("214.30", 214.3),
            ("1.0e+5", 100000.0),
            ("0x1F", 31),
            ("true", True),
            ("False", False),
            ("null", None),
            ("09:30", "09:30"),
            ("Berlin", "Berlin"),
            ("!!str 14:30", "14:30"),
        ],
    )
    def test_unambiguous_values_load_unchanged(
        self, tmp_path: Path, raw: str, loaded: object
    ) -> None:
        path = _pack_file(
            tmp_path,
            f"tool_responses:\n  get_weather:\n    - response:\n        value: {raw}\n",
        )
        sc = _load_yaml_file(path)

        response = sc.handle_tool_call(ScenarioState(), _record("get_weather"))

        assert response == {"value": loaded}
        assert type(response["value"]) is type(loaded)

    def test_a_block_scalar_is_never_reinterpreted(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  get_weather:\n    - response: |\n        NO\n",
        )
        sc = _load_yaml_file(path)

        assert sc.handle_tool_call(ScenarioState(), _record("get_weather")) == "NO\n"

    def test_an_explicitly_tagged_non_json_value_is_rejected(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  get_weather:\n    - response:\n        due: !!timestamp 2026-03-21\n",
        )

        with pytest.raises(ValueError, match=r"tool_responses\.get_weather\[0\]\.response\.due"):
            _load_yaml_file(path)

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (
                "tool_responses:\n  get_weather:\n    - response: {temp: !!float .inf}\n",
                r"response\.temp' must be a finite number",
            ),
            (
                "tool_responses:\n  get_weather:\n    - response: {days: [ok, !!binary aGk=]}\n",
                r"response\.days\[1\]' holds a bytes",
            ),
            (
                "expected_tool_calls:\n  - tool: get_weather\n    arguments: {1: Berlin}\n",
                r"arguments' has a non-string key 1",
            ),
            ("tool_responses:\n  1: []\n", "'tool_responses' has a non-string key 1"),
        ],
    )
    def test_values_that_cannot_travel_as_json_are_rejected(
        self, tmp_path: Path, body: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _load_yaml_file(_pack_file(tmp_path, body))


class TestDifficulty:
    @pytest.mark.parametrize("value", ["hard", "-1", "0", "6", "true", "2.5", '"3"'])
    def test_anything_but_an_integer_from_one_to_five_is_rejected(
        self, tmp_path: Path, value: str
    ) -> None:
        path = _pack_file(tmp_path, f"difficulty: {value}\n")

        with pytest.raises(ValueError, match=rf"'difficulty'.*1 to 5.*{re.escape(str(path))}"):
            _load_yaml_file(path)

    @pytest.mark.parametrize(
        ("body", "expected"), [("difficulty: 1\n", 1), ("difficulty: 5\n", 5), ("", None)]
    )
    def test_the_boundaries_and_an_absent_rating_load(
        self, tmp_path: Path, body: str, expected: int | None
    ) -> None:
        assert _load_yaml_file(_pack_file(tmp_path, body)).difficulty == expected

    def test_every_registered_scenario_uses_the_same_scale(self) -> None:
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

        assert {s.difficulty for s in ALL_SCENARIOS_WITH_HARDMODE} <= {None, 1, 2, 3, 4, 5}


class TestStructure:
    """A pack that cannot be graded fails to load instead of scoring as a model failure."""

    @pytest.mark.parametrize(
        ("body", "field"),
        [
            (
                "expected_tool_calls:\n  - name: get_weather\n",
                r"'name' in expected_tool_calls\[0\]",
            ),
            (
                "expected_tool_calls:\n  - arguments: {location: Berlin}\n",
                r"expected_tool_calls\[0\]\.tool",
            ),
            (
                "expected_tool_calls:\n  - tool: get_weather\n    arguments:\n",
                r"expected_tool_calls\[0\]\.arguments",
            ),
            (
                "expected_tool_calls:\n  - tool: get_weather\n    arguments: [Berlin]\n",
                r"expected_tool_calls\[0\]\.arguments",
            ),
            ("expected_tool_calls:\n  tool: get_weather\n", "'expected_tool_calls'"),
            ("expected_tool_calls:\n  - get_weather\n", r"expected_tool_calls\[0\]"),
            ("tool_responses:\n  - get_weather\n", "'tool_responses'"),
            (
                "tool_responses:\n  get_weather:\n    match: {location: Berlin}\n",
                r"tool_responses\.get_weather'",
            ),
            ("tool_responses:\n  get_weather:\n    - ok\n", r"tool_responses\.get_weather\[0\]"),
            (
                "tool_responses:\n  get_weather:\n    - match: [Berlin]\n",
                r"tool_responses\.get_weather\[0\]\.match",
            ),
            (
                "tool_responses:\n  get_weather:\n    - response: [1, 2]\n",
                r"tool_responses\.get_weather\[0\]\.response",
            ),
            (
                "tool_responses:\n  get_weather:\n    - response:\n",
                r"tool_responses\.get_weather\[0\]\.response",
            ),
            ("description: [a, b]\n", "'description'"),
        ],
    )
    def test_a_malformed_field_is_rejected_by_its_path(
        self, tmp_path: Path, body: str, field: str
    ) -> None:
        path = _pack_file(tmp_path, body)

        with pytest.raises(ValueError, match=rf"{field}.*{re.escape(str(path))}"):
            _load_yaml_file(path)

    def test_an_empty_match_still_matches_every_call(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  get_weather:\n    - match:\n      response: sunny\n",
        )
        sc = _load_yaml_file(path)

        assert sc.handle_tool_call(ScenarioState(), _record("get_weather", {"x": 1})) == "sunny"

    def test_an_expected_tool_no_yaml_scenario_is_offered_is_rejected(self, tmp_path: Path) -> None:
        path = _pack_file(tmp_path, "expected_tool_calls:\n  - tool: find_contact\n")

        with pytest.raises(ValueError, match=r"'find_contact'.*get_contacts"):
            _load_yaml_file(path)

    def test_a_response_rule_for_an_unoffered_tool_is_rejected(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path, "tool_responses:\n  find_contact:\n    - response: {name: Priya}\n"
        )

        with pytest.raises(
            ValueError,
            match=rf"'tool_responses'.*'find_contact'.*{re.escape(str(path))}.*get_contacts",
        ):
            _load_yaml_file(path)

    def test_a_match_key_the_tool_does_not_take_is_rejected(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  get_weather:\n    - match: {city: Berlin}\n"
            "      response: {temp: 18}\n",
        )

        with pytest.raises(ValueError) as exc_info:
            _load_yaml_file(path)

        message = str(exc_info.value)
        assert re.search(r"tool_responses\.get_weather\[0\]\.match' names 'city'", message)
        assert str(path) in message
        assert "its parameters: location" in message

    def test_a_match_on_a_real_parameter_fires(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  get_contacts:\n    - match: {query: Priya}\n"
            "      response: {name: Priya}\n",
        )
        sc = _load_yaml_file(path)

        response = sc.handle_tool_call(ScenarioState(), _record("get_contacts", {"query": "Priya"}))

        assert response == {"name": "Priya"}


class TestDuplicateKeys:
    """PyYAML keeps the last of two equal keys, which would silently drop a check."""

    @pytest.mark.parametrize(
        ("body", "key"),
        [
            pytest.param("title: again\n", "title", id="top-level"),
            pytest.param(
                "expected_tool_calls:\n  - tool: get_weather\n    arguments:\n"
                "      location: Berlin\n      location: Paris\n",
                "location",
                id="arguments",
            ),
            pytest.param(
                "expected_tool_calls:\n  - tool: get_weather\n"
                "expected_tool_calls:\n  - tool: get_contacts\n",
                "expected_tool_calls",
                id="expected-tool-calls",
            ),
        ],
    )
    def test_a_duplicate_key_is_rejected_with_its_location(
        self, tmp_path: Path, body: str, key: str
    ) -> None:
        path = _pack_file(tmp_path, body)

        with pytest.raises(ValueError) as exc_info:
            _load_yaml_file(path)

        message = str(exc_info.value)
        assert str(path) in message
        assert f"duplicate key {key!r}" in message
        assert "line " in message

    def test_a_key_overriding_a_merged_anchor_still_loads(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "tool_responses:\n  get_weather:\n"
            "    - match: {location: Berlin}\n"
            "      response: &berlin {temp: 18, sky: clear}\n"
            "    - match: {location: Paris}\n"
            "      response:\n        <<: *berlin\n        temp: 21\n",
        )
        sc = _load_yaml_file(path)

        response = sc.handle_tool_call(
            ScenarioState(), _record("get_weather", {"location": "Paris"})
        )

        assert response == {"temp": 21, "sky": "clear"}


class TestUnknownKeys:
    """A typo must not silently drop a check, least of all in a held-out pack."""

    @pytest.mark.parametrize(
        ("body", "key", "where", "allowed"),
        [
            pytest.param(
                "expected_tool_calls:\n  - tool: get_weather\n    argument: {location: Berlin}\n",
                "argument",
                r"expected_tool_calls\[0\]",
                "arguments, tool",
                id="expected-call",
            ),
            pytest.param(
                "tool_responses:\n  get_weather:\n    - matches: {location: Berlin}\n"
                "      response: {temp: 18}\n",
                "matches",
                r"tool_responses\.get_weather\[0\]",
                "match, response",
                id="response-rule",
            ),
            pytest.param(
                "answer_contain: ['18']\n",
                "answer_contain",
                "the scenario",
                "answer_contains, capabilities",
                id="top-level",
            ),
        ],
    )
    def test_an_unknown_key_is_named_with_its_location_and_the_allowed_set(
        self, tmp_path: Path, body: str, key: str, where: str, allowed: str
    ) -> None:
        path = _pack_file(tmp_path, body)

        with pytest.raises(ValueError) as exc_info:
            _load_yaml_file(path)

        message = str(exc_info.value)
        assert re.search(rf"Unknown key '{key}' in {where} in {re.escape(str(path))}", message)
        assert allowed in message

    def test_every_defined_key_is_accepted(self, tmp_path: Path) -> None:
        path = _pack_file(
            tmp_path,
            "difficulty: 2\ndescription: d\nheld_out: true\ncapabilities: [tool-selection]\n"
            "expected_tool_calls:\n  - tool: get_weather\n    arguments: {location: Berlin}\n"
            "tool_responses:\n  get_weather:\n    - match: {location: Berlin}\n"
            "      response: {temp: 18}\n"
            "answer_contains: ['18']\n",
        )

        assert _load_yaml_file(path).held_out is True

    def test_the_bundled_examples_use_only_defined_keys(self) -> None:
        assert len(load_yaml_scenarios(_scenarios_dir())) == 3
