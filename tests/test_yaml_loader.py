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
        path.write_text(
            "id: X\ntitle: Required fields\ncategory: A\nuser_message: hi\n" + f"{field}: []\n",
            encoding="utf-8",
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

        contact = sc.handle_tool_call(ScenarioState(), _record("find_contact", {"name": "Priya"}))

        assert contact["email"] == "priya@example.com"

    def test_the_restraint_example_fails_when_a_tool_is_used(self) -> None:
        sc = _bundled("YAML-03")
        state = ScenarioState()
        state.tool_calls.append(_record("calculator", {"expression": "60*24"}))
        state.final_answer = "1440"

        assert sc.evaluate(state).status == ScenarioStatus.FAIL
