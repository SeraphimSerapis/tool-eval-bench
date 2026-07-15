"""Tests for the declarative YAML scenario loader pilot."""

from __future__ import annotations

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


class TestYamlLoader:
    def test_loads_sample_weather_scenario(self) -> None:
        scenarios = load_yaml_scenarios(_scenarios_dir())
        assert len(scenarios) == 1
        sc = scenarios[0]
        assert sc.id == "YAML-01"
        assert sc.title == "Simple weather lookup"
        assert sc.category == Category.A
        assert sc.difficulty == 1
        assert "Berlin" in sc.user_message

    def test_handler_returns_declarative_response(self) -> None:
        scenarios = load_yaml_scenarios(_scenarios_dir())
        sc = scenarios[0]
        state = ScenarioState()
        record = _record("get_weather", {"location": "Berlin"})
        result = sc.handle_tool_call(state, record)
        assert result["location"] == "Berlin"
        assert result["condition"] == "cloudy"

    def test_handler_returns_generic_fallback_when_no_rule_matches(self) -> None:
        scenarios = load_yaml_scenarios(_scenarios_dir())
        sc = scenarios[0]
        state = ScenarioState()
        record = _record("get_weather", {"location": "Paris"})
        result = sc.handle_tool_call(state, record)
        assert result == {"result": "ok"}

    def test_evaluator_passes_on_expected_call(self) -> None:
        scenarios = load_yaml_scenarios(_scenarios_dir())
        sc = scenarios[0]
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Berlin"}))
        evaluation = sc.evaluate(state)
        assert evaluation.status == ScenarioStatus.PASS
        assert evaluation.points == 2

    def test_evaluator_fails_on_wrong_arguments(self) -> None:
        scenarios = load_yaml_scenarios(_scenarios_dir())
        sc = scenarios[0]
        state = ScenarioState()
        state.tool_calls.append(_record("get_weather", {"location": "Paris"}))
        evaluation = sc.evaluate(state)
        assert evaluation.status == ScenarioStatus.FAIL

    def test_evaluator_fails_on_extra_calls(self) -> None:
        scenarios = load_yaml_scenarios(_scenarios_dir())
        sc = scenarios[0]
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
            "expected_tool_calls: []\n"
        )

        evaluation = _load_yaml_file(path).evaluate(ScenarioState())

        assert evaluation.status == ScenarioStatus.PASS
        assert evaluation.points == 2

    def test_restraint_evaluator_fails_when_any_tool_is_called(self, tmp_path: Path) -> None:
        path = tmp_path / "restraint.yaml"
        path.write_text(
            "id: YAML-R\ntitle: Restraint\ncategory: A\nuser_message: Answer directly\n"
            "expected_tool_calls: []\n"
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
            (root / "b.yaml").write_text(yaml_b)
            (root / "a.yaml").write_text(yaml_a)
            scenarios = load_yaml_scenarios(root)
            assert [s.id for s in scenarios] == ["YAML-A", "YAML-B"]

    def test_invalid_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text("not a mapping")
            with pytest.raises(ValueError):
                _load_yaml_file(path)

    def test_missing_id_field_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "noid.yaml"
            path.write_text("title: No ID\ncategory: A\nuser_message: hi\n")
            with pytest.raises(ValueError, match="'id'"):
                _load_yaml_file(path)

    def test_missing_category_field_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nocat.yaml"
            path.write_text("id: X\ntitle: No Cat\nuser_message: hi\n")
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
        path.write_text("\n".join(f"{key}: {value}" for key, value in values.items()))

        with pytest.raises(ValueError, match=rf"{field!r}.*{path}"):
            _load_yaml_file(path)

    @pytest.mark.parametrize("field", ["id", "title", "category", "user_message"])
    def test_required_fields_must_be_non_empty_strings(self, tmp_path: Path, field: str) -> None:
        path = tmp_path / "invalid.yaml"
        path.write_text(
            "id: X\ntitle: Required fields\ncategory: A\nuser_message: hi\n" + f"{field}: []\n"
        )

        with pytest.raises(ValueError, match=rf"{field!r}.*non-empty string.*{path}"):
            _load_yaml_file(path)

    def test_invalid_category_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "badcat.yaml"
            path.write_text("id: X\ntitle: Bad Cat\ncategory: Z\nuser_message: hi\n")
            with pytest.raises(ValueError, match="Invalid category"):
                _load_yaml_file(path)

    def test_yaml_parse_error_includes_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "syntax.yaml"
            path.write_text("id: X\n  bad: : : indent\n")
            with pytest.raises(ValueError, match="YAML parse error"):
                _load_yaml_file(path)
