"""Execute the authoring guide so its tool contract cannot drift from its example."""

from pathlib import Path

from scenario_replay import replay, turn

from tool_eval_bench.domain.scenarios import ScenarioStatus


def example():
    document = (Path(__file__).parents[1] / "docs/adding-a-scenario.md").read_text(encoding="utf-8")
    source = document.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(source, "docs/adding-a-scenario.md", "exec"), namespace)  # noqa: S102 - trusted repository example
    return namespace


def test_contribution_example_runs_through_advertised_tool():
    namespace = example()
    scenario = namespace["SCENARIO"]
    assert scenario.tools_override[0]["function"]["name"] == "convert_timezone"
    assert (
        replay(
            scenario, turn(("convert_timezone", namespace["EXPECTED"])), turn(answer="01:00")
        ).status
        is ScenarioStatus.PASS
    )
    assert (
        replay(
            scenario, turn(("convert_timezone", namespace["EXPECTED"])), turn(answer="00:00")
        ).status
        is ScenarioStatus.FAIL
    )
    wrong = dict(namespace["EXPECTED"], date="2026-07-20")
    assert (
        replay(scenario, turn(("convert_timezone", wrong)), turn(answer="00:00")).status
        is ScenarioStatus.FAIL
    )
