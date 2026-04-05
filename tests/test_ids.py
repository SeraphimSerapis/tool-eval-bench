from tool_eval_bench.utils.ids import build_run_id


def test_run_id_format() -> None:
    run_id = build_run_id({"a": 1, "b": 2})
    ts, suffix = run_id.split("_")
    assert ts.endswith("Z")
    assert len(suffix) == 6
