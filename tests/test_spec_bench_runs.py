"""spec-bench repetition, steps/s, temperature, and user prompt files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_spec_per_request_metrics import DETAILED, _cli_output
from tool_eval_bench.cli.spec_bench import load_spec_prompt_file
from tool_eval_bench.runner import speculative
from tool_eval_bench.runner.speculative import (
    SpecDecodeInfo,
    SpecDecodeSample,
    parse_per_request_spec_metrics,
    pool_spec_samples,
)
from tool_eval_bench.runner.throughput import ThroughputSample
from tool_eval_bench.storage.reports.spec_decode import write_spec_decode_report

# ---------------------------------------------------------------------------
# pool_spec_samples
# ---------------------------------------------------------------------------


def _run(
    tg: int,
    total_ms: float,
    drafted: int,
    accepted: int,
    steps: int,
    *,
    ttft: float = 100.0,
    error: str | None = None,
) -> SpecDecodeSample:
    sample = SpecDecodeSample(
        pp_tokens=50,
        tg_tokens=tg,
        total_ms=total_ms,
        ttft_ms=ttft,
        tg_tps=40.0,
        draft_tokens_delta=drafted,
        accepted_tokens_delta=accepted,
        num_drafts_delta=steps,
        acceptance_source="response",
        num_spec_tokens=4,
        prompt_type="code",
        spec_method="mtp",
        baseline_tg_tps=30.0,
        error=error,
    )
    if drafted:
        sample.acceptance_rate = accepted / drafted
        sample.acceptance_length = 1.0 + accepted / steps
    return sample


def test_pooling_weights_by_tokens_not_by_run() -> None:
    long = _run(tg=100, total_ms=1100, drafted=100, accepted=80, steps=25)  # α 0.8
    short = _run(tg=20, total_ms=300, drafted=20, accepted=4, steps=5)  # α 0.2
    pooled = pool_spec_samples([long, short])
    assert pooled.runs == 2
    assert pooled.acceptance_rate == pytest.approx(84 / 120)  # not (0.8 + 0.2) / 2
    assert pooled.acceptance_length == pytest.approx(1.0 + 84 / 30)
    assert pooled.acceptance_rate_range == (pytest.approx(0.2), pytest.approx(0.8))
    # Extensive fields are per-run means, so ratios read as pooled values.
    assert pooled.tg_tokens == 60
    assert pooled.total_ms == 700.0
    assert pooled.effective_tg_tps == pytest.approx(120 / 1.2)
    assert pooled.verify_steps_per_s == pytest.approx(30 / 1.2)
    assert pooled.draft_window == pytest.approx(120 / 30)
    assert (pooled.prompt_type, pooled.spec_method, pooled.baseline_tg_tps) == (
        "code",
        "mtp",
        30.0,
    )
    assert pooled.acceptance_source == "response"


def test_pooling_drops_failed_runs_and_keeps_a_lone_failure() -> None:
    ok = _run(tg=100, total_ms=1100, drafted=100, accepted=80, steps=25)
    failed = _run(tg=0, total_ms=0, drafted=0, accepted=0, steps=0, error="boom")
    assert pool_spec_samples([failed, ok]) is ok
    assert pool_spec_samples([failed]) is failed
    assert pool_spec_samples([ok]) is ok


def test_pooling_concatenates_step_arrays_only_when_every_run_has_them() -> None:
    a = _run(tg=80, total_ms=1100, drafted=84, accepted=67, steps=12)
    b = _run(tg=80, total_ms=1100, drafted=84, accepted=67, steps=12)
    for s in (a, b):
        parsed = parse_per_request_spec_metrics(DETAILED)
        assert parsed is not None
        s.per_step_accepted = parsed.per_step_accepted
        s.per_step_drafted = parsed.per_step_drafted
    pooled = pool_spec_samples([a, b])
    assert pooled.per_step_accepted == DETAILED["per_step_accepted"] * 2
    assert pooled.per_position_acceptance == pytest.approx(a.per_position_acceptance)

    b.per_step_accepted = b.per_step_drafted = None
    assert pool_spec_samples([a, b]).per_step_accepted is None


def test_pooling_without_counters_still_pools_throughput() -> None:
    a = SpecDecodeSample(tg_tokens=100, total_ms=1100, ttft_ms=100, prompt_type="filler")
    b = SpecDecodeSample(tg_tokens=100, total_ms=900, ttft_ms=100, prompt_type="filler")
    pooled = pool_spec_samples([a, b])
    assert pooled.acceptance_rate is None
    assert pooled.acceptance_rate_range is None
    assert pooled.verify_steps_per_s is None
    assert pooled.effective_tg_tps == pytest.approx(100 / 0.9)


def test_verify_steps_per_s_needs_steps_and_time() -> None:
    assert SpecDecodeSample(num_drafts_delta=30, total_ms=1300, ttft_ms=300).verify_steps_per_s == (
        pytest.approx(30.0)
    )
    assert SpecDecodeSample(num_drafts_delta=0, total_ms=1000).verify_steps_per_s is None
    assert SpecDecodeSample(num_drafts_delta=30, total_ms=0).verify_steps_per_s is None


# ---------------------------------------------------------------------------
# run_spec_bench: runs and temperature and custom prompts
# ---------------------------------------------------------------------------


class _Client:
    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_sweep_repeats_each_cell_and_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    async def calibrate(*args: object, **kwargs: object) -> speculative.TokenizerConfig:
        return speculative.TokenizerConfig()

    async def detect(*args: object, **kwargs: object) -> SpecDecodeInfo:
        return SpecDecodeInfo(has_prometheus=True)

    calls: list[tuple[str, float]] = []

    async def measure(
        *args: object, prompt_type: str = "filler", temperature: float = 0.0, **kwargs: object
    ) -> SpecDecodeSample:
        calls.append((prompt_type, temperature))
        return _run(tg=100, total_ms=1100, drafted=100, accepted=50 + 10 * len(calls), steps=25)

    monkeypatch.setattr(speculative, "calibrate", calibrate)
    monkeypatch.setattr(speculative, "detect_spec_decoding", detect)
    monkeypatch.setattr(speculative, "measure_spec_single", measure)
    seen: list[int] = []

    async def on_sample(sample: SpecDecodeSample, idx: int, total: int) -> None:
        seen.append(sample.runs)

    samples = await speculative.run_spec_bench(
        "url",
        "m",
        client_factory=lambda **kwargs: _Client(),
        prompt_types=["filler", "code"],
        runs=3,
        temperature=0.7,
        on_sample=on_sample,
    )
    assert [c[0] for c in calls] == ["filler"] * 3 + ["code"] * 3
    assert all(c[1] == 0.7 for c in calls)
    assert len(samples) == 2 and seen == [3, 3]
    assert samples[0].acceptance_rate_range == (pytest.approx(0.6), pytest.approx(0.8))


@pytest.mark.asyncio
async def test_custom_prompt_is_sent_fixed_and_temperature_reaches_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def stream(
        client, base_url, model, messages, tg, api_key, tok_cfg=None, temperature=0.0
    ) -> ThroughputSample:
        captured["messages"] = messages
        captured["temperature"] = temperature
        return ThroughputSample(pp_tokens=12, tg_tokens=4, total_ms=100, ttft_ms=10)

    monkeypatch.setattr(speculative, "_stream_one", stream)
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        depth=4096,
        prompt_type="custom#1",
        temperature=0.5,
        custom_prompts={"custom#1": "Translate this to German: good morning"},
    )
    assert captured["temperature"] == 0.5
    assert captured["messages"][-1]["content"] == "Translate this to German: good morning"  # type: ignore[index]
    # A fixed prompt does not pretend to carry the requested depth.
    assert sample.depth == 0 and sample.prompt_type == "custom#1"


# ---------------------------------------------------------------------------
# load_spec_prompt_file
# ---------------------------------------------------------------------------


def test_prompt_file_plain_lines_json_lines_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "prompts.txt"
    path.write_text(
        "# my workload\n"
        "Summarise the following meeting notes.\n"
        "\n"
        + json.dumps({"prompt": "line one\nline two", "label": "multiline"})
        + "\n"
        + json.dumps({"prompt": "unlabelled json"})
        + "\n",
        encoding="utf-8",
    )
    prompts = load_spec_prompt_file(str(path))
    assert prompts == {
        "custom#1": "Summarise the following meeting notes.",
        "multiline": "line one\nline two",
        "custom#3": "unlabelled json",
    }
    assert load_spec_prompt_file(None) == {}


@pytest.mark.parametrize(
    "body, message",
    [
        ("{not json\n", "invalid JSON line"),
        ('{"label": "x"}\n', "needs a string 'prompt'"),
        ('{"prompt": "a", "label": "dup"}\n{"prompt": "b", "label": "dup"}\n', "duplicate"),
        ("# only a comment\n", "no prompts found"),
    ],
)
def test_prompt_file_rejects_bad_input(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_spec_prompt_file(str(path))


# ---------------------------------------------------------------------------
# CLI summary and report
# ---------------------------------------------------------------------------


def _pooled() -> SpecDecodeSample:
    pooled = pool_spec_samples(
        [
            _run(tg=100, total_ms=1100, drafted=100, accepted=80, steps=25),
            _run(tg=100, total_ms=1100, drafted=100, accepted=60, steps=25),
        ]
    )
    pooled.baseline_tg_tps = None
    return pooled


def test_cli_shows_run_count_range_steps_and_speedup_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _cli_output(monkeypatch, tmp_path, [_pooled()])
    assert "×2" in output
    assert "α=70.0%" in output and "(60–80)" in output
    assert "Steps/s" in output and "25.0" in output
    assert "Speedup ceiling: 4.0–4.0x" in output


def test_cli_omits_ceiling_when_a_baseline_was_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = _pooled()
    sample.baseline_tg_tps = 50.0
    output = _cli_output(monkeypatch, tmp_path, [sample])
    assert "Speedup ceiling" not in output
    assert "2.00x" in output


def test_report_lists_runs_temperature_range_and_steps(tmp_path: Path) -> None:
    text = write_spec_decode_report(tmp_path, "run", "m", [_pooled()], temperature=0.0).read_text(
        encoding="utf-8"
    )
    assert "- **Runs per cell**: 2" in text
    assert "- **Temperature**: 0 (greedy: a ceiling for sampled workloads)" in text
    assert "| 70.0% (60–80) |" in text
    assert "| Steps/s |" in text and "| 25.0 |" in text
    assert "- **Steps/s**" in text

    sampled = write_spec_decode_report(
        tmp_path,
        "run2",
        "m",
        [_run(tg=1, total_ms=1, drafted=1, accepted=1, steps=1)],
        temperature=0.7,
    ).read_text(encoding="utf-8")
    assert "- **Temperature**: 0.7\n" in sampled
    assert "Runs per cell" not in sampled


# ---------------------------------------------------------------------------
# dispatch wiring
# ---------------------------------------------------------------------------


def _dispatch_spec_bench(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.utils import metadata

    async def context(**kwargs):
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    kwargs: dict = {}
    monkeypatch.setattr(dispatch, "_run_spec_bench", lambda *a, **k: kwargs.update(k))
    monkeypatch.setattr(
        sys,
        "argv",
        ["tool-eval-bench", "--model", "m", "--base-url", "url", "--spec-bench", "--no-warmup"]
        + argv,
    )
    dispatch.main()
    return kwargs


def test_dispatch_passes_runs_temperature_and_file_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "p.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    kwargs = _dispatch_spec_bench(
        monkeypatch,
        ["--spec-runs", "5", "--temperature", "0.4", "--spec-prompt-file", str(path)],
    )
    assert kwargs["runs"] == 5
    assert kwargs["temperature"] == 0.4
    assert kwargs["custom_prompts"] == {"custom#1": "one", "custom#2": "two"}
    # No explicit selection: the file's prompts run after the built-in types.
    assert kwargs["prompt_types"] == ["filler", "code", "structured", "custom#1", "custom#2"]


def test_dispatch_explicit_selection_limits_file_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "p.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    kwargs = _dispatch_spec_bench(
        monkeypatch, ["--spec-prompts", "code,custom#2", "--spec-prompt-file", str(path)]
    )
    assert kwargs["prompt_types"] == ["code", "custom#2"]
    defaults = _dispatch_spec_bench(monkeypatch, [])
    assert defaults["runs"] == 3 and defaults["custom_prompts"] == {}
