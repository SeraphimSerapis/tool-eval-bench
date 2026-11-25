"""The ``--reviewer*`` flags: activation, inheritance, credentials, validation, and wiring."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from tool_eval_bench.application.run_config import build_run_config
from tool_eval_bench.cli.command_registry import REVIEWER
from tool_eval_bench.cli.commands import resolve_scenarios
from tool_eval_bench.cli.dispatch import _report_model_label, _resume_config_mismatches
from tool_eval_bench.cli.history import _paired_model_label
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.cli.retest_diffs import _build_side_kwargs
from tool_eval_bench.cli.reviewer import (
    REVIEWER_DESTS,
    build_reviewer_settings,
    prepare_reviewer_args,
    reviewer_requested,
)
from tool_eval_bench.cli.scored_run import ScoredRun
from tool_eval_bench.domain.review import ReviewerSettings

MAIN_URL = "http://bench.test:8000"


@pytest.fixture(autouse=True)
def _clean_reviewer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TOOL_EVAL_REVIEWER_BASE_URL",
        "TOOL_EVAL_REVIEWER_MODEL",
        "TOOL_EVAL_REVIEWER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _parse(*flags: str) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = make_parser()
    args = parser.parse_args(list(flags))
    prepare_reviewer_args(args, parser)
    return parser, args


def _settings(
    *flags: str,
    main_extra: dict[str, Any] | None = None,
    main_headers: dict[str, str] | None = None,
) -> ReviewerSettings:
    _, args = _parse(*flags)
    return build_reviewer_settings(
        args,
        model="critic",
        main_base_url=MAIN_URL,
        main_api_key="main-key",
        main_headers=main_headers if main_headers is not None else {"X-Team": "bench"},
        main_session_header="X-Session",
        main_extra_params=main_extra,
    )


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def test_every_reviewer_flag_is_registered_for_run_and_resume() -> None:
    assert REVIEWER == REVIEWER_DESTS
    dests = {action.dest for action in make_parser()._actions}
    assert set(REVIEWER_DESTS) <= dests


@pytest.mark.parametrize(
    "flags",
    [["--reviewer-temperature", "0"], ["--reviewer-no-think"], ["--reviewer-max-rounds", "0"]],
)
def test_false_and_zero_reviewer_values_still_enable_the_reviewer(flags: list[str]) -> None:
    assert reviewer_requested(make_parser().parse_args(flags))


def test_no_reviewer_flag_means_no_reviewer_even_with_the_environment_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOL_EVAL_REVIEWER_BASE_URL", "http://reviewer.test")
    _, args = _parse("--temperature", "0.3")

    assert not reviewer_requested(args)
    assert not hasattr(args, "_reviewer_connection")


@pytest.mark.parametrize(
    ("env", "flags", "expected"),
    [
        (
            {
                "TOOL_EVAL_REVIEWER_BASE_URL": "http://reviewer.test:8001",
                "TOOL_EVAL_REVIEWER_MODEL": "env-critic",
                "TOOL_EVAL_REVIEWER_API_KEY": "env-key",
            },
            ["--reviewer"],
            ("http://reviewer.test:8001", "env-critic", "env-key"),
        ),
        (
            {
                "TOOL_EVAL_CRITIC_BASE_URL": "http://critic.test",
                "TOOL_EVAL_CRITIC_MODEL": "provider-critic",
                "TOOL_EVAL_CRITIC_API_KEY": "provider-key",
            },
            ["--reviewer-provider", "critic", "--reviewer-model", "flag-critic"],
            ("http://critic.test", "flag-critic", "provider-key"),
        ),
        # An environment key is bound to its own URL: another origin drops it.
        (
            {
                "TOOL_EVAL_REVIEWER_BASE_URL": "http://critic.test",
                "TOOL_EVAL_REVIEWER_API_KEY": "k",
            },
            ["--reviewer-base-url", "http://elsewhere.test", "--reviewer-model", "c"],
            ("http://elsewhere.test", "c", None),
        ),
        (
            {
                "TOOL_EVAL_REVIEWER_BASE_URL": "http://critic.test",
                "TOOL_EVAL_REVIEWER_API_KEY": "k",
            },
            ["--reviewer-base-url", "http://critic.test/v1", "--reviewer-model", "c"],
            ("http://critic.test/v1", "c", "k"),
        ),
    ],
    ids=["bare-flag-and-environment", "provider", "env-key-other-origin", "env-key-same-origin"],
)
def test_the_reviewer_connection_comes_from_flags_provider_or_environment(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    flags: list[str],
    expected: tuple[str, str, str | None],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    _, args = _parse(*flags)

    connection = args._reviewer_connection
    assert (connection["base_url"], connection["model"], connection["api_key"]) == expected


# ---------------------------------------------------------------------------
# Sampling inheritance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "main_extra", "expected"),
    [
        (
            ["--reviewer-top-k", "20"],
            {"top_p": 0.9, "chat_template_kwargs": {"enable_thinking": False}},
            {"top_p": 0.9, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
        ),
        (["--reviewer-think"], {"chat_template_kwargs": {"enable_thinking": False}}, {}),
        (["--reviewer-no-think"], {}, {"chat_template_kwargs": {"enable_thinking": False}}),
        (
            [
                "--reviewer-backend-kwargs",
                '{"top_p": null, "max_tokens": 512, "chat_template_kwargs": {"x": 1, "y": null}}',
            ],
            {"top_p": 0.9, "chat_template_kwargs": {"y": 2, "z": 3}},
            {"max_tokens": 512, "chat_template_kwargs": {"z": 3, "x": 1}},
        ),
        # Request structure is never inherited; an explicit override may set it.
        ([], {"top_p": 0.9, "model": "other", "messages": [{"role": "user"}]}, {"top_p": 0.9}),
        (
            ["--reviewer-backend-kwargs", '{"stream_options": {"x": 1}}'],
            None,
            {"stream_options": {"x": 1}},
        ),
        # An explicit temperature or seed removes the inherited key.
        (
            ["--reviewer-temperature", "0", "--reviewer-seed", "7"],
            {"temperature": 0.9, "seed": 13, "top_p": 0.5},
            {"top_p": 0.5},
        ),
    ],
    ids=[
        "flag-over-inherited",
        "think",
        "no-think",
        "kwargs-merge-and-null",
        "no-structure",
        "explicit-structure",
        "temperature-and-seed",
    ],
)
def test_reviewer_extra_params_inherit_and_follow_the_reviewer_flags(
    flags: list[str], main_extra: dict[str, Any] | None, expected: dict[str, Any]
) -> None:
    settings = _settings("--reviewer-model", "critic", *flags, main_extra=main_extra)

    assert settings.extra_params == expected


def test_reviewer_sampling_fields_stay_unset_until_given() -> None:
    inherited = _settings("--reviewer-model", "critic")
    explicit = _settings(
        "--reviewer-model",
        "critic",
        "--reviewer-temperature",
        "0.0",
        "--reviewer-timeout",
        "30",
        "--reviewer-seed",
        "5",
        "--reviewer-scope",
        "final",
        "--reviewer-max-rounds",
        "2",
    )

    # Unset fields leave the service to inherit the run's final values.
    assert (inherited.temperature, inherited.timeout_seconds, inherited.seed) == (None, None, None)
    assert (explicit.temperature, explicit.timeout_seconds, explicit.seed) == (0.0, 30.0, 5)
    assert (explicit.scope, explicit.max_rounds) == ("final", 2)


def test_the_reviewer_prompt_file_is_read(tmp_path: Any) -> None:
    prompt = tmp_path / "critic.txt"
    prompt.write_text("Review strictly.\n", encoding="utf-8")

    settings = _settings("--reviewer-model", "critic", "--reviewer-system-prompt-file", str(prompt))

    assert settings.system_prompt == "Review strictly."


# ---------------------------------------------------------------------------
# Credentials: which key and headers reach the reviewer
# ---------------------------------------------------------------------------

_PROVIDER = {
    "TOOL_EVAL_BENCH_BASE_URL": "http://bench.test:8000",
    "TOOL_EVAL_BENCH_API_KEY": "bench-key",
    "TOOL_EVAL_BENCH_HEADERS": "X-Bench=1",
    "TOOL_EVAL_BENCH_SESSION_HEADER": "X-Session",
}


@pytest.mark.parametrize(
    ("env", "flags", "expected"),
    [
        # The benchmarked server: its key, headers, and session header travel.
        ({}, [], (None, "main-key", {"X-Team": "bench"}, "X-Session")),
        (
            {},
            ["--reviewer-base-url", f"{MAIN_URL}/v1"],
            (f"{MAIN_URL}/v1", "main-key", {"X-Team": "bench"}, "X-Session"),
        ),
        # Another server never receives them.
        (
            {},
            ["--reviewer-base-url", "http://bench.test:8001"],
            ("http://bench.test:8001", None, None, None),
        ),
        # The reviewer's own flags.
        (
            {},
            [
                "--reviewer-base-url",
                "http://critic.test",
                "--reviewer-api-key",
                "rk",
                "--reviewer-header",
                "X-Critic=1",
            ],
            ("http://critic.test", "rk", {"X-Critic": "1"}, None),
        ),
        # A provider's credentials are bound to the provider's URL.
        (
            _PROVIDER,
            ["--reviewer-provider", "bench", "--reviewer-base-url", "https://other.example"],
            ("https://other.example", None, None, None),
        ),
        # An environment key without its URL belongs to the benchmarked server.
        (
            {"TOOL_EVAL_REVIEWER_API_KEY": "env-reviewer-key"},
            ["--reviewer-base-url", "http://other.test:9000"],
            ("http://other.test:9000", None, None, None),
        ),
        (
            {"TOOL_EVAL_REVIEWER_API_KEY": "env-reviewer-key"},
            [],
            (None, "env-reviewer-key", {"X-Team": "bench"}, "X-Session"),
        ),
    ],
    ids=[
        "same-server",
        "same-origin-spelling",
        "other-server",
        "own-flags",
        "provider-url-override",
        "env-key-elsewhere",
        "env-key-same-server",
    ],
)
def test_credentials_reach_the_reviewer_only_where_they_belong(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    flags: list[str],
    expected: tuple[str | None, str | None, dict[str, str] | None, str | None],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    settings = _settings("--reviewer-model", "critic", *flags)

    assert (
        settings.base_url,
        settings.api_key,
        settings.extra_headers,
        settings.session_header,
    ) == expected


def test_the_reviewer_format_flag_is_kept() -> None:
    settings = _settings(
        "--reviewer-base-url", "http://critic.test", "--reviewer-format", "anthropic"
    )

    assert settings.wire_format == "anthropic"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--reviewer-temperature", "0.3"], "A reviewer needs --reviewer-model"),
        (
            ["--reviewer-model", "c", "--reviewer-max-rounds", "0"],
            "--reviewer-max-rounds must be at least 1",
        ),
        (["--reviewer-model", "c", "--reviewer-temperature", "-1"], "--reviewer-temperature"),
        (["--reviewer-model", "c", "--reviewer-top-p", "0"], "--reviewer-top-p"),
        (["--reviewer-model", "c", "--reviewer-timeout", "0"], "--reviewer-timeout"),
        (["--reviewer-model", "c", "--reviewer-backend-kwargs", "{bad"], "not valid JSON"),
        (["--reviewer-model", "c", "--reviewer-backend-kwargs", "[1]"], "must be a JSON object"),
        (
            ["--reviewer-model", "c", "--reviewer-backend-kwargs", '{"model": "other"}'],
            "must not set model",
        ),
        (
            [
                "--reviewer-model",
                "c",
                "--reviewer-backend-kwargs",
                '{"metadata": {"api_key": "x"}}',
            ],
            "must not carry credentials (metadata.api_key)",
        ),
        (["--reviewer-base-url", "ftp://critic"], "--reviewer-base-url"),
        (["--reviewer-model", "c", "--reviewer-header", "nonsense"], "--reviewer-header"),
        (
            [
                "--reviewer-model",
                "c",
                "--reviewer-system-prompt",
                "a",
                "--reviewer-system-prompt-file",
                "b",
            ],
            "mutually exclusive",
        ),
    ],
)
def test_invalid_reviewer_flags_are_rejected(
    flags: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _parse(*flags)
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The scored run, resume, and commands that label or retest a pair
# ---------------------------------------------------------------------------


def _request(args: argparse.Namespace) -> ScoredRun:
    return ScoredRun.from_args(
        args,
        model="m",
        backend="vllm",
        base_url=MAIN_URL,
        api_key=None,
        wire_format="openai",
        scenarios=resolve_scenarios(args),
        extra_params={"top_p": 0.9},
        scenario_packs=None,
    )


def test_the_scored_run_passes_the_reviewer_to_the_service() -> None:
    _, args = _parse("--scenarios", "TC-01", "--temperature", "0.4")
    args._reviewer = ReviewerSettings(model="critic", base_url="http://critic.test")

    request = _request(args)

    assert request.service_kwargs()["reviewer"] is args._reviewer
    stored = request.run_settings().reviewer
    assert stored is not None
    assert (stored["model"], stored["temperature"], stored["extra_params"]) == (
        "critic",
        0.4,
        {"top_p": 0.9},
    )


def test_resume_rejects_a_different_or_missing_reviewer() -> None:
    _, args = _parse("--scenarios", "TC-01")
    args._reviewer = ReviewerSettings(model="critic", base_url="http://critic.test")
    request = _request(args)
    stored = build_run_config(
        request.run_settings(), scenarios=request.config_scenarios(), metadata={}
    )
    common: dict[str, Any] = {
        "model": "m",
        "backend": "vllm",
        "base_url": MAIN_URL,
        "scenarios": request.scenarios,
        "extra_params": {"top_p": 0.9},
        "scenario_packs": None,
        "context_pressure": None,
    }

    assert _resume_config_mismatches(stored, args=args, **common) == []

    args._reviewer = ReviewerSettings(model="other", base_url="http://critic.test")
    assert _resume_config_mismatches(stored, args=args, **common) == ["reviewer model"]

    args._reviewer = None
    assert _resume_config_mismatches(stored, args=args, **common) == [
        "reviewer (was critic, now off)"
    ]


def test_the_cli_names_the_pair_in_summaries_and_history() -> None:
    paired = {"config": {"reviewer": {"model": "critic"}}}

    assert _report_model_label(
        "small", argparse.Namespace(_reviewer=ReviewerSettings(model="critic"))
    ) == ("small + reviewer critic")
    assert _report_model_label("small", argparse.Namespace()) == "small"
    assert _paired_model_label(paired, "m") == "m + reviewer critic"
    assert _paired_model_label({"config": {}}, "m") == "m"


def test_retest_diffs_refuses_a_reviewed_run() -> None:
    record = {"config": {"model": "m", "backend": "vllm", "reviewer": {"model": "critic"}}}

    with pytest.raises(ValueError, match="reviewed by critic"):
        _build_side_kwargs(
            "run-a",
            record,
            base_url=MAIN_URL,
            api_key=None,
            backend_override=None,
            scenarios=[],
            scenario_packs=None,
        )


def test_a_reviewer_key_of_its_own_drops_the_inherited_authorization() -> None:
    headers = {"Authorization": "Bearer main-key", "X-Team": "bench"}

    own_key = _settings("--reviewer-model", "c", "--reviewer-api-key", "rk", main_headers=headers)
    shared_key = _settings("--reviewer-model", "c", main_headers=headers)

    assert (own_key.api_key, own_key.extra_headers) == ("rk", {"X-Team": "bench"})
    assert (shared_key.api_key, shared_key.extra_headers) == ("main-key", headers)


def test_the_final_panel_names_the_reviewer() -> None:
    import io

    from rich.console import Console

    from tool_eval_bench.cli.display import _print_final_panel
    from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
    from tool_eval_bench.evals.scenarios import SCENARIOS
    from tool_eval_bench.runner.orchestrator import score_results

    summary = score_results(
        [ScenarioResult(SCENARIOS[0].id, ScenarioStatus.PASS, 2, "ok")], [SCENARIOS[0]]
    )
    shown = {}
    for reviewer in ("critic", None):
        buf = io.StringIO()
        _print_final_panel(Console(file=buf, width=120), "m", summary, 1.0, reviewer_model=reviewer)
        shown[reviewer] = buf.getvalue()

    assert "Reviewer: critic" in shown["critic"]
    assert "Reviewer" not in shown[None]


def test_the_final_panel_counts_the_reviewers_tokens() -> None:
    import io

    from rich.console import Console

    from tool_eval_bench.cli.display import _print_final_panel
    from tool_eval_bench.domain.scenarios import FailureKind, ScenarioResult, ScenarioStatus
    from tool_eval_bench.evals.scenarios import SCENARIOS
    from tool_eval_bench.runner.orchestrator import score_results

    first, second = SCENARIOS[0], SCENARIOS[1]

    def result(scenario, review, *, model_tokens=(1500, 500), timed_out=False):
        return ScenarioResult(
            scenario.id,
            ScenarioStatus.FAIL if timed_out else ScenarioStatus.PASS,
            0 if timed_out else 2,
            "timed out" if timed_out else "ok",
            prompt_tokens=model_tokens[0],
            completion_tokens=model_tokens[1],
            review=review,
            failure_kind=FailureKind.TIMEOUT if timed_out else None,
        )

    def panel(*results):
        summary = score_results(list(results), [first, second][: len(results)])
        buf = io.StringIO()
        _print_final_panel(Console(file=buf, width=160), "m", summary, 1.0)
        return buf.getvalue(), summary

    review = {"reviews": 3, "prompt_tokens": 1800, "completion_tokens": 200, "latency_ms": 9}
    paired, paired_summary = panel(result(first, review))
    # 2 points over 4,000 tokens: the pair's cost, split by who used it.
    assert "Total: 4,000 tokens (2,000 model + 2,000 reviewer)" in paired
    assert "Efficiency: 0.5 pts/1K tokens" in paired
    # The stored metrics stay the benchmarked model's own.
    assert paired_summary.total_tokens == 2000
    assert paired_summary.token_efficiency == 1.0

    # After a failed review the reviewer's count is a floor, Efficiency a ceiling.
    failed_review = {**review, "reviewer_errors": 1}
    with_failure, _ = panel(result(first, failed_review))
    assert "2,000 reviewer; usage of 1 failed review may be missing)" in with_failure
    assert "Efficiency: at most 0.5 pts/1K tokens" in with_failure
    # 2 points over 16,000 tokens is 0.125: the bound rounds up, not down to 0.1.
    big_failed = {
        "reviews": 2,
        "reviewer_errors": 1,
        "prompt_tokens": 13000,
        "completion_tokens": 1000,
    }
    rounded, _ = panel(result(first, big_failed))
    assert "Efficiency: at most 0.2 pts/1K tokens" in rounded
    reviewer_only, _ = panel(result(first, failed_review, model_tokens=(0, 0)))
    assert "the model's server reported no token usage; usage of 1 failed review" in reviewer_only

    # A timed-out scenario is out of total_tokens, so its review is out too.
    with_timeout, _ = panel(result(first, review), result(second, review, timed_out=True))
    assert "Total: 4,000 tokens (2,000 model + 2,000 reviewer)" in with_timeout

    solo, _ = panel(result(first, None))
    assert "Total: 2,000 tokens  " in solo
    assert "Efficiency: 1.0 pts/1K tokens" in solo
    assert "reviewer" not in solo

    # One side's server reported no usage: the pair's cost is unknown.
    no_reviewer_usage, _ = panel(result(first, {"reviews": 2}))
    assert "Model: 2,000 tokens" in no_reviewer_usage
    assert "the reviewer's server reported no token usage" in no_reviewer_usage
    assert "Efficiency" not in no_reviewer_usage
    no_model_usage, _ = panel(result(first, review, model_tokens=(0, 0)))
    assert "Reviewer: 2,000 tokens" in no_model_usage
    assert "the model's server reported no token usage" in no_model_usage
    assert "Efficiency" not in no_model_usage

    # A bool or a negative count is no usage.
    odd, _ = panel(result(first, {"prompt_tokens": True, "completion_tokens": -5}))
    assert "reviewer" not in odd
    assert "Total: 2,000 tokens  " in odd
