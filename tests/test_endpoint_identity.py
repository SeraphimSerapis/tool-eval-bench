"""One endpoint, one identity, whichever way its base URL is spelled.

``http://h:8000``, ``http://h:8000/``, ``http://h:8000/v1`` and ``…/v1/`` send
every request to the same ``…/v1/chat/completions``, yet used to be four
cohorts that refused to resume each other. A Gemini base is the exception: a
bare host there means ``v1beta``, so ``/v1`` really is a different API.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from tool_eval_bench.application.decision_audit import decision_judge_config
from tool_eval_bench.application.run_config import (
    RunSettings,
    build_run_config,
    resume_mismatches,
)
from tool_eval_bench.cli.dispatch import _resume_config_mismatches
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.domain.scenarios import Category, ScenarioDefinition
from tool_eval_bench.utils.fingerprint import comparison_fingerprint, with_config_fingerprint
from tool_eval_bench.utils.urls import (
    endpoint_identity,
    legacy_endpoint_identities,
    legacy_endpoint_identity,
)

SPELLINGS = [
    "http://gpu-box:8000",
    "http://gpu-box:8000/",
    "http://gpu-box:8000/v1",
    "http://gpu-box:8000/v1/",
]
GEMINI = "https://generativelanguage.googleapis.com"


def _scenario(sid: str) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=sid,
        title=sid,
        category=Category.A,
        user_message="",
        description="",
        handle_tool_call=lambda state, call: None,
        evaluate=lambda state: None,  # type: ignore[arg-type,return-value]
    )


SCENARIOS = [_scenario("TC-01")]


def _settings(base_url: str, **overrides: object) -> RunSettings:
    args = _make_parser().parse_args([])
    return RunSettings(
        model="m",
        backend="vllm",
        base_url=base_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        max_turns=args.max_turns,
        seed=args.seed,
        reference_date=args.reference_date,
        concurrency=args.parallel,
        error_rate=args.error_rate,
        alpha=args.alpha,
        extra_params=None,
        context_pressure_config=None,
        weight_by_difficulty=False,
        **overrides,  # type: ignore[arg-type]
    )


def _config(base_url: str, **overrides: object) -> dict:
    return build_run_config(_settings(base_url, **overrides), scenarios=SCENARIOS, metadata={})


class TestIdentity:
    def test_every_spelling_of_one_endpoint_shares_an_identity(self) -> None:
        assert len({endpoint_identity(url) for url in SPELLINGS}) == 1

    def test_the_bare_host_keeps_its_previous_identity(self) -> None:
        # The README's form; canonicalising to the root leaves its stored ids valid.
        assert endpoint_identity("http://gpu-box:8000") == legacy_endpoint_identity(
            "http://gpu-box:8000"
        )

    def test_a_path_prefix_still_tells_endpoints_apart(self) -> None:
        assert endpoint_identity("http://h/proxy/v1") == endpoint_identity("http://h/proxy")
        assert endpoint_identity("http://h/proxy/v1") != endpoint_identity("http://h")
        # Only one trailing /v1 is the API root; /v1/v1 requests /v1/v1/....
        assert endpoint_identity("http://h/v1/v1") != endpoint_identity("http://h/v1")

    def test_gemini_keeps_its_api_version(self) -> None:
        bare = endpoint_identity(GEMINI, wire_format="gemini")
        v1 = endpoint_identity(f"{GEMINI}/v1", wire_format="gemini")
        assert bare != v1
        assert v1 == endpoint_identity(f"{GEMINI}/v1/", wire_format="gemini")

    def test_a_pasted_messages_url_is_the_anthropic_root(self) -> None:
        root = endpoint_identity("https://gw.example/v1/messages", wire_format="anthropic")
        assert root == endpoint_identity("https://gw.example", wire_format="anthropic")
        # Under the OpenAI format a /messages path is just a path.
        assert endpoint_identity("https://gw.example/v1/messages") != endpoint_identity(
            "https://gw.example"
        )


class TestScoredRunConfig:
    def test_every_spelling_lands_in_one_cohort(self) -> None:
        configs = [_config(url) for url in SPELLINGS]
        assert len({c["config_fingerprint"] for c in configs}) == 1
        assert len({c["endpoint_id"] for c in configs}) == 1
        # The redacted URL is still stored as given.
        assert configs[2]["base_url"] == "http://***:8000/v1"

    def test_a_different_host_is_still_a_different_cohort(self) -> None:
        assert (
            _config("http://gpu-box:8000")["config_fingerprint"]
            != _config("http://other-box:8000")["config_fingerprint"]
        )

    def test_a_gemini_host_keeps_bare_and_v1_apart(self) -> None:
        assert _config(GEMINI)["endpoint_id"] != _config(f"{GEMINI}/v1")["endpoint_id"]

    def test_mode_runs_share_the_identity(self) -> None:
        fingerprints = {
            with_config_fingerprint({"model": "m", "base_url": url})["config_fingerprint"]
            for url in SPELLINGS
        }
        assert len(fingerprints) == 1

    def test_a_doubled_v1_mode_run_is_not_a_legacy_single_v1_cohort(self) -> None:
        # Before #269 a mode run's fingerprint hashed base_url with the legacy identity.
        doubled = with_config_fingerprint({"model": "m", "base_url": "http://h/v1/v1"})
        legacy_single = comparison_fingerprint(
            {"model": "m", "base_url": legacy_endpoint_identity("http://h/v1")}, {}
        )
        assert doubled["config_fingerprint"] != legacy_single


class TestResume:
    @pytest.mark.parametrize("stored_url", SPELLINGS)
    @pytest.mark.parametrize("current_url", SPELLINGS)
    def test_any_spelling_resumes_any_other(self, stored_url: str, current_url: str) -> None:
        assert resume_mismatches(_config(stored_url), _config(current_url)) == []

    def test_a_run_stored_with_the_legacy_v1_identity_resumes_from_its_url(self) -> None:
        url = "http://gpu-box:8000/v1"
        stored = {**_config(url), "endpoint_id": legacy_endpoint_identity(url)}
        assert stored["endpoint_id"] != endpoint_identity(url)

        assert resume_mismatches(stored, _config(url), base_url=url) == []

    @pytest.mark.parametrize("stored_url", SPELLINGS)
    @pytest.mark.parametrize("current_url", SPELLINGS)
    def test_a_legacy_identity_resumes_under_any_spelling(
        self, stored_url: str, current_url: str
    ) -> None:
        stored = {**_config(stored_url), "endpoint_id": legacy_endpoint_identity(stored_url)}

        assert resume_mismatches(stored, _config(current_url), base_url=current_url) == []

    @pytest.mark.parametrize(
        "group",
        [
            [
                "https://api.anthropic.com",
                "https://api.anthropic.com/v1",
                "https://api.anthropic.com/v1/messages",
            ],
            # A gateway: the /messages form is Anthropic, the /v1 form OpenAI.
            ["https://opencode.ai/zen/v1/messages", "https://opencode.ai/zen/v1"],
        ],
    )
    def test_a_legacy_messages_identity_resumes_under_any_spelling(self, group: list[str]) -> None:
        for stored_url, current_url in itertools.product(group, group):
            stored = {**_config(stored_url), "endpoint_id": legacy_endpoint_identity(stored_url)}
            current = _config(current_url)

            assert resume_mismatches(stored, current, base_url=current_url) == [], (
                stored_url,
                current_url,
            )

    def test_a_doubled_v1_does_not_accept_the_single_v1_legacy_identity(self) -> None:
        assert legacy_endpoint_identity("http://h/v1") not in legacy_endpoint_identities(
            "http://h/v1/v1"
        )
        # /v1/v1 canonicalises to the root /v1, which the legacy hash spelled
        # exactly like the /v1 endpoint. The new identity must not.
        assert legacy_endpoint_identity("http://h/v1") != endpoint_identity("http://h/v1/v1")
        stored = {**_config("http://h/v1"), "endpoint_id": legacy_endpoint_identity("http://h/v1")}
        current_url = "http://h/v1/v1"

        assert resume_mismatches(stored, _config(current_url), base_url=current_url) == [
            "base_url",
            "endpoint_id",
        ]

    def test_a_single_v1_does_not_accept_a_doubled_v1_identity(self) -> None:
        stored = _config("http://h/v1/v1")
        current_url = "http://h/v1"

        assert resume_mismatches(stored, _config(current_url), base_url=current_url) == [
            "base_url",
            "endpoint_id",
        ]

    @pytest.mark.parametrize("current_url", ["http://h/v1/v1", "http://h/v1/v1/"])
    def test_a_legacy_doubled_v1_run_still_resumes_from_its_url(self, current_url: str) -> None:
        stored = {
            **_config("http://h/v1/v1"),
            "endpoint_id": legacy_endpoint_identity("http://h/v1/v1"),
        }

        assert resume_mismatches(stored, _config(current_url), base_url=current_url) == []

    def test_a_doubled_v1_identity_ignores_a_trailing_slash(self) -> None:
        assert endpoint_identity("http://h/v1/v1") == endpoint_identity("http://h/v1/v1/")
        assert endpoint_identity(
            "http://h/v1/v1/messages", wire_format="anthropic"
        ) == endpoint_identity("http://h/v1/v1")

    @pytest.mark.parametrize(
        ("stored_url", "current_url"),
        [
            ("http://proxy/model-a/v1", "http://proxy/model-b/v1"),
            ("http://proxy/model-a/v1", "http://proxy/model-b"),
            ("http://proxy/model-a", "http://proxy/model-b/v1"),
            ("http://gpu-box:8000/v1", "http://gpu-box:8001/v1"),
            ("http://gpu-box:8000/v1", "https://gpu-box:8000/v1"),
            (GEMINI, f"{GEMINI}/v1"),
        ],
    )
    def test_a_legacy_identity_of_another_endpoint_is_refused(
        self, stored_url: str, current_url: str
    ) -> None:
        stored = {**_config(stored_url), "endpoint_id": legacy_endpoint_identity(stored_url)}

        assert "endpoint_id" in resume_mismatches(
            stored, _config(current_url), base_url=current_url
        )

    def test_the_cli_resume_check_passes_the_raw_url(self) -> None:
        url = "http://gpu-box:8000/v1"
        stored = {**_config(url), "endpoint_id": legacy_endpoint_identity(url)}

        mismatches = _resume_config_mismatches(
            stored,
            model="m",
            backend="vllm",
            base_url=url,
            scenarios=SCENARIOS,
            args=_make_parser().parse_args([]),
            extra_params=None,
            scenario_packs=None,
            context_pressure=None,
        )

        assert mismatches == []

    def test_a_legacy_identity_of_another_host_is_still_refused(self) -> None:
        stored = {
            **_config("http://other-box:8000/v1"),
            "endpoint_id": legacy_endpoint_identity("http://other-box:8000/v1"),
        }
        current_url = "http://gpu-box:8000/v1"

        assert resume_mismatches(stored, _config(current_url), base_url=current_url) == [
            "endpoint_id"
        ]

    def test_a_different_path_prefix_is_still_refused(self) -> None:
        assert resume_mismatches(_config("http://h/a/v1"), _config("http://h/b/v1")) == [
            "base_url",
            "endpoint_id",
        ]

    def test_a_judge_stored_with_the_legacy_identity_resumes(self) -> None:
        judge_url = "http://judge.test/v1"
        judge = decision_judge_config(judge_url, "j")
        current = _config("http://gpu-box:8000", decision_judge=judge)
        stored = dataclasses.replace(
            _settings("http://gpu-box:8000"),
            decision_judge={**judge, "endpoint_id": legacy_endpoint_identity(judge_url)},
        )
        previous = build_run_config(stored, scenarios=SCENARIOS, metadata={})

        assert resume_mismatches(previous, current, judge_base_url=judge_url) == []


LEGACY_JUDGE_URL = "http://judge.test:8084/v1"


def _judge_config(url: str) -> dict:
    return {**decision_judge_config(url, "j", judge_set="all"), "checks": ["c1"]}


def _legacy_judge() -> dict:
    """A judge config as stored before judge URLs were redacted and identities canonical."""
    return {
        **_judge_config(LEGACY_JUDGE_URL),
        "base_url": LEGACY_JUDGE_URL,
        "endpoint_id": legacy_endpoint_identity(LEGACY_JUDGE_URL),
    }


@pytest.mark.parametrize(
    "current_url",
    ["http://judge.test:8084/v1", "http://judge.test:8084", "http://judge.test:8084/v1/"],
)
def test_a_legacy_raw_url_judge_resumes_under_any_spelling(current_url: str) -> None:
    previous = {"decision_judge": _legacy_judge()}
    current = {"decision_judge": _judge_config(current_url)}

    assert resume_mismatches(previous, current, judge_base_url=current_url) == []


def test_a_legacy_raw_url_judge_still_refuses_another_host() -> None:
    other = "http://other.test:8084/v1"
    previous = {"decision_judge": _legacy_judge()}
    current = {"decision_judge": _judge_config(other)}

    assert resume_mismatches(previous, current, judge_base_url=other) == ["decision_judge"]


def test_a_redacted_judge_config_resumes_with_the_bare_host() -> None:
    bare = "http://judge.test:8084"
    previous = {"decision_judge": _judge_config(LEGACY_JUDGE_URL)}
    current = {"decision_judge": _judge_config(bare)}

    assert resume_mismatches(previous, current, judge_base_url=bare) == []
