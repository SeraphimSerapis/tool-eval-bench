"""Typed-decisions item set: vendored data, wire shape, soft-gold metrics, summary, report."""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import math
import shutil
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts import vendor_typed_decisions as generator
from tool_eval_bench.adapters.systemone import SystemOneAdapter
from tool_eval_bench.domain.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionBackend,
    DecisionQuestion,
    DecisionResult,
    DecisionUnsupportedError,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
    question_from_wire,
)
from tool_eval_bench.plugins.decision.evaluator import answer_distribution, score_against_gold
from tool_eval_bench.plugins.decision.metrics import argmax, brier_score, kl_divergence
from tool_eval_bench.plugins.decision.plugin import DecisionPlugin
from tool_eval_bench.plugins.decision.typed_decisions import (
    WORKFLOWS,
    DatasetIntegrityError,
    GoldAnswer,
    TypedDecisionCase,
    dataset_info,
    load_cases,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src/tool_eval_bench/plugins/decision/vendor/typed_decisions"


def _raw_rows() -> list[dict[str, Any]]:
    """Rows read straight from the vendored file, bypassing the loader."""
    text = gzip.decompress((DATA_DIR / "test.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines()]


# ---------------------------------------------------------------------------
# Vendored data
# ---------------------------------------------------------------------------


class TestVendoredData:
    def test_all_400_test_cases_load_with_five_questions_across_four_workflows(self) -> None:
        cases = load_cases()
        assert len(cases) == 400
        assert Counter(c.workflow for c in cases) == dict.fromkeys(WORKFLOWS, 100)
        assert all(len(c.questions) == 5 and set(c.gold) == set(c.questions) for c in cases)
        kinds = Counter(type(q).__name__ for c in cases for q in c.questions.values())
        assert kinds == {"YesNoQuestion": 600, "ChoiceQuestion": 600, "ScoreQuestion": 800}

    def test_every_gold_label_is_a_most_likely_gold_answer(self) -> None:
        for case in load_cases():
            for gold in case.gold.values():
                assert gold.probabilities[gold.label] == max(gold.probabilities.values())

    def test_provenance_ships_next_to_the_data(self) -> None:
        info = dataset_info()
        assert info.dataset == generator.DATASET == "LocalLLaMA/typed-decisions"
        assert info.revision == generator.REVISION
        assert (info.split, info.license) == ("test", "Apache-2.0")
        notice = (DATA_DIR / "NOTICE").read_text(encoding="utf-8")
        assert info.revision in notice and info.url in notice and "Apache License 2.0" in notice
        license_text = (DATA_DIR / "LICENSE").read_text(encoding="utf-8")
        assert "Apache License" in license_text and "Version 2.0, January 2004" in license_text

    def test_package_data_ships_the_license_and_notice_with_the_data(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as file:
            patterns = tomllib.load(file)["tool"]["setuptools"]["package-data"]["tool_eval_bench"]
        shipped = {
            name
            for name in ("LICENSE", "NOTICE", "manifest.json", "test.jsonl.gz")
            if any(
                fnmatch.fnmatch(f"plugins/decision/vendor/typed_decisions/{name}", p)
                for p in patterns
            )
        }
        assert shipped == {"LICENSE", "NOTICE", "manifest.json", "test.jsonl.gz"}

    def _copy(self, tmp_path: Path) -> Path:
        target = tmp_path / "typed_decisions"
        shutil.copytree(DATA_DIR, target)
        return target

    def _rewrite(self, directory: Path, rows: list[dict[str, Any]], *, fix_hash: bool) -> None:
        jsonl = generator.encode_jsonl(rows)
        (directory / "test.jsonl.gz").write_bytes(generator.gzip_deterministic(jsonl))
        if fix_hash:
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["data"]["jsonl_sha256"] = hashlib.sha256(jsonl).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_an_edited_row_fails_loudly(self, tmp_path: Path) -> None:
        directory = self._copy(tmp_path)
        rows = _raw_rows()
        rows[7]["gold"]["urgency"]["label"] = "0"
        self._rewrite(directory, rows, fix_hash=False)
        with pytest.raises(DatasetIntegrityError, match="sha256"):
            load_cases(directory)  # type: ignore[arg-type]

    def test_a_dropped_row_fails_even_with_a_matching_hash(self, tmp_path: Path) -> None:
        directory = self._copy(tmp_path)
        self._rewrite(directory, _raw_rows()[1:], fix_hash=True)
        with pytest.raises(DatasetIntegrityError, match="399 rows, expected 400"):
            load_cases(directory)  # type: ignore[arg-type]

    def test_the_unmodified_copy_still_loads(self, tmp_path: Path) -> None:
        assert len(load_cases(self._copy(tmp_path))) == 400  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "damage",
        [
            pytest.param(lambda b: b[: len(b) // 2], id="truncated"),
            pytest.param(lambda b: b[:20] + bytes([b[20] ^ 0xFF]) + b[21:], id="bit-flip"),
            pytest.param(lambda b: b"", id="empty"),
        ],
    )
    def test_a_damaged_file_is_an_integrity_error(
        self, tmp_path: Path, damage: Callable[[bytes], bytes]
    ) -> None:
        directory = self._copy(tmp_path)
        path = directory / "test.jsonl.gz"
        path.write_bytes(damage(path.read_bytes()))
        with pytest.raises(DatasetIntegrityError):
            load_cases(directory)  # type: ignore[arg-type]

    def test_a_missing_data_file_or_manifest_is_an_integrity_error(self, tmp_path: Path) -> None:
        directory = self._copy(tmp_path)
        (directory / "test.jsonl.gz").unlink()
        with pytest.raises(DatasetIntegrityError, match="unreadable"):
            load_cases(directory)  # type: ignore[arg-type]
        (directory / "manifest.json").unlink()
        for read in (load_cases, dataset_info):
            with pytest.raises(DatasetIntegrityError, match="manifest is unreadable"):
                read(directory)  # type: ignore[operator]

    @pytest.mark.parametrize("key", ["jsonl_sha256", "rows", "path"])
    def test_a_manifest_missing_a_data_field_is_an_integrity_error(
        self, tmp_path: Path, key: str
    ) -> None:
        directory = self._copy(tmp_path)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["data"][key]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(DatasetIntegrityError, match=key):
            load_cases(directory)  # type: ignore[arg-type]

    @pytest.mark.parametrize("content", ['{"split": "test"}', "[]", "not json"])
    def test_a_manifest_without_provenance_is_an_integrity_error(
        self, tmp_path: Path, content: str
    ) -> None:
        directory = self._copy(tmp_path)
        (directory / "manifest.json").write_text(content, encoding="utf-8")
        with pytest.raises(DatasetIntegrityError, match="manifest"):
            dataset_info(directory)  # type: ignore[arg-type]

    def test_a_duplicate_case_id_behind_a_matching_hash_is_an_integrity_error(
        self, tmp_path: Path
    ) -> None:
        directory = self._copy(tmp_path)
        rows = _raw_rows()
        rows[1]["id"] = rows[0]["id"]
        self._rewrite(directory, rows, fix_hash=True)
        with pytest.raises(DatasetIntegrityError, match="duplicate case ids"):
            load_cases(directory)  # type: ignore[arg-type]

    @pytest.mark.parametrize("drop", ["question", "gold"])
    def test_a_case_without_five_questions_with_gold_is_an_integrity_error(
        self, tmp_path: Path, drop: str
    ) -> None:
        directory = self._copy(tmp_path)
        rows = _raw_rows()
        name = next(iter(rows[5]["questions"]))
        if drop == "question":
            del rows[5]["questions"][name]
            del rows[5]["gold"][name]
        else:
            del rows[5]["gold"][name]
        self._rewrite(directory, rows, fix_hash=True)
        with pytest.raises(DatasetIntegrityError, match="does not have 5 questions with gold"):
            load_cases(directory)  # type: ignore[arg-type]

    def test_a_malformed_row_behind_a_matching_hash_is_an_integrity_error(
        self, tmp_path: Path
    ) -> None:
        directory = self._copy(tmp_path)
        rows = _raw_rows()
        del rows[3]["gold"]
        self._rewrite(directory, rows, fix_hash=True)
        with pytest.raises(DatasetIntegrityError, match="malformed"):
            load_cases(directory)  # type: ignore[arg-type]


class TestGenerator:
    def _row(self, **overrides: Any) -> dict[str, Any]:
        row = {
            "id": "customer_service_000001",
            "workflow": "customer_service",
            "split": "test",
            "state": '{"thread": []}',
            "questions": '{"q": {"type": "noul", "instructions": "i"}}',
            "gold": '{"q": {"label": "true", "probabilities": {"true": 1.0, "false": 0.0}}}',
            "factors": '{"tone": "calm"}',
            "label_agreement": "{}",
            "n_questions": 1,
        }
        return {**row, **overrides}

    def test_keeps_only_the_benchmark_columns_and_decodes_json(self) -> None:
        [out] = generator.convert_rows([self._row()])
        assert list(out) == ["id", "workflow", "state", "questions", "gold"]
        assert out["state"] == {"thread": []}
        assert out["questions"]["q"] == {"type": "noul", "instructions": "i"}

    def test_refuses_a_row_from_the_train_split(self) -> None:
        with pytest.raises(ValueError, match="not from the test split"):
            generator.convert_rows([self._row(split="train")])

    def test_refuses_a_column_of_the_wrong_type(self) -> None:
        with pytest.raises(ValueError, match="questions is not a string"):
            generator.convert_rows([self._row(questions={"q": {}})])


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


class TestWireShape:
    def test_every_question_round_trips_through_the_typed_form(self) -> None:
        for row in _raw_rows():
            for raw in row["questions"].values():
                assert question_from_wire(raw).to_wire() == raw

    def test_yes_no_criteria_are_sent_only_when_present(self) -> None:
        assert "criteria" not in YesNoQuestion("Is it?").to_wire()
        assert YesNoQuestion("Is it?", {"true": "a", "false": "b"}).to_wire()["criteria"] == {
            "true": "a",
            "false": "b",
        }

    @pytest.mark.parametrize(
        "raw",
        [
            {"type": "choice", "instructions": "i", "criteria": ["a", "b"]},
            {"type": "score", "instructions": "i", "criteria": {"0": "a"}},
            {"type": "noul", "instructions": "i", "criteria": {"true": 1}},
            {"type": "choice", "criteria": {"a": "b"}},
            {"type": "rank", "instructions": "i"},
        ],
    )
    def test_malformed_questions_are_rejected(self, raw: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            question_from_wire(raw)

    @pytest.mark.asyncio
    async def test_one_request_per_case_whose_body_is_the_row_state_and_questions(self) -> None:
        rows = _raw_rows()[:3] + _raw_rows()[100:101]
        by_id = {c.id: c for c in load_cases()}
        cases = [by_id[r["id"]] for r in rows]
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            case = next(c for c in cases if c.state == body["state"])
            return httpx.Response(200, json={"answers": _gold_answers_wire(case)})

        adapter = SystemOneAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await DecisionPlugin().run(
            adapter, model="d1", base_url="http://host:8084", cases=cases
        )

        assert len(bodies) == len(rows)
        for body, row in zip(bodies, rows, strict=True):
            assert body == {"model": "d1", "state": row["state"], "questions": row["questions"]}
        assert result.details["total"] == 5 * len(rows) and result.details["errors"] == 0


def _gold_answers_wire(case: TypedDecisionCase) -> dict[str, Any]:
    """A server response that repeats the gold distribution for every question."""
    answers: dict[str, Any] = {}
    for name, question in case.questions.items():
        probs = dict(case.gold[name].probabilities)
        if isinstance(question, YesNoQuestion):
            answers[name] = {"type": "noul", "noul": probs["true"]}
        elif isinstance(question, ScoreQuestion):
            answers[name] = {"type": "score", "score": 0.0, "probabilities": probs}
        else:
            answers[name] = {"type": "choice", "choice": argmax(probs), "probabilities": probs}
    return answers


# ---------------------------------------------------------------------------
# Soft-gold metrics
# ---------------------------------------------------------------------------


class TestSoftMetrics:
    def test_kl_by_hand(self) -> None:
        # 0.5 ln(0.5/0.25) + 0.5 ln(0.5/0.75)
        expected = 0.5 * math.log(2) + 0.5 * math.log(2 / 3)
        assert kl_divergence({"a": 0.5, "b": 0.5}, {"a": 0.25, "b": 0.75}) == pytest.approx(
            expected
        )
        assert kl_divergence({"a": 0.3, "b": 0.7}, {"a": 0.3, "b": 0.7}) == pytest.approx(0.0)

    def test_kl_ignores_answers_the_gold_rules_out(self) -> None:
        assert kl_divergence({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}) == pytest.approx(
            math.log(2)
        )

    def test_kl_floors_a_zero_or_missing_prediction_instead_of_going_infinite(self) -> None:
        gold = {"a": 0.5, "b": 0.5}
        expected = 0.5 * math.log(0.5) + 0.5 * math.log(0.5 / 1e-12)
        assert kl_divergence(gold, {"a": 1.0, "b": 0.0}) == pytest.approx(expected)
        assert kl_divergence(gold, {"a": 1.0}) == pytest.approx(expected)
        assert math.isfinite(kl_divergence(gold, {}))

    def test_brier_against_a_soft_gold(self) -> None:
        assert brier_score({"a": 0.6, "b": 0.4}, {"a": 1.0, "b": 0.0}) == pytest.approx(0.32)
        assert brier_score({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == pytest.approx(2.0)
        # Mass on an answer the gold does not list still counts.
        assert brier_score({"a": 1.0}, {"a": 0.5, "x": 0.5}) == pytest.approx(0.5)

    def test_a_gold_tie_is_scored_against_the_dataset_label(self) -> None:
        question = ChoiceQuestion("q", {"a": "", "b": ""})
        gold = GoldAnswer(label="b", probabilities={"a": 0.5, "b": 0.5})
        picks_a = score_against_gold(question, ChoiceAnswer("a", {"a": 0.7, "b": 0.3}), gold)
        picks_b = score_against_gold(question, ChoiceAnswer("b", {"a": 0.3, "b": 0.7}), gold)
        assert not picks_a.correct and picks_b.correct
        assert picks_a.kl == pytest.approx(picks_b.kl)

    def test_a_predicted_tie_goes_to_the_first_option(self) -> None:
        question = ChoiceQuestion("q", {"b": "", "a": ""})
        gold = GoldAnswer(label="b", probabilities={"a": 0.1, "b": 0.9})
        score = score_against_gold(question, ChoiceAnswer("a", {"a": 0.5, "b": 0.5}), gold)
        assert score.predicted == "b" and score.correct and score.confidence == 0.5

    def test_the_predicted_label_is_the_argmax_not_the_servers_choice_field(self) -> None:
        question = ChoiceQuestion("q", {"a": "", "b": ""})
        gold = GoldAnswer(label="b", probabilities={"a": 0.2, "b": 0.8})
        score = score_against_gold(question, ChoiceAnswer("a", {"a": 0.4, "b": 0.6}), gold)
        assert score.predicted == "b" and score.correct

    def test_yes_no_and_score_distributions_use_the_gold_keys(self) -> None:
        assert answer_distribution(YesNoQuestion("q"), YesNoAnswer(0.25)) == {
            "true": 0.25,
            "false": 0.75,
        }
        levels = ScoreQuestion("q", ("x", "y", "z"))
        # A level the server left out is probability 0, and an extra one is dropped.
        assert answer_distribution(levels, ScoreAnswer(1.0, {"0": 0.2, "1": 0.8, "9": 0.0})) == {
            "0": 0.2,
            "1": 0.8,
            "2": 0.0,
        }
        tie = score_against_gold(
            YesNoQuestion("q"),
            YesNoAnswer(0.5),
            GoldAnswer("true", {"true": 0.6, "false": 0.4}),
        )
        assert tie.predicted == "true"

    def test_an_answer_of_the_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            answer_distribution(YesNoQuestion("q"), ChoiceAnswer("a", {"a": 1.0}))

    def test_a_uniform_predictor_matches_the_cards_uniform_reference(self) -> None:
        # The card scores a model that puts equal probability on every option at
        # KL 0.444 and Brier 0.238.  Matching it pins both definitions.
        kl: list[float] = []
        brier: list[float] = []
        for case in load_cases():
            for name, question in case.questions.items():
                answer = _uniform(question)
                score = score_against_gold(question, answer, case.gold[name])
                kl.append(score.kl)
                brier.append(score.brier)
        assert round(sum(kl) / len(kl), 3) == 0.444
        assert round(sum(brier) / len(brier), 3) == 0.238


def _uniform(question: DecisionQuestion) -> DecisionAnswer:
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer("", {k: 1 / len(question.options) for k in question.options})
    if isinstance(question, ScoreQuestion):
        n = len(question.levels)
        return ScoreAnswer(0.0, {str(i): 1 / n for i in range(n)})
    return YesNoAnswer(0.5)


# ---------------------------------------------------------------------------
# Plugin run and summary
# ---------------------------------------------------------------------------

Answerer = Callable[[str, DecisionQuestion, GoldAnswer], DecisionAnswer]


def _echo_gold(_name: str, question: DecisionQuestion, gold: GoldAnswer) -> DecisionAnswer:
    probs = dict(gold.probabilities)
    if isinstance(question, YesNoQuestion):
        return YesNoAnswer(probs["true"])
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer(0.0, probs)
    return ChoiceAnswer(argmax(probs), probs)


class FakeTypedBackend(DecisionBackend):
    """Answers each question through ``answer``, looking the case up by its state."""

    def __init__(
        self,
        cases: list[TypedDecisionCase],
        answer: Answerer = _echo_gold,
        *,
        fail_ids: frozenset[str] = frozenset(),
        unsupported: bool = False,
    ) -> None:
        self._by_state = {json.dumps(c.state, sort_keys=True): c for c in cases}
        self.answer = answer
        self.fail_ids = fail_ids
        self.unsupported = unsupported
        self.calls = 0

    async def aclose(self) -> None:
        return None

    async def decide(
        self,
        *,
        model: str,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        base_url: str = "",
    ) -> DecisionResult:
        self.calls += 1
        if self.unsupported:
            raise DecisionUnsupportedError("no endpoint")
        case = self._by_state[json.dumps(state, sort_keys=True)]
        if case.id in self.fail_ids:
            raise TimeoutError("synthetic")
        answers = {n: self.answer(n, q, case.gold[n]) for n, q in questions.items()}
        return DecisionResult(answers, input_tokens=100, elapsed_ms=40.0)


def _hand_case(case_id: str, workflow: str) -> TypedDecisionCase:
    """A small case with known gold: yes/no gold true, choice gold b, score gold 1."""
    return TypedDecisionCase(
        id=case_id,
        workflow=workflow,
        state={"id": case_id},
        questions={
            "flag": YesNoQuestion("flag?"),
            "pick": ChoiceQuestion("pick?", {"a": "", "b": ""}),
            "level": ScoreQuestion("level?", ("low", "mid", "high")),
        },
        gold={
            "flag": GoldAnswer("true", {"false": 0.2, "true": 0.8}),
            "pick": GoldAnswer("b", {"a": 0.4, "b": 0.6}),
            "level": GoldAnswer("1", {"0": 0.2, "1": 0.6, "2": 0.2}),
        },
    )


def _right_on_yes_no_only(
    name: str, question: DecisionQuestion, gold: GoldAnswer
) -> DecisionAnswer:
    if isinstance(question, YesNoQuestion):
        return YesNoAnswer(0.9)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer("a", {"a": 0.9, "b": 0.1})
    return ScoreAnswer(0.0, {"0": 0.9, "1": 0.05, "2": 0.05})


async def _run(backend: DecisionBackend, **kwargs: Any) -> Any:
    return await DecisionPlugin().run(backend, model="m", base_url="u", **kwargs)  # type: ignore[arg-type]


class TestTypedRun:
    @pytest.mark.asyncio
    async def test_the_default_run_sends_all_400_cases_and_records_the_dataset(self) -> None:
        cases = load_cases()
        backend = FakeTypedBackend(cases)
        result = await _run(backend)
        d = result.details
        assert backend.calls == 400
        assert (d["total"], d["cases"]) == (2000, 400)
        assert d["dataset"] == "LocalLLaMA/typed-decisions"
        assert d["dataset_revision"] == generator.REVISION
        assert result.metadata["dataset_revision"] == generator.REVISION
        # Repeating the gold distribution is a perfect match in distribution...
        assert d["kl"] == 0.0 and d["brier"] == 0.0
        # ...and misses only where the gold ties and its label is not listed first.
        assert 2000 - 35 <= d["correct"] < 2000
        assert result.total_tokens == 100 * 400
        assert d["latency_ms"]["p50"] == 40.0

    @pytest.mark.asyncio
    async def test_summary_breaks_down_by_type_workflow_and_question(self) -> None:
        cases = [_hand_case("a1", "alpha"), _hand_case("a2", "alpha"), _hand_case("b1", "beta")]
        result = await _run(FakeTypedBackend(cases, _right_on_yes_no_only), cases=cases)
        d = result.details
        assert [*d["by_type"]] == ["noul", "choice", "score"]
        assert d["by_type"]["noul"]["accuracy"] == 100.0
        assert d["by_type"]["choice"]["accuracy"] == 0.0
        assert d["by_type"]["score"]["correct"] == 0
        assert d["by_workflow"]["alpha"] == {
            "correct": 2,
            "total": 6,
            "accuracy": 33.33,
            "kl": d["by_workflow"]["alpha"]["kl"],
            "brier": d["by_workflow"]["alpha"]["brier"],
        }
        assert d["by_workflow"]["beta"]["total"] == 3
        assert set(d["by_question"]) == {
            "alpha/flag",
            "alpha/pick",
            "alpha/level",
            "beta/flag",
            "beta/pick",
            "beta/level",
        }
        # yes/no: 0.8 ln(0.8/0.9) + 0.2 ln(0.2/0.1)
        flag_kl = 0.8 * math.log(0.8 / 0.9) + 0.2 * math.log(2)
        assert d["by_type"]["noul"]["kl"] == pytest.approx(flag_kl, abs=1e-4)
        assert d["accuracy"] == pytest.approx(100 / 3, abs=0.01)
        # Every answer is 90% confident and one in three is right.
        assert d["calibration"]["ece"] == pytest.approx(0.9 - 1 / 3, abs=1e-3)
        assert d["calibration"]["high_confidence_errors"] == 6

    @pytest.mark.asyncio
    async def test_a_failed_request_costs_every_decision_of_its_case(self) -> None:
        cases = [_hand_case("a1", "alpha"), _hand_case("b1", "beta")]
        backend = FakeTypedBackend(cases, fail_ids=frozenset({"b1"}))
        result = await _run(backend, cases=cases)
        d = result.details
        assert d["errors"] == 3 and d["incomplete"] and d["status"] == "incomplete"
        assert d["error_kinds"] == {"TimeoutError": 3}
        assert d["accuracy"] == 50.0
        # The failed workflow has no distribution to compare, which is not a KL of 0.
        assert d["by_workflow"]["beta"]["kl"] is None
        assert d["by_workflow"]["alpha"]["kl"] == d["kl"]

    @pytest.mark.asyncio
    async def test_an_answer_of_the_wrong_type_costs_its_whole_case(self) -> None:
        # The real adapter rejects a malformed response as a whole, so a fake that
        # gets one answer wrong must fail the case the same way.
        cases = [_hand_case("a1", "alpha"), _hand_case("b1", "beta")]

        def wrong_pick(name: str, question: DecisionQuestion, gold: GoldAnswer) -> DecisionAnswer:
            return YesNoAnswer(0.5) if name == "pick" else _echo_gold(name, question, gold)

        result = await _run(FakeTypedBackend(cases, wrong_pick), cases=cases)
        assert all(r["is_error"] for r in result.item_results)
        assert result.details["error_kinds"] == {"ValueError": 6}

    @pytest.mark.asyncio
    async def test_an_unsupported_endpoint_aborts(self) -> None:
        cases = [_hand_case("a1", "alpha")]
        with pytest.raises(DecisionUnsupportedError):
            await _run(FakeTypedBackend(cases, unsupported=True), cases=cases)

    @pytest.mark.asyncio
    async def test_progress_counts_decisions(self) -> None:
        cases = [_hand_case("a1", "alpha"), _hand_case("b1", "beta")]
        seen: list[tuple[int, int]] = []

        async def on_progress(current: int, total: int, info: dict[str, Any]) -> None:
            seen.append((current, total))

        await _run(FakeTypedBackend(cases), cases=cases, on_progress=on_progress, concurrency=2)
        assert seen == [(i, 6) for i in range(1, 7)]

    @pytest.mark.asyncio
    async def test_a_chat_only_adapter_is_refused_with_a_pointer(self) -> None:
        with pytest.raises(ValueError, match="build_decision_adapter"):
            await _run(object())  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            await _run(FakeTypedBackend([]), cases=[], concurrency=0)

    @pytest.mark.parametrize(
        ("accuracy", "stars"),
        [(47.0, "★ "), (47.01, "★★ "), (73.49, "★★ "), (73.5, "★★★ "), (100.0, "★★★ ")],
    )
    def test_the_rating_bands_are_the_cards_two_reference_points(
        self, accuracy: float, stars: str
    ) -> None:
        from tool_eval_bench.plugins.decision.plugin import _rating

        assert _rating(accuracy, answered=1).startswith(stars)

    @pytest.mark.asyncio
    async def test_a_run_with_no_successful_case_is_unrated(self) -> None:
        from tool_eval_bench.plugins.decision.plugin import NO_SUCCESSFUL_CASES

        cases = [_hand_case("a1", "alpha"), _hand_case("b1", "beta")]
        backend = FakeTypedBackend(cases, fail_ids=frozenset({"a1", "b1"}))
        result = await _run(backend, cases=cases)
        # 0% here measures the connection, not the model, so it earns no stars.
        assert result.rating == NO_SUCCESSFUL_CASES and "★" not in result.rating
        assert result.details["incomplete"]

    @pytest.mark.asyncio
    async def test_one_successful_case_is_enough_for_a_rating(self) -> None:
        cases = [_hand_case("a1", "alpha"), _hand_case("b1", "beta")]
        result = await _run(FakeTypedBackend(cases, fail_ids=frozenset({"b1"})), cases=cases)
        assert result.rating.startswith("★")

    @pytest.mark.asyncio
    async def test_an_empty_case_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cases must not be empty"):
            await _run(FakeTypedBackend([]), cases=[])


# ---------------------------------------------------------------------------
# Report and CLI
# ---------------------------------------------------------------------------


class TestTypedReport:
    @pytest.mark.asyncio
    async def test_report_credits_the_dataset_and_has_every_section(self) -> None:
        cases = [_hand_case("a1", "alpha"), _hand_case("b1", "beta")]
        plugin = DecisionPlugin()
        result = await plugin.run(
            FakeTypedBackend(cases, _right_on_yes_no_only),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            cases=cases,
        )
        report = "\n".join(plugin.render_report_section(result))
        assert "LocalLLaMA/typed-decisions" in report
        assert generator.REVISION[:12] in report
        assert "Apache-2.0" in report and "NOTICE" in report
        assert "agreement with that teacher, not correctness" in report
        for heading in (
            "### By Question Type",
            "### By Workflow",
            "### By Question",
            "### Calibration",
            "### Latency",
            "### Disagreements with Gold",
            "### Full Trace",
        ):
            assert heading in report, heading
        assert "a1/pick" in report and "pred  a=0.900 b=0.100" in report
        # Gold is stored false-first, as in the dataset; the trace aligns it with pred.
        assert "pred  true=0.900 false=0.100\n      gold  true=0.800 false=0.200" in report

    @pytest.mark.asyncio
    async def test_an_all_error_run_renders_without_distances(self) -> None:
        cases = [_hand_case("a1", "alpha")]
        plugin = DecisionPlugin()
        result = await plugin.run(
            FakeTypedBackend(cases, fail_ids=frozenset({"a1"})),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            cases=cases,
        )
        report = "\n".join(plugin.render_report_section(result))
        assert "**KL from gold:** n/a" in report
        assert "flag (noul)  ERROR  TimeoutError: synthetic" in report
        assert "**Rating:** Incomplete: no successful cases" in report
        # No request came back, so there is no latency to report as 0 ms.
        assert "### Latency" not in report and "### Calibration" not in report


class TestDecisionRunner:
    def test_run_prints_credits_persists_and_fingerprints_the_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from rich.console import Console

        import tool_eval_bench.adapters.factory as factory
        from tool_eval_bench.cli import plugin_runners

        cases = load_cases()
        monkeypatch.setattr(
            factory, "build_decision_adapter", lambda **_kw: FakeTypedBackend(cases)
        )
        monkeypatch.setattr(plugin_runners, "_metadata_for_storage", lambda value: {})
        persisted: list[dict[str, Any]] = []
        monkeypatch.setattr(plugin_runners, "_persist_plugin_run", persisted.append)

        args = argparse.Namespace(parallel=4, timeout=5.0, format=None)
        console = Console(record=True, width=200)
        plugin_runners._run_decision_benchmark(
            console, "m", "Display", "http://h", None, args, output_dir=str(tmp_path)
        )

        output = console.export_text()
        assert "Accuracy by question type" in output and "Accuracy by workflow" in output
        assert "Reliability" in output and "decisions)" in output
        assert "LocalLLaMA/typed-decisions" in output
        [run] = persisted
        assert run["run_type"] == "decision"
        assert run["config"]["dataset"] == "LocalLLaMA/typed-decisions"
        assert run["config"]["dataset_revision"] == generator.REVISION
        assert run["config"]["config_fingerprint"]
        assert run["scores"]["dataset_revision"] == generator.REVISION
        [report] = list(tmp_path.rglob("*.md"))
        text = report.read_text(encoding="utf-8")
        assert "Third-party data" in text and generator.REVISION in text

    def test_an_unsupported_endpoint_exits_with_a_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rich.console import Console

        import tool_eval_bench.adapters.factory as factory
        from tool_eval_bench.cli import plugin_runners

        monkeypatch.setattr(
            factory,
            "build_decision_adapter",
            lambda **_kw: FakeTypedBackend([], unsupported=True),
        )
        console = Console(record=True, width=160)
        with pytest.raises(SystemExit):
            plugin_runners._run_decision_benchmark(
                console,
                "m",
                "Display",
                "http://h",
                None,
                argparse.Namespace(parallel=1, timeout=5.0, format=None),
            )
        assert "no endpoint" in console.export_text()

    def test_damaged_data_exits_with_a_message_instead_of_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rich.console import Console

        from tool_eval_bench.cli import plugin_runners
        from tool_eval_bench.plugins.decision import typed_decisions

        def broken(*_a: Any) -> Any:
            raise DatasetIntegrityError("typed-decisions manifest is unreadable: gone")

        monkeypatch.setattr(typed_decisions, "dataset_info", broken)
        console = Console(record=True, width=160)
        with pytest.raises(SystemExit):
            plugin_runners._run_decision_benchmark(
                console,
                "m",
                "Display",
                "http://h",
                None,
                argparse.Namespace(parallel=1, timeout=5.0, format=None),
            )
        assert "Decision error: typed-decisions manifest is unreadable" in console.export_text()

    def test_data_that_fails_its_check_inside_the_run_exits_with_a_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rich.console import Console

        import tool_eval_bench.adapters.factory as factory
        from tool_eval_bench.cli import plugin_runners
        from tool_eval_bench.plugins.decision import typed_decisions

        def broken(*_a: Any) -> Any:
            raise DatasetIntegrityError("typed-decisions data has sha256 abc")

        backend = FakeTypedBackend([])
        monkeypatch.setattr(typed_decisions, "load_cases", broken)
        monkeypatch.setattr(factory, "build_decision_adapter", lambda **_kw: backend)
        console = Console(record=True, width=160)
        with pytest.raises(SystemExit) as exit_info:
            plugin_runners._run_decision_benchmark(
                console,
                "m",
                "Display",
                "http://h",
                None,
                argparse.Namespace(parallel=1, timeout=5.0, format=None),
            )
        assert exit_info.value.code == 1
        output = console.export_text()
        assert "Decision error: typed-decisions data has sha256 abc" in output
        assert "Traceback" not in output and backend.calls == 0

    def test_another_revision_gets_another_fingerprint(self) -> None:
        from tool_eval_bench.cli.helpers import with_config_fingerprint

        config = {"model": "m", "base_url": "http://h", "mode": "decision"}
        a = with_config_fingerprint({**config, "dataset_revision": "aaa"})
        b = with_config_fingerprint({**config, "dataset_revision": "bbb"})
        assert a["config_fingerprint"] != b["config_fingerprint"]
