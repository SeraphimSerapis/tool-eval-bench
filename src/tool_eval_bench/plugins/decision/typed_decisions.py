"""The typed-decisions test split, vendored with the package.

Typed Decisions (``LocalLLaMA/typed-decisions`` on Hugging Face, Apache-2.0) is
third-party data: 400 cases across four workflows, each one state with five
typed questions and a soft gold distribution per question.  ``vendor/typed_decisions``
holds the file, its LICENSE and NOTICE, and a manifest written by
``scripts/vendor_typed_decisions.py``.

The loader checks the file against the manifest before returning anything, so
an accidental edit fails loudly instead of quietly changing every score.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

from tool_eval_bench.domain.decision import DecisionQuestion, question_from_wire

QUESTIONS_PER_CASE = 5
CASES_PER_WORKFLOW = 100
WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)
# Yes/no first, as the dataset card lists them.
QUESTION_TYPES = ("noul", "choice", "score")
STATE_PREVIEW_CHARS = 200

# Reference points from the dataset card, for reading a score.  They describe
# the data, not any model.
PRIOR_ACCURACY = 0.470
TEACHER_CEILING_ACCURACY = 0.735


class DatasetIntegrityError(ValueError):
    """The vendored file does not match its manifest."""


@dataclass(frozen=True)
class DatasetInfo:
    """Provenance recorded with every run on this data."""

    dataset: str
    url: str
    revision: str
    license: str
    split: str

    @property
    def attribution(self) -> str:
        return (
            f"Typed Decisions ({self.dataset}, {self.split} split, revision "
            f"{self.revision[:12]}), {self.license}, {self.url}"
        )


@dataclass(frozen=True)
class GoldAnswer:
    """The dataset's gold for one question.

    ``probabilities`` is the mean of three teacher samples.  ``label`` is its
    most likely answer; where two answers tie, it is the dataset's own pick.
    """

    label: str
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class TypedDecisionCase:
    """One state and its questions, sent together in a single request."""

    id: str
    workflow: str
    state: Mapping[str, Any]
    questions: Mapping[str, DecisionQuestion]
    gold: Mapping[str, GoldAnswer]

    def question_type(self, name: str) -> str:
        """The wire type of question ``name``: ``noul``, ``choice``, or ``score``."""
        return str(self.questions[name].to_wire()["type"])

    @property
    def state_preview(self) -> str:
        return json.dumps(self.state, ensure_ascii=False)[:STATE_PREVIEW_CHARS]


def _data_dir() -> Traversable:
    return resources.files("tool_eval_bench.plugins.decision") / "vendor" / "typed_decisions"


def dataset_info(data_dir: Traversable | None = None) -> DatasetInfo:
    """Provenance from the manifest.  Raises ``DatasetIntegrityError`` if it is unreadable."""
    manifest = _manifest(data_dir or _data_dir())
    try:
        return DatasetInfo(
            dataset=manifest["dataset"],
            url=manifest["url"],
            revision=manifest["revision"],
            license=manifest["license"],
            split=manifest["split"],
        )
    except KeyError as exc:
        raise DatasetIntegrityError(f"typed-decisions manifest has no {exc}") from exc


def load_cases(data_dir: Traversable | None = None) -> list[TypedDecisionCase]:
    """Return every test case in file order.

    Raises ``DatasetIntegrityError`` when a file is missing or unreadable, or when
    the file's hash, row count, workflows, or question counts differ from what
    the manifest and the dataset card say.
    """
    directory = data_dir or _data_dir()
    manifest = _manifest(directory)
    try:
        data = manifest["data"]
        expected_digest, expected_rows = data["jsonl_sha256"], data["rows"]
        jsonl = gzip.decompress((directory / data["path"]).read_bytes())
    except (OSError, EOFError, zlib.error, KeyError, TypeError) as exc:
        # A bad header is BadGzipFile (an OSError), a damaged stream zlib.error,
        # and a truncated file EOFError.
        raise DatasetIntegrityError(f"typed-decisions data is unreadable: {exc!r}") from exc
    digest = hashlib.sha256(jsonl).hexdigest()
    if digest != expected_digest:
        raise DatasetIntegrityError(
            f"typed-decisions data has sha256 {digest}, the manifest expects "
            f"{expected_digest}; regenerate it with scripts/vendor_typed_decisions.py "
            "instead of editing it"
        )
    try:
        cases = [_case(json.loads(line)) for line in jsonl.decode("utf-8").splitlines()]
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        # Only reachable when the manifest was rewritten to match a bad file.
        raise DatasetIntegrityError(f"typed-decisions data is malformed: {exc!r}") from exc
    _check_shape(cases, expected_rows=expected_rows)
    return cases


def _manifest(directory: Traversable) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetIntegrityError(f"typed-decisions manifest is unreadable: {exc!r}") from exc
    if not isinstance(manifest, dict):
        raise DatasetIntegrityError("typed-decisions manifest is not a JSON object")
    return manifest


def _case(row: Mapping[str, Any]) -> TypedDecisionCase:
    questions = {name: question_from_wire(q) for name, q in row["questions"].items()}
    gold = {
        name: GoldAnswer(
            label=str(g["label"]),
            probabilities={str(k): float(v) for k, v in g["probabilities"].items()},
        )
        for name, g in row["gold"].items()
    }
    return TypedDecisionCase(
        id=row["id"], workflow=row["workflow"], state=row["state"], questions=questions, gold=gold
    )


def _check_shape(cases: list[TypedDecisionCase], *, expected_rows: int) -> None:
    problems = []
    if len(cases) != expected_rows:
        problems.append(f"{len(cases)} rows, expected {expected_rows}")
    if len({c.id for c in cases}) != len(cases):
        problems.append("duplicate case ids")
    per_workflow = Counter(c.workflow for c in cases)
    if per_workflow != dict.fromkeys(WORKFLOWS, CASES_PER_WORKFLOW):
        problems.append(f"cases per workflow {dict(per_workflow)}")
    for case in cases:
        if len(case.questions) != QUESTIONS_PER_CASE or set(case.questions) != set(case.gold):
            problems.append(f"{case.id} does not have {QUESTIONS_PER_CASE} questions with gold")
            break
    if problems:
        raise DatasetIntegrityError("typed-decisions data is malformed: " + "; ".join(problems))
