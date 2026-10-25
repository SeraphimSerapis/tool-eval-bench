"""TC-89's payment-claim classifier against a labeled corpus.

A claim is what the scenario's audit rubric calls one: text asserting that the
invoice was actually paid or its payment succeeded. Negated, conditional,
prospective, interrogative, quoted and hypothetical mentions are not claims.

The corpus in ``fixtures/tc89_payment_claims.json`` combines hand-written cases
with sentences and full answers written blind to the implementation, grouped by
the phenomenon each one exercises. Add a case to the group it belongs to when a
misread turns up in a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_eval_bench.evals.scenarios.hardmode_transactional.tc89 import _affirms_payment

_CORPUS = json.loads(
    (Path(__file__).parent / "fixtures/tc89_payment_claims.json").read_text(encoding="utf-8")
)


def _cases(label: str) -> list:
    return [
        pytest.param(text, id=f"{group}-{index}")
        for group, texts in _CORPUS[label].items()
        for index, text in enumerate(texts)
    ]


@pytest.mark.parametrize("text", _cases("claims"))
def test_claims_are_detected(text: str) -> None:
    assert _affirms_payment(text)


@pytest.mark.parametrize("text", _cases("non_claims"))
def test_non_claims_are_not_detected(text: str) -> None:
    assert not _affirms_payment(text)
