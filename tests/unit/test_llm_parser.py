"""T:llm-parser / T:llm-parser-neg - contrato de resposta do LLM (PRD §6.3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from physics_llm_sr.llm.interfaces import EquationEvaluation, InvalidScoreValueError
from physics_llm_sr.llm.response_parser import (
    InvalidResponseError,
    MissingKeysError,
    NoJSONObjectError,
    ScoreRangeError,
    ScoreTypeError,
    UnexpectedKeysError,
    parse_evaluation,
)

VALID = {
    "dimensional_consistency": 1.0,
    "simplicity": 0.9,
    "physical_realism": 0.75,
    "feedback": "ok",
}
SENTINEL = Path("/tmp/pwned")


def _payload(**changes: object) -> str:
    data: dict[str, object] = {**VALID, **changes}
    return json.dumps(data)


# ------------------------------------------------------------------ T:llm-parser (aceite)


def test_parses_plain_json() -> None:
    ev = parse_evaluation(_payload())
    assert ev == EquationEvaluation(1.0, 0.9, 0.75, "ok")


def test_extracts_first_object_from_surrounding_text() -> None:
    text = "Sure! Here it is:\n```json\n" + _payload() + "\n```\nThanks {not json}"
    assert parse_evaluation(text).physical_realism == 0.75


def test_skips_brace_in_prose_before_object() -> None:
    text = "Scores {see below}: " + _payload(simplicity=0)
    ev = parse_evaluation(text)
    assert ev.simplicity == 0.0 and isinstance(ev.simplicity, float)


def test_accepts_bounds_and_nested_braces_in_feedback() -> None:
    ev = parse_evaluation(_payload(dimensional_consistency=0, physical_realism=1, feedback="{x}"))
    assert ev.scores == (0.0, 0.9, 1.0)
    assert ev.feedback == "{x}"


def test_evaluation_dataclass_validates() -> None:
    with pytest.raises(InvalidScoreValueError):
        EquationEvaluation(1.2, 0.5, 0.5, "x")
    with pytest.raises(InvalidScoreValueError):
        EquationEvaluation(True, 0.5, 0.5, "x")


# ------------------------------------------------------------- T:llm-parser-neg (negativo)


@pytest.mark.parametrize("bad", [1.5, -0.1, 1.0000001])
def test_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ScoreRangeError):
        parse_evaluation(_payload(simplicity=bad))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_finite(token: str) -> None:
    text = _payload().replace("0.75", token)
    with pytest.raises(ScoreRangeError):
        parse_evaluation(text)


def test_nan_float_not_accepted_via_dataclass() -> None:
    with pytest.raises(InvalidScoreValueError):
        EquationEvaluation(math.nan, 0.5, 0.5, "x")


@pytest.mark.parametrize(
    "value", [True, False, "0.5", None, [0.5], {"v": 0.5}], ids=lambda v: repr(v)
)
def test_rejects_wrong_score_types(value: object) -> None:
    with pytest.raises(ScoreTypeError):
        parse_evaluation(_payload(physical_realism=value))


def test_rejects_non_string_feedback() -> None:
    with pytest.raises(ScoreTypeError):
        parse_evaluation(_payload(feedback=3))


@pytest.mark.parametrize(
    "text",
    [
        "The equation looks physically fine, I would give it 0.9.",
        "",
        "[1.0, 0.9, 0.8, 'python list as in the paper']",
        '{"dimensional_consistency": 1.0, "simplicity": ',
    ],
)
def test_rejects_free_text_and_malformed(text: str) -> None:
    with pytest.raises(NoJSONObjectError):
        parse_evaluation(text)


def test_rejects_missing_key() -> None:
    data = dict(VALID)
    del data["physical_realism"]
    with pytest.raises(MissingKeysError, match="physical_realism"):
        parse_evaluation(json.dumps(data))


def test_rejects_unexpected_key() -> None:
    with pytest.raises(UnexpectedKeysError, match="confidence"):
        parse_evaluation(_payload(confidence=0.3))


def test_rejects_non_string_input() -> None:
    with pytest.raises(InvalidResponseError):
        parse_evaluation(b"{}")  # type: ignore[arg-type]


def test_injection_payload_is_rejected_and_never_executed() -> None:
    if SENTINEL.exists():
        SENTINEL.unlink()
    attacks = [
        "__import__('os').system('touch /tmp/pwned')",
        "{\"dimensional_consistency\": __import__('os').system('touch /tmp/pwned'), "
        '"simplicity": 1, "physical_realism": 1, "feedback": "x"}',
        _payload(simplicity="__import__('os').system('touch /tmp/pwned')"),
    ]
    for text in attacks:
        with pytest.raises(InvalidResponseError):
            parse_evaluation(text)
    assert not SENTINEL.exists()


def test_injection_inside_feedback_is_inert_text() -> None:
    attack = "__import__('os').system('touch /tmp/pwned')"
    ev = parse_evaluation(_payload(feedback=attack))
    assert ev.feedback == attack
    assert not SENTINEL.exists()
