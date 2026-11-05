"""Canonização: chave de cache determinística, simplificação separada e forma estrutural."""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.symbolic.canonicalization import (
    DEFAULT_SIGNIFICANT_DIGITS,
    canonical_key,
    canonicalize,
    round_significant,
    simplify,
    structural_normal_form,
)
from physics_llm_sr.symbolic.expression import (
    Const,
    Var,
    evaluate,
    parse_infix,
    random_expression,
)

FEATURES = ("mass", "length", "height", "damping", "t")


def _p(text: str):  # type: ignore[no-untyped-def]
    return parse_infix(text, variables=FEATURES)


def test_round_significant() -> None:
    assert round_significant(9.80665, 3) == 9.81
    assert round_significant(-0.000123456, 2) == -0.00012
    assert round_significant(0.0, 4) == 0.0
    assert round_significant(-0.0, 4) == 0.0
    assert round_significant(123456.0, 2) == 120000.0


def test_commutative_children_are_sorted_deterministically() -> None:
    a = _p("(t + mass) * cos(length)")
    b = _p("cos(length) * (mass + t)")
    assert canonicalize(a) == canonicalize(b)
    assert canonical_key(a) == canonical_key(b)


def test_non_commutative_order_is_preserved() -> None:
    assert canonical_key(_p("t - mass")) != canonical_key(_p("mass - t"))
    assert canonical_key(_p("t / mass")) != canonical_key(_p("mass / t"))


def test_constants_rounded_to_significant_digits() -> None:
    assert canonical_key(_p("9.8100001 * t")) == canonical_key(_p("9.81 * t"))
    assert canonical_key(_p("9.81 * t"), significant_digits=2) == canonical_key(_p("9.8 * t"), 2)
    assert DEFAULT_SIGNIFICANT_DIGITS >= 6


def test_canonical_key_does_not_simplify() -> None:
    assert canonical_key(_p("t + 0")) != canonical_key(_p("t"))
    assert canonical_key(_p("t * 1")) != canonical_key(_p("t"))
    assert canonical_key(_p("2 * 3")) != canonical_key(_p("6"))


def test_canonical_key_is_stable_string() -> None:
    key = canonical_key(_p("t + mass"))
    assert isinstance(key, str)
    assert key == canonical_key(_p("mass + t"))
    assert key == "add mass t"


def test_simplify_trivial_identities() -> None:
    assert simplify(_p("t + 0")) == Var("t")
    assert simplify(_p("0 + t")) == Var("t")
    assert simplify(_p("t - 0")) == Var("t")
    assert simplify(_p("t * 1")) == Var("t")
    assert simplify(_p("1 * t")) == Var("t")
    assert simplify(_p("t / 1")) == Var("t")
    assert simplify(_p("t ** 1")) == _p("t ** 1")  # pow recorta em power_clip
    assert simplify(_p("-(-t)")) == Var("t")
    assert simplify(_p("t * 0")) == Const(0.0)
    assert simplify(_p("2 * 3 + t")) == _p("6 + t")


def test_simplify_preserves_protected_evaluation_on_random_trees() -> None:
    rng = np.random.default_rng(3)
    xd = rng.uniform(-3.0, 3.0, size=(64, len(FEATURES)))
    for _ in range(200):
        expr = random_expression(rng, FEATURES, max_depth=4)
        np.testing.assert_allclose(
            evaluate(simplify(expr), xd, FEATURES), evaluate(expr, xd, FEATURES), rtol=1e-9
        )


def test_canonicalize_preserves_evaluation_up_to_rounding() -> None:
    rng = np.random.default_rng(4)
    xd = rng.uniform(0.1, 3.0, size=(32, len(FEATURES)))
    for _ in range(100):
        expr = random_expression(rng, FEATURES, max_depth=3, operators=["add", "mul", "sub"])
        np.testing.assert_allclose(
            evaluate(canonicalize(expr, 15), xd, FEATURES),
            evaluate(expr, xd, FEATURES),
            rtol=1e-9,
            atol=1e-9,
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sqrt(2*9.81*height)", "pow(19.62*height, 0.5)"),
        ("length*cos(sqrt(10/mass)*t)", "length*cos(pow(10/mass, 0.5)*t)"),
        ("exp(0.5*log(19.62*height))", "pow(19.62*height, 0.5)"),
        ("exp(log(19.62*height)*0.5)", "pow(19.62*height, 0.5)"),
        ("t + 0", "t + 0"),  # sem simplificação algébrica
        ("neg(2.0) * t", "(-2.0) * t"),
    ],
)
def test_structural_normal_form_gap10(text: str, expected: str) -> None:
    assert structural_normal_form(_p(text)) == _p(expected)
