"""T:ast - AST imutável, avaliação vetorizada e conversões com roundtrip."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from physics_llm_sr.symbolic.expression import (
    Const,
    Expr,
    ExpressionError,
    Op,
    Var,
    depth,
    evaluate,
    from_prefix,
    from_sympy,
    parse_infix,
    random_expression,
    size,
    to_infix,
    to_prefix,
    to_sympy,
    variables,
)
from physics_llm_sr.symbolic.operators import UnknownOperatorError

FEATURES = ("mass", "length", "height", "damping", "t")

GROUND_TRUTHS = [
    "sqrt(2*9.81*height)",
    "length*cos(sqrt(10/mass)*t)",
    "1.0*exp(-damping*t/2)*cos(6.283185307179586*length-6.283185307179586*t)",
]


@pytest.fixture(scope="module")
def xd() -> np.ndarray:
    rng = np.random.default_rng(12345)
    n = 200
    return np.column_stack(
        [
            rng.uniform(0.1, 10.0, n),  # mass
            rng.uniform(0.01, 0.5, n),  # length
            rng.uniform(1.0, 100.0, n),  # height
            rng.uniform(0.0, 1.0, n),  # damping
            rng.uniform(0.0, 5.0, n),  # t
        ]
    )


def _reference(gt_index: int, xd: np.ndarray) -> np.ndarray:
    mass, length, height, damping, t = xd.T
    out: np.ndarray
    if gt_index == 0:
        out = np.sqrt(2 * 9.81 * height)
    elif gt_index == 1:
        out = length * np.cos(np.sqrt(10 / mass) * t)
    else:
        w = 6.283185307179586
        out = 1.0 * np.exp(-damping * t / 2) * np.cos(w * length - w * t)
    return out


@pytest.mark.parametrize("idx", range(len(GROUND_TRUTHS)))
def test_parse_infix_of_ground_truth_matches_numpy_reference(idx: int, xd: np.ndarray) -> None:
    expr = parse_infix(GROUND_TRUTHS[idx], variables=FEATURES)
    np.testing.assert_allclose(evaluate(expr, xd, FEATURES), _reference(idx, xd), rtol=1e-12)


@pytest.mark.parametrize("gt", GROUND_TRUTHS)
def test_roundtrip_prefix_is_structurally_exact(gt: str, xd: np.ndarray) -> None:
    expr = parse_infix(gt, variables=FEATURES)
    back = from_prefix(to_prefix(expr))
    assert back == expr
    np.testing.assert_array_equal(evaluate(back, xd, FEATURES), evaluate(expr, xd, FEATURES))


@pytest.mark.parametrize("gt", GROUND_TRUTHS)
def test_roundtrip_infix_is_structurally_exact(gt: str, xd: np.ndarray) -> None:
    expr = parse_infix(gt, variables=FEATURES)
    back = parse_infix(to_infix(expr), variables=FEATURES)
    assert back == expr
    np.testing.assert_array_equal(evaluate(back, xd, FEATURES), evaluate(expr, xd, FEATURES))


@pytest.mark.parametrize("gt", GROUND_TRUTHS)
def test_roundtrip_sympy_preserves_evaluation(gt: str, xd: np.ndarray) -> None:
    expr = parse_infix(gt, variables=FEATURES)
    back = from_sympy(to_sympy(expr), variables=FEATURES)
    np.testing.assert_allclose(
        evaluate(back, xd, FEATURES), evaluate(expr, xd, FEATURES), rtol=1e-9, atol=1e-12
    )


def test_from_sympy_handles_canonical_sympy_forms(xd: np.ndarray) -> None:
    import sympy

    mass, t, length = sympy.symbols("mass t length")
    sym = 3 * length / mass - sympy.sqrt(t) + sympy.log(mass) ** 2 - sympy.Rational(1, 2)
    expr = from_sympy(sym, variables=FEATURES)
    m, ln, tt = xd[:, 0], xd[:, 1], xd[:, 4]
    ref = 3 * ln / m - np.sqrt(tt) + np.log(m) ** 2 - 0.5
    np.testing.assert_allclose(evaluate(expr, xd, FEATURES), ref, rtol=1e-12)


def test_from_sympy_rejects_unknown_function() -> None:
    import sympy

    x = sympy.Symbol("mass")
    with pytest.raises(UnknownOperatorError):
        from_sympy(sympy.tan(x), variables=FEATURES)


def test_mixed_roundtrip_chain_on_random_trees(xd: np.ndarray) -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        expr = random_expression(rng, FEATURES, max_depth=5)
        assert from_prefix(to_prefix(expr)) == expr
        assert parse_infix(to_infix(expr), variables=FEATURES) == expr


def test_size_and_depth() -> None:
    expr = parse_infix("sqrt(2*9.81*height)", variables=FEATURES)
    # sqrt(mul(mul(2, 9.81), height)) -> 6 nós; folha tem profundidade 0
    assert size(expr) == 6
    assert depth(expr) == 3
    assert size(Var("t")) == 1
    assert depth(Const(1.0)) == 0
    assert variables(expr) == frozenset({"height"})


def test_nodes_are_immutable_and_hashable() -> None:
    expr = Op("add", (Var("t"), Const(1.0)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        expr.name = "sub"  # type: ignore[misc]
    assert hash(expr) == hash(Op("add", (Var("t"), Const(1.0))))
    assert isinstance(expr, Expr)


def test_op_validates_registry_and_arity() -> None:
    with pytest.raises(UnknownOperatorError):
        Op("tan", (Var("t"),))
    with pytest.raises(ExpressionError):
        Op("add", (Var("t"),))
    with pytest.raises(ExpressionError):
        Const(float("nan"))
    with pytest.raises(ExpressionError):
        Var("exp")  # colide com nome de operador


def test_evaluate_shape_and_unknown_variable(xd: np.ndarray) -> None:
    assert evaluate(Const(2.5), xd, FEATURES).shape == (xd.shape[0],)
    with pytest.raises(ExpressionError):
        evaluate(Var("velocity"), xd, FEATURES)
    with pytest.raises(ExpressionError):
        evaluate(Var("t"), xd[:, :2], FEATURES)


def test_evaluate_is_finite_on_protected_cases() -> None:
    x_zero = np.zeros((3, 1))
    expr = parse_infix("log(x) + 1/x + exp(1000*(x+1)) + (x-1)**0.5", variables=("x",))
    out = evaluate(expr, x_zero, ("x",))
    assert np.all(np.isfinite(out))


def test_from_prefix_rejects_malformed_sequences() -> None:
    with pytest.raises(ExpressionError):
        from_prefix(["add", "t"])
    with pytest.raises(ExpressionError):
        from_prefix(["t", "t"])
    with pytest.raises(ExpressionError):
        from_prefix([])
    with pytest.raises(ExpressionError):
        from_prefix(["add", "t", "x"], variables=("t",))
