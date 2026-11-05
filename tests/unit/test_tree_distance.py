"""T:tree-distance e T:tree-distance-neg - distância estrutural normalizada (GAP-10)."""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.config.schemas import SymbolicConfig
from physics_llm_sr.symbolic.expression import (
    Const,
    Expr,
    ExpressionError,
    Op,
    Var,
    parse_infix,
    random_expression,
)
from physics_llm_sr.symbolic.operators import UnknownOperatorError
from physics_llm_sr.symbolic.tree_distance import (
    TreeDistanceParams,
    tree_distance,
    tree_distance_terms,
)

NAMES = ("x", "y", "a", "t", "h")


def _p(text: str) -> Expr:
    return parse_infix(text, variables=NAMES)


# Exemplos TD-01..TD-18 de references/scientific_assumptions.md (GAP-10).
# (id, T1, T2, alpha, descend, canonicalize, S, N)
GAP10_EXAMPLES = [
    ("TD-01", "x", "x", 1.0, True, False, 0.0, 1),
    ("TD-02", "x", "y", 1.0, True, False, 1.0, 1),
    ("TD-03", "2.0", "2.5", 1.0, True, False, 0.5, 1),
    ("TD-04", "2.0", "5.0", 1.0, True, False, 1.0, 1),
    ("TD-05", "2.0", "5.0", 0.1, True, False, 0.3, 1),
    ("TD-06", "x + y", "y + x", 1.0, True, False, 0.0, 3),
    ("TD-07", "x - y", "y - x", 1.0, True, False, 2.0, 3),
    ("TD-08", "x + 0", "x", 1.0, True, False, 1.0, 1),
    ("TD-09", "x * y", "x + y", 1.0, True, False, 1.0, 3),
    ("TD-10", "sin(x)", "cos(x)", 1.0, True, False, 1.0, 2),
    ("TD-11", "sin(x)", "x * y", 1.0, True, False, 1.0, 1),
    ("TD-12", "x * y", "x + y", 1.0, False, False, 1.0, 1),
    ("TD-13", "x * (y + 2)", "(y + 2.5) * x", 1.0, True, False, 0.5, 5),
    ("TD-14", "x + x", "x + y", 1.0, True, False, 1.0, 3),
    ("TD-15", "exp(-0.5 * (a * t))", "exp(-0.4 * (t * a))", 1.0, True, False, 0.1, 6),
    ("TD-16", "pow(19.62 * h, 0.5)", "sqrt(19.6 * h)", 1.0, True, True, 0.02, 5),
    ("TD-17", "pow(19.62 * h, 0.5)", "exp(0.5 * log(19.62 * h))", 1.0, True, True, 0.0, 5),
    ("TD-18", "pow(19.62 * h, 0.5)", "9.81 * t", 1.0, True, True, 3.0, 3),
]


@pytest.mark.parametrize(
    ("case", "t1", "t2", "alpha", "descend", "canon", "s_expected", "n_expected"),
    GAP10_EXAMPLES,
    ids=[e[0] for e in GAP10_EXAMPLES],
)
def test_gap10_worked_examples(
    case: str,
    t1: str,
    t2: str,
    alpha: float,
    descend: bool,
    canon: bool,
    s_expected: float,
    n_expected: int,
) -> None:
    params = TreeDistanceParams(
        alpha=alpha, descend_on_operator_mismatch=descend, canonicalize=canon
    )
    a, b = _p(t1), _p(t2)
    s, n = tree_distance_terms(a, b, params)
    assert s == pytest.approx(s_expected, abs=1e-12)
    assert n == n_expected
    assert tree_distance(a, b, params) == pytest.approx(s_expected / n_expected, abs=1e-12)
    assert tree_distance(b, a, params) == pytest.approx(s_expected / n_expected, abs=1e-12)


def test_identical_is_zero() -> None:
    gt = parse_infix(
        "1.0*exp(-damping*t/2)*cos(6.283185307179586*length-6.283185307179586*t)",
        variables=("damping", "t", "length"),
    )
    assert tree_distance(gt, gt) == 0.0


def test_commutative_swap_is_zero() -> None:
    assert tree_distance(_p("x * (y + t)"), _p("(t + y) * x")) == 0.0


@pytest.mark.parametrize(("t1", "t2"), [("x - y", "y - x"), ("x / y", "y / x")])
def test_non_commutative_swap_is_positive(t1: str, t2: str) -> None:
    assert tree_distance(_p(t1), _p(t2)) > 0.0


def test_structural_not_numerical_equivalence() -> None:
    assert tree_distance(_p("x + 0"), _p("x")) > 0.0
    assert tree_distance(_p("x * 1"), _p("x")) > 0.0


def test_constant_penalty_is_bounded_and_monotone_in_alpha() -> None:
    a, b = _p("2.0 * x"), _p("1000.0 * x")
    # S = min(1, alpha*|dv|) <= 1 em uma de 3 comparações
    assert tree_distance(a, b, TreeDistanceParams(alpha=1.0)) == pytest.approx(1 / 3)
    small = tree_distance(_p("2.0 * x"), _p("2.1 * x"), TreeDistanceParams(alpha=0.1))
    big = tree_distance(_p("2.0 * x"), _p("2.1 * x"), TreeDistanceParams(alpha=1.0))
    assert 0.0 < small < big < 1 / 3
    assert tree_distance(a, b, TreeDistanceParams(alpha=0.0)) == 0.0


def test_default_params_come_from_config() -> None:
    cfg = SymbolicConfig()
    params = TreeDistanceParams.from_config(cfg)
    assert params.alpha == cfg.tree_distance_constant_penalty
    assert params == TreeDistanceParams()
    assert params.canonicalize is True
    assert params.descend_on_operator_mismatch is True


def test_ground_truth_reference_forms_match_after_canonicalization() -> None:
    feats = ("mass", "length", "height", "damping", "t")
    for gt, ref in [
        ("sqrt(2*9.81*height)", "pow(19.62*height, 0.5)"),
        ("length*cos(sqrt(10/mass)*t)", "length*cos(pow(10/mass, 0.5)*t)"),
    ]:
        assert tree_distance(parse_infix(gt, feats), parse_infix(ref, feats)) == 0.0


def test_property_random_trees_in_unit_interval_symmetric_and_reflexive() -> None:
    rng = np.random.default_rng(2024)
    for i in range(500):
        params = TreeDistanceParams(
            alpha=float(rng.uniform(0.0, 1.0)),
            descend_on_operator_mismatch=bool(i % 2),
            canonicalize=bool(i % 3),
        )
        a = random_expression(rng, NAMES, max_depth=5)
        b = random_expression(rng, NAMES, max_depth=5)
        d = tree_distance(a, b, params)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(tree_distance(b, a, params), abs=1e-12)
        assert tree_distance(a, a, params) == 0.0


# --------------------------------------------------------------------- T:tree-distance-neg


class _Alien(Expr):
    __slots__ = ()


def _op_bypassing_validation(name: str, children: tuple[Expr, ...]) -> Op:
    node = object.__new__(Op)
    object.__setattr__(node, "name", name)
    object.__setattr__(node, "children", children)
    return node


def test_unknown_node_type_rejected() -> None:
    with pytest.raises(ExpressionError):
        tree_distance(_Alien(), Var("x"))
    with pytest.raises(ExpressionError):
        tree_distance(Var("x"), "x")  # type: ignore[arg-type]


def test_unknown_operator_rejected() -> None:
    alien = _op_bypassing_validation("tan", (Var("x"),))
    with pytest.raises(UnknownOperatorError):
        tree_distance(alien, Op("sin", (Var("x"),)))
    with pytest.raises(UnknownOperatorError):
        tree_distance(Const(1.0), alien)


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        TreeDistanceParams(alpha=1.5)
    with pytest.raises(ValueError):
        TreeDistanceParams(alpha=-0.1)
