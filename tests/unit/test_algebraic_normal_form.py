"""Forma normal algébrica (métrica secundária ``tree_distance_algebraic``).

Propriedades: preserva a função (fora dos pontos em que a semântica protegida ativa), é
idempotente e determinística. Casos reais das rodadas O14/O15 (queda livre e onda amortecida).
"""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.symbolic.canonicalization import (
    ALGEBRAIC_MAX_ITERATIONS,
    algebraic_normal_form,
    structural_normal_form,
)
from physics_llm_sr.symbolic.expression import (
    Const,
    Expr,
    Op,
    Var,
    parse_infix,
    random_expression,
    size,
)
from physics_llm_sr.symbolic.operators import DEFAULT_PROTECTION, OPERATORS
from physics_llm_sr.symbolic.tree_distance import tree_distance

FEATURES = ("mass", "length", "height", "damping", "t")

PROPERTY_RTOL = 1e-6
"""Tolerância relativa do teste de preservação, sobre a maior magnitude intermediária no ponto."""

N_RANDOM_TREES = 600
N_POINTS = 64


def _p(text: str) -> Expr:
    return parse_infix(text, variables=FEATURES)


def anf(text: str) -> Expr:
    return algebraic_normal_form(_p(text))


# ----------------------------------------------------------------- avaliação instrumentada


def _raw_eval(expr: Expr, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Avalia com a semântica protegida e devolve (valor, proteção_ativa, magnitude_max).

    ``proteção_ativa`` marca pontos em que algum operador protegido saiu da semântica
    matemática (divisão com |den| < eps, log com |x| < eps, exp recortado, potência recortada
    ou não finita, resultado intermediário não finito).
    """
    p = DEFAULT_PROTECTION
    index = {n: i for i, n in enumerate(FEATURES)}
    n = x.shape[0]

    def rec(node: Expr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(node, Const):
            v = np.full(n, node.value)
            return v, np.zeros(n, bool), np.abs(v)
        if isinstance(node, Var):
            v = x[:, index[node.name]].copy()
            return v, np.zeros(n, bool), np.abs(v)
        assert isinstance(node, Op)
        parts = [rec(c) for c in node.children]
        args = [q[0] for q in parts]
        active = np.logical_or.reduce([q[1] for q in parts])
        mag = np.maximum.reduce([q[2] for q in parts])
        with np.errstate(all="ignore"):
            raw = OPERATORS[node.name].func(p, *args)
            if node.name == "div":
                active |= np.abs(args[1]) < p.division_epsilon
            elif node.name == "log":
                active |= np.abs(args[0]) < p.log_epsilon
            elif node.name == "exp":
                active |= np.abs(args[0]) > p.exp_clip
            elif node.name == "pow":
                b = args[1]
                integer = b == np.round(b)
                exact = np.power(np.where(integer, args[0], np.abs(args[0])), b)
                active |= ~np.isfinite(exact) | (np.abs(exact) > p.power_clip)
            active |= ~np.isfinite(raw)
        out = np.nan_to_num(raw, nan=0.0, posinf=p.power_clip, neginf=-p.power_clip)
        return out, active, np.maximum(mag, np.abs(out))

    return rec(expr)


def _domain(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-3.0, 3.0, size=(N_POINTS, len(FEATURES)))


# --------------------------------------------------------------------------- propriedades


def test_preserves_function_on_random_trees() -> None:
    rng = np.random.default_rng(20261115)
    checked = 0
    for _ in range(N_RANDOM_TREES):
        expr = random_expression(rng, FEATURES, max_depth=4)
        x = _domain(rng)
        normal = algebraic_normal_form(expr)
        y0, a0, m0 = _raw_eval(expr, x)
        y1, a1, m1 = _raw_eval(normal, x)
        ok = ~(a0 | a1)
        scale = np.maximum(np.maximum(m0, m1), 1.0)
        err = np.abs(y0 - y1)
        bad = ok & (err > PROPERTY_RTOL * scale)
        assert not bad.any(), (expr, normal, x[bad][0], y0[bad][0], y1[bad][0])
        checked += int(ok.sum())
    assert checked > N_RANDOM_TREES * N_POINTS // 4


def test_idempotent_and_deterministic_on_random_trees() -> None:
    rng = np.random.default_rng(7)
    for _ in range(N_RANDOM_TREES):
        expr = random_expression(rng, FEATURES, max_depth=4)
        once = algebraic_normal_form(expr)
        assert algebraic_normal_form(once) == once, (expr, once)
        assert algebraic_normal_form(expr) == once


def test_iteration_limit_is_positive() -> None:
    assert ALGEBRAIC_MAX_ITERATIONS >= 2


def test_size_never_explodes() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        expr = random_expression(rng, FEATURES, max_depth=5)
        assert size(algebraic_normal_form(expr)) <= 4 * size(expr) + 8


def test_structural_normal_form_is_unchanged() -> None:
    # a forma estrutural (GAP-10) NÃO simplifica: x + 0 continua diferente de x
    assert structural_normal_form(_p("t + 0")) != structural_normal_form(_p("t"))
    assert structural_normal_form(_p("t / 2.0")) == Op("div", (Var("t"), Const(2.0)))


# --------------------------------------------------------------------------- identidades


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("height / t + height / t", "2 * height / t"),
        ("height / t + height / t", "2 * (height / t)"),
        ("t - t", "0"),
        ("t * 1", "t"),
        ("1.0 * t", "t"),
        ("t + 0", "t"),
        ("t / 4.0", "0.25 * t"),
        ("-(-t)", "t"),
        ("2 * (3 * t)", "6 * t"),
        ("-t * 3", "(-3) * t"),
        ("cos(-t)", "cos(t)"),
        ("cos(length - t)", "cos(t - length)"),
        ("sin(-t)", "-sin(t)"),
        ("t / exp(damping)", "t * exp(-damping)"),
        ("exp(t) * exp(damping)", "exp(t + damping)"),
        ("2 * (t - length)", "2 * t - 2 * length"),
        ("cos(6.0 * (t - length))", "cos(6.0 * length - 6.0 * t)"),
        ("t * t", "pow(t, 2)"),
        ("mass * t / mass", "t"),
        ("sqrt(4 * height)", "2 * sqrt(height)"),
        ("1 / sqrt(mass)", "pow(mass, -0.5)"),
        ("exp(0.5 * log(height))", "sqrt(height)"),
    ],
)
def test_identities_reach_same_normal_form(left: str, right: str) -> None:
    assert anf(left) == anf(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("sin(t - length)", "sin(length - t)"),  # sin é ímpar, não par
        ("t / mass", "mass / t"),
        ("exp(3 * log(t))", "pow(t, 3)"),  # |t|³ ≠ t³ para t < 0
        ("t * (length + mass)", "t * length + t * mass"),  # só constantes distribuem
    ],
)
def test_non_identities_or_out_of_scope_stay_different(left: str, right: str) -> None:
    assert anf(left) != anf(right)


# --------------------------------------------------------------------------- casos reais


def _alg_distance(found: str, target: str) -> float:
    return tree_distance(anf(found), anf(target))


def test_real_free_fall_cases() -> None:
    assert _alg_distance("height / t + height / t", "2*height/t") == 0.0
    assert _alg_distance("t / 0.1019367991845056", "9.81*t") <= 1e-12
    assert _alg_distance("(height + height) / t", "2*height/t") == 0.0


def test_real_damped_wave_deap_case_is_small() -> None:
    ref = "1.0*exp(-damping*t/2)*cos(6.283185307179586*length-6.283185307179586*t)"
    found = "cos(6.299*(t - length)) / exp(0.4965*damping*t)"
    d = _alg_distance(found, ref)
    # só as constantes diferem: 2*|6.299-2π| + |0.4965-0.5| sobre 15 comparações
    expected = (2 * abs(6.299 - 6.283185307179586) + abs(0.4965 - 0.5)) / 15
    assert d == pytest.approx(expected, rel=1e-9)
    assert tree_distance(_p(found), _p(ref)) > 0.5  # a primária continua alta


def test_real_damped_wave_raw_deap_tree() -> None:
    ref = "1.0*exp(-damping*t/2)*cos(6.283185307179586*length-6.283185307179586*t)"
    raw = (
        "cos((-4.2179694348501195) * (log((-4.452222398260974)) * (t - length)))"
        " / exp(t * damping * 0.7046152672499026 * 0.7046152672499026)"
    )
    assert _alg_distance(raw, ref) < 0.01


def test_wrong_expression_stays_far() -> None:
    ref = "1.0*exp(-damping*t/2)*cos(6.283185307179586*length-6.283185307179586*t)"
    assert _alg_distance("cos(length)", ref) > 0.5
