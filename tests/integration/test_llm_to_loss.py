"""Integração LLM -> perda: os escores mudam o ranking só quando w3 > 0 (base da onda 5)."""

from __future__ import annotations

import numpy as np

from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import CombinedObjective
from physics_llm_sr.symbolic.expression import Expr, parse_infix

FEATURES = ("x",)
CTX = PhysicalContext("s", {"x": "x"}, "exp", "x", {"x": "m"})
X = np.array([[1.0], [2.0], [3.0]])
Y = np.array([2.0, 3.0, 4.0])


def _setup() -> tuple[list[Expr], FakeEquationEvaluator]:
    # Duas expressões com MESMO erro (0) e MESMO tamanho (3 nós), escores opostos.
    a = parse_infix("x + 1", FEATURES)  # erro 0, 3 nós
    b = parse_infix("x - -1", FEATURES)  # erro 0, 3 nós (Const -1 literal), chave canônica distinta
    fake = FakeEquationEvaluator.from_table(
        {
            a: EquationEvaluation(1.0, 1.0, 1.0, "bom"),
            b: EquationEvaluation(0.0, 0.0, 0.0, "ruim"),
        }
    )
    return [a, b], fake


def _obj(w: tuple[float, float, float], fake: FakeEquationEvaluator) -> CombinedObjective:
    return CombinedObjective(w, fake, CTX, FEATURES)


def test_scores_change_ranking_only_when_w3_positive() -> None:
    (a, b), fake = _setup()
    out = _obj((0.4, 0.2, 0.4), fake).evaluate_population([b, a], X, Y)
    assert out[0].e == out[1].e == 0.0
    assert out[0].s == out[1].s
    assert out[1].total < out[0].total  # a (escores altos) vence b

    fake0 = FakeEquationEvaluator.from_table({})
    tie = _obj((0.5, 0.5, 0.0), fake0).evaluate_population([b, a], X, Y)
    assert tie[0].total == tie[1].total
    assert fake0.calls == 0
