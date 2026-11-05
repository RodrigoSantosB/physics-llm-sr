"""Adaptador PySR sem Julia: importar não inicializa o Julia; conversão e callback em Python."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.interfaces import EquationEvaluation, EvaluationFailedError
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import CombinedObjective
from physics_llm_sr.sr.pysr_adapter import (
    ObjectiveScale,
    PenaltyCallback,
    UnsupportedPysrError,
    julia_text_to_expr,
    pysr_operators,
)
from physics_llm_sr.symbolic.expression import parse_infix, size, to_infix

FEATURES = ("mass", "height")
CTX = PhysicalContext("s", {"mass": "m", "height": "h"}, "exp", "height", {"mass": "kg"})
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]])
Y = np.array([1.0, 2.0, 3.0])
GOOD = EquationEvaluation(0.9, 0.9, 0.9, "g")
BAD = EquationEvaluation(0.1, 0.1, 0.1, "b")


def test_import_does_not_load_julia() -> None:
    code = (
        "import sys\n"
        "import physics_llm_sr.sr.pysr_adapter\n"
        "bad = [m for m in ('pysr', 'juliacall') if m in sys.modules]\n"
        "print(bad); sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_julia_text_to_expr_maps_variables_and_power() -> None:
    expr = julia_text_to_expr("cos(x2) * (x1 ^ 0.5) - neg(-1.5)", FEATURES)
    assert to_infix(expr) == to_infix(parse_infix("cos(height) * mass ** 0.5 - neg(-1.5)"))
    assert size(julia_text_to_expr("x1 * -0.5", FEATURES)) == 3  # constante negativa = 1 nó


def test_pysr_operators_follow_gap06_names() -> None:
    binary, unary = pysr_operators(["add", "sub", "mul", "div", "pow", "neg", "exp", "cos"])
    assert binary == ["+", "-", "*", "/", "^"]
    assert unary == ["neg", "exp", "cos"]


def _callback(evaluator: FakeEquationEvaluator, **kw: object) -> PenaltyCallback:
    obj = CombinedObjective((1 / 3, 1 / 3, 1 / 3), evaluator, CTX, FEATURES, **kw)  # type: ignore[arg-type]
    return PenaltyCallback(obj, X, Y, FEATURES, ObjectiveScale.from_objective(obj, Y))


def test_callback_scores_simplified_form_with_memo() -> None:
    fake = FakeEquationEvaluator.from_function(
        lambda e, c, v: GOOD if "cos" in to_infix(e) else BAD
    )
    cb = _callback(fake)
    assert cb("cos(0.3) * x1") == pytest.approx(0.9)  # cos(constante) é dobrado: sem cos
    assert cb("cos(x1) * 2.0001") == pytest.approx(0.1)
    assert cb("cos(x1) * 2.0002") == pytest.approx(0.1)  # mesma chave a 3 algarismos
    assert fake.calls == 2 and cb.memo_hits == 1
    assert cb("x1 + Inf") == math.inf and cb.parse_errors == 1


def test_callback_failure_returns_inf_and_stops_calling_llm() -> None:
    fake = FakeEquationEvaluator.from_table({})  # sem default: EvaluationFailedError
    cb = _callback(fake)
    assert cb("x1") == math.inf
    assert isinstance(cb.failure, EvaluationFailedError)
    assert cb("x2") == math.inf
    assert fake.calls == 1  # após a falha o avaliador não é mais chamado


def test_objective_scale_reads_combined_objective() -> None:
    obj = CombinedObjective(
        (0.6, 0.1, 0.3), FakeEquationEvaluator.from_function(lambda e, c, v: GOOD), CTX,
        FEATURES, normalization="normalized", max_nodes=31, error_metric="huber",
        huber_delta=2.0,
    )  # fmt: skip
    scale = ObjectiveScale.from_objective(obj, Y)
    assert scale.weights == (0.6, 0.1, 0.3) and scale.huber and scale.huber_delta == 2.0
    assert scale.normalized and scale.max_nodes == 31.0 and scale.var_y == float(np.var(Y))
    with pytest.raises(UnsupportedPysrError):
        ObjectiveScale.from_objective(object(), Y)  # type: ignore[arg-type]
