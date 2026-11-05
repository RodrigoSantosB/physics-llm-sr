"""Conversão ``_Program -> AST``: avaliação nativa do gplearn == ``symbolic.evaluate``.

Tolerância: ``rtol = atol = 1e-9``. Programas cujo resultado nativo tem valor não finito
ou acima de ``power_clip`` são excluídos (a AST satura em ``±power_clip``; o gplearn,
não), e a exclusão é limitada para o teste não ficar vazio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gplearn.functions import _Function

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.optimization.objective import build_objective
from physics_llm_sr.sr.gplearn_adapter import (
    GplearnRegressor,
    UnsupportedGplearnError,
    build_estimator,
    make_function_set,
    program_to_expr,
)
from physics_llm_sr.symbolic.expression import Const, Op, Var, evaluate
from physics_llm_sr.symbolic.operators import DEFAULT_PROTECTION, ProtectionParams

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
RTOL = ATOL = 1e-9
#: População grande o suficiente para cobrir todos os operadores.
POPULATION = 300
GENERATIONS = 2
#: Fração mínima de programas comparáveis (não saturados).
MIN_COMPARABLE_FRACTION = 0.8


def test_native_and_ast_evaluation_coincide() -> None:
    cfg = load_config(SMOKE)
    cfg = cfg.model_copy(
        update={"sr": cfg.sr.model_copy(update={"population": POPULATION, "generations": 1})}
    )
    data = generate_dataset(cfg, "free_fall")
    est = build_estimator(cfg, data.feature_names, DEFAULT_PROTECTION, seed=7)
    est.set_params(generations=GENERATIONS)
    est.fit(data.X, data.y_observed)
    programs = est._programs[-1]
    used: set[str] = set()
    compared = 0
    for program in programs:
        native = program.execute(data.X)
        if not np.all(np.isfinite(native)) or np.max(np.abs(native)) >= (
            DEFAULT_PROTECTION.power_clip
        ):
            continue
        expr = program_to_expr(program.program, data.feature_names)
        ours = evaluate(expr, data.X, data.feature_names)
        np.testing.assert_allclose(ours, native, rtol=RTOL, atol=ATOL)
        used.update(n.name for n in program.program if isinstance(n, _Function))
        compared += 1
    assert compared >= MIN_COMPARABLE_FRACTION * len(programs)
    assert set(cfg.sr.operators["gplearn"]) <= used  # todos os operadores exercitados


def test_program_to_expr_maps_nodes() -> None:
    fs = {f if isinstance(f, str) else f.name: f for f in make_function_set(["add", "exp"])}
    from gplearn.functions import _function_map

    program = [_function_map["add"], fs["exp"], 1, np.float64(-0.5)]
    expr = program_to_expr(program, ("a", "b"))
    assert expr == Op("add", (Op("exp", (Var("b"),)), Const(-0.5)))


def test_unsupported_operator_and_protection_rejected() -> None:
    with pytest.raises(UnsupportedGplearnError):
        make_function_set(["pow"])
    other = ProtectionParams(0.01, 1.0, 0.001, 100.0, 1e6)
    with pytest.raises(UnsupportedGplearnError):
        make_function_set(["div"], other)


def test_predict_matches_native_gplearn_predict() -> None:
    cfg = load_config(SMOKE)
    data = generate_dataset(cfg, "free_fall")
    objective = build_objective(cfg, None, None, "gplearn", True, data.feature_names)
    reg = GplearnRegressor()
    reg.fit(data, objective, cfg)
    assert reg.estimator is not None
    np.testing.assert_allclose(
        reg.predict(data.X), reg.estimator.predict(data.X), rtol=RTOL, atol=ATOL
    )
