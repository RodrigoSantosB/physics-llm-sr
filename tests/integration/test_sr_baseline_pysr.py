"""C:baseline-sr (T:baseline-sr) para o adaptador PySR: busca sem LLM (w3 = 0)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import build_objective
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng
from physics_llm_sr.sr.interfaces import SymbolicRegressor
from physics_llm_sr.sr.pysr_adapter import PREDICT_REL_TOL, PySRRegressor
from physics_llm_sr.symbolic.expression import Expr, evaluate, to_infix

pytestmark = pytest.mark.julia

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"


def _train(cfg: ExperimentConfig) -> Dataset:
    data = generate_dataset(cfg, "free_fall")
    return split_by_simulation(data, cfg.data.split, derive_rng(cfg.seeds.master, "split"))["train"]


def test_baseline_pysr_trains_without_llm() -> None:
    cfg = load_config(SMOKE)
    train = _train(cfg)
    fake = FakeEquationEvaluator.from_table({}, default=EquationEvaluation(1, 1, 1, "x"))
    ctx = get_scenario("free_fall", cfg.physics).physical_context()
    objective = build_objective(cfg, fake, ctx, "pysr", True, train.feature_names)

    reg = PySRRegressor()
    assert isinstance(reg, SymbolicRegressor)
    result = reg.fit(train, objective, cfg)

    assert fake.calls == 0  # baseline: avaliador nunca chamado
    assert reg.callback is None  # w3 = 0: callback não registrado
    assert result.metadata.extra is not None
    assert result.metadata.extra["julia_loss_llm_calls"] == 0
    assert isinstance(result.best_expression, Expr)
    assert to_infix(result.best_expression)
    pred = reg.predict(train.X)
    assert pred.shape == train.y_observed.shape
    assert np.all(np.isfinite(pred))
    # predict via AST == predict nativo do PySR (domínio válido no treino)
    np.testing.assert_allclose(pred, reg.predict_native(train.X), rtol=PREDICT_REL_TOL)
    assert len(result.history) >= 1
    assert result.n_generations == len(result.history)
    assert result.stop_reason in ("max_generations", "early_stopping")
    assert result.metadata.algorithm == "pysr"
    assert result.metadata.integration_status == "baseline_only"
    assert not result.metadata.llm_in_loop
    assert all(r.n_llm_calls == 0 for r in result.history)
    assert all(math.isfinite(r.best_loss) for r in result.history)
    losses = [r.best_loss for r in result.history]
    assert losses == sorted(losses, reverse=True)  # hall of fame: nunca piora
    # paridade: perda Julia da melhor == L do objective (Huber, GAP-05)
    assert objective.error_metric == "huber"
    best = objective.evaluate_population([result.best_expression], train.X, train.y_observed)[0]
    assert math.isclose(best.total, result.history[-1].best_loss, rel_tol=1e-9)
    assert result.metadata.extra["audit"]["ok"]
    assert np.allclose(pred, evaluate(result.best_expression, train.X, train.feature_names))


def test_baseline_pysr_is_deterministic() -> None:
    cfg = load_config(SMOKE)
    train = _train(cfg)
    runs = []
    for _ in range(2):
        objective = build_objective(cfg, None, None, "pysr", True, train.feature_names)
        result = PySRRegressor().fit(train, objective, cfg)
        runs.append((to_infix(result.best_expression), [r.best_loss for r in result.history]))
    assert runs[0] == runs[1]
