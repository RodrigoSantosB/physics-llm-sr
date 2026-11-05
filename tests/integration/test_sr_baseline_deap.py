"""T:baseline-sr (DEAP): queda livre smoke, objective baseline, avaliador nunca chamado."""

from __future__ import annotations

import math

import numpy as np

from physics_llm_sr.config.schemas import DataConfig, ExperimentConfig, SRConfig
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import build_objective
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng
from physics_llm_sr.sr.deap_adapter import DeapRegressor
from physics_llm_sr.sr.interfaces import SymbolicRegressor
from physics_llm_sr.symbolic.expression import Expr, evaluate, variables

CFG = ExperimentConfig(
    mode=Mode.SMOKE,
    track=Track.PAPER_REPLICATION,
    data=DataConfig(n_simulations=10, n_time_points=20),
    sr=SRConfig(population=20, generations=4),
)


def test_deap_baseline_free_fall_smoke() -> None:
    data = generate_dataset(CFG, "free_fall")
    train = split_by_simulation(data, CFG.data.split, derive_rng(42, "split"))["train"]
    ctx = get_scenario("free_fall", CFG.physics).physical_context()
    fake = FakeEquationEvaluator.from_table({})
    objective = build_objective(CFG, fake, ctx, "deap", True, train.feature_names)

    reg = DeapRegressor()
    assert isinstance(reg, SymbolicRegressor)
    result = reg.fit(train, objective, CFG)

    assert fake.calls == 0
    assert isinstance(result.best_expression, Expr)
    assert variables(result.best_expression) <= set(train.feature_names)
    pred = reg.predict(train.X)
    assert pred.shape == (train.X.shape[0],) and np.all(np.isfinite(pred))
    assert np.allclose(pred, evaluate(result.best_expression, train.X, train.feature_names))
    assert len(result.history) >= 1 and result.n_generations >= 1
    assert all(math.isfinite(r.best_loss) or r.best_loss == math.inf for r in result.history)
    assert result.stop_reason in {"max_generations", "early_stopping"}
    meta = reg.get_metadata()
    assert meta.algorithm == "deap" and not meta.llm_in_loop
    assert meta.integration_status == "baseline_only"
    assert meta.extra is not None and meta.extra["crossover_prob"] == 0.05
    assert meta.extra["tournament_size"] == 3 and meta.extra["max_depth"] == 8
