"""T:sr-interface - contrato do Protocol SymbolicRegressor e dos registros."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset, Mode, Track
from physics_llm_sr.sr.interfaces import (
    CandidateObjective,
    FitResult,
    GenerationRecord,
    LossBreakdown,
    RegressorMetadata,
    SymbolicRegressor,
)
from physics_llm_sr.symbolic.expression import Expr, Var, to_infix


class _Objective:
    def evaluate_population(self, exprs: Sequence[Expr], X: Any, y: Any) -> Sequence[LossBreakdown]:  # noqa: N803
        return [LossBreakdown(total=1.0, e=0.5, s=float(1), c=0.5, evaluation=None) for _ in exprs]


class _Dummy:
    def __init__(self) -> None:
        self._best: Expr = Var("t")
        self._history: list[GenerationRecord] = []

    def fit(
        self, dataset: Dataset, objective: CandidateObjective, config: ExperimentConfig
    ) -> FitResult:
        (b,) = objective.evaluate_population([self._best], dataset.X, dataset.y_observed)
        self._history.append(
            GenerationRecord(
                generation=0,
                best_loss=b.total,
                best_e=b.e,
                best_s=b.s,
                best_c=b.c,
                best_expression=to_infix(self._best),
                n_llm_calls=0,
                n_cache_hits=0,
                elapsed=0.0,
            )
        )
        return FitResult(
            best_expression=self._best, history=tuple(self._history), metadata=self.get_metadata()
        )

    def predict(self, X: Any) -> Any:  # noqa: N803
        return np.asarray(X)[:, 0]

    def get_best_expression(self) -> Expr:
        return self._best

    def get_search_history(self) -> Sequence[GenerationRecord]:
        return tuple(self._history)

    def get_metadata(self) -> RegressorMetadata:
        return RegressorMetadata(
            algorithm="dummy", llm_in_loop=False, integration_status="baseline_only"
        )


def _dataset() -> Dataset:
    x = np.arange(10.0).reshape(5, 2)
    y = x[:, 0]
    return Dataset(
        X=x,
        y_true=y,
        y_observed=y,
        feature_names=("a", "b"),
        target_name="y",
        scenario="free_fall",
        simulation_ids=np.arange(5),
        t=np.arange(5.0),
    )


def test_dummy_satisfies_protocols_at_runtime() -> None:
    assert isinstance(_Dummy(), SymbolicRegressor)
    assert isinstance(_Objective(), CandidateObjective)
    assert not isinstance(object(), SymbolicRegressor)


def test_fit_roundtrip_and_metadata() -> None:
    reg = _Dummy()
    cfg = ExperimentConfig(mode=Mode.SMOKE, track=Track.PAPER_REPLICATION)
    result = reg.fit(_dataset(), _Objective(), cfg)
    assert result.best_expression == Var("t")
    assert len(reg.get_search_history()) == 1
    rec = reg.get_search_history()[0]
    assert (rec.best_loss, rec.best_e, rec.best_s, rec.best_c) == (1.0, 0.5, 1.0, 0.5)
    assert rec.best_expression == "t"
    assert result.metadata.llm_in_loop is False
    assert result.metadata.integration_status == "baseline_only"
    assert reg.predict(_dataset().X).shape == (5,)


def test_interfaces_do_not_import_optimization() -> None:
    assert "physics_llm_sr.optimization.objective" not in sys.modules
    import physics_llm_sr.sr.interfaces as mod

    src = open(mod.__file__, encoding="utf-8").read()
    assert "physics_llm_sr.optimization" not in src
