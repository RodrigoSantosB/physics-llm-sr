"""T:sr-interface - contrato do Protocol SymbolicRegressor e dos registros."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset, Mode, Track
from physics_llm_sr.sr.interfaces import (
    CandidateObjective,
    FitResult,
    GenerationRecord,
    LossBreakdown,
    RegressorMetadata,
    SymbolicRegressor,
    ensure_feature_alignment,
)
from physics_llm_sr.symbolic.expression import Expr, Var, to_infix


class _Objective:
    feature_names: tuple[str, ...] = ("a", "b")

    def evaluate_population(self, exprs: Sequence[Expr], X: Any, y: Any) -> Sequence[LossBreakdown]:  # noqa: N803
        return [LossBreakdown(total=1.0, e=0.5, s=float(1), c=0.5, evaluation=None) for _ in exprs]


class _Dummy:
    def __init__(self) -> None:
        self._best: Expr = Var("t")
        self._history: list[GenerationRecord] = []

    def fit(
        self,
        train: Dataset,
        objective: CandidateObjective,
        config: ExperimentConfig,
        validation: Dataset | None = None,
    ) -> FitResult:
        ensure_feature_alignment(train, objective)
        (b,) = objective.evaluate_population([self._best], train.X, train.y_observed)
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
            best_expression=self._best,
            history=tuple(self._history),
            metadata=self.get_metadata(),
            elapsed_seconds=0.0,
            n_generations=1,
            stop_reason="max_generations",
            llm_counters={"llm_calls": 0.0},
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
    # Subprocesso: outros testes (optimization) importam o módulo no mesmo processo.
    code = (
        "import sys, physics_llm_sr.sr.interfaces;"
        "sys.exit('physics_llm_sr.optimization.objective' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
    import physics_llm_sr.sr.interfaces as mod

    src = open(mod.__file__, encoding="utf-8").read()
    assert "physics_llm_sr.optimization" not in src


def test_fit_result_new_fields() -> None:
    cfg = ExperimentConfig(mode=Mode.SMOKE, track=Track.PAPER_REPLICATION)
    r = _Dummy().fit(_dataset(), _Objective(), cfg)
    assert r.stop_reason == "max_generations"
    assert r.n_generations == 1
    assert r.elapsed_seconds == 0.0
    assert r.llm_counters["llm_calls"] == 0.0


def test_fit_accepts_validation_by_keyword() -> None:
    cfg = ExperimentConfig(mode=Mode.SMOKE, track=Track.PAPER_REPLICATION)
    _Dummy().fit(_dataset(), _Objective(), cfg, validation=_dataset())


def test_feature_alignment_ok_and_mismatch() -> None:
    ensure_feature_alignment(_dataset(), _Objective())

    class _Other(_Objective):
        feature_names = ("b", "a")

    with pytest.raises(ValueError, match="feature_names"):
        ensure_feature_alignment(_dataset(), _Other())


def test_objective_without_feature_names_is_not_a_candidate_objective() -> None:
    class _NoNames:
        def evaluate_population(self, exprs: Any, X: Any, y: Any) -> Any:  # noqa: N803
            return []

    assert not isinstance(_NoNames(), CandidateObjective)
