"""Métrica secundária ``tree_distance_algebraic``: métricas, runner, relatório e critérios."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr.config.schemas import SymbolicConfig
from physics_llm_sr.evaluation.metrics import structural_metrics, tree_distance_algebraic
from physics_llm_sr.experiments.structural import structural_row
from physics_llm_sr.symbolic.expression import parse_infix
from physics_llm_sr.workflows import baseline_study as bs
from physics_llm_sr.workflows import convergence_search as cs

FEATURES = ("mass", "length", "height", "damping", "t")
FREE_FALL = ("sqrt(2*9.81*height)", "9.81*t", "2*height/t", f"t/{1.0 / 9.81!r}")
WAVE_GT = "1.0*exp(-damping*t/2)*cos(6.283185307179586*length-6.283185307179586*t)"


def _p(text: str):  # type: ignore[no-untyped-def]
    return parse_infix(text, variables=FEATURES)


def test_structural_metrics_adds_algebraic_key_and_keeps_others() -> None:
    gt, *eq = (_p(f) for f in FREE_FALL)
    out = structural_metrics(_p("height / t + height / t"), gt, eq)
    assert set(out) == {"tree_distance", "tree_distance_equivalent_min", "tree_distance_algebraic"}
    assert out["tree_distance_algebraic"] == 0.0
    assert out["tree_distance_equivalent_min"] > 0.0  # a secundária antiga não reconhece u + u
    assert (
        out["tree_distance"]
        == structural_metrics(_p("height / t + height / t"), gt)["tree_distance"]
    )


def test_algebraic_distance_real_cases() -> None:
    gt, *eq = (_p(f) for f in FREE_FALL)
    assert tree_distance_algebraic(_p("t/0.1019367991845056"), gt, eq) <= 1e-12
    assert tree_distance_algebraic(_p("t/0.10193679917007942"), gt, eq) <= 1e-9
    wave = _p("cos(6.299*(t - length)) / exp(0.4965*damping*t)")
    d = tree_distance_algebraic(wave, _p(WAVE_GT))
    assert 0.0 < d < 0.005
    assert tree_distance_algebraic(_p("cos(length)"), _p(WAVE_GT)) > 0.5


def test_structural_row_feeds_results_column() -> None:
    gt, *eq = (_p(f) for f in FREE_FALL)
    row = structural_row(_p("height / t + height / t"), gt, eq, SymbolicConfig())
    assert row["tree_distance_algebraic"] == 0.0


def test_consistency_criteria_use_min_of_three_distances() -> None:
    crit = bs.ConsistencyCriteria(max_one_minus_r2=1e-3, max_tree_distance=0.0)
    assert crit.use_algebraic
    assert bs.classify(1e-9, 1.0, crit) == bs.CLASS_FITS
    assert bs.classify(1e-9, 1.0, crit, equivalent_distance=0.0) == bs.CLASS_RECOVERS
    assert bs.classify(1e-9, 1.0, crit, algebraic_distance=0.0) == bs.CLASS_RECOVERS
    off = bs.ConsistencyCriteria(max_one_minus_r2=1e-3, max_tree_distance=0.0, use_algebraic=False)
    assert bs.classify(1e-9, 1.0, off, algebraic_distance=0.0) == bs.CLASS_FITS


def test_convergence_criterion_uses_algebraic_by_default() -> None:
    crit = cs.ConvergenceCriterion()
    assert crit.use_algebraic
    assert crit.repetition_succeeds(1e-5, 1.0, 1.0, 0.01)
    assert crit.repetition_succeeds(1e-5, 1.0, 0.0)  # assinatura antiga continua válida
    assert not crit.repetition_succeeds(1e-5, 1.0, 1.0, math.nan)
    off = cs.ConvergenceCriterion(use_algebraic=False)
    assert not off.repetition_succeeds(1e-5, 1.0, 1.0, 0.01)


def _result(tmp_path: Path, with_column: bool) -> bs.StudyResult:
    design = bs.StudyDesign(
        algorithm="gplearn",
        scenarios=("free_fall",),
        repetitions=1,
        output_root=tmp_path,
        n_simulations=6,
        n_time_points=20,
    )
    row = {
        "scenario": "free_fall", "sr": "gplearn", "repetition": 0, "seed": 1,
        "noise_type": "none", "noise_location": "target", "noise_level": 0.0, "status": "ok",
        "expression": "height / t + height / t", "ground_truth": FREE_FALL[0],
        "test_true_mae": 0.0, "test_true_mse": 0.0, "test_true_one_minus_r2": 0.0,
        "tree_distance": 1.0, "tree_distance_equivalent_min": 0.5, "exact_recovery": False,
        "n_generations": 5, "stop_reason": "max_generations", "cell_seconds": 0.1,
        "cell_key": "k0",
    }  # fmt: skip
    if with_column:
        row["tree_distance_algebraic"] = 0.0
    run = bs.ExecutedRun(
        phase=bs.PHASE1,
        condition="sem ruído",
        design=design,
        config=bs.build_config(design),
        run_dir=tmp_path / "run",
        results=pd.DataFrame([row]),
        history=pd.DataFrame(),
    )
    return bs.StudyResult(phase=bs.PHASE1, design=design, runs=(run,))


@pytest.mark.parametrize("with_column", [True, False])
def test_consistency_report_has_algebraic_column_and_tolerates_old_parquet(
    tmp_path: Path, with_column: bool
) -> None:
    report = bs.consistency_report(_result(tmp_path, with_column))
    assert "distancia_algebrica" in report.columns
    assert tuple(report.columns) == bs.CONSISTENCY_COLUMNS
    # parquet antigo sem a coluna: recalculada a partir da expressão e das formas declaradas
    assert report["distancia_algebrica"].iloc[0] == 0.0
    assert report["classificacao"].iloc[0] == bs.CLASS_RECOVERS
    strict = bs.consistency_report(
        _result(tmp_path, with_column),
        bs.ConsistencyCriteria(max_one_minus_r2=1e-3, max_tree_distance=0.0, use_algebraic=False),
    )
    assert strict["classificacao"].iloc[0] == bs.CLASS_FITS


def test_successes_read_algebraic_column(tmp_path: Path) -> None:
    report = bs.consistency_report(_result(tmp_path, True))
    assert cs.successes(report, cs.ConvergenceCriterion()) == [True]
    assert cs.successes(report, cs.ConvergenceCriterion(use_algebraic=False)) == [False]
    old = report.drop(columns=["distancia_algebrica"])
    assert cs.successes(old, cs.ConvergenceCriterion()) == [False]
