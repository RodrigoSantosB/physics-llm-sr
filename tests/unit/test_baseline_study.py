"""T:baseline-study — desenho experimental, controle positivo, relatório de consistência e logs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.workflows import baseline_study as bs

SCENARIOS = ("free_fall", "harmonic_motion", "damped_wave")


def tiny_design(tmp_path: Path, algorithm: str = "gplearn", **over: object) -> bs.StudyDesign:
    """Desenho mínimo (segundos): dados pequenos e orçamento de busca ridículo."""
    args: dict[str, object] = {
        "algorithm": algorithm,
        "repetitions": 1,
        "master_seed": 7,
        "search_budget": bs.SearchBudget(population=10, generations=3),
        "n_simulations": 6,
        "n_time_points": 20,
        "output_root": tmp_path / "estudo",
    }
    args.update(over)
    return bs.StudyDesign(**args)  # type: ignore[arg-type]


def test_study_design_defaults_and_validation(tmp_path: Path) -> None:
    design = bs.StudyDesign(algorithm="gplearn", output_root=tmp_path)
    assert design.scenarios == SCENARIOS
    assert design.repetitions == 3 and design.loss_formulation == "literal"
    assert design.noise.noise_type == "none" and design.noise.sigma_fraction == 0.0
    assert len(design.design_hash()) == 10
    assert design.design_hash() == bs.StudyDesign(algorithm="gplearn").design_hash()
    assert design.design_hash() != replace(design, master_seed=1).design_hash()
    for bad in (
        {"algorithm": "random_forest"},
        {"scenarios": ()},
        {"scenarios": ("free_fall", "pendulo")},
        {"repetitions": 0},
        {"loss_formulation": "braco"},
        {"hyperparameters": (("data.n_simulations", 3),)},
    ):
        with pytest.raises(bs.StudyDesignError):
            bs.StudyDesign(**{"algorithm": "gplearn", **bad})  # type: ignore[arg-type]
    with pytest.raises(bs.StudyDesignError):
        bs.NoiseCondition("zero_mean", 0.0)
    with pytest.raises(bs.StudyDesignError):
        bs.NoiseCondition("none", 0.01)
    with pytest.raises(bs.StudyDesignError):
        bs.SearchBudget(population=0)


def test_loss_formulation_maps_to_track(tmp_path: Path) -> None:
    literal = bs.build_config(tiny_design(tmp_path))
    normalized = bs.build_config(tiny_design(tmp_path, loss_formulation="normalized"))
    assert literal.track.value == "paper_replication" and literal.loss.normalization == "literal"
    assert normalized.track.value == "extended_analysis"
    assert literal.data.noise.sigma_fraction == 0.0
    assert literal.experiments.baseline_only and literal.llm.backend == "fake"


def test_phase1_refuses_noise(tmp_path: Path) -> None:
    noisy = tiny_design(tmp_path, noise=bs.NoiseCondition("zero_mean", 0.01, "target"))
    with pytest.raises(bs.CleanPhaseError, match="σ = 0"):
        bs.phase1_clean_validation(noisy)
    assert not (tmp_path / "estudo").exists()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_positive_control_reference_is_exact(scenario: str, tmp_path: Path) -> None:
    table = bs.positive_control(scenario, tiny_design(tmp_path))
    assert not table.empty and set(table["cenario"]) == {scenario}
    primary = table[table["forma_primaria"]].iloc[0]
    assert primary["distancia_estrutural"] == 0.0
    assert primary["um_menos_r2"] <= bs.POSITIVE_CONTROL_TOLERANCE
    assert (table["um_menos_r2"] <= bs.POSITIVE_CONTROL_TOLERANCE).all()
    assert (table["distancia_equivalente_min"] == 0.0).all()
    assert table["aprovado"].all() and table["dados_limpos"].all()


def test_classification_follows_visible_thresholds() -> None:
    crit = bs.ConsistencyCriteria(max_one_minus_r2=1e-3, max_tree_distance=0.0)
    assert bs.classify(1e-9, 0.0, crit) == bs.CLASS_RECOVERS
    assert bs.classify(5e-4, 0.0, crit) == bs.CLASS_RECOVERS
    assert bs.classify(1e-9, 2.0, crit) == bs.CLASS_FITS
    assert bs.classify(0.2, 0.0, crit) == bs.CLASS_FAILS
    assert bs.classify(0.2, 3.0, crit) == bs.CLASS_FAILS
    assert bs.classify(float("nan"), float("nan"), crit) == bs.CLASS_FAILS
    assert bs.classify(1e-9, 0.0, crit, status="error") == bs.CLASS_ERROR
    relaxed = bs.ConsistencyCriteria(max_one_minus_r2=1e-3, max_tree_distance=3.0)
    assert bs.classify(1e-9, 2.0, relaxed) == bs.CLASS_RECOVERS
    with pytest.raises(bs.StudyDesignError):
        bs.ConsistencyCriteria(max_one_minus_r2=-1.0)


def _fake_result(tmp_path: Path) -> bs.StudyResult:
    """Resultado fabricado: uma repetição que acerta e outra que não (sem rodar busca)."""
    design = tiny_design(tmp_path, scenarios=("free_fall",), repetitions=2)
    cfg = bs.build_config(design)
    rows = []
    for rep, (expr, dist, r2) in enumerate(
        [("sqrt(2*9.81*height)", 0.0, 0.0), ("height*0.5", 4.0, 0.3)]
    ):
        rows.append(
            {
                "scenario": "free_fall", "sr": "gplearn", "repetition": rep, "seed": 11 + rep,
                "noise_type": "zero_mean", "noise_location": "target", "noise_level": 0.0,
                "status": "ok", "expression": expr, "ground_truth": "sqrt(2*9.81*height)",
                "test_true_mae": r2, "test_true_mse": r2, "test_true_one_minus_r2": r2,
                "tree_distance": dist, "tree_distance_equivalent_min": dist,
                "exact_recovery": dist == 0.0, "n_generations": 5, "stop_reason": "max_generations",
                "cell_seconds": 0.1, "cell_key": f"k{rep}",
            }
        )  # fmt: skip
    run = bs.ExecutedRun(
        phase=bs.PHASE1,
        condition="sem ruído",
        design=design,
        config=cfg,
        run_dir=tmp_path / "run",
        results=pd.DataFrame(rows),
        history=pd.DataFrame(),
    )
    return bs.StudyResult(phase=bs.PHASE1, design=design, runs=(run,))


def test_consistency_report_schema_and_classes(tmp_path: Path) -> None:
    report = bs.consistency_report(_fake_result(tmp_path))
    assert tuple(report.columns) == bs.CONSISTENCY_COLUMNS
    assert list(report["classificacao"]) == [bs.CLASS_RECOVERS, bs.CLASS_FAILS]
    assert report["equacao_referencia"].eq("sqrt(2*9.81*height)").all()
    # erro máximo no grid denso: 0 (a própria equação) e > 0 (equação errada), fora do treino
    assert float(report["erro_max_abs_grid"].iloc[0]) < 1e-9
    assert float(report["erro_max_abs_grid"].iloc[1]) > 1.0
    strict = bs.consistency_report(
        _fake_result(tmp_path), bs.ConsistencyCriteria(max_one_minus_r2=1e-12, max_tree_distance=0)
    )
    assert strict["classificacao"].iloc[0] == bs.CLASS_RECOVERS


def test_phase1_approval_uses_fraction_of_accepted_classes(tmp_path: Path) -> None:
    report = bs.consistency_report(_fake_result(tmp_path))
    half = bs.phase1_approval(report, min_fraction=0.5)
    assert bool(half.loc[0, "aprovado"]) and half.loc[0, "fracao_aceita"] == 0.5
    full = bs.phase1_approval(report, min_fraction=1.0)
    assert not bool(full.loc[0, "aprovado"])
    assert set(full.columns) >= {"cenario", "algoritmo", "n", "fracao_aceita", "aprovado"}


def test_study_log_is_valid_jsonl_one_line_per_execution(tmp_path: Path) -> None:
    result = _fake_result(tmp_path)
    run = result.runs[0]
    run.run_dir.mkdir(parents=True)
    bs.write_run_outputs(run, result.design.criteria)
    lines = (run.run_dir / bs.STUDY_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(run.results) == 2
    first = json.loads(lines[0])
    assert {"fase", "cenario", "algoritmo", "semente", "sigma", "hiperparametros"} <= set(first)
    assert {"metricas", "status", "desenho_hash"} <= set(first)
    assert first["sigma"] == 0.0 and first["status"]["classificacao"] == bs.CLASS_RECOVERS
    csv = pd.read_csv(run.run_dir / bs.CONSISTENCY_FILE)
    assert tuple(csv.columns) == bs.CONSISTENCY_COLUMNS and len(csv) == 2
    assert np.isfinite(first["metricas"]["erro_max_abs_grid"])
