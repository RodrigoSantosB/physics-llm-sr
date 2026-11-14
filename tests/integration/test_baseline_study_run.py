"""T:baseline-study (integração) — Fase 1 real, sem ruído, nos três cenários (tamanhos mínimos)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.workflows import baseline_study as bs

SCENARIOS = ("free_fall", "harmonic_motion", "damped_wave")


@pytest.mark.parametrize("algorithm", ["deap", "gplearn"])
def test_phase1_runs_three_scenarios_without_noise(
    algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    design = bs.StudyDesign(
        algorithm=algorithm,
        repetitions=2,
        master_seed=3,
        search_budget=bs.SearchBudget(population=10, generations=3, max_cell_seconds=None),
        n_simulations=6,
        n_time_points=20,
    )
    result = bs.phase1_clean_validation(design)
    run = result.runs[0]
    assert run.run_dir == tmp_path / "results/nb05_baseline_local" / bs.PHASE1 / algorithm / (
        design.design_hash()
    )
    for name in ("run.json", "results.parquet", bs.CONSISTENCY_FILE, bs.STUDY_LOG_FILE):
        assert (run.run_dir / name).is_file(), name
    results = pd.read_parquet(run.run_dir / "results.parquet")
    assert len(results) == 3 * 2 and set(results["scenario"]) == set(SCENARIOS)
    assert (results["status"] == "ok").all() and (results["noise_level"] == 0.0).all()
    assert (results["llm"] == "none").all()  # nenhum LLM na matriz
    report = bs.consistency_report(result)
    assert tuple(report.columns) == bs.CONSISTENCY_COLUMNS and len(report) == 6
    assert (report["sigma"] == 0.0).all() and report["erro_max_abs_grid"].notna().all()
    lines = (run.run_dir / bs.STUDY_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and all(json.loads(x)["fase"] == bs.PHASE1 for x in lines)
    # retomada: nova chamada não reexecuta nada e mantém uma linha por execução
    again = bs.phase1_clean_validation(design)
    assert again.runs[0].summary is not None and again.runs[0].summary.executed == 0
    assert len((run.run_dir / bs.STUDY_LOG_FILE).read_text(encoding="utf-8").splitlines()) == 6


def test_phase2_noise_sweep_has_same_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    design = bs.StudyDesign(
        algorithm="gplearn",
        scenarios=("free_fall",),
        repetitions=1,
        search_budget=bs.SearchBudget(population=10, generations=2, max_cell_seconds=None),
        n_simulations=6,
        n_time_points=20,
    )
    result = bs.phase2_noise_sweep(design, levels=[0.01, 0.05], locations=["target"])
    assert len(result.runs) == 2
    report = bs.consistency_report(result)
    assert tuple(report.columns) == bs.CONSISTENCY_COLUMNS
    assert sorted(report["sigma"]) == [0.01, 0.05]
    assert set(report["fase"]) == {bs.PHASE2_NOISE}
    grid = bs.phase2_hyperparameter_sweep(design, {"sr.gplearn.p_crossover": [0.5, 0.9]})
    assert len(grid.runs) == 2 and grid.runs[0].run_dir != grid.runs[1].run_dir
    assert tuple(bs.consistency_report(grid).columns) == bs.CONSISTENCY_COLUMNS


def test_phase2_algorithm_comparison_and_combined_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    design = bs.StudyDesign(
        algorithm="gplearn",
        scenarios=("free_fall",),
        repetitions=1,
        search_budget=bs.SearchBudget(population=10, generations=2, max_cell_seconds=None),
        n_simulations=6,
        n_time_points=20,
    )
    phase1 = bs.phase1_clean_validation(design)
    other = bs.phase2_algorithm_comparison(design, ["deap", "gplearn"])
    assert [r.condition for r in other.runs] == ["deap", "gplearn"]
    combined = pd.concat(
        [bs.consistency_report(phase1), bs.consistency_report(other)], ignore_index=True
    )
    assert set(combined["fase"]) == {bs.PHASE1, bs.PHASE2_ALGORITHMS}
    with pytest.raises(bs.StudyDesignError):
        bs.combine_results([phase1, other])
