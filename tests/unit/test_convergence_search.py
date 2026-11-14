"""T:convergence-search — critério, espaço de busca, etapas, refinamento, confirmação e export."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from physics_llm_sr.workflows import baseline_study as bs
from physics_llm_sr.workflows import convergence_search as cs

TINY: dict[str, Any] = {"n_simulations": 6, "n_time_points": 20, "max_cell_seconds": None}
TINY_BUDGET = {"sr.population": 10, "sr.generations": 2}
LOOSE = cs.ConvergenceCriterion(max_tree_distance=1.0, max_one_minus_r2=1e9)
"""Critério que qualquer execução válida atinge (testa a mecânica, não a ciência)."""
IMPOSSIBLE = cs.ConvergenceCriterion(max_tree_distance=0.0, max_one_minus_r2=0.0)


def _cand(label: str, **extra: Any) -> cs.Candidate:
    return cs.make_candidate(label, {**TINY_BUDGET, **extra})


# --------------------------------------------------------------------------- critério


def test_criterion_defaults_and_repetition_rule() -> None:
    crit = cs.ConvergenceCriterion()
    assert (crit.max_tree_distance, crit.max_one_minus_r2) == (0.05, 1e-3)
    assert crit.min_success_fraction == pytest.approx(2 / 3)
    # mínimo entre a distância primária e a equivalente
    assert crit.repetition_succeeds(1e-5, 1.0, 0.0)
    assert crit.repetition_succeeds(1e-5, 0.04, 1.0)
    assert not crit.repetition_succeeds(1e-5, 1.0, 0.06)
    assert not crit.repetition_succeeds(2e-3, 0.0, 0.0)  # ajuste numérico ruim
    assert not crit.repetition_succeeds(math.nan, 0.0, 0.0)
    assert not crit.repetition_succeeds(1e-5, math.nan, math.nan)
    assert crit.converges(2, 3) and crit.converges(3, 3)
    assert not crit.converges(1, 3) and not crit.converges(0, 0)
    for bad in ({"max_tree_distance": -1.0}, {"min_success_fraction": 0.0}):
        with pytest.raises(cs.ConvergenceSearchError):
            cs.ConvergenceCriterion(**bad)


# --------------------------------------------------------------------------- espaço de busca


def test_candidate_paths_and_declared_deviations() -> None:
    with pytest.raises(cs.ConvergenceSearchError):
        cs.Candidate("x", {"data.n_simulations": 3})
    with pytest.raises(cs.ConvergenceSearchError):
        cs.Candidate(" ", {})
    cand = cs.make_candidate(
        "c",
        {
            "sr.gplearn.const_range": (-10.0, 10.0),  # desvio (H-03 = (-1, 1))
            "sr.gplearn.tournament_size": 3,  # igual ao baseline literal
            "sr.early_stopping.enabled": False,  # desvio
            "sr.pysr.ncycles_per_iteration": 380,  # igual ao artigo/piloto
        },
    )
    assert cand.deviations == ("sr.gplearn.const_range", "sr.early_stopping.enabled")


def test_default_search_spaces_are_ordered_and_valid() -> None:
    for algorithm in bs.ALGORITHMS:
        space = cs.SearchSpace.default(algorithm)
        assert len(space) >= 4
        assert space.candidates[0].hypothesis == "H1"  # parada antecipada primeiro
        for cand in space.candidates:
            design = cs.design_for(
                algorithm, "free_fall", cand, repetitions=1, master_seed=1, output_root="x"
            )
            bs.build_config(design)  # caminhos existem e a config valida
    with pytest.raises(cs.ConvergenceSearchError):
        cs.SearchSpace("gplearn", (_cand("a"), _cand("a")))


def test_split_overrides_routes_budget_hyperparameters_and_loss() -> None:
    budget, hyper, formulation = cs.split_overrides(
        {
            "sr.population": 500,
            "sr.early_stopping.enabled": False,
            "sr.pysr.niterations": 7,
            "sr.gplearn.tournament_size": 20,
            "loss.normalization": "normalized",
        }
    )
    assert (budget.population, budget.early_stopping, budget.pysr_niterations) == (500, False, 7)
    assert budget.generations == bs.SearchBudget().generations
    assert hyper == (("sr.gplearn.tournament_size", 20),)
    assert formulation == "normalized"
    assert cs.estimated_cost("gplearn", {"sr.population": 10, "sr.generations": 3}) == 30.0


def test_rank_table_orders_by_convergence_then_success_then_distance_then_cost() -> None:
    base = {c: 0 for c in cs.TABLE_COLUMNS}
    rows = [
        {**base, "rotulo": "a", "converge": False, "taxa_sucesso": 0.33,
         "mediana_distancia_min": 0.0, "mediana_um_menos_r2": 0.0, "custo_estimado": 1.0},
        {**base, "rotulo": "b", "converge": True, "taxa_sucesso": 0.67,
         "mediana_distancia_min": 0.1, "mediana_um_menos_r2": 0.0, "custo_estimado": 9.0},
        {**base, "rotulo": "c", "converge": True, "taxa_sucesso": 0.67,
         "mediana_distancia_min": 0.1, "mediana_um_menos_r2": 0.0, "custo_estimado": 2.0},
        {**base, "rotulo": "d", "converge": True, "taxa_sucesso": 1.0,
         "mediana_distancia_min": math.nan, "mediana_um_menos_r2": 0.0, "custo_estimado": 99.0},
    ]  # fmt: skip
    ranked = cs.rank_table(pd.DataFrame(rows))
    assert list(ranked["rotulo"]) == ["d", "c", "b", "a"]


# --------------------------------------------------------------------------- etapas


def test_run_stage_stops_early_on_success_and_writes_logs(tmp_path: Path) -> None:
    cands = [_cand("primeira"), _cand("segunda"), _cand("terceira")]
    result = cs.run_stage(
        "gplearn", "free_fall", cands, repetitions=2, master_seed=5, parallel=1,
        criterion=LOOSE, stop_on_success=True, output_root=tmp_path, **TINY,
    )  # fmt: skip
    assert result.stopped_early
    assert list(result.table["rotulo"]) == ["primeira"]
    row = result.table.iloc[0]
    assert row["converge"] and row["n_sucesso"] == 2 and row["n_execucoes"] == 2
    assert list(result.table.columns) == list(cs.TABLE_COLUMNS)
    log = result.stage_dir / cs.STAGE_LOG_FILE
    lines = [json.loads(x) for x in log.read_text("utf-8").splitlines()]
    assert len(lines) == 2 and all(x["rotulo"] == "primeira" and x["sucesso"] for x in lines)
    assert (result.stage_dir / cs.SEARCH_TABLE_FILE).exists()
    assert result.stage_dir == tmp_path / "gplearn" / "free_fall" / cs.STAGE_SCREENING


def test_run_stage_without_success_runs_all_candidates_in_parallel(tmp_path: Path) -> None:
    cands = [_cand("a"), _cand("b", **{"sr.gplearn.tournament_size": 5})]
    result = cs.run_stage(
        "gplearn", "free_fall", cands, repetitions=1, master_seed=5, parallel=2,
        criterion=IMPOSSIBLE, stop_on_success=True, output_root=tmp_path, **TINY,
    )  # fmt: skip
    assert not result.stopped_early
    assert set(result.table["rotulo"]) == {"a", "b"}
    assert not result.table["converge"].any()
    # mesmas sementes para todas as candidatas da etapa (comparação pareada)
    seeds = {label: set(rep["semente_busca"]) for label, rep in result.reports.items()}
    assert seeds["a"] == seeds["b"]


def test_refine_returns_cheapest_converging_configuration(tmp_path: Path) -> None:
    best = cs.make_candidate("melhor", {"sr.population": 12, "sr.generations": 4})
    cands = cs.cheaper_candidates("gplearn", best, ((0.5, 0.5), (0.5, 1.0)))
    costs = [cs.estimated_cost("gplearn", c.overrides) for c in cands]
    assert costs == sorted(costs) and cands[-1] is best
    chosen, stage = cs.refine(
        "gplearn", "free_fall", best, repetitions=1, master_seed=3,
        factors=((0.5, 0.5), (0.5, 1.0)), criterion=LOOSE, output_root=tmp_path, **TINY,
    )  # fmt: skip
    assert chosen.overrides["sr.population"] == 6 and chosen.overrides["sr.generations"] == 2
    assert stage.stage == cs.STAGE_REFINEMENT and len(stage.table) == 3
    none, _ = cs.refine(
        "gplearn", "free_fall", best, repetitions=1, master_seed=3,
        factors=((0.5, 0.5),), criterion=IMPOSSIBLE, output_root=tmp_path, **TINY,
    )  # fmt: skip
    assert none is best


def test_confirm_uses_new_seeds_disjoint_from_search(tmp_path: Path) -> None:
    cand = _cand("c")
    search = cs.run_stage(
        "gplearn", "free_fall", [cand], repetitions=2, master_seed=11,
        criterion=LOOSE, output_root=tmp_path, **TINY,
    )  # fmt: skip
    conf = cs.confirm(
        "gplearn", "free_fall", cand, n_repetitions=3, seed_offset=1000,
        search_master_seed=11, used_seeds=search.seeds, criterion=LOOSE,
        output_root=tmp_path, **TINY,
    )  # fmt: skip
    assert conf.stage == cs.STAGE_CONFIRMATION
    assert len(conf.seeds) == 3 and not (conf.seeds & search.seeds)
    with pytest.raises(cs.ConvergenceSearchError):
        cs.confirm(
            "gplearn", "free_fall", cand, n_repetitions=3, seed_offset=1000,
            search_master_seed=11, used_seeds=conf.seeds, output_root=tmp_path, **TINY,
        )  # fmt: skip
    with pytest.raises(cs.ConvergenceSearchError):
        cs.confirm("gplearn", "free_fall", cand, seed_offset=0, output_root=tmp_path, **TINY)


def test_export_best_writes_files_loadable_by_notebook_05(tmp_path: Path) -> None:
    cand = cs.make_candidate(
        "vencedora",
        {
            **TINY_BUDGET,
            "sr.early_stopping.enabled": False,
            "sr.gplearn.const_range": (-10.0, 10.0),
        },
    )
    stage = cs.run_stage(
        "gplearn", "free_fall", [cand], repetitions=1, master_seed=2,
        criterion=LOOSE, output_root=tmp_path, **TINY,
    )  # fmt: skip
    conf = cs.confirm(
        "gplearn", "free_fall", cand, n_repetitions=1, search_master_seed=2,
        used_seeds=stage.seeds, criterion=LOOSE, output_root=tmp_path, **TINY,
    )  # fmt: skip
    paths = cs.export_best(
        "gplearn", "free_fall", cand, [stage], confirmation=conf, output_root=tmp_path
    )
    root = tmp_path / "gplearn" / "free_fall"
    assert paths == {
        "yaml": root / cs.BEST_YAML,
        "json": root / cs.BEST_JSON,
        "table": root / cs.SEARCH_TABLE_FILE,
    }
    data = json.loads(paths["json"].read_text("utf-8"))
    assert data["search_budget"]["population"] == 10
    assert data["search_budget"]["early_stopping"] is False
    assert data["hyperparameters"] == {"sr.gplearn.const_range": [-10.0, 10.0]}
    assert data["desvios_do_baseline_literal"] == [
        "sr.population", "sr.generations", "sr.early_stopping.enabled", "sr.gplearn.const_range"
    ]  # fmt: skip
    assert data["confirmacao"]["n_execucoes"] == 1
    assert "ORCAMENTO = SearchBudget(" in data["notebook_05"]
    table = pd.read_csv(paths["table"])
    assert set(table["etapa"]) == {cs.STAGE_SCREENING, cs.STAGE_CONFIRMATION}
    for path in (paths["json"], paths["yaml"]):
        budget, hyper, formulation = cs.load_best_config(path)
        design = bs.StudyDesign(
            algorithm="gplearn", search_budget=budget, hyperparameters=hyper,
            loss_formulation=formulation,  # type: ignore[arg-type]
        )  # fmt: skip
        assert bs.build_config(design).sr.gplearn.const_range == (-10.0, 10.0)


# --------------------------------------------------------------------------- expressividade


def test_expressiveness_diagnosis_free_fall_and_loss_blocking() -> None:
    small: dict[str, Any] = {"n_simulations": 6, "n_time_points": 20}
    ff = cs.expressiveness_diagnosis("gplearn", "free_fall", **small)
    gt = ff[ff["tipo"] == "referência"].iloc[0]
    eq = ff[ff["tipo"] == "equivalente"].iloc[0]
    assert not gt["expressavel"] and gt["operadores_ausentes"] == ("sqrt",)  # GAP-06
    assert eq["expressavel"] and eq["nos"] == 3 and eq["perda_L"] == pytest.approx(1.5)
    assert not cs.loss_blocks_convergence(ff)
    hm = cs.expressiveness_diagnosis("gplearn", "harmonic_motion", **small)
    assert cs.loss_blocks_convergence(hm)  # perda literal: a constante vence a referência
    normalized = cs.Candidate("n", {"loss.normalization": "normalized"})
    hm_n = cs.expressiveness_diagnosis("gplearn", "harmonic_motion", normalized, **small)
    assert not cs.loss_blocks_convergence(hm_n)
