"""Tabelas de exibição dos notebooks sobre um run smoke (backend simulado, sem LLM real)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr import colab
from physics_llm_sr.analysis import llm_inspection
from physics_llm_sr.analysis import notebook_views as nv
from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV


@pytest.fixture(scope="module")
def run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[pd.DataFrame, object]]:
    root = tmp_path_factory.mktemp("views")
    mp = pytest.MonkeyPatch()
    mp.setenv(PROJECT_ROOT_ENV, str(root))
    try:
        path = colab.build_single_llm_config(
            scenario="free_fall",
            experiment="benchmark",
            llm="mistral",
            algorithms=["deap", "gplearn"],
            repetitions=3,
            prompt="E",
            prompts=["E"],
            braco="normalized",
            noise_levels=[0.01],
            noise_locations=["target"],
            out_dir=root,
            smoke=True,
        )
        cfg = load_config(path)
        run_dir = colab.notebook_run_dir(cfg)
        colab.run_notebook_stage(cfg, ["exp2"], run_dir, arm="baseline")
        colab.run_notebook_stage(cfg, ["exp2"], run_dir, arm="llm")
        yield colab.load_results(run_dir), cfg
    finally:
        mp.undo()


def test_arm_runs_are_complementary_and_resumable(run: tuple[pd.DataFrame, object]) -> None:
    results, _ = run
    assert (results["llm"] == "none").sum() == 4 and (results["llm"] != "none").sum() == 4
    assert set(results["status"]) == {"ok"}


def test_metrics_summary_has_mean_std_and_n(run: tuple[pd.DataFrame, object]) -> None:
    out = nv.metrics_summary(run[0], ("scenario", "sr", "llm"))
    assert {"mae_mean", "mae_std", "tree_distance_mean", "n"} <= set(out.columns)
    assert set(out["n"]) == {2} and len(out) == 4
    assert nv.metrics_summary(pd.DataFrame()).empty


def test_equations_and_side_by_side(run: tuple[pd.DataFrame, object]) -> None:
    eq = nv.equations_table(run[0])
    assert {"expression", "ground_truth", "tree_distance"} <= set(eq.columns) and len(eq) == 8
    side = nv.side_by_side_equations(run[0])
    assert len(side) == 4 and {"expression_sem_llm", "expression_com_llm", "mudou"} <= set(
        side.columns
    )
    assert side["mudou"].dtype == bool


def test_evaluations_of_discovered_come_from_cache_only(
    run: tuple[pd.DataFrame, object],
) -> None:
    results, cfg = run
    records = llm_inspection.read_cache_records(llm_inspection.cache_file(cfg))  # type: ignore[arg-type]
    assert len(records) > 0
    checkpoints = {m.name: m.checkpoint for m in cfg.llm.models}  # type: ignore[attr-defined]
    out = nv.evaluations_of_discovered(
        results,
        records,
        checkpoints,
        significant_digits=cfg.llm.cache_significant_digits,  # type: ignore[attr-defined]
        key_mode=cfg.llm.cache_key_mode,  # type: ignore[attr-defined]
    )
    assert len(out) == 4 and set(out["llm"]) == {"mistral"}
    assert out["found"].all()  # a equação final foi avaliada durante a busca
    assert out["penalty"].between(0, 1).all()
    empty = nv.evaluations_of_discovered(
        results, records.iloc[0:0], checkpoints, significant_digits=3, key_mode="exact"
    )
    assert not empty["found"].any() and empty["penalty"].isna().all()


def test_cost_summary(run: tuple[pd.DataFrame, object]) -> None:
    out = nv.cost_summary(run[0])
    assert set(out["sr"]) == {"deap", "gplearn"}
    assert (out["llm_calls"] > 0).all() and (out["cells_error"] == 0).all()
    assert np.allclose(out["sr_seconds"], out["train_seconds"] - out["llm_seconds"])


def test_baseline_vs_published_uses_only_baseline_rows(run: tuple[pd.DataFrame, object]) -> None:
    out = nv.baseline_vs_published(run[0])
    assert set(out.columns) >= {"table", "metric", "published", "n", "classe"}
    assert (out["n"] > 0).all()
    assert set(out["table"]) <= {3, 4, 5}
    assert nv.baseline_vs_published(pd.DataFrame()).empty


def test_status_table_counts_cells_per_algorithm(run: tuple[pd.DataFrame, object]) -> None:
    out = nv.status_table(run[0])
    assert {"sr", "ok", "error", "train_seconds_mean"} <= set(out.columns)
    assert dict(zip(out["sr"], out["ok"], strict=True)) == {"deap": 4, "gplearn": 4}
    assert (out["error"] == 0).all() and nv.status_table(pd.DataFrame()).empty


def test_default_noise_rows_keep_only_the_standard_condition(
    run: tuple[pd.DataFrame, object],
) -> None:
    results, cfg = run
    noisy = results.assign(noise_level=[0.01, 0.05] * 4)
    kept = nv.default_noise_rows(noisy, cfg)  # type: ignore[arg-type]
    assert len(kept) == 4 and set(kept["noise_level"]) == {0.01}


def test_convergence_history_has_config_label_and_filters(
    run: tuple[pd.DataFrame, object],
) -> None:
    results, cfg = run
    run_dir = colab.notebook_run_dir(cfg)  # type: ignore[arg-type]
    history = pd.read_parquet(run_dir / "history.parquet")
    out = nv.convergence_history(results, history, group=("scenario", "sr", "llm"))
    assert {"config", "generation", "best_loss"} <= set(out.columns)
    assert set(out["config"]) >= {"free_fall / deap / none", "free_fall / deap / mistral"}
    assert nv.convergence_history(results.iloc[0:0], history).empty


def test_paired_repetitions_has_with_and_without_llm_columns(
    run: tuple[pd.DataFrame, object],
) -> None:
    out = nv.paired_repetitions(run[0], metrics=("mae", "tree_distance"))
    assert len(out) == 4 and {"sr", "repetition", "llm", "prompt"} <= set(out.columns)
    assert {"mae_sem_llm", "mae_com_llm", "mae_diff", "tree_distance_sem_llm"} <= set(out.columns)
    assert np.allclose(out["mae_diff"], out["mae_sem_llm"] - out["mae_com_llm"])
    assert nv.paired_repetitions(pd.DataFrame()).empty
