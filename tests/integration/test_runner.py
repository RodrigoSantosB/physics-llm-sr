"""C:runner (T:runner): smoke completo gera os artefatos com as colunas exigidas.

Rótulos de modo/trilha corretos em tudo; smoke nunca rotulado como resultado completo;
baseline e LLM pareados (mesmo dataset, mesmo split, mesma semente de busca).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.evaluation.paper_comparison import RESULTS_COLUMNS
from physics_llm_sr.experiments.runner import run_experiments

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
REQUIRED = [c.strip() for c in RESULTS_COLUMNS.split(",")]
EXTRA = [
    "status",
    "error",
    "cell_key",
    "experiments",
    "train_mae",
    "train_mse",
    "train_one_minus_r2",
    "validation_mae",
    "test_mae",
    "test_mse",
    "test_one_minus_r2",
    "test_true_mae",
    "expression",
    "complexity",
    "n_generations",
    "stop_reason",
    "train_seconds",
    "llm_calls",
    "llm_cache_hits",
    "gt_disclosed",
    "integration_status",
    "data_hash",
    "exact_recovery",
]
HISTORY_COLUMNS = ["config", "generation", "best_loss", "best_e", "best_s", "best_c", "cell_key"]


@pytest.fixture()
def smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    summary = run_experiments(load_config(SMOKE))
    assert summary.failed == 0
    return summary.run_dir


def test_smoke_run_writes_results_with_required_columns(smoke_run: Path) -> None:
    results = pd.read_parquet(smoke_run / "results.parquet")
    missing = [c for c in [*REQUIRED, *EXTRA] if c not in results.columns]
    assert not missing, missing
    # 1 cenário x 2 SR x 1 repetição x (baseline + fake)
    assert len(results) == 4
    assert (results["status"] == "ok").all()
    assert (results["mode"] == "smoke").all()
    assert (results["track"] == "paper_replication").all()
    assert results["run_id"].nunique() == 1
    # métrica principal = teste; treino separado
    assert (results["mae"] == results["test_mae"]).all()
    base = results[results["llm"] == "none"]
    llm = results[results["llm"] != "none"]
    assert (base["prompt"] == "none").all()
    assert (base["integration_status"] == "baseline_only").all()
    assert (base["llm_calls"] == 0).all()
    assert set(llm["llm"]) == {"fake"} and set(llm["prompt"]) == {"E"}
    assert (llm["integration_status"] == "integrated").all()
    assert not llm["gt_disclosed"].any()
    assert results["expression"].str.len().min() > 0
    assert results["tree_distance"].between(0, 1).all()
    assert (results["tree_distance_equivalent_min"] <= results["tree_distance"]).all()


def test_baseline_and_llm_are_paired(smoke_run: Path) -> None:
    results = pd.read_parquet(smoke_run / "results.parquet")
    keys = ["scenario", "sr", "repetition", "noise_type", "noise_location", "noise_level"]
    merged = results[results["llm"] == "none"].merge(
        results[results["llm"] != "none"], on=keys, suffixes=("_b", "_l")
    )
    assert len(merged) == 2
    assert (merged["seed_b"] == merged["seed_l"]).all()
    assert (merged["data_hash_b"] == merged["data_hash_l"]).all()
    assert (merged["train_hash_b"] == merged["train_hash_l"]).all()


def test_history_manifests_and_labels(smoke_run: Path) -> None:
    history = pd.read_parquet(smoke_run / "history.parquet")
    assert [c for c in HISTORY_COLUMNS if c not in history.columns] == []
    assert history["cell_key"].nunique() == 4
    assert (history["mode"] == "smoke").all()
    manifests = sorted((smoke_run / "cells").glob("*/manifest.json"))
    assert len(manifests) == 4
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["mode"] == "smoke" and data["track"] == "paper_replication"
        assert data["config_hash"] and data["data_hashes"] and data["discovered_expressions"]
    run_manifest = json.loads((smoke_run / "manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["mode"] == "smoke"


def test_smoke_is_never_labelled_complete(smoke_run: Path) -> None:
    report = json.loads((smoke_run / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "smoke"
    assert report["reduced_run"] is True
    assert report["complete_result"] is False
    comparison = pd.read_csv(smoke_run / "comparison.csv")
    assert len(comparison) > 0
    assert comparison["reduced_run"].all()
    assert (comparison["mode"] == "smoke").all()
    assert "numericamente_proxima" not in set(comparison["classe"])
    assert sorted(p.name for p in (smoke_run / "figures").glob("*.png"))
