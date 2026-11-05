"""C:e2e (T:e2e, PRD §12.4): pipeline mínimo de ponta a ponta com avaliador simulado.

gerar dados → baseline → integrado (Fake) → métricas → distância → persistir → replot →
repetir com a mesma config → artefatos metodológicos idênticos (``results.parquet`` sem
colunas de tempo/run_id/custo de cache).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.experiments.runner import methodological_view, replot, run_experiments
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.symbolic.expression import Expr, size, variables

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"


def _config() -> ExperimentConfig:
    data = load_config(SMOKE).model_dump(mode="json")
    data["experiments"]["run"] = ["baseline", "llm_integration", "noise_robustness"]
    data["experiments"]["noise_model"] = "fake"
    data["data"]["exp4"] = {"levels": [0.01, 0.05], "locations": ["target"]}
    return parse_config(data)


def _score(expr: Expr, ctx: PhysicalContext, variant: str) -> EquationEvaluation:
    """Avaliador simulado determinístico: favorece expressões curtas que usam ``height``."""
    uses_height = 1.0 if "height" in variables(expr) else 0.0
    simplicity = 1.0 / (1.0 + size(expr))
    return EquationEvaluation(uses_height, simplicity, 0.5, "simulado")


def _run(cfg: ExperimentConfig, run_dir: Path) -> Path:
    fake = FakeEquationEvaluator.from_function(_score)
    summary = run_experiments(cfg, run_dir=run_dir, evaluator_factory=lambda m: fake)
    assert summary.failed == 0
    return summary.run_dir


def test_minimal_pipeline_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cfg = _config()
    first = _run(cfg, tmp_path / "a")
    results = pd.read_parquet(first / "results.parquet")

    # dados gerados + baseline + integrado + condições de ruído (célula padrão deduplicada)
    assert len(results) == 8
    assert {"none", "fake"} == set(results["llm"])
    assert set(results["noise_level"]) == {0.01, 0.05}
    shared = results[(results["noise_level"] == 0.01) & (results["llm"] == "fake")]
    assert set(shared["experiments"]) == {"llm_integration,noise_robustness"}
    # métricas e distância calculadas
    for col in ("train_mae", "test_mae", "one_minus_r2", "tree_distance"):
        assert results[col].notna().all(), col
    integrated = results[results["llm"] == "fake"]
    assert (integrated["llm_requested"] > 0).all()
    # ruído parte do mesmo dataset limpo: níveis diferentes -> dados diferentes, mesmo split
    by_level = results[results["llm"] == "none"].groupby("noise_level")["data_hash"].first()
    assert by_level.nunique() == 2

    # replot só a partir dos arquivos persistidos (sem células, sem treino)
    shutil.rmtree(first / "figures")
    shutil.rmtree(first / "cells")
    figures = replot(first)
    assert figures and all(p.is_file() for p in figures)
    assert any("noise_zero_mean" in p.name for p in figures)

    # repetir com a mesma config -> artefatos metodológicos idênticos
    second = _run(cfg, tmp_path / "b")
    a = methodological_view(results)
    b = methodological_view(pd.read_parquet(second / "results.parquet"))
    pd.testing.assert_frame_equal(a, b)
    ha = pd.read_parquet(first / "history.parquet").drop(columns=["run_id", "elapsed"])
    hb = pd.read_parquet(second / "history.parquet").drop(columns=["run_id", "elapsed"])
    pd.testing.assert_frame_equal(ha, hb)
