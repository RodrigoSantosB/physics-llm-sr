"""C:piloto (T:piloto): a config do piloto, com o backend trocado para ``fake``, roda ponta a
ponta rápido nos DOIS braços e gera a tabela comparativa rotulada ``pilot``/reduzida.

O PySR fica fora da execução do teste (``algorithms``) para não depender de Julia; a matriz
completa (com PySR) é conferida em ``build_matrix``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.experiments.runner import build_matrix, run_experiments

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
FAST_SR = ["deap", "gplearn"]
REPS = 3


def _with_fake_backend(name: str) -> ExperimentConfig:
    data = load_config(CONFIGS / f"{name}.yaml").model_dump(mode="json")
    data["llm"]["backend"] = "fake"  # fixture: nenhum Ollama/transformers
    return parse_config(data)


@pytest.mark.parametrize(
    ("name", "track", "normalization"),
    [
        ("pilot", "paper_replication", "literal"),
        ("pilot_extended", "extended_analysis", "normalized"),
    ],
)
def test_pilot_arm_runs_end_to_end_with_fake_backend(
    name: str, track: str, normalization: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cfg = _with_fake_backend(name)
    planned = build_matrix(cfg)
    # free_fall x {deap, gplearn, pysr} x mistral/E x 3 repetições, + baseline pareado
    assert len(planned) == 3 * REPS * 2
    assert {p.cell.sr for p in planned} == {"deap", "gplearn", "pysr"}
    assert {(p.cell.llm, p.cell.prompt) for p in planned} == {("mistral", "E"), ("none", "none")}

    summary = run_experiments(cfg, algorithms=FAST_SR)
    assert summary.failed == 0 and summary.executed == len(FAST_SR) * REPS * 2
    run_dir = summary.run_dir
    assert run_dir.parent.name == Path(cfg.output_dir).name  # braços em diretórios separados

    results = pd.read_parquet(run_dir / "results.parquet")
    assert set(results["mode"]) == {"pilot"} and set(results["track"]) == {track}
    assert set(results["normalization"]) == {normalization}
    assert sorted(results["repetition"].unique()) == list(range(REPS))

    comparison = pd.read_csv(run_dir / "comparison.csv")
    assert set(comparison["mode"]) == {"pilot"} and set(comparison["track"]) == {track}
    assert comparison["reduced_run"].all()
    assert "numericamente_proxima" not in set(comparison["classe"])
    t5 = comparison[
        (comparison["table"] == 5)
        & (comparison["scenario"] == "free_fall")
        & (comparison["llm"].isin(["mistral", "none"]))
        & (comparison["sr"].isin(FAST_SR))
    ]
    assert not t5.empty and (t5["n"] == REPS).all()

    paired = pd.read_csv(run_dir / "paired_comparison.csv")
    assert set(paired["sr"]) == set(FAST_SR) and (paired["n"] == REPS).all()

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["reduced_run"] is True and report["complete_result"] is False
    assert "pilot" in report["label"]
