"""``complete_result`` (report.json) só para uma execução completa de reprodução.

Exige TODAS as condições: ``mode=full``, ``track=paper_replication``, zero erros, execução
não abortada e matriz executada (células ``ok``) == matriz planejada completa da config
(sem ``--only``/``--sr`` filtrando). Cada teste quebra exatamente uma condição.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.experiments.reporting import report
from physics_llm_sr.experiments.runner import run_experiments
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
OK = EquationEvaluation(0.5, 0.5, 0.5, "ok")


@pytest.fixture(autouse=True)
def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))


def _cfg(mode: str = "full", track: str = "paper_replication") -> ExperimentConfig:
    data = load_config(SMOKE).model_dump(mode="json")
    data["mode"] = mode
    data["track"] = track
    data["experiments"]["algorithms"] = ["deap"]
    if mode == "full":
        data["llm"]["backend"] = "transformers"  # avaliador injetado: nenhum modelo carregado
        data["llm"]["models"][0]["revision"] = "0" * 40
    return parse_config(data)


def _run(cfg: ExperimentConfig, run_dir: Path, **kw: Any) -> dict[str, Any]:
    fake = FakeEquationEvaluator.from_table({}, default=OK)
    run_experiments(cfg, run_dir=run_dir, evaluator_factory=lambda m: fake, **kw)
    info: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return info


def test_full_paper_replication_complete_matrix_is_complete(tmp_path: Path) -> None:
    info = _run(_cfg(), tmp_path / "run")
    assert info["complete_result"] is True
    assert info["incomplete_reasons"] == []


def test_smoke_is_never_complete(tmp_path: Path) -> None:
    info = _run(_cfg(mode="smoke"), tmp_path / "run")
    assert info["complete_result"] is False


def test_extended_track_is_never_complete(tmp_path: Path) -> None:
    info = _run(_cfg(track="extended_analysis"), tmp_path / "run")
    assert info["complete_result"] is False


def test_filtered_matrix_with_only_is_not_complete(tmp_path: Path) -> None:
    info = _run(_cfg(), tmp_path / "run", only=["exp1"])
    assert info["complete_result"] is False
    assert any("matriz" in r for r in info["incomplete_reasons"])


def test_filtered_matrix_with_sr_is_not_complete(tmp_path: Path) -> None:
    data = _cfg().model_dump(mode="json")
    data["experiments"]["algorithms"] = ["deap", "gplearn"]
    info = _run(parse_config(data), tmp_path / "run", algorithms=["deap"])
    assert info["complete_result"] is False
    assert any("matriz" in r for r in info["incomplete_reasons"])


def test_cell_error_makes_run_incomplete(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _run(_cfg(), run)
    result = next((run / "cells").glob("*/result.json"))
    row = json.loads(result.read_text(encoding="utf-8"))
    row.update(status="error", error="falha simulada")
    result.write_text(json.dumps(row), encoding="utf-8")
    info = report(run)
    assert info["complete_result"] is False
    assert any("erro" in r for r in info["incomplete_reasons"])


def test_aborted_run_is_not_complete(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _run(_cfg(), run)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    meta["status"] = "aborted"
    (run / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    info = report(run)
    assert info["complete_result"] is False
    assert any("abortada" in r for r in info["incomplete_reasons"])
