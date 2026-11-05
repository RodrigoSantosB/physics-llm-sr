"""Política de falhas do runner (parte de C:retomada).

Falha do LLM (``EvaluationFailedError``) marca a célula ``error`` com motivo (nunca
silenciada); smoke/pilot continuam e reportam; full + paper_replication aborta. Limite de
tempo por célula e adaptador indisponível seguem a mesma política.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.core_types import Mode, PhysicalContext
from physics_llm_sr.experiments import sr_registry
from physics_llm_sr.experiments.runner import ExperimentAbortedError, run_experiments
from physics_llm_sr.llm.interfaces import EquationEvaluation, EvaluationFailedError
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.symbolic.expression import Expr

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"


class FailingEvaluator:
    def evaluate(self, expression: Expr, ctx: PhysicalContext, variant: str) -> EquationEvaluation:
        raise EvaluationFailedError("resposta inválida após 2 tentativas (simulado)")

    def evaluate_many(
        self, expressions: Sequence[Expr], ctx: PhysicalContext, variant: str
    ) -> list[EquationEvaluation]:
        raise EvaluationFailedError("resposta inválida após 2 tentativas (simulado)")


@pytest.fixture(autouse=True)
def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))


def test_llm_failure_marks_cell_error_and_smoke_continues(tmp_path: Path) -> None:
    cfg = load_config(SMOKE)
    summary = run_experiments(
        cfg, run_dir=tmp_path / "run", evaluator_factory=lambda m: FailingEvaluator()
    )
    assert (summary.executed, summary.failed) == (4, 2)
    results = pd.read_parquet(tmp_path / "run" / "results.parquet")
    errors = results[results["status"] == "error"]
    assert set(errors["llm"]) == {"fake"}
    assert errors["error"].str.contains("EvaluationFailedError").all()
    assert (errors["stop_reason"] == "error").all()
    assert errors["mae"].isna().all()  # nenhuma nota/métrica substituta
    assert (results.loc[results["llm"] == "none", "status"] == "ok").all()
    report = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    assert report["n_error"] == 2 and len(report["errors"]) == 2

    # Retomada tenta de novo só as células com erro.
    ok = FakeEquationEvaluator.from_table({}, default=EquationEvaluation(1, 1, 1, "ok"))
    again = run_experiments(
        cfg, run_dir=tmp_path / "run", resume=True, evaluator_factory=lambda m: ok
    )
    assert (again.skipped, again.executed, again.failed) == (2, 2, 0)
    final = pd.read_parquet(tmp_path / "run" / "results.parquet")
    assert (final["status"] == "ok").all() and len(final) == 4


def test_full_paper_replication_aborts_on_llm_failure(tmp_path: Path) -> None:
    data = load_config(SMOKE).model_dump(mode="json")
    data["mode"] = "full"
    data["llm"]["backend"] = "transformers"  # avaliador injetado: nenhum modelo é carregado
    data["llm"]["models"][0]["revision"] = "0" * 40
    full = parse_config(data)
    assert full.mode is Mode.FULL
    with pytest.raises(ExperimentAbortedError, match="EvaluationFailedError"):
        run_experiments(
            full, run_dir=tmp_path / "run", evaluator_factory=lambda m: FailingEvaluator()
        )
    results = pd.read_parquet(tmp_path / "run" / "results.parquet")
    assert list(results["status"]).count("error") == 1  # parou na primeira falha


def test_cell_time_limit_is_reported(tmp_path: Path) -> None:
    cfg = load_config(SMOKE)
    limited = cfg.model_copy(
        update={"experiments": cfg.experiments.model_copy(update={"max_cell_seconds": 1e-9})}
    )
    summary = run_experiments(limited, run_dir=tmp_path / "run", only=["exp1"])
    assert summary.failed == summary.executed == 2
    results = pd.read_parquet(tmp_path / "run" / "results.parquet")
    assert (results["stop_reason"] == "time_limit").all()
    assert results["error"].str.contains("CellTimeLimitError").all()


def test_unavailable_adapter_is_a_clear_cell_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sr_registry.ADAPTERS, "gplearn", ("physics_llm_sr.sr.nao_existe", "Nada"))
    cfg = load_config(SMOKE)
    summary = run_experiments(cfg, run_dir=tmp_path / "run", only=["exp1"])
    assert summary.failed == 1
    results = pd.read_parquet(tmp_path / "run" / "results.parquet")
    err = results[results["sr"] == "gplearn"]["error"].iloc[0]
    assert "AdapterUnavailableError" in err and "indisponível" in err
