"""C:retomada (T:retomada): interrupção após k células, retomada com ``resume``.

Células concluídas não são reexecutadas (contador) e os resultados metodológicos são
idênticos aos de uma execução contínua.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.experiments.cells import Cell
from physics_llm_sr.experiments.execution import CellOutcome
from physics_llm_sr.experiments.runner import (
    find_resumable_run,
    methodological_view,
    run_experiments,
)

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
K = 2


class Interrupted(Exception):
    pass


def test_resume_skips_completed_cells_and_matches_continuous_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cfg = load_config(SMOKE)
    continuous = run_experiments(cfg, run_dir=tmp_path / "continuous")
    assert continuous.executed == 4

    first: list[str] = []

    def crash_after_k(cell: Cell, outcome: CellOutcome) -> None:
        first.append(cell.key())
        if len(first) == K:
            raise Interrupted

    interrupted = tmp_path / "interrupted"
    with pytest.raises(Interrupted):
        run_experiments(cfg, run_dir=interrupted, on_cell_done=crash_after_k)
    assert len(list((interrupted / "cells").glob("*/result.json"))) == K
    assert not (interrupted / "results.parquet").exists()

    second: list[str] = []
    summary = run_experiments(
        cfg,
        run_dir=interrupted,
        resume=True,
        on_cell_done=lambda cell, _: second.append(cell.key()),
    )
    assert summary.skipped == K
    assert summary.executed == 4 - K == len(second)
    assert not set(first) & set(second)

    a = methodological_view(pd.read_parquet(continuous.run_dir / "results.parquet"))
    b = methodological_view(pd.read_parquet(interrupted / "results.parquet"))
    pd.testing.assert_frame_equal(a, b)
    ha = pd.read_parquet(continuous.run_dir / "history.parquet").drop(
        columns=["run_id", "elapsed", "n_llm_calls", "n_cache_hits"]
    )
    hb = pd.read_parquet(interrupted / "history.parquet").drop(
        columns=["run_id", "elapsed", "n_llm_calls", "n_cache_hits"]
    )
    pd.testing.assert_frame_equal(ha, hb)

    # Retomar de novo: nada a fazer.
    again = run_experiments(cfg, run_dir=interrupted, resume=True)
    assert (again.executed, again.skipped) == (0, 4)


def test_resume_without_run_dir_finds_latest_run_with_same_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cfg = load_config(SMOKE)
    with pytest.raises(Interrupted):
        run_experiments(cfg, on_cell_done=lambda c, o: (_ for _ in ()).throw(Interrupted()))
    found = find_resumable_run(cfg)
    assert found is not None
    summary = run_experiments(cfg, resume=True)
    assert summary.run_dir == found
    assert (summary.executed, summary.skipped) == (3, 1)


def test_resume_refuses_different_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cfg = load_config(SMOKE)
    run_experiments(cfg, run_dir=tmp_path / "run", only=["exp1"], write_report=False)
    other = cfg.model_copy(update={"seeds": cfg.seeds.model_copy(update={"master": 7})})
    with pytest.raises(ValueError, match="retomada exige a mesma config"):
        run_experiments(other, run_dir=tmp_path / "run", resume=True)
