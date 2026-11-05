"""CLI ``physics-llm-sr run|report`` sobre o smoke (parte de C:runner)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_llm_sr.cli import EXIT_OK, EXIT_USAGE, main
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV

SMOKE = str(Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml")


def test_cli_run_resume_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    assert main(["run", "--config", SMOKE, "--only", "exp1", "--sr", "deap"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "executadas=1" in out and "mode=smoke" in out
    assert main(["run", "--config", SMOKE, "--resume", "--only", "exp1,exp2"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "puladas=1" in out and "executadas=3" in out
    run_dir = next((tmp_path / "results" / "smoke").iterdir())
    assert main(["report", str(run_dir)]) == EXIT_OK
    assert "EXECUÇÃO REDUZIDA" in capsys.readouterr().out


def test_cli_rejects_unknown_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    assert main(["run", "--config", SMOKE, "--only", "exp9"]) == EXIT_USAGE
    assert main(["run", "--config", SMOKE, "--only", "exp7"]) == EXIT_USAGE
