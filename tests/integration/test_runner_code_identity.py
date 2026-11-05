"""Identidade do código na execução: ``run.json`` grava git_sha/git_dirty e a retomada com
código diferente é recusada, salvo ``allow_code_change`` (registrado e avisado no report).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from physics_llm_sr.cli import EXIT_OK, EXIT_USAGE, main
from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.experiments import runner
from physics_llm_sr.experiments.runner import CodeChangedError, run_experiments

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"


def _identity(sha: str, dirty: bool = False) -> dict[str, Any]:
    return {"git_sha": sha, "git_dirty": dirty, "git_diff_hash": None}


@pytest.fixture(autouse=True)
def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))


def _read(run_dir: Path, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / name).read_text(encoding="utf-8"))
    return data


def test_run_json_records_code_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "code_identity", lambda: _identity("a" * 40, True))
    run_experiments(
        load_config(SMOKE), run_dir=tmp_path / "run", only=["exp1"], algorithms=["deap"]
    )
    info = _read(tmp_path / "run", "run.json")
    assert info["git_sha"] == "a" * 40 and info["git_dirty"] is True


def test_resume_with_different_code_is_refused_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(SMOKE)
    run = tmp_path / "run"
    monkeypatch.setattr(runner, "code_identity", lambda: _identity("a" * 40))
    run_experiments(cfg, run_dir=run, only=["exp1"], algorithms=["deap"])
    monkeypatch.setattr(runner, "code_identity", lambda: _identity("b" * 40))
    with pytest.raises(CodeChangedError, match="--allow-code-change"):
        run_experiments(cfg, run_dir=run, resume=True)
    # mesma identidade: retomada normal
    monkeypatch.setattr(runner, "code_identity", lambda: _identity("a" * 40))
    run_experiments(cfg, run_dir=run, resume=True, only=["exp1"], algorithms=["deap"])
    # código diferente com permissão explícita: registrado e avisado
    monkeypatch.setattr(runner, "code_identity", lambda: _identity("b" * 40))
    run_experiments(cfg, run_dir=run, resume=True, allow_code_change=True)
    info = _read(run, "run.json")
    assert info["git_sha"] == "a" * 40  # identidade da criação preservada
    assert [c["to"]["git_sha"] for c in info["code_changes"]] == ["b" * 40]
    report = _read(run, "report.json")
    assert any("código" in w for w in report["warnings"])


def test_cli_allow_code_change_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "code_identity", lambda: _identity("a" * 40))
    base = ["run", "--config", str(SMOKE), "--only", "exp1", "--sr", "deap"]
    assert main(base) == EXIT_OK
    monkeypatch.setattr(runner, "code_identity", lambda: _identity("c" * 40))
    assert main([*base, "--resume"]) == EXIT_USAGE
    assert main([*base, "--resume", "--allow-code-change"]) == EXIT_OK


def test_cli_relative_resume_resolved_against_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    base = ["run", "--config", str(SMOKE), "--only", "exp1", "--sr", "deap"]
    assert main(base) == EXIT_OK
    run_dir = next((tmp_path / "results" / "smoke").iterdir())
    relative = str(run_dir.relative_to(tmp_path))
    assert main([*base, "--resume", relative]) == EXIT_OK
    assert not (elsewhere / relative).exists()  # não criou diretório relativo ao cwd
    assert _read(run_dir, "run.json")["last_invocation"]["resume"] is True
