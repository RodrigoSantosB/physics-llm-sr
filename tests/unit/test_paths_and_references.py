"""Colab: raiz de projeto e diretório de referências configuráveis."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_llm_sr.config.paths import PROJECT_ROOT_ENV, project_root, resolve_project_path
from physics_llm_sr.config.schemas import LLMConfig
from physics_llm_sr.evaluation.paper_comparison import (
    REFERENCES_DIR_ENV,
    load_paper_results,
    resolve_references_dir,
)
from physics_llm_sr.llm.cache import CACHE_FILENAME, EvaluationCache
from physics_llm_sr.reproducibility.artifacts import create_run_dir

REPO_REFERENCES = Path(__file__).resolve().parents[2] / "references"


def test_project_root_env_then_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert project_root() == tmp_path.resolve()
    other = tmp_path / "drive"
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(other))
    assert resolve_project_path(".cache/llm") == other.resolve() / ".cache" / "llm"
    assert resolve_project_path(tmp_path / "abs") == tmp_path / "abs"


def test_cache_and_output_dirs_resolve_against_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cache = EvaluationCache.from_config(LLMConfig(cache_dir="rel/cache"))
    assert cache is not None and cache.path == tmp_path.resolve() / "rel/cache" / CACHE_FILENAME
    run = create_run_dir("results/x", "r1")
    assert run == tmp_path.resolve() / "results/x/r1" and run.is_dir()


def test_references_dir_explicit_env_and_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(REFERENCES_DIR_ENV, raising=False)
    assert resolve_references_dir(REPO_REFERENCES) == REPO_REFERENCES.resolve()
    monkeypatch.chdir(tmp_path)
    assert resolve_references_dir() == REPO_REFERENCES.resolve()  # raiz do repo
    monkeypatch.setenv(REFERENCES_DIR_ENV, str(REPO_REFERENCES))
    assert resolve_references_dir() == REPO_REFERENCES.resolve()
    monkeypatch.setenv(REFERENCES_DIR_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=REFERENCES_DIR_ENV):
        resolve_references_dir()
    assert len(load_paper_results(REPO_REFERENCES)) > 0
