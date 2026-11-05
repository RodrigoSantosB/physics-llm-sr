"""C:colab (T:colab): plano de ambiente do Colab (dry-run) e configs do piloto Colab."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from physics_llm_sr import colab
from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.core_types import Mode, Track

REPO = Path(__file__).resolve().parents[2]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PROMPTS_ENV = "PHYSICS_LLM_SR_PROMPTS_DIR"
REFERENCES_ENV = "PHYSICS_LLM_SR_REFERENCES_DIR"


def _kinds(plan: colab.EnvironmentPlan) -> list[str]:
    return [a.kind for a in plan.actions]


def test_dry_run_colab_returns_plan_without_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    before = dict(os.environ)
    plan = colab.setup_environment(
        repo_dir=REPO, repo_url="", mount_drive=True, in_colab=True, dry_run=True
    )
    assert dict(os.environ) == before  # nada foi definido
    assert not plan.executed and plan.in_colab
    kinds = _kinds(plan)
    assert {"pip", "env", "drive"} <= set(kinds)
    pip = next(a for a in plan.actions if a.kind == "pip")
    assert pip.command is not None and "-e" in pip.command
    assert any(part.endswith("[hf]") for part in pip.command)
    env_vars = {a.name for a in plan.actions if a.kind == "env"}
    assert {PROJECT_ROOT_ENV, PROMPTS_ENV, REFERENCES_ENV} <= env_vars
    root = next(a for a in plan.actions if a.kind == "env" and a.name == PROJECT_ROOT_ENV)
    assert "drive" in str(root.value).lower()  # resultados/cache no Drive


def test_dry_run_without_drive_has_no_drive_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    plan = colab.setup_environment(repo_dir=REPO, mount_drive=False, in_colab=True, dry_run=True)
    assert "drive" not in _kinds(plan)
    assert "pip" in _kinds(plan)


def test_dry_run_clone_only_when_url_given() -> None:
    no_url = colab.setup_environment(repo_dir=REPO / "nao-existe", in_colab=True, dry_run=True)
    assert "clone" not in _kinds(no_url)
    assert "repo_missing" in [a.kind for a in no_url.actions]  # avisa em vez de clonar
    url = "https://example.invalid/repo.git"
    with_url = colab.setup_environment(
        repo_dir=REPO / "nao-existe", repo_url=url, in_colab=True, dry_run=True
    )
    clone = next(a for a in with_url.actions if a.kind == "clone")
    assert clone.command is not None and url in clone.command


def test_dry_run_local_does_not_pip_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    plan = colab.setup_environment(repo_dir=REPO, in_colab=False, dry_run=True)
    assert not plan.in_colab
    assert "pip" not in _kinds(plan) and "drive" not in _kinds(plan)


def test_hf_token_never_in_plan_and_warns_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_segredo_nao_imprimir")
    plan = colab.setup_environment(repo_dir=REPO, in_colab=True, dry_run=True)
    assert "hf_segredo_nao_imprimir" not in repr(plan)
    text = " ".join(a.description for a in plan.actions)
    assert "HF_TOKEN" in text and "Llama" in text
    assert "Julia" in text  # PySR instala Julia na primeira importação


def test_real_run_sets_environment(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    plan = colab.setup_environment(repo_dir=REPO, in_colab=False, dry_run=False, environ=env)
    assert plan.executed
    assert Path(env[PROJECT_ROOT_ENV]) == REPO
    assert Path(env[PROMPTS_ENV]) == REPO / "prompts"
    assert Path(env[REFERENCES_ENV]) == REPO / "references"
    mine = {PROJECT_ROOT_ENV: str(tmp_path)}
    colab.setup_environment(repo_dir=REPO, in_colab=False, dry_run=False, environ=mine)
    assert mine[PROJECT_ROOT_ENV] == str(tmp_path)  # valor do usuário é respeitado


def test_find_repo_root_from_notebooks_dir() -> None:
    assert colab.find_repo_root(REPO / "notebooks") == REPO


@pytest.mark.parametrize("name", ["colab_pilot", "colab_pilot_extended"])
def test_colab_pilot_configs_are_valid_transformers_pinned(name: str) -> None:
    cfg = load_config(REPO / "configs" / f"{name}.yaml")
    assert cfg.mode is Mode.PILOT
    assert cfg.llm.backend == "transformers"
    assert cfg.llm.models
    for model in cfg.llm.models:
        assert model.revision and SHA_RE.match(model.revision), model
        assert model.quantization == "fp16"


def test_colab_pilot_arms() -> None:
    assert load_config(REPO / "configs" / "colab_pilot.yaml").track is Track.PAPER_REPLICATION
    ext = load_config(REPO / "configs" / "colab_pilot_extended.yaml")
    assert ext.track is Track.EXTENDED_ANALYSIS and ext.loss.normalization == "normalized"


@pytest.mark.parametrize("profile", colab.PROFILES)
@pytest.mark.parametrize("arm", colab.ARMS)
def test_notebook_config_every_profile_and_arm(profile: str, arm: str) -> None:
    cfg = colab.notebook_config(profile, arm, repo_dir=REPO)
    if arm == "replication":
        assert cfg.track is Track.PAPER_REPLICATION and cfg.loss.normalization == "literal"
    else:
        assert cfg.track is Track.EXTENDED_ANALYSIS and cfg.loss.normalization == "normalized"


def test_notebook_config_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="RUN_PROFILE"):
        colab.notebook_config("full", "replication", repo_dir=REPO)
    with pytest.raises(ValueError, match="ARM"):
        colab.notebook_config("smoke", "outro", repo_dir=REPO)


def test_resolve_stage_existing_run_uses_stored_config_and_does_not_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cfg = colab.notebook_config("smoke", "replication", repo_dir=REPO)
    run_dir = tmp_path / "meu_run"
    summary_dir, summary = colab.run_notebook_stage(cfg, ["exp1"], run_dir, execute=False)
    assert summary is None and summary_dir == run_dir and not run_dir.exists()
    loaded_cfg, path = colab.resolve_stage("pilot", "extended", "")
    assert loaded_cfg.output_dir == "results/pilot_extended"
    assert path == tmp_path / "results/pilot_extended" / colab.NOTEBOOK_RUN_NAME
    assert colab.load_results(run_dir).empty


def test_build_evaluator_fake_backend() -> None:
    cfg = colab.notebook_config("smoke", "replication", repo_dir=REPO)
    evaluator = colab.build_evaluator(cfg)
    assert evaluator.backend.model_metadata()["checkpoint"] == "fake-evaluator"
