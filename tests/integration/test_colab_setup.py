"""C:colab (T:colab): plano de ambiente do Colab (dry-run) e configs do piloto Colab."""

from __future__ import annotations

import os
import re
import subprocess
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


def _ordered(plan: colab.EnvironmentPlan) -> list[str]:
    """Tipos das ações sem avisos e sem a repetição de ``env``."""
    out: list[str] = []
    for kind in _kinds(plan):
        if kind in {"warning", "env_kept"}:
            continue
        if not out or out[-1] != kind:
            out.append(kind)
    return out


def test_dry_run_colab_order_is_drive_clone_check_pip_env() -> None:
    plan = colab.setup_environment(
        repo_dir="/content/drive/MyDrive/physics-llm-sr",
        repo_url="https://example.invalid/repo.git",
        mount_drive=True,
        in_colab=True,
        dry_run=True,
    )
    assert _ordered(plan) == ["drive", "clone", "check_repo", "pip", "env"]


def test_dry_run_colab_order_without_drive_or_url() -> None:
    plan = colab.setup_environment(repo_dir=REPO, in_colab=True, dry_run=True)
    assert _ordered(plan) == ["check_repo", "pip", "env"]
    assert plan.blocking_errors == []  # o repositório existe


def test_dry_run_clone_only_when_url_given() -> None:
    no_url = colab.setup_environment(repo_dir=REPO / "nao-existe", in_colab=True, dry_run=True)
    assert "clone" not in _kinds(no_url)
    assert "repo_missing" not in _kinds(no_url)  # não é mais um aviso: é erro bloqueante
    url = "https://example.invalid/repo.git"
    with_url = colab.setup_environment(
        repo_dir=REPO / "nao-existe", repo_url=url, in_colab=True, dry_run=True
    )
    clone = next(a for a in with_url.actions if a.kind == "clone")
    assert clone.command is not None and url in clone.command
    assert with_url.blocking_errors == []  # o clone vai criar o diretório


def test_missing_repo_is_blocking_with_actionable_message() -> None:
    plan = colab.setup_environment(repo_dir=REPO / "nao-existe", in_colab=True, dry_run=True)
    assert len(plan.blocking_errors) == 1
    msg = plan.blocking_errors[0]
    assert str(REPO / "nao-existe") in msg and "REPO_URL" in msg and "Drive" in msg
    with pytest.raises(colab.RepoNotFoundError, match="REPO_URL"):
        colab.setup_environment(repo_dir=REPO / "nao-existe", in_colab=True, dry_run=False)


def test_missing_repo_is_checked_after_drive_mount_not_before() -> None:
    """Com o Drive a montar, o plano não declara o repositório ausente (ainda não dá para saber)."""
    plan = colab.setup_environment(
        repo_dir="/content/drive/MyDrive/nao-existe", mount_drive=True, in_colab=True, dry_run=True
    )
    assert plan.blocking_errors == []
    assert _ordered(plan)[:2] == ["drive", "check_repo"]


def test_execute_stops_before_pip_when_repo_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        calls.append(tuple(cmd))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(colab.RepoNotFoundError):
        colab.setup_environment(
            repo_dir=tmp_path / "vazio", in_colab=True, dry_run=False, environ={}
        )
    assert calls == []  # nenhum pip em caminho inexistente


def test_hf_home_defaults_into_drive_project_when_mounted() -> None:
    plan = colab.setup_environment(
        repo_dir=REPO, mount_drive=True, drive_dir="/d/proj", in_colab=True, dry_run=True
    )
    hf = next(a for a in plan.actions if a.kind == "env" and a.name == "HF_HOME")
    assert hf.value == "/d/proj/hf_home"
    custom = colab.setup_environment(
        repo_dir=REPO, mount_drive=True, hf_home="/x/hf", in_colab=True, dry_run=True
    )
    assert next(a.value for a in custom.actions if a.name == "HF_HOME") == "/x/hf"
    plain = colab.setup_environment(repo_dir=REPO, in_colab=True, dry_run=True)
    assert all(a.name != "HF_HOME" for a in plain.actions)


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


def test_detect_colab_is_false_without_google_package() -> None:
    assert colab.detect_colab() is False


def test_clone_is_shallow_on_branch_and_pulls_when_repo_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = colab.DEFAULT_REPO_URL
    assert url == "https://github.com/RodrigoSantosB/physics-llm-sr.git"
    plan = colab.setup_environment(
        repo_dir="/content/physics-llm-sr", repo_url=url, in_colab=True, dry_run=True
    )
    clone = next(a for a in plan.actions if a.kind == "clone")
    assert clone.command is not None
    assert clone.command[:6] == ("git", "clone", "--depth", "1", "-b", "main")
    # repositório já existente com .git: git pull --ff-only (sem clonar)
    (tmp_path / ".git").mkdir()
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **_: calls.append(tuple(cmd)))
    existing = colab.EnvironmentPlan(in_colab=True, repo_dir=tmp_path)
    existing.actions.append(colab.SetupAction("clone", "x", ("git", "clone", url, str(tmp_path))))
    colab._execute(existing, {})
    assert calls == [("git", "-C", str(tmp_path), "pull", "--ff-only")]
