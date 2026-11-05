"""Apoio ao notebook principal: preparo de ambiente (local/Colab) e escolha de configs.

Sem lógica científica. :func:`setup_environment` monta (e, fora do ``dry_run``, executa) o
plano de preparo: instalar o pacote no Colab, montar o Drive, definir as variáveis de
ambiente que fazem resultados e cache persistirem. :func:`notebook_config` escolhe a
config por ``RUN_PROFILE`` e ``ARM``; :func:`run_notebook_stage` chama o runner com
retomada num diretório de execução fixo por perfil/braço (cada etapa recarrega o que já
foi calculado dos arquivos).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV, resolve_project_path
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.reproducibility.manifests import (
    TRACKED_LIBRARIES,
    julia_version,
    library_versions,
)

if TYPE_CHECKING:  # imports pesados ficam adiados: o módulo é importável antes do pip install
    from physics_llm_sr.experiments.runner import RunSummary
    from physics_llm_sr.llm.scoring import LLMEquationEvaluator

RESULTS_FILE = "results.parquet"
PROMPTS_DIR_ENV = "PHYSICS_LLM_SR_PROMPTS_DIR"
REFERENCES_DIR_ENV = "PHYSICS_LLM_SR_REFERENCES_DIR"
REPO_DIR_ENV = "PHYSICS_LLM_SR_REPO_DIR"
HF_TOKEN_ENV = "HF_TOKEN"
HF_SECRET_NAME = "HF_TOKEN"
COLAB_MODULE = "google.colab"
DEFAULT_COLAB_REPO_DIR = "/content/physics-llm-sr"
DEFAULT_DRIVE_MOUNT = "/content/drive"
DEFAULT_DRIVE_PROJECT = "/content/drive/MyDrive/physics-llm-sr"
DEFAULT_COLAB_WORK_DIR = "/content/physics-llm-sr-work"
DEFAULT_EXTRAS = "hf"
NOTEBOOK_RUN_NAME = "notebook_run"

PROFILES = ("smoke", "pilot", "colab_pilot", "replication")
ARMS = ("replication", "extended")
#: (perfil, braço) -> arquivo em ``configs/``.
CONFIG_FILES: dict[tuple[str, str], str] = {
    ("smoke", "replication"): "smoke.yaml",
    ("smoke", "extended"): "smoke.yaml",  # derivado (track extended + perda normalizada)
    ("pilot", "replication"): "pilot.yaml",
    ("pilot", "extended"): "pilot_extended.yaml",
    ("colab_pilot", "replication"): "colab_pilot.yaml",
    ("colab_pilot", "extended"): "colab_pilot_extended.yaml",
    ("replication", "replication"): "replication.yaml",
    ("replication", "extended"): "extended.yaml",
}
#: Experimentos sempre permitidos nas etapas 07/11 (o par baseline x LLM do piloto).
CORE_EXPERIMENTS = ("baseline", "llm_integration")
#: Redução do smoke para as etapas 12-14 caberem em poucos minutos.
SMOKE_PROMPTS = ["A", "D", "E"]
SMOKE_NOISE_LEVELS = [0.01, 0.05]
SMOKE_NOISE_LOCATIONS = ["target"]
SMOKE_OUTPUT_DIR = "results/notebook_smoke"
ALL_EXPERIMENTS = [
    "baseline",
    "llm_integration",
    "prompt_sensitivity",
    "noise_robustness",
    "biased_noise",
]


@dataclass(frozen=True)
class SetupAction:
    """Uma ação do plano de preparo (``kind``: pip, clone, drive, env, info, warning...)."""

    kind: str
    description: str
    command: tuple[str, ...] | None = None
    name: str | None = None
    value: str | None = None


@dataclass
class EnvironmentPlan:
    """Plano de preparo do ambiente; ``executed`` é falso no ``dry_run``."""

    in_colab: bool
    repo_dir: Path
    actions: list[SetupAction] = field(default_factory=list)
    executed: bool = False


def detect_colab() -> bool:
    """Verdadeiro se o código roda no Google Colab."""
    try:
        return importlib.util.find_spec(COLAB_MODULE) is not None
    except ModuleNotFoundError:  # pacote pai ``google`` ausente
        return False


def find_repo_root(start: Path | None = None) -> Path:
    """Raiz do repositório: ancestral de ``start`` (default cwd) com pyproject e configs/."""
    if start is None and os.environ.get(REPO_DIR_ENV):
        return Path(os.environ[REPO_DIR_ENV]).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError(f"raiz do repositório não encontrada a partir de {here}")


def _plan_colab(
    plan: EnvironmentPlan, repo_url: str, mount_drive: bool, drive_dir: str, extras: str
) -> str:
    """Acrescenta as ações do Colab e devolve a raiz de projeto escolhida."""
    repo = plan.repo_dir
    if not repo.is_dir():
        if repo_url:
            plan.actions.append(
                SetupAction(
                    "clone",
                    f"clonar {repo_url} em {repo}",
                    ("git", "clone", repo_url, str(repo)),
                )
            )
        else:
            plan.actions.append(
                SetupAction(
                    "repo_missing",
                    f"{repo} não existe e REPO_URL está vazio: suba o repositório ao Drive "
                    "(ou ao /content) e ajuste REPO_DIR",
                )
            )
    spec = f"{repo}[{extras}]" if extras else str(repo)
    plan.actions.append(
        SetupAction(
            "pip",
            f"instalar o pacote em modo editável com o extra [{extras}]",
            (sys.executable, "-m", "pip", "install", "-q", "-e", spec),
        )
    )
    root = DEFAULT_COLAB_WORK_DIR
    if mount_drive:
        root = drive_dir
        plan.actions.append(
            SetupAction("drive", f"montar o Google Drive em {DEFAULT_DRIVE_MOUNT}", name="mount")
        )
    plan.actions.append(
        SetupAction(
            "warning",
            "Llama 2 (meta-llama/Llama-2-7b-chat-hf) é um repositório restrito: aceite a licença "
            "no Hugging Face e crie o segredo HF_TOKEN no Colab (ícone de chave). "
            "O token nunca é impresso.",
        )
    )
    plan.actions.append(
        SetupAction(
            "warning",
            "PySR: a primeira importação instala o Julia e as dependências (vários minutos).",
        )
    )
    return root


def setup_environment(
    *,
    repo_dir: str | Path | None = None,
    repo_url: str = "",
    mount_drive: bool = False,
    drive_dir: str = DEFAULT_DRIVE_PROJECT,
    in_colab: bool | None = None,
    dry_run: bool = False,
    extras: str = DEFAULT_EXTRAS,
    environ: MutableMapping[str, str] | None = None,
) -> EnvironmentPlan:
    """Prepara o ambiente (ou, com ``dry_run=True``, só devolve o plano).

    No Colab: clona (se ``repo_url``), ``pip install -e <repo>[hf]``, monta o Drive
    (opcional) e define ``PHYSICS_LLM_SR_PROJECT_ROOT`` (Drive: resultados e cache
    persistem), ``PHYSICS_LLM_SR_PROMPTS_DIR`` e ``PHYSICS_LLM_SR_REFERENCES_DIR``.
    Localmente: só as variáveis de ambiente (a raiz do projeto = raiz do repositório).
    Variáveis já definidas pelo usuário são respeitadas.
    """
    env: MutableMapping[str, str] = os.environ if environ is None else environ
    colab = detect_colab() if in_colab is None else in_colab
    if repo_dir is not None:
        repo = Path(repo_dir)
    else:
        repo = Path(DEFAULT_COLAB_REPO_DIR) if colab else find_repo_root()
    plan = EnvironmentPlan(in_colab=colab, repo_dir=repo)
    root = str(repo)
    if colab:
        root = _plan_colab(plan, repo_url, mount_drive, drive_dir, extras)
    wanted = {
        PROJECT_ROOT_ENV: root,
        REPO_DIR_ENV: str(repo),
        PROMPTS_DIR_ENV: str(repo / "prompts"),
        REFERENCES_DIR_ENV: str(repo / "references"),
    }
    for name, value in wanted.items():
        if name in env:
            plan.actions.append(
                SetupAction("env_kept", f"{name} já definida; mantida", name=name, value=env[name])
            )
        else:
            plan.actions.append(
                SetupAction("env", f"definir {name}", name=name, value=value),
            )
    if not dry_run:
        _execute(plan, env)
        plan.executed = True
    return plan


def _execute(plan: EnvironmentPlan, env: MutableMapping[str, str]) -> None:
    for action in plan.actions:
        if action.kind in {"clone", "pip"} and action.command:
            subprocess.run(action.command, check=True)
        elif action.kind == "drive":
            from google.colab import drive  # type: ignore[import-not-found,unused-ignore]

            drive.mount(DEFAULT_DRIVE_MOUNT)
        elif action.kind == "env" and action.name and action.value is not None:
            env[action.name] = action.value
            if action.name == PROJECT_ROOT_ENV:
                Path(action.value).mkdir(parents=True, exist_ok=True)


def login_huggingface() -> bool:
    """Login no Hugging Face com ``HF_TOKEN`` (segredo do Colab ou env); não imprime o token."""
    token = os.environ.get(HF_TOKEN_ENV)
    if not token and detect_colab():
        try:
            from google.colab import userdata  # type: ignore[import-not-found,unused-ignore]

            token = userdata.get(HF_SECRET_NAME)
        except Exception:  # segredo ausente ou sem acesso: o chamador avisa
            token = None
    if not token:
        return False
    from huggingface_hub import login  # type: ignore[import-not-found,unused-ignore]

    login(token=token, add_to_git_credential=False)
    return True


def environment_report() -> dict[str, Any]:
    """Versões e hardware para a etapa 01 (sem segredos)."""
    from physics_llm_sr.reproducibility.manifests import hardware_info

    versions = library_versions(TRACKED_LIBRARIES)
    versions["physics-llm-sr"] = _version("physics-llm-sr")
    cuda: bool | None = None
    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda = bool(torch.cuda.is_available())
    return {
        "cuda_disponível": cuda,
        "python": sys.version.split()[0],
        "libraries": versions,
        "julia": julia_version(),
        "hardware": hardware_info(),
        "colab": detect_colab(),
        "hf_token_present": bool(os.environ.get(HF_TOKEN_ENV)),
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# ------------------------------------------------------------------ configs do notebook


def notebook_config(profile: str, arm: str, repo_dir: str | Path | None = None) -> ExperimentConfig:
    """Config de ``RUN_PROFILE`` x ``ARM``.

    ``smoke`` roda os cinco experimentos em escala mínima (avaliador simulado) e o braço
    ``extended`` é derivado da config smoke (track extended, perda normalizada).
    """
    if profile not in PROFILES:
        raise ValueError(f"RUN_PROFILE inválido: {profile!r}; válidos: {PROFILES}")
    if arm not in ARMS:
        raise ValueError(f"ARM inválido: {arm!r}; válidos: {ARMS}")
    repo = Path(repo_dir) if repo_dir is not None else find_repo_root()
    cfg = load_config(repo / "configs" / CONFIG_FILES[(profile, arm)])
    if profile != "smoke":
        return cfg
    data = cfg.model_dump(mode="json")
    data["experiments"]["run"] = list(ALL_EXPERIMENTS)
    data["llm"]["prompt_variants"] = list(SMOKE_PROMPTS)
    for exp in ("exp4", "exp5"):
        data["data"][exp]["levels"] = list(SMOKE_NOISE_LEVELS)
        data["data"][exp]["locations"] = list(SMOKE_NOISE_LOCATIONS)
    data["output_dir"] = f"{SMOKE_OUTPUT_DIR}/{arm}"
    if arm == "extended":
        data["track"] = "extended_analysis"
        data["loss"]["normalization"] = "normalized"
    return parse_config(data)


def notebook_run_dir(cfg: ExperimentConfig) -> Path:
    """Diretório fixo da execução do notebook (``output_dir/notebook_run``)."""
    return resolve_project_path(cfg.output_dir) / NOTEBOOK_RUN_NAME


def stage_allowed(cfg: ExperimentConfig, experiments: Iterable[str]) -> list[str]:
    """Experimentos pedidos que a config prevê (baseline e llm_integration sempre)."""
    from physics_llm_sr.experiments.runner import resolve_experiments

    wanted = resolve_experiments(list(experiments), cfg)
    allowed = set(cfg.experiments.run) | set(CORE_EXPERIMENTS)
    return [name for name in wanted if name in allowed]


def resolve_stage(
    profile: str, arm: str, existing_run: str | Path = ""
) -> tuple[ExperimentConfig, Path]:
    """Config e diretório de execução de uma etapa.

    Com ``existing_run`` (um ``run_dir`` já gerado, p.ex. do piloto) usa a config gravada
    nele e não executa nada; senão, config de ``profile``/``arm`` e o diretório fixo do
    notebook. Garante a raiz de projeto (``PHYSICS_LLM_SR_PROJECT_ROOT``) se ainda não houver.
    """
    if str(existing_run):
        from physics_llm_sr.experiments.reporting import load_run_config

        path = Path(existing_run).expanduser().resolve()
        return load_run_config(path), path
    os.environ.setdefault(PROJECT_ROOT_ENV, str(find_repo_root()))
    cfg = notebook_config(profile, arm)
    return cfg, notebook_run_dir(cfg)


def load_results(run_dir: str | Path) -> pd.DataFrame:
    """``results.parquet`` consolidado da execução (DataFrame vazio se ainda não existe)."""
    path = Path(run_dir) / RESULTS_FILE
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def build_evaluator(cfg: ExperimentConfig, model: str | None = None) -> LLMEquationEvaluator:
    """Avaliador LLM da config (mesma construção do runner); ``model`` default = o primeiro."""
    from physics_llm_sr.llm.inference import build_backend
    from physics_llm_sr.llm.scoring import LLMEquationEvaluator

    name = model or (cfg.experiments.models or [m.name for m in cfg.llm.models])[0]
    return LLMEquationEvaluator.from_config(cfg.llm, build_backend(cfg.llm, name, track=cfg.track))


def run_notebook_stage(
    cfg: ExperimentConfig,
    experiments: Iterable[str],
    run_dir: str | Path | None = None,
    execute: bool = True,
    **kwargs: Any,
) -> tuple[Path, RunSummary | None]:
    """Roda (com retomada) os ``experiments`` da etapa em ``run_dir`` e devolve o diretório.

    Células já concluídas são recarregadas, não recalculadas. Experimentos fora da config
    (p.ex. Exp3-5 num piloto) são ignorados: devolve ``summary=None`` e o diretório.
    """
    path = Path(run_dir) if run_dir is not None else notebook_run_dir(cfg)
    only = stage_allowed(cfg, experiments)
    if not only or not execute:
        return path, None
    from physics_llm_sr.experiments.reporting import consolidate
    from physics_llm_sr.experiments.runner import run_experiments

    # O relatório completo (comparação com o artigo, figuras) custa dezenas de segundos:
    # só a etapa 16 o gera; as demais apenas consolidam os resultados em results.parquet.
    kwargs.setdefault("write_report", False)
    summary = run_experiments(cfg, run_dir=path, resume=True, only=only, **kwargs)
    consolidate(summary.run_dir)
    return summary.run_dir, summary


def profile_summary(cfg: ExperimentConfig) -> Mapping[str, Any]:
    """Resumo legível da config para exibição."""
    reduced = cfg.mode is not Mode.FULL or cfg.track is not Track.PAPER_REPLICATION
    return {
        "reduzida (não é reprodução completa)": reduced,
        "mode": cfg.mode.value,
        "track": cfg.track.value,
        "normalization": cfg.loss.normalization,
        "backend": cfg.llm.backend,
        "models": [m.name for m in cfg.llm.models],
        "algorithms": cfg.experiments.algorithms,
        "scenarios": cfg.experiments.scenarios,
        "repetitions": cfg.experiments.repetitions,
        "population": cfg.sr.population,
        "generations": cfg.sr.generations,
        "output_dir": cfg.output_dir,
    }
