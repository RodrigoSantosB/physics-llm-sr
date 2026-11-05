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
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
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
GATED_PREFIX = "meta-llama/"  # repositórios restritos (Llama 2)
HF_HOME_ENV = "HF_HOME"
DEFAULT_COLAB_HF_HOME = "/content/hf_home"  # disco local: o Drive trunca arquivos de ~14,5 GB
HF_MODULES = ("transformers", "huggingface_hub")
PYPROJECT = "pyproject.toml"
COLAB_MODULE = "google.colab"
DEFAULT_COLAB_REPO_DIR = "/content/physics-llm-sr"
DEFAULT_DRIVE_MOUNT = "/content/drive"
DEFAULT_DRIVE_PROJECT = "/content/drive/MyDrive/physics-llm-sr-runs"  # só resultados/cache/HF_HOME
DEFAULT_REPO_URL = "https://github.com/RodrigoSantosB/physics-llm-sr.git"
DEFAULT_REPO_BRANCH = "main"
DEFAULT_COLAB_WORK_DIR = "/content/physics-llm-sr-work"
DEFAULT_EXTRAS = "hf"
NOTEBOOK_RUN_NAME = "notebook_run"

PROFILES = ("smoke", "pilot", "colab_pilot", "colab_hoje", "replication")
ARMS = ("replication", "extended")
#: (perfil, braço) -> arquivo em ``configs/``.
CONFIG_FILES: dict[tuple[str, str], str] = {
    ("smoke", "replication"): "smoke.yaml",
    ("smoke", "extended"): "smoke.yaml",  # derivado (track extended + perda normalizada)
    ("pilot", "replication"): "pilot.yaml",
    ("pilot", "extended"): "pilot_extended.yaml",
    ("colab_pilot", "replication"): "colab_pilot.yaml",
    ("colab_pilot", "extended"): "colab_pilot_extended.yaml",
    ("colab_hoje", "replication"): "colab_hoje_queda_livre_literal.yaml",  # controle: só P1
    ("colab_hoje", "extended"): "colab_hoje_queda_livre.yaml",  # principal
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


#: Etapas de prioridade da execução de hoje (``colab_hoje``) -> experimentos (aliases do runner).
HOJE_STAGES: dict[str, tuple[str, ...]] = {
    "P1": ("exp1", "exp2"),
    "P2": ("exp3",),
    "P3": ("exp4",),
    "P4": ("exp5",),
}
#: HIPÓTESES de chamadas LLM únicas por célula com LLM (a medir; cache reduz entre repetições).
DEFAULT_CALLS_PER_CELL: dict[str, int] = {"deap": 300, "gplearn": 1000, "pysr": 200}
#: Algoritmos cujo avaliador roda em lote (``llm.batch_size``); PySR chama em série (callback).
BATCHED_ALGORITHMS = frozenset({"deap", "gplearn"})
SECONDS_PER_HOUR = 3600.0


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
    #: Problemas já conhecidos no plano que impedem a execução (repositório ausente).
    blocking_errors: list[str] = field(default_factory=list)


class RepoNotFoundError(FileNotFoundError):
    """O repositório do projeto não existe no caminho esperado (erro bloqueante do Colab)."""


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


def _repo_missing_message(repo: Path) -> str:
    return (
        f"repositório não encontrado em {repo} (falta pyproject.toml). Coloque o código do projeto "
        "no Drive (pasta inteira com .git, ou `git bundle` + `git clone`; veja o README) e ajuste "
        "REPO_DIR, ou preencha REPO_URL para clonar, e confira MOUNT_DRIVE = True se o "
        "repositório está no Drive."
    )


def _plan_colab(
    plan: EnvironmentPlan,
    repo_url: str,
    mount_drive: bool,
    drive_dir: str,
    extras: str,
    hf_home: str,
    repo_branch: str = DEFAULT_REPO_BRANCH,
) -> tuple[str, str | None]:
    """Acrescenta as ações do Colab (drive, clone, check_repo, pip) e devolve (raiz, HF_HOME).

    A ordem importa: o repositório pode estar no Drive, então a verificação vem depois da
    montagem; sem repositório o plano termina em erro bloqueante antes do pip.
    """
    repo = plan.repo_dir
    root = DEFAULT_COLAB_WORK_DIR
    if mount_drive:
        root = drive_dir
        plan.actions.append(
            SetupAction("drive", f"montar o Google Drive em {DEFAULT_DRIVE_MOUNT}", name="mount")
        )
    if repo_url:
        plan.actions.append(
            SetupAction(
                "clone",
                f"clonar {repo_url} (branch {repo_branch}, --depth 1) em {repo}; "
                "se a pasta já existe, `git pull --ff-only`",
                ("git", "clone", "--depth", "1", "-b", repo_branch, repo_url, str(repo)),
                value=str(repo),
            )
        )
    plan.actions.append(
        SetupAction("check_repo", f"verificar que {repo} contém o repositório (bloqueante)")
    )
    if not mount_drive and not repo_url and not (repo / PYPROJECT).is_file():
        plan.blocking_errors.append(_repo_missing_message(repo))
    spec = f"{repo}[{extras}]" if extras else str(repo)
    plan.actions.append(
        SetupAction(
            "pip",
            f"instalar o pacote em modo editável com o extra [{extras}]",
            (sys.executable, "-m", "pip", "install", "-q", "-e", spec),
        )
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
    hf = hf_home or DEFAULT_COLAB_HF_HOME
    plan.actions.append(
        SetupAction(
            "warning",
            f"HF_HOME = {hf} (sobrescreve o valor anterior). Se transformers/huggingface_hub "
            "já foi importado nesta sessão com outro HF_HOME, reinicie o ambiente de execução. "
            "Cache no Drive só com >= 20 GB livres por modelo (não recomendado: a cota e o FUSE "
            "truncam os pesos e geram 'header too large').",
        )
    )
    return root, hf


def setup_environment(
    *,
    repo_dir: str | Path | None = None,
    repo_url: str = "",
    repo_branch: str = DEFAULT_REPO_BRANCH,
    mount_drive: bool = False,
    drive_dir: str = DEFAULT_DRIVE_PROJECT,
    in_colab: bool | None = None,
    dry_run: bool = False,
    extras: str = DEFAULT_EXTRAS,
    hf_home: str = "",
    environ: MutableMapping[str, str] | None = None,
) -> EnvironmentPlan:
    """Prepara o ambiente (ou, com ``dry_run=True``, só devolve o plano).

    No Colab, na ordem: monta o Drive (opcional), clona (se ``repo_url``), verifica o
    repositório (ausente: :class:`RepoNotFoundError`), ``pip install -e <repo>[hf]`` e define
    ``HF_HOME`` (``hf_home``, ou o disco local ``/content/hf_home``; sempre sobrescreve, pois
    pesos no Drive truncam), além de ``PHYSICS_LLM_SR_PROJECT_ROOT`` (Drive: resultados e
    cache persistem), ``PHYSICS_LLM_SR_PROMPTS_DIR`` e ``PHYSICS_LLM_SR_REFERENCES_DIR``.
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
    hf: str | None = None
    if colab:
        root, hf = _plan_colab(plan, repo_url, mount_drive, drive_dir, extras, hf_home, repo_branch)
    wanted = {
        PROJECT_ROOT_ENV: root,
        REPO_DIR_ENV: str(repo),
        PROMPTS_DIR_ENV: str(repo / "prompts"),
        REFERENCES_DIR_ENV: str(repo / "references"),
    }
    if hf is not None:
        wanted[HF_HOME_ENV] = hf
    for name, value in wanted.items():
        if name in env and name != HF_HOME_ENV:
            plan.actions.append(
                SetupAction("env_kept", f"{name} já definida; mantida", name=name, value=env[name])
            )
        else:
            plan.actions.append(
                SetupAction("env", f"definir {name}", name=name, value=value),
            )
    if not dry_run:
        if plan.blocking_errors:
            raise RepoNotFoundError(plan.blocking_errors[0])
        _execute(plan, env)
        plan.executed = True
    return plan


def _execute(plan: EnvironmentPlan, env: MutableMapping[str, str]) -> None:
    for action in plan.actions:
        if action.kind == "clone" and action.command:
            if not plan.repo_dir.is_dir():
                subprocess.run(action.command, check=True)
            elif (plan.repo_dir / ".git").exists():
                subprocess.run(("git", "-C", str(plan.repo_dir), "pull", "--ff-only"), check=True)
        elif action.kind == "check_repo":
            if not (plan.repo_dir / PYPROJECT).is_file():
                raise RepoNotFoundError(_repo_missing_message(plan.repo_dir))
        elif action.kind == "pip" and action.command:
            subprocess.run(action.command, check=True)
        elif action.kind == "drive":
            from google.colab import drive  # type: ignore[import-not-found,unused-ignore]

            drive.mount(DEFAULT_DRIVE_MOUNT)
        elif action.kind == "env" and action.name and action.value is not None:
            env[action.name] = action.value
            if action.name in {PROJECT_ROOT_ENV, HF_HOME_ENV}:
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


def environment_report(include_cuda: bool = True) -> dict[str, Any]:
    """Versões e hardware para a etapa 01 (sem segredos).

    ``include_cuda=False`` não importa o torch (notebook 01, sem LLM): ``cuda_disponível = None``.
    """
    from physics_llm_sr.reproducibility.manifests import hardware_info

    versions = library_versions(TRACKED_LIBRARIES)
    versions["physics-llm-sr"] = _version("physics-llm-sr")
    cuda: bool | None = None
    if include_cuda and importlib.util.find_spec("torch") is not None:
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


def ensure_project_root() -> Path:
    """Raiz de projeto (``PHYSICS_LLM_SR_PROJECT_ROOT``; default: raiz do repositório)."""
    os.environ.setdefault(PROJECT_ROOT_ENV, str(find_repo_root()))
    return Path(os.environ[PROJECT_ROOT_ENV]).expanduser().resolve()


def resolve_stage(
    profile: str,
    arm: str,
    existing_run: str | Path = "",
    config_path: str | Path | None = None,
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
    cfg = load_config(config_path) if config_path else notebook_config(profile, arm)
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


def stage_experiments(cfg: ExperimentConfig, stage: str) -> list[str]:
    """Experimentos (aliases) da etapa ``P1``-``P4`` que estão em ``experiments.run``."""
    from physics_llm_sr.experiments.matrix import ALIASES

    return [e for e in HOJE_STAGES[stage] if ALIASES[e] in cfg.experiments.run]


def plan_table(cfg: ExperimentConfig) -> pd.DataFrame:
    """Células por experimento x modelo (baseline = ``none``) da matriz resolvida da config."""
    from physics_llm_sr.experiments.matrix import EXPERIMENTS

    rows = []
    for name in cfg.experiments.run:
        counts: dict[str, int] = {}
        for cell in EXPERIMENTS[name](cfg):
            if cfg.experiments.baseline_only and not cell.is_baseline:
                continue
            counts[cell.llm] = counts.get(cell.llm, 0) + 1
        # contagem BRUTA por experimento (o baseline pareado se repete entre experimentos)
        by_model = {f"modelo:{m}": n for m, n in sorted(counts.items())}
        rows.append({"experimento": name, **by_model})
    out: pd.DataFrame = pd.DataFrame(rows).fillna(0)
    return out


def ensure_hf_access(cfg: ExperimentConfig) -> list[str]:
    """Garante o ``HF_TOKEN`` quando um modelo restrito (Llama 2) será usado.

    Devolve os modelos restritos selecionados. Sem token, levanta ``RuntimeError`` com o passo
    a passo (antes de carregar qualquer modelo). Mistral e Falcon não exigem token.
    """
    from physics_llm_sr.experiments.cells import selected_models

    if cfg.llm.backend != "transformers":
        return []
    used = set(selected_models(cfg))
    if {"noise_robustness", "biased_noise"} & set(cfg.experiments.run):
        used.add(cfg.experiments.noise_model)
    gated = sorted(
        m.name for m in cfg.llm.models if m.name in used and m.checkpoint.startswith(GATED_PREFIX)
    )
    if gated and not login_huggingface():
        raise RuntimeError(
            f"modelo(s) restrito(s) {gated} exigem HF_TOKEN. Peça acesso em "
            "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf (aceite a licença), crie o "
            "token em huggingface.co/settings/tokens e salve-o no Colab em Segredos (ícone de "
            "chave) como HF_TOKEN com 'Acesso ao notebook' ligado. Ou retire 'llama2' de MODELOS."
        )
    return gated


# ------------------------------------------------------------------ estimativa de tempo

LATENCY_BATCH_CALLS = 10
LATENCY_SERIAL_CALLS = 5
LATENCY_SCENARIO = "free_fall"
LATENCY_PROMPT = "E"
LATENCY_TEMPLATE = "sqrt(2*{c}*height)"  # constantes distintas: sem acerto de cache na medição


def measure_llm_latency(
    cfg: ExperimentConfig,
    n_batch: int = LATENCY_BATCH_CALLS,
    n_serial: int = LATENCY_SERIAL_CALLS,
) -> dict[str, float]:
    """Mede s/chamada do avaliador da config: ``n_batch`` em um lote e ``n_serial`` isoladas.

    Cache desligado e uma avaliação de aquecimento (carga do modelo) fora da medição.
    """
    import time

    from physics_llm_sr.llm.inference import build_backend
    from physics_llm_sr.llm.scoring import LLMEquationEvaluator
    from physics_llm_sr.physics.registry import get_scenario
    from physics_llm_sr.symbolic.expression import parse_infix

    llm = cfg.llm.model_copy(update={"cache_enabled": False})
    name = (cfg.experiments.models or [m.name for m in cfg.llm.models])[0]
    evaluator = LLMEquationEvaluator.from_config(llm, build_backend(llm, name, track=cfg.track))
    ctx = get_scenario(LATENCY_SCENARIO, cfg.physics).physical_context()
    features = ("mass", "length", "height", "damping", "t")

    def exprs(start: int, count: int) -> list[Any]:
        texts = [LATENCY_TEMPLATE.format(c=f"{9.0 + 0.01 * (start + i):.2f}") for i in range(count)]
        return [parse_infix(t, features) for t in texts]

    evaluator.evaluate_many(exprs(1000, 1), ctx, LATENCY_PROMPT)  # aquecimento
    t0 = time.perf_counter()
    evaluator.evaluate_many(exprs(0, n_batch), ctx, LATENCY_PROMPT)
    batched = (time.perf_counter() - t0) / n_batch
    t0 = time.perf_counter()
    for expr in exprs(n_batch, n_serial):
        evaluator.evaluate(expr, ctx, LATENCY_PROMPT)
    serial = (time.perf_counter() - t0) / n_serial
    return {"batched_s_per_call": batched, "serial_s_per_call": serial}


def estimate_runtime(
    config_path: str | Path | ExperimentConfig,
    seconds_per_call_batched: float,
    seconds_per_call_serial: float,
    calls_per_cell: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Tempo projetado de LLM por etapa P1-P4 (``HOJE_STAGES``) a partir da matriz da config.

    Só entram as etapas cujos experimentos estão em ``experiments.run``. Conta as células
    NOVAS de cada etapa (as já pedidas por etapas anteriores são deduplicadas, como na
    retomada do runner). Só células com LLM geram chamadas: ``calls_per_cell[sr]`` (default
    ``DEFAULT_CALLS_PER_CELL``, hipóteses) vezes o custo por chamada (lote para DEAP/gplearn,
    série para PySR). Não inclui o tempo de busca do SR.
    """
    from physics_llm_sr.experiments.matrix import build_matrix

    cfg = config_path if isinstance(config_path, ExperimentConfig) else load_config(config_path)
    calls = {**DEFAULT_CALLS_PER_CELL, **(dict(calls_per_cell) if calls_per_cell else {})}

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for stage in HOJE_STAGES:
        exps = stage_experiments(cfg, stage)
        if not exps:
            continue
        new = [p for p in build_matrix(cfg, only=list(exps)) if p.cell.key() not in seen]
        seen.update(p.cell.key() for p in new)
        llm_cells = [p.cell for p in new if not p.cell.is_baseline]
        n_calls = sum(calls[c.sr] for c in llm_cells)
        seconds = sum(
            calls[c.sr]
            * (seconds_per_call_batched if c.sr in BATCHED_ALGORITHMS else seconds_per_call_serial)
            for c in llm_cells
        )
        rows.append(
            {
                "stage": stage,
                "experiments": "+".join(exps),
                "cells": len(new),
                "llm_cells": len(llm_cells),
                "llm_calls": n_calls,
                "llm_seconds": seconds,
                "llm_hours": seconds / SECONDS_PER_HOUR,
            }
        )
    out: pd.DataFrame = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------------ presets e config derivada

DERIVED_DIR = "configs_derivadas"
HASH_LENGTH = 10
OUTPUT_LAYOUTS = ("suffix", "subdir")
ARM_TRACKS = {
    "replication": {"track": "paper_replication", "normalization": "literal"},
    "extended": {"track": "extended_analysis", "normalization": "normalized"},
}


@dataclass(frozen=True)
class Preset:
    """Config base (arquivo em ``configs/``) e ajustes fixos do preset."""

    config: str
    description: str
    patch: Mapping[str, Any] = field(default_factory=dict)
    models: tuple[str, ...] | None = None


#: Presets do notebook (``PRESET``). Sobrescritas do usuário (``MODELOS`` etc.) valem por cima.
PRESETS: dict[str, Preset] = {
    "hoje": Preset(
        "colab_hoje_queda_livre.yaml",
        "queda livre, Mistral, P1-P4 (extended, PySR reduzido): cerca de 4 h no A100",
    ),
    "hoje_literal": Preset(
        "colab_hoje_queda_livre_literal.yaml",
        "controle literal (paper_replication) da execução de hoje: só P1",
    ),
    "3_modelos_queda_livre": Preset(
        "colab_hoje_queda_livre.yaml",
        "hoje com Mistral, Llama 2 (exige HF_TOKEN) e Falcon",
        models=("mistral", "llama2", "falcon"),
    ),
    "completo": Preset(
        "replication.yaml",
        "artigo inteiro: 3 cenários x 3 SR x 3 LLMs x 25 repetições, prompts A-H, ruído completo "
        "(dias de GPU)",
    ),
    "completo_extended": Preset(
        "extended.yaml", "idem, braço extended_analysis (perda normalizada, desvio declarado)"
    ),
    "smoke": Preset(
        "smoke.yaml",
        "validação mínima com avaliador simulado (minutos, sem LLM real)",
        {
            "experiments": {"run": ALL_EXPERIMENTS},
            "llm": {"prompt_variants": SMOKE_PROMPTS},
            "data": {
                "exp4": {"levels": SMOKE_NOISE_LEVELS, "locations": SMOKE_NOISE_LOCATIONS},
                "exp5": {"levels": SMOKE_NOISE_LEVELS, "locations": SMOKE_NOISE_LOCATIONS},
            },
            "output_dir": f"{SMOKE_OUTPUT_DIR}/replication",
        },
    ),
}


#: Preset -> (perfil legado, braço) usados por etapas que comparam braços (etapa 10).
PRESET_PROFILE: dict[str, tuple[str, str]] = {
    "hoje": ("colab_hoje", "extended"),
    "hoje_literal": ("colab_hoje", "replication"),
    "3_modelos_queda_livre": ("colab_hoje", "extended"),
    "completo": ("replication", "replication"),
    "completo_extended": ("replication", "extended"),
    "smoke": ("smoke", "replication"),
}


def preset_profile(preset: str) -> tuple[str, str]:
    """(``RUN_PROFILE``, ``ARM``) do preset; erro claro se o nome não existe."""
    if preset not in PRESETS:
        raise ValueError(f"PRESET inválido: {preset!r}; válidos: {tuple(PRESETS)}")
    return PRESET_PROFILE[preset]


def _deep_update(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value if not isinstance(value, Mapping) else dict(value)


def build_run_config(
    base_profile: str,
    *,
    models: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    experiments: Sequence[str] | None = None,
    algorithms: Sequence[str] | None = None,
    repetitions: int | None = None,
    prompts: Sequence[str] | None = None,
    noise_levels: Sequence[float] | None = None,
    noise_locations: Sequence[str] | None = None,
    arm: str | None = None,
    out_dir: str | Path,
    repo_dir: str | Path | None = None,
    patch: Mapping[str, Any] | None = None,
    output_dir: str | None = None,
    output_layout: str = "suffix",
    output_prefix: str = "",
    derived_dir_name: str = DERIVED_DIR,
    smoke: bool = False,
) -> Path:
    """Config derivada de um preset (ou arquivo de ``configs/``) com sobrescritas validadas.

    ``base_profile`` é uma chave de :data:`PRESETS` ou o nome de um YAML em ``configs/``.
    ``None`` mantém o valor do base. ``experiments`` aceita ``exp1``..``exp5`` ou os nomes
    completos; ``noise_levels`` vale para Exp4 e Exp5; ``arm`` (``replication``|``extended``)
    troca track e normalização. A config final é validada por :func:`load_config` (erro claro
    do schema) e gravada em ``<out_dir>/configs_derivadas/<hash>.yaml``. O hash cobre o
    conteúdo derivado: se houve sobrescrita, ``output_dir`` ganha o sufixo ``_<hash>`` para que
    execuções diferentes nunca se misturem; sem sobrescritas, vale o ``output_dir`` do base.

    Extras dos notebooks: ``patch`` (atualização profunda aplicada ao base, sem efeito no
    diretório de resultados), ``noise_locations`` (Exp4 e Exp5), ``output_dir`` (substitui o
    base), ``output_prefix`` (insere ``<prefixo>`` após ``results/``, p.ex. ``nb03_``),
    ``output_layout`` (``"suffix"``: ``<base>_<hash>`` só com sobrescrita; ``"subdir"``:
    sempre ``<base>/<hash>``), ``derived_dir_name`` (subpasta da config derivada) e ``smoke``
    (tamanhos mínimos, backend simulado, sem PySR: :func:`_apply_smoke`).
    Só composição de config: nenhuma lógica científica.
    """
    import hashlib
    import json

    import yaml

    from physics_llm_sr.experiments.matrix import resolve_experiments

    preset = PRESETS.get(base_profile)
    base_file = preset.config if preset else base_profile
    repo = Path(repo_dir) if repo_dir is not None else find_repo_root()
    base_path = repo / "configs" / base_file
    data: dict[str, Any] = load_config(base_path).model_dump(mode="json")
    if preset is not None:
        _deep_update(data, preset.patch)
        if models is None and preset.models is not None:
            models = preset.models
    if patch:
        _deep_update(data, patch)
    if output_dir is not None:
        data["output_dir"] = output_dir
    if output_prefix:
        data["output_dir"] = data["output_dir"].replace("results/", f"results/{output_prefix}", 1)
    base_output = data["output_dir"]
    overridden = False

    def _set(path: tuple[str, ...], value: Any) -> None:
        nonlocal overridden
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        overridden = True

    if arm is not None:
        if arm not in ARM_TRACKS:
            raise ValueError(f"arm inválido: {arm!r}; válidos: {tuple(ARM_TRACKS)}")
        _set(("track",), ARM_TRACKS[arm]["track"])
        _set(("loss", "normalization"), ARM_TRACKS[arm]["normalization"])
        if arm == "replication":
            data["llm"].update(evaluation_scope="all", top_k=None, cache_key_mode="exact")
    if models is not None:
        _set(("experiments", "models"), list(models))
        if data["experiments"]["noise_model"] not in models and models:
            _set(("experiments", "noise_model"), list(models)[0])
    if scenarios is not None:
        _set(("experiments", "scenarios"), list(scenarios))
    if experiments is not None:
        _set(("experiments", "run"), resolve_experiments(experiments, parse_config(data)))
    if algorithms is not None:
        _set(("experiments", "algorithms"), list(algorithms))
    if repetitions is not None:
        _set(("experiments", "repetitions"), repetitions)
    if prompts is not None:
        _set(("llm", "prompt_variants"), list(prompts))
    if noise_levels is not None:
        _set(("data", "exp4", "levels"), list(noise_levels))
        _set(("data", "exp5", "levels"), list(noise_levels))
    if noise_locations is not None:
        _set(("data", "exp4", "locations"), list(noise_locations))
        _set(("data", "exp5", "locations"), list(noise_locations))
    if smoke:
        _apply_smoke(data)
        overridden = True
    digest = hashlib.sha256(
        json.dumps({**data, "output_dir": base_output}, sort_keys=True).encode()
    ).hexdigest()[:HASH_LENGTH]
    if output_layout not in OUTPUT_LAYOUTS:
        raise ValueError(f"output_layout inválido: {output_layout!r}; válidos: {OUTPUT_LAYOUTS}")
    if output_layout == "subdir":
        data["output_dir"] = f"{base_output}/{digest}"
    elif overridden:
        data["output_dir"] = f"{base_output}_{digest}"
    parse_config(data)  # valida: ConfigError traz o campo e o motivo
    target_dir = Path(out_dir).expanduser() / derived_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{digest}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    load_config(path)
    return path


# ------------------------------------------------------------------ notebooks 01 e 02
NOTEBOOK_SMOKE_ENV = "PHYSICS_LLM_SR_NOTEBOOK_SMOKE"
NB01_CONFIG = "nb01_baseline.yaml"
NB02_CONFIG = "nb02_um_llm.yaml"
NB01_OUTPUT = "results/nb01_baseline"
NB01_EXPERIMENTS = ("exp1", "exp4", "exp5")  # Exp1 + células baseline de Exp4 e Exp5
#: ``BRACO`` dos notebooks -> braço de :data:`ARM_TRACKS`.
NB_BRACOS = {"literal": "replication", "normalized": "extended"}
#: ``EXPERIMENTO`` do notebook 02 -> alias do runner.
NB02_EXPERIMENTS = {
    "benchmark": "exp2",
    "prompts": "exp3",
    "ruido": "exp4",
    "ruido_enviesado": "exp5",
}
NB_SCENARIOS = ("free_fall", "harmonic_motion", "damped_wave")
NB_ALGORITHMS = ("deap", "gplearn", "pysr")
NB_LLMS = ("mistral", "llama2", "falcon")
NB_PROMPTS = tuple("ABCDEFGH")
NB_NOISE_LOCATIONS = ("inputs", "target", "both")
#: Reduções do modo smoke (``make notebook-smoke``): minutos, sem LLM real e sem Julia.
SMOKE_NB_REPETITIONS = 2
SMOKE_NB_SIMULATIONS = 5
SMOKE_NB_TIME_POINTS = 20
SMOKE_NB_POPULATION = 10
SMOKE_NB_GENERATIONS = 3
SMOKE_NB_NOISE_LEVELS = 2
SMOKE_NB_CACHE_DIR_SUFFIX = "_smoke"


class NotebookParameterError(ValueError):
    """Parâmetro de notebook inválido (mensagem em português, antes de qualquer execução)."""


def smoke_requested() -> bool:
    """Indica se ``PHYSICS_LLM_SR_NOTEBOOK_SMOKE`` pede o modo smoke dos notebooks."""
    return os.environ.get(NOTEBOOK_SMOKE_ENV, "").strip().lower() in {"1", "true", "yes", "sim"}


def _apply_smoke(data: dict[str, Any]) -> None:
    """Reduz a config (dict) a tamanhos mínimos, backend simulado e sem PySR (Julia)."""
    ex = data["experiments"]
    data["mode"] = "smoke"
    data["data"].update(n_simulations=SMOKE_NB_SIMULATIONS, n_time_points=SMOKE_NB_TIME_POINTS)
    data["sr"].update(population=SMOKE_NB_POPULATION, generations=SMOKE_NB_GENERATIONS)
    ex["repetitions"] = min(ex["repetitions"], SMOKE_NB_REPETITIONS)
    ex["algorithms"] = [a for a in ex["algorithms"] if a != "pysr"] or ["deap"]
    ex["max_cell_seconds"] = None
    for exp in ("exp4", "exp5"):
        data["data"][exp]["levels"] = data["data"][exp]["levels"][:SMOKE_NB_NOISE_LEVELS]
    llm = data["llm"]
    llm.update(backend="fake", batch_size=1, max_new_tokens=64)
    llm["cache_dir"] = f"{llm['cache_dir']}{SMOKE_NB_CACHE_DIR_SUFFIX}"
    for model in llm["models"]:
        model.update(revision=None, quantization=None, infra="local")


def _check_choice(name: str, value: object, valid: Sequence[str]) -> None:
    if not isinstance(value, str) or value not in valid:
        raise NotebookParameterError(f"{name} inválido: {value!r}; use um de {tuple(valid)}")


def _check_list(name: str, values: object, valid: Sequence[str]) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Sequence) or not values:
        raise NotebookParameterError(f"{name} deve ser uma lista não vazia de {tuple(valid)}")
    for value in values:
        _check_choice(name, value, valid)
    return list(values)


def _check_common(
    algorithms: object,
    repetitions: object,
    noise_levels: object,
    noise_locations: object,
    braco: object,
) -> list[str]:
    algos = _check_list("ALGORITMOS", algorithms, NB_ALGORITHMS)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise NotebookParameterError(f"REPETICOES deve ser um inteiro >= 1, não {repetitions!r}")
    if (
        isinstance(noise_levels, str)
        or not isinstance(noise_levels, Sequence)
        or not noise_levels
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in noise_levels
        )
    ):
        raise NotebookParameterError(
            f"NIVEIS_RUIDO deve ser uma lista de números > 0 (fração do desvio-padrão), "
            f"não {noise_levels!r}"
        )
    _check_list("LOCALIZACOES", noise_locations, NB_NOISE_LOCATIONS)
    _check_choice("BRACO", braco, tuple(NB_BRACOS))
    return algos


def build_baseline_config(
    *,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    repetitions: int,
    noise_levels: Sequence[float],
    noise_locations: Sequence[str],
    braco: str,
    out_dir: str | Path,
    smoke: bool = False,
    repo_dir: str | Path | None = None,
) -> Path:
    """Config derivada do notebook 01 (baseline sem LLM), validada antes de rodar.

    Parte de ``configs/nb01_baseline.yaml`` (``baseline_only``: nenhuma célula com LLM). Os
    parâmetros em português são validados (:class:`NotebookParameterError`) e o resultado
    passa pelo schema. Saída: ``results/nb01_baseline/<hash>``; config em
    ``<out_dir>/configs_derivadas/nb01/<hash>.yaml``.
    """
    scen = _check_list("CENARIOS", scenarios, NB_SCENARIOS)
    algos = _check_common(algorithms, repetitions, noise_levels, noise_locations, braco)
    return build_run_config(
        NB01_CONFIG,
        scenarios=scen,
        algorithms=algos,
        repetitions=repetitions,
        noise_levels=list(noise_levels),
        noise_locations=list(noise_locations),
        arm=NB_BRACOS[braco],
        out_dir=out_dir,
        repo_dir=repo_dir,
        output_layout="subdir",
        derived_dir_name=f"{DERIVED_DIR}/nb01",
        smoke=smoke,
    )


def build_single_llm_config(
    *,
    scenario: str,
    experiment: str,
    llm: str,
    algorithms: Sequence[str],
    repetitions: int,
    prompt: str,
    prompts: Sequence[str],
    braco: str,
    noise_levels: Sequence[float],
    noise_locations: Sequence[str],
    out_dir: str | Path,
    smoke: bool = False,
    repo_dir: str | Path | None = None,
) -> Path:
    """Config derivada do notebook 02 (um LLM, um cenário, um experimento).

    ``experiment``: ``benchmark`` (Exp2), ``prompts`` (Exp3, usa ``prompts``), ``ruido``
    (Exp4) ou ``ruido_enviesado`` (Exp5); ``prompt`` vale para Exp2, Exp4 e Exp5. Validação em
    português (:class:`NotebookParameterError`) antes de rodar; o Llama 2 exige ``HF_TOKEN``
    (verificado por :func:`ensure_hf_access` com a config resultante). Saída:
    ``results/nb02_<cenario>_<experimento>_<llm>/<hash>``.
    """
    _check_choice("CENARIO", scenario, NB_SCENARIOS)
    _check_choice("EXPERIMENTO", experiment, tuple(NB02_EXPERIMENTS))
    _check_choice("LLM", llm, NB_LLMS)
    _check_choice("PROMPT", prompt, NB_PROMPTS)
    algos = _check_common(algorithms, repetitions, noise_levels, noise_locations, braco)
    chosen_prompts = _check_list("PROMPTS", prompts, NB_PROMPTS)
    base = {"benchmark_prompt": prompt, "noise_prompt": prompt}
    return build_run_config(
        NB02_CONFIG,
        models=[llm],
        scenarios=[scenario],
        experiments=[NB02_EXPERIMENTS[experiment]],
        algorithms=algos,
        repetitions=repetitions,
        prompts=chosen_prompts if experiment == "prompts" else [prompt],
        noise_levels=list(noise_levels),
        noise_locations=list(noise_locations),
        arm=NB_BRACOS[braco],
        out_dir=out_dir,
        repo_dir=repo_dir,
        patch={"experiments": base},
        output_dir=f"results/nb02_{scenario}_{experiment}_{llm}",
        output_layout="subdir",
        derived_dir_name=f"{DERIVED_DIR}/nb02",
        smoke=smoke,
    )


#: HIPÓTESES de segundos por célula baseline em CPU (a medir; PySR inclui a partida do Julia).
DEFAULT_CPU_SECONDS_PER_CELL: dict[str, float] = {"deap": 10.0, "gplearn": 20.0, "pysr": 120.0}


def estimate_cpu_runtime(
    cfg: ExperimentConfig, seconds_per_cell: Mapping[str, float] | None = None
) -> pd.DataFrame:
    """Tempo projetado em CPU por algoritmo para a matriz da config (notebook 01).

    ``seconds_per_cell`` sobrescreve :data:`DEFAULT_CPU_SECONDS_PER_CELL` (hipóteses). Cada
    célula é limitada por ``experiments.max_cell_seconds`` quando definido.
    """
    from physics_llm_sr.experiments.matrix import build_matrix

    per_cell = {
        **DEFAULT_CPU_SECONDS_PER_CELL,
        **(dict(seconds_per_cell) if seconds_per_cell else {}),
    }
    cap = cfg.experiments.max_cell_seconds
    counts: dict[str, int] = {}
    for planned in build_matrix(cfg):
        counts[planned.cell.sr] = counts.get(planned.cell.sr, 0) + 1
    rows = []
    for sr in cfg.experiments.algorithms:
        each = per_cell[sr] if cap is None else min(per_cell[sr], cap)
        n = counts.get(sr, 0)
        rows.append(
            {
                "sr": sr,
                "cells": n,
                "seconds_per_cell": each,
                "seconds": n * each,
                "hours": n * each / SECONDS_PER_HOUR,
            }
        )
    out: pd.DataFrame = pd.DataFrame(rows)
    return out


def require_gpu(cfg: ExperimentConfig) -> None:
    """Exige GPU para o backend ``transformers`` (7B em fp16 não roda em CPU): erro claro."""
    if cfg.llm.backend != "transformers":
        return
    if not environment_report()["cuda_disponível"]:
        raise RuntimeError(
            "o backend transformers (7B em fp16) exige GPU: no Colab use Ambiente de execução > "
            "Alterar tipo > GPU A100 (High-RAM) e reinicie. Para validar só o encanamento, use o "
            "modo smoke (PHYSICS_LLM_SR_NOTEBOOK_SMOKE=1)."
        )
