"""Estudo de baseline local (sem LLM): desenho, execução, controle positivo e consistência.

Fluxo (cada etapa é uma chamada de função, sem duplicação entre cenários):

1. :class:`StudyDesign` fixa o **desenho experimental**: algoritmo de regressão simbólica,
   cenários (equações de referência), repetições, semente mestra, formulação da função de perda,
   orçamento de busca e condição de ruído.
2. :func:`positive_control` avalia a própria equação de referência (e suas formas equivalentes)
   pelo mesmo caminho de métricas, sobre dados limpos: valida dados e métricas antes de culpar o
   algoritmo.
3. :func:`phase1_clean_validation` (regime sem ruído, σ = 0) executa o baseline reutilizando o
   runner do projeto (retomável) e grava manifesto, ``results.parquet``, ``consistency.csv`` e
   ``study_log.jsonl`` em ``<output_root>/<fase>/<algoritmo>/<hash do desenho>/``.
4. :func:`consistency_report` compara o modelo descoberto com a equação de referência e classifica
   cada repetição com limiares visíveis (:class:`ConsistencyCriteria`).
5. Fase 2 (refinamento progressivo; só depois da Fase 1 aprovada): :func:`phase2_noise_sweep`,
   :func:`phase2_hyperparameter_sweep` e :func:`phase2_algorithm_comparison` geram conjuntos de
   execuções separados, com o mesmo esquema de saída.

Nada aqui reimplementa regressão simbólica, geração de dados ou métricas: tudo vem de
``sr``, ``data``, ``evaluation``, ``symbolic`` e ``experiments``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from physics_llm_sr.colab import ARM_TRACKS, find_repo_root
from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV, project_root, resolve_project_path
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.evaluation.metrics import regression_metrics, structural_metrics
from physics_llm_sr.experiments.data_provider import DataProvider
from physics_llm_sr.experiments.reporting import consolidate
from physics_llm_sr.experiments.runner import RunSummary, run_experiments
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng
from physics_llm_sr.symbolic.expression import evaluate, parse_infix
from physics_llm_sr.symbolic.operators import ProtectionParams
from physics_llm_sr.symbolic.tree_distance import TreeDistanceParams

# --------------------------------------------------------------------------- constantes

ALGORITHMS = ("deap", "gplearn", "pysr")
SCENARIOS = ("free_fall", "harmonic_motion", "damped_wave")
NOISE_TYPES = ("none", "zero_mean", "biased")
NOISE_LOCATIONS = ("inputs", "target", "both")
LOSS_FORMULATIONS = ("literal", "normalized")
#: Caminhos ``loss.*`` aceitos em ``hyperparameters`` (formulação da perda do baseline, w3 = 0):
#: denominador comum de ``s`` (só em normalized/extended) e pesos do baseline. Ambos são desvios.
LOSS_HYPERPARAMETERS = ("loss.max_nodes", "loss.baseline_weights")
BASE_CONFIG = "nb01_baseline.yaml"
"""Config base (em ``configs/``): sem LLM (``baseline_only``), pesos do baseline 0,5/0,5/0."""

PHASE1 = "fase1_validacao_limpa"
PHASE2_NOISE = "fase2_ruido"
PHASE2_HYPERPARAMETERS = "fase2_hiperparametros"
PHASE2_ALGORITHMS = "fase2_algoritmos"
CLEAN_CONDITION = "sem ruído (σ = 0)"

HASH_LENGTH = 10
CONSISTENCY_FILE = "consistency.csv"
STUDY_LOG_FILE = "study_log.jsonl"
RESULTS_FILE = "results.parquet"

POSITIVE_CONTROL_TOLERANCE = 1e-9
"""Tolerância numérica do controle positivo para ``1 - R²`` (arredondamento de ponto flutuante:
a equação de referência reproduz o alvo analítico a ~1e-16; 1e-9 deixa folga e continua
rejeitando qualquer erro estrutural)."""

GRID_SIMULATIONS = 200
GRID_TIME_POINTS = 50
"""Grid denso de verificação: ``GRID_SIMULATIONS`` simulações novas (semente própria, fora do
treino) × ``GRID_TIME_POINTS`` instantes igualmente espaçados em ``[0, t_fim]`` (domínio físico)."""

CLASS_RECOVERS = "recupera estrutura"
CLASS_FITS = "ajusta numericamente sem recuperar estrutura"
CLASS_FAILS = "não converge"
CLASS_ERROR = "erro de execução"

MINIMAL_ENV = "PHYSICS_LLM_SR_NB05_MINIMO"
MINIMAL_POPULATION = 20
MINIMAL_GENERATIONS = 5
MINIMAL_SIMULATIONS = 20
MINIMAL_TIME_POINTS = 20

CONSISTENCY_COLUMNS: tuple[str, ...] = (
    "fase",
    "condicao",
    "cenario",
    "algoritmo",
    "repeticao",
    "semente_busca",
    "sigma",
    "tipo_ruido",
    "localizacao_ruido",
    "equacao_descoberta",
    "equacao_referencia",
    "mae_teste",
    "mse_teste",
    "um_menos_r2_teste",
    "erro_max_abs_grid",
    "distancia_estrutural",
    "distancia_equivalente_min",
    "recuperacao_exata",
    "geracoes",
    "motivo_parada",
    "tempo_s",
    "status_execucao",
    "classificacao",
    "hash_desenho",
)
"""Esquema fixo (ordem e nomes) do relatório de consistência e de ``consistency.csv``."""

_EXPERIMENT_BY_NOISE = {
    "none": "baseline",
    "zero_mean": "noise_robustness",
    "biased": "biased_noise",
}
_HASH_EXCLUDED = ("output_root", "allow_code_change", "criteria")


class StudyDesignError(ValueError):
    """Parâmetro inválido do desenho experimental (mensagem em português)."""


class CleanPhaseError(StudyDesignError):
    """A Fase 1 (regime sem ruído) recusa qualquer condição com ruído."""


# --------------------------------------------------------------------------- desenho


@dataclass(frozen=True)
class SearchBudget:
    """Orçamento de busca: tamanho da população, gerações e critério de parada.

    ``early_stopping``: para quando a melhoria relativa da melhor perda fica abaixo de
    ``min_relative_improvement`` por ``patience`` gerações seguidas (GAP: 0,1% e 3 gerações).
    Os campos ``pysr_*`` valem só para o PySR, que roda **reduzido** (desvio declarado: busca em
    Julia mais cara por indivíduo); seus resultados não são comparáveis em esforço com os de
    DEAP/gplearn. ``max_cell_seconds``: tempo máximo por execução (``None`` = sem limite).
    """

    population: int = 100
    generations: int = 50
    early_stopping: bool = True
    min_relative_improvement: float = 0.001
    patience: int = 3
    max_cell_seconds: float | None = 1800.0
    pysr_niterations: int = 10
    pysr_population_size: int = 50
    pysr_ncycles_per_iteration: int = 50

    def __post_init__(self) -> None:
        """Valida os limites."""
        for name in (
            "population",
            "generations",
            "patience",
            "pysr_niterations",
            "pysr_population_size",
            "pysr_ncycles_per_iteration",
        ):
            if getattr(self, name) < 1:
                raise StudyDesignError(
                    f"SearchBudget.{name} deve ser >= 1, não {getattr(self, name)}"
                )
        if self.min_relative_improvement <= 0:
            raise StudyDesignError("SearchBudget.min_relative_improvement deve ser > 0")
        if self.max_cell_seconds is not None and self.max_cell_seconds <= 0:
            raise StudyDesignError("SearchBudget.max_cell_seconds deve ser > 0 ou None")

    @classmethod
    def minimal(cls) -> SearchBudget:
        """Orçamento mínimo para validar o encanamento (segundos; sem valor científico)."""
        return cls(
            population=MINIMAL_POPULATION,
            generations=MINIMAL_GENERATIONS,
            max_cell_seconds=None,
            pysr_niterations=2,
            pysr_population_size=10,
            pysr_ncycles_per_iteration=5,
        )


@dataclass(frozen=True)
class NoiseCondition:
    """Condição de ruído: tipo, σ como fração do desvio-padrão do sinal e localização.

    ``none``: regime sem ruído (σ = 0). ``zero_mean``: ruído gaussiano de média zero,
    ``σ = sigma_fraction · std(sinal)``. ``biased``: média ``0,01 · média do atributo`` e o mesmo σ.
    ``location``: ``inputs``, ``target`` ou ``both``.
    """

    noise_type: Literal["none", "zero_mean", "biased"] = "none"
    sigma_fraction: float = 0.0
    location: Literal["inputs", "target", "both"] = "target"

    def __post_init__(self) -> None:
        """Valida tipo, localização e a coerência entre tipo e σ."""
        if self.noise_type not in NOISE_TYPES:
            raise StudyDesignError(
                f"tipo de ruído inválido: {self.noise_type!r}; use {NOISE_TYPES}"
            )
        if self.location not in NOISE_LOCATIONS:
            raise StudyDesignError(
                f"localização inválida: {self.location!r}; use {NOISE_LOCATIONS}"
            )
        if self.noise_type == "none" and self.sigma_fraction != 0.0:
            raise StudyDesignError("noise_type='none' exige sigma_fraction = 0")
        if self.noise_type != "none" and not self.sigma_fraction > 0.0:
            raise StudyDesignError(f"noise_type={self.noise_type!r} exige sigma_fraction > 0")

    @property
    def is_clean(self) -> bool:
        """Verdadeiro no regime sem ruído."""
        return self.noise_type == "none"

    def label(self) -> str:
        """Rótulo legível da condição."""
        if self.is_clean:
            return CLEAN_CONDITION
        return f"{self.noise_type}/{self.location}/σ={self.sigma_fraction:g}"


@dataclass(frozen=True)
class ConsistencyCriteria:
    """Critério de consistência (limiares visíveis; nenhuma tolerância escondida).

    - ``max_one_minus_r2``: ajuste numérico aceito quando ``1 − R²`` (teste limpo) ≤ limiar.
    - ``max_tree_distance``: estrutura recuperada quando a distância estrutural primária à
      equação de referência ≤ limiar (0 = mesma estrutura, GAP-11).
    """

    max_one_minus_r2: float = 1e-4
    max_tree_distance: float = 0.0

    def __post_init__(self) -> None:
        """Valida os limiares (não negativos)."""
        if self.max_one_minus_r2 < 0 or self.max_tree_distance < 0:
            raise StudyDesignError("os limiares do critério devem ser >= 0")


@dataclass(frozen=True)
class StudyDesign:
    """Desenho experimental completo e reproduzível.

    ``loss_formulation``: ``literal`` = perda do artigo (MSE em unidades SI; trilha
    ``paper_replication``); ``normalized`` = perda normalizada (desvio declarado; trilha
    ``extended_analysis``). Cada repetição usa uma **semente derivada** de ``master_seed`` (dados,
    divisão treino/validação/teste e busca). ``hyperparameters``: sobrescritas de
    ``sr.*`` como pares ``("sr.gplearn.p_crossover", 0.5)`` (uso da Fase 2). ``allow_code_change``:
    permite retomar execuções feitas com outro estado do código (fica registrado em ``run.json``).
    """

    algorithm: str
    scenarios: tuple[str, ...] = SCENARIOS
    repetitions: int = 3
    master_seed: int = 42
    loss_formulation: Literal["literal", "normalized"] = "literal"
    search_budget: SearchBudget = field(default_factory=SearchBudget)
    noise: NoiseCondition = field(default_factory=NoiseCondition)
    output_root: Path | str = "results/nb05_baseline_local"
    n_simulations: int = 500
    n_time_points: int = 100
    hyperparameters: tuple[tuple[str, Any], ...] = ()
    criteria: ConsistencyCriteria = field(default_factory=ConsistencyCriteria)
    allow_code_change: bool = True

    def __post_init__(self) -> None:
        """Valida todos os campos antes de qualquer execução."""
        if self.algorithm not in ALGORITHMS:
            raise StudyDesignError(f"algoritmo inválido: {self.algorithm!r}; use {ALGORITHMS}")
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise StudyDesignError("scenarios deve ser uma tupla não vazia e sem repetições")
        for name in self.scenarios:
            if name not in SCENARIOS:
                raise StudyDesignError(f"cenário inválido: {name!r}; use {SCENARIOS}")
        if self.repetitions < 1 or self.n_simulations < 1 or self.n_time_points < 1:
            raise StudyDesignError("repetitions, n_simulations e n_time_points devem ser >= 1")
        if self.loss_formulation not in LOSS_FORMULATIONS:
            raise StudyDesignError(
                f"formulação da função de perda inválida: {self.loss_formulation!r}; "
                f"use {LOSS_FORMULATIONS}"
            )
        for path, _ in self.hyperparameters:
            if not (path.startswith("sr.") or path in LOSS_HYPERPARAMETERS):
                raise StudyDesignError(
                    f"hiperparâmetro {path!r} fora de 'sr.*'/{list(LOSS_HYPERPARAMETERS)}: "
                    "só a busca e a formulação da perda do baseline podem variar na Fase 2"
                )

    def design_hash(self) -> str:
        """Hash do desenho (ignora saída, retomada e limiares: não mudam as execuções)."""
        data = {k: v for k, v in asdict(self).items() if k not in _HASH_EXCLUDED}
        text = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


@dataclass(frozen=True)
class ExecutedRun:
    """Um conjunto de execuções (um desenho numa fase) e seus arquivos persistidos."""

    phase: str
    condition: str
    design: StudyDesign
    config: ExperimentConfig
    run_dir: Path
    results: pd.DataFrame
    history: pd.DataFrame
    summary: RunSummary | None = None


@dataclass(frozen=True)
class StudyResult:
    """Resultado de uma fase: uma ou mais execuções com o mesmo esquema de saída."""

    phase: str
    design: StudyDesign
    runs: tuple[ExecutedRun, ...]

    @property
    def results(self) -> pd.DataFrame:
        """Linhas de resultado de todas as execuções (uma por cenário × repetição)."""
        frames = [r.results.assign(_condicao=r.condition, _fase=r.phase) for r in self.runs]
        out: pd.DataFrame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return out

    @property
    def run_dirs(self) -> list[Path]:
        """Diretórios das execuções."""
        return [r.run_dir for r in self.runs]


# --------------------------------------------------------------------------- ambiente


def minimal_mode_requested() -> bool:
    """Indica se ``PHYSICS_LLM_SR_NB05_MINIMO`` pede orçamento e dados mínimos (testes/make)."""
    return os.environ.get(MINIMAL_ENV, "").strip().lower() in {"1", "true", "yes", "sim"}


def environment_report() -> dict[str, Any]:
    """Resumo do ambiente local: Python, pacote, raiz de resultados, Julia/PySR e módulos de LLM.

    Não importa PySR (a primeira importação pode demorar); só verifica se está instalado.
    """
    return {
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "raiz_de_resultados": str(project_root()),
        "variavel_raiz": os.environ.get(PROJECT_ROOT_ENV, "(não definida: usa o diretório atual)"),
        "pysr_instalado": importlib.util.find_spec("pysr") is not None,
        "julia_no_path": shutil.which("julia") is not None,
        "modulos_de_llm_importados": sorted(
            m for m in ("transformers", "torch") if m in sys.modules
        ),
    }


def require_algorithm(algorithm: str) -> None:
    """Falha cedo, com instrução clara, se o algoritmo escolhido não está utilizável.

    ``pysr`` exige o pacote ``pysr`` e o Julia (instalado pelo ``juliapkg`` na primeira importação).
    """
    if algorithm not in ALGORITHMS:
        raise StudyDesignError(f"algoritmo inválido: {algorithm!r}; use {ALGORITHMS}")
    if algorithm == "pysr":
        if importlib.util.find_spec("pysr") is None:
            raise StudyDesignError(
                "ALGORITMO='pysr' exige o pacote pysr (pip install pysr) e o Julia; "
                "use 'gplearn' ou 'deap' para rodar sem Julia"
            )
        import pysr  # noqa: F401  (a primeira importação instala/inicia o Julia)


# --------------------------------------------------------------------------- configuração


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Atribui ``value`` em ``data`` seguindo ``a.b.c``; o caminho precisa existir."""
    node: Any = data
    keys = path.split(".")
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise StudyDesignError(f"hiperparâmetro inexistente: {path!r}")
        node = node[key]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise StudyDesignError(f"hiperparâmetro inexistente: {path!r}")
    node[keys[-1]] = value


def build_config(design: StudyDesign) -> ExperimentConfig:
    """Configuração validada do desenho (parte de ``configs/nb01_baseline.yaml``).

    Só células sem LLM (``baseline_only``, backend simulado nunca instanciado). Sem ruído o
    experimento é o baseline com ``sigma_fraction = 0``; com ruído, o Exp. 4 (média zero) ou o
    Exp. 5 (enviesado) restrito à condição pedida.
    """
    base = find_repo_root() / "configs" / BASE_CONFIG
    data: dict[str, Any] = load_config(base).model_dump(mode="json")
    track = ARM_TRACKS["replication" if design.loss_formulation == "literal" else "extended"]
    data["track"] = track["track"]
    data["loss"]["normalization"] = track["normalization"]
    if design.loss_formulation == "literal":
        data["llm"].update(evaluation_scope="all", top_k=None, cache_key_mode="exact")
    data["seeds"]["master"] = design.master_seed
    data["data"]["n_simulations"] = design.n_simulations
    data["data"]["n_time_points"] = design.n_time_points
    noise = design.noise
    data["data"]["noise"] = {
        "sigma_fraction": noise.sigma_fraction,
        "location": noise.location,
    }
    if noise.noise_type == "zero_mean":
        data["data"]["exp4"] = {"levels": [noise.sigma_fraction], "locations": [noise.location]}
    elif noise.noise_type == "biased":
        data["data"]["exp5"]["levels"] = [noise.sigma_fraction]
        data["data"]["exp5"]["locations"] = [noise.location]
    budget = design.search_budget
    data["sr"]["population"] = budget.population
    data["sr"]["generations"] = budget.generations
    data["sr"]["early_stopping"] = {
        "enabled": budget.early_stopping,
        "min_relative_improvement": budget.min_relative_improvement,
        "patience": budget.patience,
    }
    data["sr"]["pysr"].update(
        niterations=budget.pysr_niterations,
        population_size=budget.pysr_population_size,
        ncycles_per_iteration=budget.pysr_ncycles_per_iteration,
    )
    data["experiments"].update(
        run=[_EXPERIMENT_BY_NOISE[noise.noise_type]],
        scenarios=list(design.scenarios),
        algorithms=[design.algorithm],
        repetitions=design.repetitions,
        max_cell_seconds=budget.max_cell_seconds,
        baseline_only=True,
    )
    for path, value in design.hyperparameters:
        _set_path(data, path, value)
    return parse_config(data)


# --------------------------------------------------------------------------- execução


def _run_dir(design: StudyDesign, phase: str) -> Path:
    root = resolve_project_path(design.output_root)
    return root / phase / design.algorithm / design.design_hash()


def _execute(design: StudyDesign, phase: str, condition: str) -> ExecutedRun:
    """Executa (com retomada) um desenho e grava consistência e log padronizados."""
    require_algorithm(design.algorithm)
    config = build_config(design)
    run_dir = _run_dir(design, phase)
    summary = run_experiments(
        config,
        run_dir=run_dir,
        resume=True,
        write_report=False,
        allow_code_change=design.allow_code_change,
    )
    results, history = consolidate(summary.run_dir)
    run = ExecutedRun(phase, condition, design, config, summary.run_dir, results, history, summary)
    write_run_outputs(run, design.criteria)
    return run


def phase1_clean_validation(design: StudyDesign) -> StudyResult:
    """Fase 1 — validação limpa: baseline estritamente sem ruído (σ = 0).

    Levanta :class:`CleanPhaseError` se o desenho trouxer qualquer ruído. Retomável: repetir a
    chamada com o mesmo desenho reaproveita as repetições já concluídas.
    """
    if not design.noise.is_clean:
        raise CleanPhaseError(
            "a Fase 1 exige o regime sem ruído (σ = 0); recebido "
            f"{design.noise.label()}. Use NoiseCondition() ou a Fase 2 (phase2_noise_sweep)."
        )
    run = _execute(design, PHASE1, CLEAN_CONDITION)
    return StudyResult(PHASE1, design, (run,))


def combine_results(results: Sequence[StudyResult]) -> StudyResult:
    """Une resultados da mesma fase (ex.: uma chamada por cenário) num só :class:`StudyResult`."""
    if not results:
        raise StudyDesignError("combine_results exige ao menos um resultado")
    phases = {r.phase for r in results}
    if len(phases) != 1:
        raise StudyDesignError(f"só resultados da mesma fase podem ser unidos: {sorted(phases)}")
    runs = tuple(run for r in results for run in r.runs)
    return StudyResult(results[0].phase, results[0].design, runs)


def phase2_noise_sweep(
    design: StudyDesign,
    levels: Sequence[float],
    locations: Sequence[str] = ("target",),
    noise_type: Literal["zero_mean", "biased"] = "zero_mean",
) -> StudyResult:
    """Fase 2a — ruído controlado: uma execução por (localização × nível de σ)."""
    runs = []
    for location in locations:
        for level in levels:
            noise = NoiseCondition(noise_type, float(level), location)  # type: ignore[arg-type]
            runs.append(_execute(replace(design, noise=noise), PHASE2_NOISE, noise.label()))
    return StudyResult(PHASE2_NOISE, design, tuple(runs))


def phase2_hyperparameter_sweep(
    design: StudyDesign, grid: Mapping[str, Sequence[Any]]
) -> StudyResult:
    """Fase 2b — hiperparâmetros: produto cartesiano de ``grid`` (``{"sr.population": [50]}``)."""
    keys = list(grid)
    runs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        pairs = tuple(zip(keys, values, strict=True))
        label = ", ".join(f"{k.removeprefix('sr.')}={v}" for k, v in pairs)
        runs.append(
            _execute(
                replace(design, hyperparameters=(*design.hyperparameters, *pairs)),
                PHASE2_HYPERPARAMETERS,
                label,
            )
        )
    return StudyResult(PHASE2_HYPERPARAMETERS, design, tuple(runs))


def phase2_algorithm_comparison(design: StudyDesign, algorithms: Sequence[str]) -> StudyResult:
    """Fase 2c — algoritmos: o mesmo desenho executado com cada algoritmo."""
    runs = [_execute(replace(design, algorithm=alg), PHASE2_ALGORITHMS, alg) for alg in algorithms]
    return StudyResult(PHASE2_ALGORITHMS, design, tuple(runs))


# --------------------------------------------------------------------------- controle positivo


def _protection(cfg: ExperimentConfig) -> ProtectionParams:
    return ProtectionParams.from_config(cfg.symbolic)


def dense_grid(cfg: ExperimentConfig, scenario_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Grid denso no domínio físico, fora da amostra de treino: ``(X, y_referência)``.

    Simulações novas (semente ``grid/<cenário>``) com instantes igualmente espaçados em
    ``[0, t_fim]``; o alvo é a solução analítica do cenário.
    """
    scenario = get_scenario(scenario_name, cfg.physics)
    rng = derive_rng(cfg.seeds.master, f"grid/{scenario_name}")
    xs, ys = [], []
    for _ in range(GRID_SIMULATIONS):
        params = scenario.sample_parameters(rng, cfg.physics.sampling)
        t = np.linspace(0.0, scenario.time_end(params), GRID_TIME_POINTS)
        traj = scenario.trajectory(params, t)
        xs.append(traj.X)
        ys.append(traj.y_true)
    return np.concatenate(xs), np.concatenate(ys)


def _max_abs_grid_error(
    expression: str, cfg: ExperimentConfig, scenario_name: str, grid: tuple[np.ndarray, np.ndarray]
) -> float:
    scenario = get_scenario(scenario_name, cfg.physics)
    try:
        yhat = evaluate(parse_infix(expression), grid[0], scenario.feature_names, _protection(cfg))
        return float(np.max(np.abs(yhat - grid[1])))
    except Exception:  # expressão inválida/não avaliável: sem erro numérico definido
        return math.nan


def positive_control(scenario: str, design: StudyDesign | None = None) -> pd.DataFrame:
    """Avalia a equação de referência (e formas equivalentes) nos dados limpos: controle positivo.

    Usa o mesmo caminho de métricas do baseline (``regression_metrics``,
    ``structural_metrics``). Esperado: distância estrutural 0 e ``1 − R²`` ≤
    :data:`POSITIVE_CONTROL_TOLERANCE` (erro de ponto flutuante). Uma linha por forma; a primeira
    (``forma_primaria``) é a equação de referência do artigo. Se falhar, o problema está nos dados
    ou nas métricas, não no algoritmo.
    """
    design = design or StudyDesign(algorithm="gplearn", scenarios=(scenario,))
    cfg = build_config(design)
    provider = DataProvider(cfg)
    data: Dataset = provider.clean(scenario, 0)
    clean = bool(np.array_equal(data.y_observed, data.y_true))
    sc = get_scenario(scenario, cfg.physics)
    forms = sc.equivalent_ground_truths()
    gt = parse_infix(forms[0])
    equivalents = [parse_infix(f) for f in forms[1:]]
    params = TreeDistanceParams.from_config(cfg.symbolic)
    grid = dense_grid(cfg, scenario)
    rows = []
    for index, form in enumerate(forms):
        expr = parse_infix(form)
        yhat = evaluate(expr, data.X, data.feature_names, _protection(cfg))
        reg = regression_metrics(data.y_observed, yhat)
        struct = structural_metrics(expr, gt, equivalents, params)
        fits = reg["one_minus_r2"] <= POSITIVE_CONTROL_TOLERANCE
        same = struct["tree_distance_equivalent_min"] == 0.0
        primary_ok = struct["tree_distance"] == 0.0 if index == 0 else True
        rows.append(
            {
                "cenario": scenario,
                "forma": form,
                "forma_primaria": index == 0,
                "mae": reg["mae"],
                "mse": reg["mse"],
                "um_menos_r2": reg["one_minus_r2"],
                "erro_max_abs_grid": _max_abs_grid_error(form, cfg, scenario, grid),
                "distancia_estrutural": struct["tree_distance"],
                "distancia_equivalente_min": struct["tree_distance_equivalent_min"],
                "dados_limpos": clean,
                "aprovado": bool(fits and same and primary_ok and clean),
            }
        )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


# --------------------------------------------------------------------------- consistência


def classify(
    one_minus_r2: float,
    tree_distance: float,
    criteria: ConsistencyCriteria,
    status: str = "ok",
) -> str:
    """Classifica uma repetição pelos limiares visíveis de ``criteria``.

    - :data:`CLASS_RECOVERS`: ``1 − R²`` ≤ limiar de ajuste **e** distância ≤ limiar de estrutura;
    - :data:`CLASS_FITS`: ajuste dentro do limiar, mas estrutura acima do limiar;
    - :data:`CLASS_FAILS`: ajuste fora do limiar (ou métricas não finitas);
    - :data:`CLASS_ERROR`: a execução falhou (``status != 'ok'``).
    """
    if status != "ok":
        return CLASS_ERROR
    if not math.isfinite(one_minus_r2) or one_minus_r2 > criteria.max_one_minus_r2:
        return CLASS_FAILS
    if math.isfinite(tree_distance) and tree_distance <= criteria.max_tree_distance:
        return CLASS_RECOVERS
    return CLASS_FITS


def _num(row: Mapping[Any, Any], key: str) -> float:
    value = row.get(key)
    return math.nan if value is None or pd.isna(value) else float(value)


def _report_rows(run: ExecutedRun, criteria: ConsistencyCriteria) -> list[dict[str, Any]]:
    cfg = run.config
    grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    design_hash = run.design.design_hash()
    for record in run.results.to_dict("records"):
        scenario = str(record["scenario"])
        status = str(record.get("status", "error"))
        expression = record.get("expression")
        expression = None if expression is None or pd.isna(expression) else str(expression)
        if expression is not None and scenario not in grids:
            grids[scenario] = dense_grid(cfg, scenario)
        grid_error = (
            _max_abs_grid_error(expression, cfg, scenario, grids[scenario])
            if expression is not None and status == "ok"
            else math.nan
        )
        r2 = _num(record, "test_true_one_minus_r2")
        dist = _num(record, "tree_distance")
        exact = record.get("exact_recovery")
        rows.append(
            {
                "fase": run.phase,
                "condicao": run.condition,
                "cenario": scenario,
                "algoritmo": record.get("sr", run.design.algorithm),
                "repeticao": int(record.get("repetition", 0)),
                "semente_busca": int(record["seed"]) if not pd.isna(record.get("seed")) else -1,
                "sigma": _num(record, "noise_level"),
                "tipo_ruido": record.get("noise_type"),
                "localizacao_ruido": record.get("noise_location"),
                "equacao_descoberta": expression,
                "equacao_referencia": record.get("ground_truth"),
                "mae_teste": _num(record, "test_true_mae"),
                "mse_teste": _num(record, "test_true_mse"),
                "um_menos_r2_teste": r2,
                "erro_max_abs_grid": grid_error,
                "distancia_estrutural": dist,
                "distancia_equivalente_min": _num(record, "tree_distance_equivalent_min"),
                "recuperacao_exata": None if exact is None or pd.isna(exact) else bool(exact),
                "geracoes": _num(record, "n_generations"),
                "motivo_parada": record.get("stop_reason"),
                "tempo_s": _num(record, "cell_seconds"),
                "status_execucao": status,
                "classificacao": classify(r2, dist, criteria, status),
                "hash_desenho": design_hash,
            }
        )
    return rows


def consistency_report(
    result: StudyResult, criteria: ConsistencyCriteria | None = None
) -> pd.DataFrame:
    """Relatório de consistência: uma linha por cenário × repetição (× condição na Fase 2).

    Compara o modelo descoberto com a equação de referência: métricas no **teste limpo**
    (MAE, MSE, ``1 − R²`` contra o alvo verdadeiro), erro máximo absoluto num grid denso do
    domínio físico (fora da amostra de treino), distância estrutural primária e distância mínima
    a formas equivalentes (secundária), recuperação exata, gerações, motivo de parada, tempo e a
    classificação pelos limiares de ``criteria`` (padrão: ``result.design.criteria``).
    Esquema fixo: :data:`CONSISTENCY_COLUMNS`.
    """
    crit = criteria or result.design.criteria
    rows = [row for run in result.runs for row in _report_rows(run, crit)]
    report: pd.DataFrame = pd.DataFrame(rows, columns=list(CONSISTENCY_COLUMNS))
    return report


def phase1_approval(
    report: pd.DataFrame,
    min_fraction: float = 1.0,
    accepted: Sequence[str] = (CLASS_RECOVERS, CLASS_FITS),
) -> pd.DataFrame:
    """Critério de aprovação da Fase 1, por cenário × algoritmo.

    Aprovado quando a fração de repetições em ``accepted`` é ≥ ``min_fraction``. Padrão: o modelo
    recupera o **comportamento** esperado ("recupera estrutura" ou "ajusta numericamente");
    passe ``accepted=(CLASS_RECOVERS,)`` para exigir também a estrutura. Devolve ``n``,
    contagem por classe, ``fracao_aceita`` e ``aprovado``.
    """
    if not 0.0 <= min_fraction <= 1.0:
        raise StudyDesignError("min_fraction deve estar em [0, 1]")
    rows = []
    for (scenario, algorithm), sub in report.groupby(["cenario", "algoritmo"], sort=False):
        counts = sub["classificacao"].value_counts()
        fraction = float(sub["classificacao"].isin(list(accepted)).mean())
        rows.append(
            {
                "cenario": scenario,
                "algoritmo": algorithm,
                "n": len(sub),
                **{
                    c: int(counts.get(c, 0))
                    for c in (CLASS_RECOVERS, CLASS_FITS, CLASS_FAILS, CLASS_ERROR)
                },
                "fracao_aceita": fraction,
                "aprovado": fraction >= min_fraction,
            }
        )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


# --------------------------------------------------------------------------- logs padronizados


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def write_run_outputs(run: ExecutedRun, criteria: ConsistencyCriteria) -> None:
    """Grava ``consistency.csv`` e ``study_log.jsonl`` (uma linha por execução) em ``run_dir``.

    Os arquivos são regenerados por inteiro a partir de ``results`` (idempotente na retomada).
    """
    result = StudyResult(run.phase, run.design, (run,))
    report = consistency_report(result, criteria)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(run.run_dir / CONSISTENCY_FILE, index=False)
    hyper = {
        "formulacao_perda": run.design.loss_formulation,
        "orcamento_de_busca": asdict(run.design.search_budget),
        **dict(run.design.hyperparameters),
    }
    lines = []
    for row in report.to_dict("records"):
        entry = {
            "fase": row["fase"],
            "condicao": row["condicao"],
            "cenario": row["cenario"],
            "algoritmo": row["algoritmo"],
            "repeticao": row["repeticao"],
            "semente": row["semente_busca"],
            "semente_mestra": run.design.master_seed,
            "sigma": row["sigma"],
            "tipo_ruido": row["tipo_ruido"],
            "localizacao_ruido": row["localizacao_ruido"],
            "hiperparametros": hyper,
            "metricas": {
                k: row[k]
                for k in (
                    "mae_teste",
                    "mse_teste",
                    "um_menos_r2_teste",
                    "erro_max_abs_grid",
                    "distancia_estrutural",
                    "distancia_equivalente_min",
                    "geracoes",
                    "tempo_s",
                )
            },
            "equacao_descoberta": row["equacao_descoberta"],
            "equacao_referencia": row["equacao_referencia"],
            "status": {
                "execucao": row["status_execucao"],
                "classificacao": row["classificacao"],
                "motivo_parada": row["motivo_parada"],
            },
            "desenho_hash": row["hash_desenho"],
        }
        lines.append(json.dumps(_json_safe(entry), ensure_ascii=False, default=str))
    (run.run_dir / STUDY_LOG_FILE).write_text("\n".join(lines) + ("\n" if lines else ""), "utf-8")
