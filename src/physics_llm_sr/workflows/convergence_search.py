"""Busca de configuração para fazer o baseline de regressão simbólica convergir (sem LLM).

Antes de inserir o LLM na seleção, o baseline precisa recuperar a equação de referência (ou uma
forma fisicamente equivalente declarada pelo cenário) no regime sem ruído. Este módulo organiza a
busca dessa configuração em etapas, **reutilizando** :mod:`~physics_llm_sr.workflows.baseline_study`
(desenho experimental, execução retomável, relatório de consistência e logs). Nada aqui
reimplementa regressão simbólica, geração de dados ou métricas.

Etapas:

1. :func:`expressiveness_diagnosis` — o conjunto de operadores consegue escrever a referência (ou
   uma forma equivalente)? Qual a perda ``L`` de cada forma e de candidatas triviais (constante,
   uma variável)? Se uma trivial tem ``L`` menor que a referência, nenhuma busca converge sob
   essa formulação da perda.
2. :func:`run_stage` — triagem: lista **ordenada** de configurações candidatas
   (:class:`SearchSpace`), ``repetitions`` execuções cada, uma linha por configuração.
3. :func:`refine` — a partir da melhor, procura a configuração **mais barata** (população ×
   gerações menores) que ainda converge.
4. :func:`confirm` — reexecuta a configuração escolhida em **sementes novas** (semente mestra
   deslocada), disjuntas das usadas na busca: evita escolher a semente sortuda.
5. :func:`export_best` — grava ``best_config.yaml``/``best_config.json`` (pronto para o
   ``SearchBudget``/``hyperparameters`` do notebook 05) e ``search_table.csv``.

Critério de convergência (:class:`ConvergenceCriterion`, limiares visíveis): uma repetição é bem
sucedida quando ``min(distância estrutural primária, distância equivalente) ≤ 0,05`` **e**
``1 − R²`` (teste limpo) ``≤ 1e-3``; a configuração converge quando a fração de repetições bem
sucedidas é ``≥ 2/3``.

Configurações candidatas são dicionários de caminhos de configuração (``sr.*`` e
``loss.normalization``) → valor. Caminhos de orçamento (``sr.population``, ``sr.generations``,
``sr.early_stopping.*``, ``sr.pysr.niterations``/``population_size``/``ncycles_per_iteration``)
viram o :class:`~physics_llm_sr.workflows.baseline_study.SearchBudget`; ``loss.normalization``
vira a formulação da função de perda; o resto vira ``StudyDesign.hyperparameters``.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.experiments.data_provider import DataProvider
from physics_llm_sr.optimization.objective import build_objective
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import Expr, Op, Var, parse_infix, size
from physics_llm_sr.workflows import baseline_study as bs

# --------------------------------------------------------------------------- constantes

OUTPUT_ROOT = "results/convergence_search"
"""Raiz padrão dos resultados (relativa a ``PHYSICS_LLM_SR_PROJECT_ROOT`` ou ao diretório atual)."""

STAGE_SCREENING = "triagem"
STAGE_REFINEMENT = "refinamento"
STAGE_CONFIRMATION = "confirmacao"

STAGE_LOG_FILE = "study_log.jsonl"
SEARCH_TABLE_FILE = "search_table.csv"
BEST_YAML = "best_config.yaml"
BEST_JSON = "best_config.json"

DEFAULT_CONFIRMATION_OFFSET = 1000
"""Deslocamento padrão da semente mestra na confirmação (sementes novas, fora da busca)."""

MAX_WORKERS = 8
"""Teto de processos paralelos (DEAP/gplearn). PySR roda sempre em série (Julia)."""

SERIAL_ALGORITHMS = frozenset({"pysr"})

DEFAULT_REFINE_FACTORS: tuple[tuple[float, float], ...] = (
    (0.25, 0.5),
    (0.25, 1.0),
    (0.5, 0.5),
    (0.5, 1.0),
    (1.0, 0.5),
)
"""Fatores (população, gerações) testados no refinamento, além da própria configuração."""

LOSS_NORMALIZATION_PATH = "loss.normalization"

MINIMAL_ENV = "PHYSICS_LLM_SR_NB06_MINIMO"
"""Variável de ambiente que pede dados e orçamento mínimos ao notebook 06 (testes)."""

_BUDGET_PATHS: dict[str, str] = {
    "sr.population": "population",
    "sr.generations": "generations",
    "sr.early_stopping.enabled": "early_stopping",
    "sr.early_stopping.min_relative_improvement": "min_relative_improvement",
    "sr.early_stopping.patience": "patience",
    "sr.pysr.niterations": "pysr_niterations",
    "sr.pysr.population_size": "pysr_population_size",
    "sr.pysr.ncycles_per_iteration": "pysr_ncycles_per_iteration",
}
"""Caminhos de configuração que pertencem ao :class:`SearchBudget` (campo correspondente)."""

TABLE_COLUMNS: tuple[str, ...] = (
    "etapa",
    "ordem",
    "rotulo",
    "algoritmo",
    "cenario",
    "converge",
    "n_execucoes",
    "n_sucesso",
    "taxa_sucesso",
    "mediana_um_menos_r2",
    "mediana_distancia_estrutural",
    "min_distancia_estrutural",
    "mediana_distancia_equivalente",
    "min_distancia_equivalente",
    "mediana_distancia_min",
    "geracoes_media",
    "tempo_medio_s",
    "custo_estimado",
    "expressao_tipica",
    "expressoes",
    "sementes",
    "desvios",
    "config",
)
"""Esquema fixo da tabela de uma etapa (uma linha por configuração)."""


class ConvergenceSearchError(ValueError):
    """Parâmetro inválido da busca de convergência (mensagem em português)."""


def minimal_mode_requested() -> bool:
    """Indica se ``PHYSICS_LLM_SR_NB06_MINIMO`` pede dados e orçamento mínimos (testes)."""
    return os.environ.get(MINIMAL_ENV, "").strip().lower() in {"1", "true", "yes", "sim"}


# --------------------------------------------------------------------------- critério


@dataclass(frozen=True)
class ConvergenceCriterion:
    """Critério de convergência do baseline (limiares visíveis, nenhuma tolerância escondida).

    - ``max_tree_distance``: limiar para ``min(distância estrutural primária, distância mínima a
      formas equivalentes)``; 0 = mesma estrutura (GAP-11). 0,05 tolera constantes próximas
      (α = 1: ``|Δc| ≤ 0,15`` numa árvore de 3 nós).
    - ``max_one_minus_r2``: limiar de ``1 − R²`` no teste limpo.
    - ``min_success_fraction``: fração mínima de repetições bem sucedidas para a configuração
      convergir.
    """

    max_tree_distance: float = 0.05
    max_one_minus_r2: float = 1e-3
    min_success_fraction: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        """Valida os limiares."""
        if self.max_tree_distance < 0 or self.max_one_minus_r2 < 0:
            raise ConvergenceSearchError("os limiares do critério devem ser >= 0")
        if not 0.0 < self.min_success_fraction <= 1.0:
            raise ConvergenceSearchError("min_success_fraction deve estar em (0, 1]")

    def repetition_succeeds(
        self, one_minus_r2: float, tree_distance: float, equivalent_distance: float
    ) -> bool:
        """Uma repetição é bem sucedida? (métricas não finitas = fracasso)."""
        distances = [d for d in (tree_distance, equivalent_distance) if math.isfinite(d)]
        if not distances or not math.isfinite(one_minus_r2):
            return False
        return min(distances) <= self.max_tree_distance and one_minus_r2 <= self.max_one_minus_r2

    def converges(self, n_success: int, n_runs: int) -> bool:
        """Diz se a configuração converge (``n_success / n_runs ≥ min_success_fraction``)."""
        if n_runs <= 0:
            return False
        # tolerância de ponto flutuante: 2/3 de 3 execuções = 2 sucessos
        return n_success / n_runs >= self.min_success_fraction - 1e-12


# --------------------------------------------------------------------------- espaço de busca


@dataclass(frozen=True)
class Candidate:
    """Uma configuração candidata: rótulo, sobrescritas de configuração e desvios declarados.

    ``overrides``: caminho (``sr.*`` ou ``loss.normalization``) → valor. ``deviations``: caminhos
    cujo valor difere da configuração literal do baseline (``configs/nb01_baseline.yaml``, valores
    do artigo ou hipóteses H-02/H-03); calculado por :func:`make_candidate`. ``hypothesis``: qual
    hipótese de não convergência a candidata testa (H1..H5).
    """

    label: str
    overrides: Mapping[str, Any]
    deviations: tuple[str, ...] = ()
    hypothesis: str = ""

    def __post_init__(self) -> None:
        """Valida o rótulo e os caminhos."""
        if not self.label.strip():
            raise ConvergenceSearchError("Candidate.label não pode ser vazio")
        for path in self.overrides:
            if not (path.startswith("sr.") or path == LOSS_NORMALIZATION_PATH):
                raise ConvergenceSearchError(
                    f"caminho {path!r} fora de 'sr.*'/'loss.normalization': "
                    "só a busca e a formulação da perda podem variar"
                )

    def as_dict(self) -> dict[str, Any]:
        """Representação serializável (JSON/YAML)."""
        return {
            "rotulo": self.label,
            "hipotese": self.hypothesis,
            "overrides": {k: _plain(v) for k, v in self.overrides.items()},
            "desvios": list(self.deviations),
        }


def _plain(value: Any) -> Any:
    """Converta tuplas/numpy em tipos JSON/YAML simples."""
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _base_config_dict() -> dict[str, Any]:
    from physics_llm_sr.colab import find_repo_root

    base = find_repo_root() / "configs" / bs.BASE_CONFIG
    data: dict[str, Any] = load_config(base).model_dump(mode="json")
    return data


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise ConvergenceSearchError(f"caminho de configuração inexistente: {path!r}")
        node = node[key]
    return node


def deviations_from_base(overrides: Mapping[str, Any]) -> tuple[str, ...]:
    """Caminhos de ``overrides`` cujo valor difere da configuração literal do baseline.

    Referência: ``configs/nb01_baseline.yaml`` (valores do artigo e hipóteses H-02/H-03), exceto o
    orçamento do PySR, cuja referência é o do artigo/piloto (H-02: 1 população de
    ``sr.population`` indivíduos, ``sr.generations`` iterações, 380 ciclos).
    """
    base = _base_config_dict()
    paper_pysr = {
        "sr.pysr.niterations": base["sr"]["generations"],
        "sr.pysr.population_size": base["sr"]["population"],
        "sr.pysr.ncycles_per_iteration": 380,
    }
    out = []
    for path, value in overrides.items():
        reference = paper_pysr[path] if path in paper_pysr else _get_path(base, path)
        if _plain(value) != _plain(reference):
            out.append(path)
    return tuple(out)


def make_candidate(label: str, overrides: Mapping[str, Any], hypothesis: str = "") -> Candidate:
    """Candidata com os desvios calculados contra a configuração literal do baseline."""
    return Candidate(label, dict(overrides), deviations_from_base(overrides), hypothesis)


def _with_sqrt(algorithm: str) -> list[str]:
    ops: list[str] = list(_base_config_dict()["sr"]["operators"][algorithm])
    return [*ops, "sqrt"] if "sqrt" not in ops else ops


@dataclass(frozen=True)
class SearchSpace:
    """Lista ORDENADA de configurações candidatas para um algoritmo.

    A ordem é a das hipóteses mais prováveis primeiro; :func:`run_stage` executa nessa ordem e,
    com ``stop_on_success=True``, para na primeira que converge.
    """

    algorithm: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        """Valida o algoritmo e a unicidade dos rótulos."""
        if self.algorithm not in bs.ALGORITHMS:
            raise ConvergenceSearchError(
                f"algoritmo inválido: {self.algorithm!r}; use {bs.ALGORITHMS}"
            )
        labels = [c.label for c in self.candidates]
        if len(set(labels)) != len(labels):
            raise ConvergenceSearchError("rótulos de candidatas repetidos")

    def __len__(self) -> int:
        """Número de candidatas."""
        return len(self.candidates)

    @classmethod
    def default(cls, algorithm: str) -> SearchSpace:
        """Espaço padrão por algoritmo (hipóteses H1–H5, combinando as mais prováveis primeiro).

        H1 parada antecipada agressiva; H2 faixa de constantes; H3 pressão seletiva (torneio);
        H4 orçamento (população/gerações); H5 conjunto de operadores sem ``sqrt``. DEAP: também
        probabilidades de crossover/mutação do artigo (0,05/0,01) que quase não exploram.
        PySR: orçamento reduzido (populações, tamanho, ciclos, iterações).
        """
        builders: dict[str, Callable[[], list[Candidate]]] = {
            "gplearn": _gplearn_space,
            "deap": _deap_space,
            "pysr": _pysr_space,
        }
        if algorithm not in builders:
            raise ConvergenceSearchError(f"algoritmo inválido: {algorithm!r}; use {bs.ALGORITHMS}")
        return cls(algorithm, tuple(builders[algorithm]()))

    @classmethod
    def minimal(cls, algorithm: str) -> SearchSpace:
        """Espaço mínimo (segundos) para validar o encanamento; sem valor científico."""
        tiny = {"sr.population": bs.MINIMAL_POPULATION, "sr.generations": bs.MINIMAL_GENERATIONS}
        if algorithm == "pysr":
            tiny = {
                "sr.pysr.population_size": 10,
                "sr.pysr.niterations": 2,
                "sr.pysr.ncycles_per_iteration": 5,
            }
        return cls(
            algorithm,
            (
                make_candidate("mínimo: literal", dict(tiny), "encanamento"),
                make_candidate("mínimo: sem parada", {**tiny, **_NO_ES}, "encanamento"),
            ),
        )

    def within_cost(self, max_cost: float) -> SearchSpace:
        """Só as candidatas com custo estimado ≤ ``max_cost`` (ver :func:`estimated_cost`)."""
        kept = tuple(
            c for c in self.candidates if estimated_cost(self.algorithm, c.overrides) <= max_cost
        )
        if not kept:
            raise ConvergenceSearchError(f"nenhuma candidata com custo estimado <= {max_cost:g}")
        return replace(self, candidates=kept)


_NO_ES = {"sr.early_stopping.enabled": False}
_PATIENT_ES = {"sr.early_stopping.enabled": True, "sr.early_stopping.patience": 20}


def _gplearn_space() -> list[Candidate]:
    sq = {"sr.operators.gplearn": _with_sqrt("gplearn")}
    rng10 = {"sr.gplearn.const_range": (-10.0, 10.0)}
    rng20 = {"sr.gplearn.const_range": (0.0, 20.0)}
    mut = {
        "sr.gplearn.p_crossover": 0.7,
        "sr.gplearn.p_subtree_mutation": 0.1,
        "sr.gplearn.p_hoist_mutation": 0.05,
        "sr.gplearn.p_point_mutation": 0.1,
    }
    pop = {"sr.population": 1000, "sr.generations": 100}
    big = {"sr.population": 2000, "sr.generations": 200}
    t7 = {"sr.gplearn.tournament_size": 7}
    t20 = {"sr.gplearn.tournament_size": 20}
    specs: list[tuple[str, str, dict[str, Any]]] = [
        ("H1 sem parada antecipada", "H1", {**_NO_ES}),
        ("H1+H2 sem parada, constantes (-10,10)", "H1+H2", {**_NO_ES, **rng10}),
        ("H1+H2 sem parada, constantes (0,20)", "H1+H2", {**_NO_ES, **rng20}),
        ("H1+H2+H3 constantes (-10,10), torneio 7", "H1+H2+H3", {**_NO_ES, **rng10, **t7}),
        ("H1+H2+H3+H4 (-10,10), torneio 7, 1000x100", "H1-H4", {**_NO_ES, **rng10, **t7, **pop}),
        (
            "H1-H4 (-10,10), torneio 20, mutação pontual 0,1, 1000x100",
            "H1-H4",
            {**_NO_ES, **rng10, **t20, **mut, **pop},
        ),
        (
            "H1-H4 (0,20), torneio 20, mutação pontual 0,1, 1000x100",
            "H1-H4",
            {**_NO_ES, **rng20, **t20, **mut, **pop},
        ),
        (
            "H1-H5 (-10,10), torneio 20, sqrt, 1000x100",
            "H1-H5",
            {**_NO_ES, **rng10, **t20, **pop, **sq},
        ),
        (
            "H1-H4 (-10,10), torneio 20, mutação 0,1, 2000x200",
            "H1-H4",
            {**_NO_ES, **rng10, **t20, **mut, **big},
        ),
        (
            "paciência 20 (-10,10), torneio 20, 1000x100",
            "H1'-H4",
            {**_PATIENT_ES, **rng10, **t20, **pop},
        ),
    ]
    return [make_candidate(label, ov, hyp) for label, hyp, ov in specs]


def _deap_space() -> list[Candidate]:
    sq = {"sr.operators.deap": _with_sqrt("deap")}
    var = {"sr.deap.crossover_prob": 0.7, "sr.deap.mutation_prob": 0.2}
    var9 = {"sr.deap.crossover_prob": 0.9, "sr.deap.mutation_prob": 0.1}
    pop = {"sr.population": 500, "sr.generations": 100}
    big = {"sr.population": 1000, "sr.generations": 200}
    t7 = {"sr.deap.tournament_size": 7}
    specs: list[tuple[str, str, dict[str, Any]]] = [
        ("H1 sem parada antecipada", "H1", {**_NO_ES}),
        ("H1 + cx 0,7 / mut 0,2", "H1+DEAP", {**_NO_ES, **var}),
        ("H1 + cx 0,9 / mut 0,1", "H1+DEAP", {**_NO_ES, **var9}),
        ("H1+H3 cx 0,7 / mut 0,2, torneio 7", "H1+H3", {**_NO_ES, **var, **t7}),
        ("H1+H3+H4 cx 0,7 / mut 0,2, torneio 7, 500x100", "H1-H4", {**_NO_ES, **var, **t7, **pop}),
        (
            "H1-H5 cx 0,7 / mut 0,2, torneio 7, sqrt, 500x100",
            "H1-H5",
            {**_NO_ES, **var, **t7, **pop, **sq},
        ),
        ("H1-H4 cx 0,7 / mut 0,2, torneio 7, 1000x200", "H1-H4", {**_NO_ES, **var, **t7, **big}),
        (
            "paciência 20, cx 0,7 / mut 0,2, torneio 7, 500x100",
            "H1'-H4",
            {**_PATIENT_ES, **var, **t7, **pop},
        ),
    ]
    return [make_candidate(label, ov, hyp) for label, hyp, ov in specs]


def _pysr_space() -> list[Candidate]:
    sq = {"sr.operators.pysr": _with_sqrt("pysr")}
    paper = {
        "sr.pysr.niterations": 50,
        "sr.pysr.population_size": 100,
        "sr.pysr.ncycles_per_iteration": 380,
    }
    multi = {
        "sr.pysr.populations": 8,
        "sr.pysr.population_size": 50,
        "sr.pysr.ncycles_per_iteration": 200,
        "sr.pysr.niterations": 20,
    }
    specs: list[tuple[str, str, dict[str, Any]]] = [
        ("H1 sem parada antecipada (orçamento reduzido)", "H1", {**_NO_ES}),
        (
            "H1+H4 ciclos 380, 20 iterações",
            "H1+H4",
            {**_NO_ES, "sr.pysr.ncycles_per_iteration": 380, "sr.pysr.niterations": 20},
        ),
        ("H1+H4 orçamento do artigo (1x100, 380 ciclos, 50 it.)", "H1+H4", {**_NO_ES, **paper}),
        ("H1+H4 8 populações x 50, 200 ciclos, 20 it.", "H1+H4", {**_NO_ES, **multi}),
        ("H1+H4+H5 8 populações x 50, sqrt", "H1-H5", {**_NO_ES, **multi, **sq}),
    ]
    return [make_candidate(label, ov, hyp) for label, hyp, ov in specs]


# --------------------------------------------------------------------------- desenho


def split_overrides(
    overrides: Mapping[str, Any], base_budget: bs.SearchBudget | None = None
) -> tuple[bs.SearchBudget, tuple[tuple[str, Any], ...], str]:
    """Separa as sobrescritas em ``(SearchBudget, hyperparameters, formulação da perda)``."""
    budget = base_budget or bs.SearchBudget()
    budget_kwargs: dict[str, Any] = {}
    hyper: list[tuple[str, Any]] = []
    loss_formulation = "literal"
    for path, value in overrides.items():
        if path in _BUDGET_PATHS:
            budget_kwargs[_BUDGET_PATHS[path]] = value
        elif path == LOSS_NORMALIZATION_PATH:
            loss_formulation = str(value)
        else:
            hyper.append((path, value))
    return replace(budget, **budget_kwargs), tuple(hyper), loss_formulation


def estimated_cost(algorithm: str, overrides: Mapping[str, Any]) -> float:
    """Custo nominal de busca (avaliações de indivíduos), sem considerar a parada antecipada.

    DEAP/gplearn: ``população × gerações``. PySR: ``populações × tamanho × ciclos × iterações``.
    """
    budget, hyper, _ = split_overrides(overrides)
    if algorithm == "pysr":
        base = _base_config_dict()["sr"]["pysr"]
        pops = dict(hyper).get("sr.pysr.populations", base["populations"])
        return float(
            pops
            * budget.pysr_population_size
            * budget.pysr_ncycles_per_iteration
            * budget.pysr_niterations
        )
    return float(budget.population * budget.generations)


def design_for(
    algorithm: str,
    scenario: str,
    candidate: Candidate,
    *,
    repetitions: int,
    master_seed: int,
    output_root: Path | str,
    n_simulations: int = 500,
    n_time_points: int = 100,
    max_cell_seconds: float | None = 1800.0,
) -> bs.StudyDesign:
    """``StudyDesign`` (Fase 1, sem ruído) de uma candidata."""
    base = bs.SearchBudget(max_cell_seconds=max_cell_seconds)
    budget, hyper, formulation = split_overrides(candidate.overrides, base)
    return bs.StudyDesign(
        algorithm=algorithm,
        scenarios=(scenario,),
        repetitions=repetitions,
        master_seed=master_seed,
        loss_formulation=formulation,  # type: ignore[arg-type]
        search_budget=budget,
        output_root=output_root,
        n_simulations=n_simulations,
        n_time_points=n_time_points,
        hyperparameters=hyper,
    )


# --------------------------------------------------------------------------- execução


@dataclass(frozen=True)
class StageResult:
    """Resultado de uma etapa: tabela (uma linha por configuração) e relatórios por repetição."""

    stage: str
    algorithm: str
    scenario: str
    table: pd.DataFrame
    reports: Mapping[str, pd.DataFrame]
    candidates: tuple[Candidate, ...]
    criterion: ConvergenceCriterion
    stage_dir: Path
    elapsed_seconds: float
    stopped_early: bool = False

    @property
    def best(self) -> Candidate:
        """Melhor candidata (primeira linha da tabela ordenada)."""
        if self.table.empty:
            raise ConvergenceSearchError("etapa sem execuções")
        label = str(self.table.iloc[0]["rotulo"])
        return next(c for c in self.candidates if c.label == label)

    @property
    def converged(self) -> pd.DataFrame:
        """Linhas das configurações que convergem."""
        out: pd.DataFrame = self.table[self.table["converge"]]
        return out

    @property
    def seeds(self) -> set[int]:
        """Sementes de busca usadas na etapa (todas as repetições)."""
        return {
            int(s) for rep in self.reports.values() for s in rep["semente_busca"] if int(s) >= 0
        }


def _run_candidate(design: bs.StudyDesign) -> pd.DataFrame:
    """Executa uma candidata (todas as repetições) e devolve o relatório de consistência."""
    result = bs.phase1_clean_validation(design)
    return bs.consistency_report(result)


def _median(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(finite) if finite else math.nan


def _min(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(finite) if finite else math.nan


def _num(value: Any) -> float:
    return math.nan if value is None or pd.isna(value) else float(value)


def successes(report: pd.DataFrame, criterion: ConvergenceCriterion) -> list[bool]:
    """Sucesso de cada repetição do relatório de consistência pelo ``criterion``."""
    return [
        criterion.repetition_succeeds(
            _num(row["um_menos_r2_teste"]),
            _num(row["distancia_estrutural"]),
            _num(row["distancia_equivalente_min"]),
        )
        if row["status_execucao"] == "ok"
        else False
        for row in report.to_dict("records")
    ]


def summarize(
    report: pd.DataFrame,
    candidate: Candidate,
    criterion: ConvergenceCriterion,
    *,
    stage: str,
    order: int,
    algorithm: str,
    scenario: str,
) -> dict[str, Any]:
    """Uma linha da tabela da etapa a partir do relatório de consistência da candidata."""
    ok = successes(report, criterion)
    n_success = int(sum(ok))
    n = len(report)
    dist = [_num(v) for v in report["distancia_estrutural"]]
    eq = [_num(v) for v in report["distancia_equivalente_min"]]
    dmin = [_min([a, b]) for a, b in zip(dist, eq, strict=True)]
    expressions = [str(e) for e in report["equacao_descoberta"] if e is not None and not pd.isna(e)]
    typical = Counter(expressions).most_common(1)[0][0] if expressions else ""
    if expressions and Counter(expressions).most_common(1)[0][1] == 1:
        # nenhuma repetida: a de menor distância (a mais representativa do sucesso)
        finite = [(d, e) for d, e in zip(dmin, report["equacao_descoberta"], strict=True)]
        typical = str(min(finite, key=lambda p: p[0] if math.isfinite(p[0]) else math.inf)[1])
    return {
        "etapa": stage,
        "ordem": order,
        "rotulo": candidate.label,
        "algoritmo": algorithm,
        "cenario": scenario,
        "converge": criterion.converges(n_success, n),
        "n_execucoes": n,
        "n_sucesso": n_success,
        "taxa_sucesso": n_success / n if n else 0.0,
        "mediana_um_menos_r2": _median(report["um_menos_r2_teste"]),
        "mediana_distancia_estrutural": _median(dist),
        "min_distancia_estrutural": _min(dist),
        "mediana_distancia_equivalente": _median(eq),
        "min_distancia_equivalente": _min(eq),
        "mediana_distancia_min": _median(dmin),
        "geracoes_media": float(np.nanmean(report["geracoes"].astype(float))) if n else math.nan,
        "tempo_medio_s": float(np.nanmean(report["tempo_s"].astype(float))) if n else math.nan,
        "custo_estimado": estimated_cost(algorithm, candidate.overrides),
        "expressao_tipica": typical,
        "expressoes": " | ".join(expressions),
        "sementes": ",".join(str(int(s)) for s in report["semente_busca"]),
        "desvios": ",".join(candidate.deviations),
        "config": json.dumps(candidate.as_dict()["overrides"], ensure_ascii=False, sort_keys=True),
    }


def rank_table(table: pd.DataFrame) -> pd.DataFrame:
    """Ordene as configurações da melhor para a pior.

    Prioridade: converge, taxa de sucesso, distância mínima mediana, ``1 − R²``, custo estimado e
    tempo médio.
    """
    if table.empty:
        return table
    ranked = table.assign(
        _d=table["mediana_distancia_min"].fillna(math.inf),
        _r=table["mediana_um_menos_r2"].fillna(math.inf),
    ).sort_values(
        ["converge", "taxa_sucesso", "_d", "_r", "custo_estimado", "tempo_medio_s"],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    )
    out: pd.DataFrame = ranked.drop(columns=["_d", "_r"]).reset_index(drop=True)
    return out


def _stage_dir(output_root: Path | str, algorithm: str, scenario: str, stage: str) -> Path:
    return resolve_project_path(output_root) / algorithm / scenario / stage


def _write_stage_log(
    stage_dir: Path,
    stage: str,
    reports: Mapping[str, pd.DataFrame],
    candidates: Sequence[Candidate],
    criterion: ConvergenceCriterion,
) -> None:
    by_label = {c.label: c for c in candidates}
    lines = []
    for label, report in reports.items():
        ok = successes(report, criterion)
        for row, success in zip(report.to_dict("records"), ok, strict=True):
            entry = {
                "etapa": stage,
                "rotulo": label,
                "config": by_label[label].as_dict()["overrides"],
                "desvios": list(by_label[label].deviations),
                "criterio": asdict(criterion),
                "sucesso": bool(success),
                **{k: row[k] for k in bs.CONSISTENCY_COLUMNS},
            }
            lines.append(json.dumps(bs._json_safe(entry), ensure_ascii=False, default=str))
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / STAGE_LOG_FILE
    if stage == STAGE_CONFIRMATION and path.exists():
        # a confirmação acumula configurações: mantém as linhas de outras candidatas
        kept = [
            line
            for line in path.read_text("utf-8").splitlines()
            if line.strip() and json.loads(line).get("rotulo") not in reports
        ]
        lines = kept + lines
    path.write_text("\n".join(lines) + ("\n" if lines else ""), "utf-8")


def _write_stage_table(stage_dir: Path, stage: str, table: pd.DataFrame) -> pd.DataFrame:
    """Grava ``search_table.csv`` da etapa (a confirmação acumula candidatas por rótulo)."""
    path = stage_dir / SEARCH_TABLE_FILE
    out = table
    if stage == STAGE_CONFIRMATION and path.exists():
        old = pd.read_csv(path)
        old = old[~old["rotulo"].isin(table["rotulo"])]
        out = pd.concat([old, table], ignore_index=True) if not old.empty else table
    out.to_csv(path, index=False)
    return out


def run_stage(
    algorithm: str,
    scenario: str,
    candidates: Sequence[Candidate] | SearchSpace,
    repetitions: int = 3,
    master_seed: int = 42,
    parallel: int = 1,
    *,
    criterion: ConvergenceCriterion | None = None,
    stop_on_success: bool = False,
    stage: str = STAGE_SCREENING,
    output_root: Path | str = OUTPUT_ROOT,
    n_simulations: int = 500,
    n_time_points: int = 100,
    max_cell_seconds: float | None = 1800.0,
    progress: Callable[[str], None] | None = None,
) -> StageResult:
    """Executa uma etapa: cada candidata, na ordem dada, com ``repetitions`` repetições.

    ``master_seed``: semente mestra (as sementes de cada repetição são derivadas dela).
    ``parallel``: processos simultâneos (≤ :data:`MAX_WORKERS`; PySR sempre em série). As
    candidatas rodam em lotes de ``parallel`` na ordem da lista; com ``stop_on_success=True`` a
    etapa para ao fim do primeiro lote em que alguma configuração atinge o critério.
    Devolve :class:`StageResult` com a tabela ordenada (:func:`rank_table`) e grava
    ``<raiz>/<algoritmo>/<cenário>/<etapa>/study_log.jsonl`` e ``search_table.csv``.
    """
    crit = criterion or ConvergenceCriterion()
    cands = tuple(candidates.candidates if isinstance(candidates, SearchSpace) else candidates)
    if not cands:
        raise ConvergenceSearchError("run_stage exige ao menos uma candidata")
    if repetitions < 1:
        raise ConvergenceSearchError("repetitions deve ser >= 1")
    workers = 1 if algorithm in SERIAL_ALGORITHMS else max(1, min(parallel, MAX_WORKERS))
    stage_dir = _stage_dir(output_root, algorithm, scenario, stage)
    designs = [
        design_for(
            algorithm,
            scenario,
            c,
            repetitions=repetitions,
            master_seed=master_seed,
            output_root=stage_dir,
            n_simulations=n_simulations,
            n_time_points=n_time_points,
            max_cell_seconds=max_cell_seconds,
        )
        for c in cands
    ]
    bs.require_algorithm(algorithm)
    start = time.perf_counter()
    reports: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    stopped = False
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        for first in range(0, len(cands), workers):
            batch = list(range(first, min(first + workers, len(cands))))
            if executor is None:
                outs = [_run_candidate(designs[i]) for i in batch]
            else:
                outs = list(executor.map(_run_candidate, [designs[i] for i in batch]))
            batch_converged = False
            for i, report in zip(batch, outs, strict=True):
                reports[cands[i].label] = report
                row = summarize(
                    report,
                    cands[i],
                    crit,
                    stage=stage,
                    order=i,
                    algorithm=algorithm,
                    scenario=scenario,
                )
                rows.append(row)
                batch_converged = batch_converged or bool(row["converge"])
                if progress is not None:
                    progress(
                        f"[{stage}] {cands[i].label}: {row['n_sucesso']}/{row['n_execucoes']} "
                        f"sucesso; típica {row['expressao_tipica']}; {row['tempo_medio_s']:.1f} s"
                    )
            if stop_on_success and batch_converged and batch[-1] < len(cands) - 1:
                stopped = True
                break
    finally:
        if executor is not None:
            executor.shutdown()
    table = rank_table(pd.DataFrame(rows, columns=list(TABLE_COLUMNS)))
    _write_stage_log(stage_dir, stage, reports, cands, crit)
    _write_stage_table(stage_dir, stage, table)
    return StageResult(
        stage=stage,
        algorithm=algorithm,
        scenario=scenario,
        table=table,
        reports=reports,
        candidates=cands,
        criterion=crit,
        stage_dir=stage_dir,
        elapsed_seconds=time.perf_counter() - start,
        stopped_early=stopped,
    )


def cheaper_candidates(
    algorithm: str,
    best: Candidate,
    factors: Sequence[tuple[float, float]] = DEFAULT_REFINE_FACTORS,
) -> list[Candidate]:
    """Versões mais baratas de ``best``: população e gerações (iterações no PySR) reduzidas.

    Devolve ``best`` e as reduções, ordenadas por custo estimado crescente (rótulos únicos).
    """
    budget, _, _ = split_overrides(best.overrides)
    pop_path, gen_path = (
        ("sr.pysr.population_size", "sr.pysr.niterations")
        if algorithm == "pysr"
        else ("sr.population", "sr.generations")
    )
    pop0 = budget.pysr_population_size if algorithm == "pysr" else budget.population
    gen0 = budget.pysr_niterations if algorithm == "pysr" else budget.generations
    out = {best.label: best}
    for fp, fg in factors:
        pop = max(2 if algorithm == "pysr" else 1, round(pop0 * fp))
        gen = max(1, round(gen0 * fg))
        if (pop, gen) == (pop0, gen0):
            continue
        overrides = {**best.overrides, pop_path: pop, gen_path: gen}
        label = f"{best.label} | {pop}x{gen}"
        out[label] = make_candidate(label, overrides, best.hypothesis)
    return sorted(out.values(), key=lambda c: estimated_cost(algorithm, c.overrides))


def refine(
    algorithm: str,
    scenario: str,
    best: Candidate,
    repetitions: int = 3,
    master_seed: int = 42,
    parallel: int = 1,
    *,
    factors: Sequence[tuple[float, float]] = DEFAULT_REFINE_FACTORS,
    criterion: ConvergenceCriterion | None = None,
    output_root: Path | str = OUTPUT_ROOT,
    **kwargs: Any,
) -> tuple[Candidate, StageResult]:
    """Refinamento: a configuração MAIS BARATA (custo estimado) que ainda converge.

    Executa ``best`` e suas reduções (:func:`cheaper_candidates`) com as mesmas sementes da
    triagem e devolve ``(escolhida, etapa)``. Se nenhuma convergir, devolve ``best``.
    """
    crit = criterion or ConvergenceCriterion()
    cands = cheaper_candidates(algorithm, best, factors)
    result = run_stage(
        algorithm,
        scenario,
        cands,
        repetitions,
        master_seed,
        parallel,
        criterion=crit,
        stage=STAGE_REFINEMENT,
        output_root=output_root,
        **kwargs,
    )
    converged = result.converged
    if converged.empty:
        return best, result
    cheapest = converged.sort_values(["custo_estimado", "tempo_medio_s"], kind="mergesort").iloc[0][
        "rotulo"
    ]
    return next(c for c in cands if c.label == cheapest), result


def confirm(
    algorithm: str,
    scenario: str,
    config: Candidate,
    n_repetitions: int = 5,
    seed_offset: int = DEFAULT_CONFIRMATION_OFFSET,
    *,
    search_master_seed: int = 42,
    used_seeds: Iterable[int] = (),
    criterion: ConvergenceCriterion | None = None,
    output_root: Path | str = OUTPUT_ROOT,
    parallel: int = 1,
    n_simulations: int = 500,
    n_time_points: int = 100,
    max_cell_seconds: float | None = 1800.0,
    progress: Callable[[str], None] | None = None,
) -> StageResult:
    """Confirmação em SEMENTES NOVAS, disjuntas das usadas na busca.

    Em série (``parallel = 1``): um desenho com ``n_repetitions`` repetições e semente mestra
    ``search_master_seed + seed_offset``. Em paralelo (``parallel > 1``, DEAP/gplearn): uma
    repetição por desenho, com sementes mestras ``search_master_seed + seed_offset + k``
    (``k = 0..n_repetitions-1``), executadas em processos simultâneos. Nos dois modos os dados, a
    divisão treino/validação/teste e a busca de cada repetição mudam com a semente.
    ``used_seeds``: sementes de busca já usadas (triagem/refinamento); se alguma reaparecer, a
    confirmação é recusada (:class:`ConvergenceSearchError`) — evita escolher a semente sortuda.
    """
    if seed_offset == 0:
        raise ConvergenceSearchError("seed_offset = 0 repetiria as sementes da busca")
    if n_repetitions < 1:
        raise ConvergenceSearchError("n_repetitions deve ser >= 1")
    crit = criterion or ConvergenceCriterion()
    data: dict[str, Any] = {
        "n_simulations": n_simulations,
        "n_time_points": n_time_points,
        "max_cell_seconds": max_cell_seconds,
    }
    workers = 1 if algorithm in SERIAL_ALGORITHMS else max(1, min(parallel, MAX_WORKERS))
    if workers == 1:
        result = run_stage(
            algorithm,
            scenario,
            [config],
            n_repetitions,
            search_master_seed + seed_offset,
            1,
            criterion=crit,
            stage=STAGE_CONFIRMATION,
            output_root=output_root,
            progress=progress,
            **data,
        )
    else:
        result = _confirm_parallel(
            algorithm,
            scenario,
            config,
            [search_master_seed + seed_offset + k for k in range(n_repetitions)],
            crit,
            output_root,
            workers,
            data,
            progress,
        )
    overlap = result.seeds & {int(s) for s in used_seeds}
    if overlap:
        raise ConvergenceSearchError(
            f"a confirmação repetiu sementes da busca: {sorted(overlap)}; mude seed_offset"
        )
    return result


def _confirm_parallel(
    algorithm: str,
    scenario: str,
    config: Candidate,
    master_seeds: Sequence[int],
    criterion: ConvergenceCriterion,
    output_root: Path | str,
    workers: int,
    data: Mapping[str, Any],
    progress: Callable[[str], None] | None,
) -> StageResult:
    """Uma repetição por semente mestra, em processos simultâneos (ver :func:`confirm`)."""
    stage_dir = _stage_dir(output_root, algorithm, scenario, STAGE_CONFIRMATION)
    designs = [
        design_for(
            algorithm,
            scenario,
            config,
            repetitions=1,
            master_seed=seed,
            output_root=stage_dir,
            **data,
        )
        for seed in master_seeds
    ]
    bs.require_algorithm(algorithm)
    start = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as executor:
        parts = list(executor.map(_run_candidate, designs))
    report = pd.concat(parts, ignore_index=True)
    report["repeticao"] = list(range(len(report)))
    row = summarize(
        report,
        config,
        criterion,
        stage=STAGE_CONFIRMATION,
        order=0,
        algorithm=algorithm,
        scenario=scenario,
    )
    if progress is not None:
        progress(
            f"[{STAGE_CONFIRMATION}] {config.label}: {row['n_sucesso']}/{row['n_execucoes']} "
            f"sucesso; típica {row['expressao_tipica']}; {row['tempo_medio_s']:.1f} s"
        )
    table = pd.DataFrame([row], columns=list(TABLE_COLUMNS))
    reports = {config.label: report}
    _write_stage_log(stage_dir, STAGE_CONFIRMATION, reports, [config], criterion)
    _write_stage_table(stage_dir, STAGE_CONFIRMATION, table)
    return StageResult(
        stage=STAGE_CONFIRMATION,
        algorithm=algorithm,
        scenario=scenario,
        table=table,
        reports=reports,
        candidates=(config,),
        criterion=criterion,
        stage_dir=stage_dir,
        elapsed_seconds=time.perf_counter() - start,
    )


# --------------------------------------------------------------------------- exportação


def best_config_dict(
    algorithm: str,
    scenario: str,
    candidate: Candidate,
    *,
    criterion: ConvergenceCriterion | None = None,
    confirmation: StageResult | None = None,
    max_cell_seconds: float | None = 1800.0,
) -> dict[str, Any]:
    """Dicionário da configuração vencedora, pronto para o notebook 05.

    ``search_budget``: argumentos de ``SearchBudget(...)``; ``hyperparameters``: caminho → valor
    (``StudyDesign(hyperparameters=tuple(d.items()))``); ``loss_formulation``.
    """
    budget, hyper, formulation = split_overrides(
        candidate.overrides, bs.SearchBudget(max_cell_seconds=max_cell_seconds)
    )
    out: dict[str, Any] = {
        "algoritmo": algorithm,
        "cenario": scenario,
        "rotulo": candidate.label,
        "hipotese": candidate.hypothesis,
        "loss_formulation": formulation,
        "search_budget": {f.name: _plain(getattr(budget, f.name)) for f in fields(budget)},
        "hyperparameters": {k: _plain(v) for k, v in hyper},
        "overrides": {k: _plain(v) for k, v in candidate.overrides.items()},
        "desvios_do_baseline_literal": list(candidate.deviations),
        "criterio": asdict(criterion or ConvergenceCriterion()),
    }
    if confirmation is not None and not confirmation.table.empty:
        row = confirmation.table.iloc[0]
        out["confirmacao"] = {
            "converge": bool(row["converge"]),
            "n_sucesso": int(row["n_sucesso"]),
            "n_execucoes": int(row["n_execucoes"]),
            "taxa_sucesso": float(row["taxa_sucesso"]),
            "expressao_tipica": str(row["expressao_tipica"]),
            "tempo_medio_s": float(row["tempo_medio_s"]),
            "sementes": str(row["sementes"]),
        }
    out["notebook_05"] = notebook05_snippet(out)
    return out


def notebook05_snippet(best: Mapping[str, Any]) -> str:
    """Bloco para colar na Etapa 1 do notebook 05 (``ORCAMENTO`` e ``HIPERPARAMETROS``)."""
    budget = ", ".join(f"{k}={v!r}" for k, v in best["search_budget"].items())
    hyper = ",\n".join(f"    {k!r}: {v!r}" for k, v in best["hyperparameters"].items())
    return (
        f"ALGORITMO = {best['algoritmo']!r}\n"
        f"FORMULACAO_PERDA = {best['loss_formulation']!r}\n"
        f"ORCAMENTO = SearchBudget({budget})\n"
        f"HIPERPARAMETROS = {{\n{hyper}\n}}\n"
        "# em StudyDesign(...): hyperparameters=tuple(HIPERPARAMETROS.items())"
    )


def export_best(
    algorithm: str,
    scenario: str,
    candidate: Candidate,
    stages: Sequence[StageResult | pd.DataFrame],
    *,
    criterion: ConvergenceCriterion | None = None,
    confirmation: StageResult | None = None,
    output_root: Path | str = OUTPUT_ROOT,
    max_cell_seconds: float | None = 1800.0,
) -> dict[str, Path]:
    """Grava ``best_config.yaml``, ``best_config.json`` e ``search_table.csv``.

    Destino: ``<raiz>/<algoritmo>/<cenário>/``. ``search_table.csv`` junta as tabelas de todas as
    etapas (triagem, refinamento e confirmação). Devolve os caminhos gravados.
    """
    target = resolve_project_path(output_root) / algorithm / scenario
    target.mkdir(parents=True, exist_ok=True)
    best = best_config_dict(
        algorithm,
        scenario,
        candidate,
        criterion=criterion,
        confirmation=confirmation,
        max_cell_seconds=max_cell_seconds,
    )
    paths = {
        "yaml": target / BEST_YAML,
        "json": target / BEST_JSON,
        "table": target / SEARCH_TABLE_FILE,
    }
    paths["json"].write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", "utf-8")
    paths["yaml"].write_text(
        yaml.safe_dump(best, allow_unicode=True, sort_keys=False, width=100), "utf-8"
    )
    tables = [s if isinstance(s, pd.DataFrame) else s.table for s in stages]
    if confirmation is not None and all(confirmation is not s for s in stages):
        tables.append(confirmation.table)
    frames = [t for t in tables if not t.empty]
    combined = (
        pd.concat(frames, ignore_index=True).drop_duplicates(["etapa", "rotulo", "sementes"])
        if frames
        else pd.DataFrame(columns=list(TABLE_COLUMNS))
    )
    combined.to_csv(paths["table"], index=False)
    return paths


def load_stage_table(
    algorithm: str, scenario: str, stage: str, output_root: Path | str = OUTPUT_ROOT
) -> pd.DataFrame:
    """Tabela gravada de uma etapa (``<raiz>/<algoritmo>/<cenário>/<etapa>/search_table.csv``)."""
    path = _stage_dir(output_root, algorithm, scenario, stage) / SEARCH_TABLE_FILE
    table: pd.DataFrame = pd.read_csv(path)
    return table


def candidate_from_row(row: Mapping[str, Any], hypothesis: str = "") -> Candidate:
    """Reconstrói a :class:`Candidate` de uma linha de tabela de etapa (coluna ``config``)."""
    overrides = json.loads(str(row["config"]))
    return make_candidate(str(row["rotulo"]), overrides, hypothesis)


def load_best_config(path: Path | str) -> tuple[bs.SearchBudget, tuple[tuple[str, Any], ...], str]:
    """Lê ``best_config.json``/``.yaml`` → ``(SearchBudget, hyperparameters, formulação)``."""
    p = Path(path)
    text = p.read_text("utf-8")
    data = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
    budget = bs.SearchBudget(**data["search_budget"])
    hyper = tuple(
        (k, tuple(v) if k.endswith("const_range") else v)
        for k, v in data["hyperparameters"].items()
    )
    return budget, hyper, str(data["loss_formulation"])


# --------------------------------------------------------------------------- expressividade


def _operators(expr: Expr) -> set[str]:
    if isinstance(expr, Op):
        out = {expr.name}
        for child in expr.children:
            out |= _operators(child)
        return out
    return set()


def _variables(expr: Expr) -> set[str]:
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, Op):
        return set().union(*(_variables(c) for c in expr.children))
    return set()


@dataclass(frozen=True)
class ExpressivenessRow:
    """Uma forma avaliada no diagnóstico de expressividade."""

    forma: str
    tipo: str
    nos: int
    operadores_ausentes: tuple[str, ...]
    expressavel: bool
    e: float
    s: float
    perda_L: float  # noqa: N815 - L da perda composta (notação do artigo)


def expressiveness_diagnosis(
    algorithm: str,
    scenario: str,
    candidate: Candidate | None = None,
    *,
    master_seed: int = 42,
    n_simulations: int = 500,
    n_time_points: int = 100,
    extra_forms: Sequence[str] = (),
) -> pd.DataFrame:
    """Diagnóstico de expressividade e da perda ``L`` de formas de referência e triviais.

    Para a equação de referência e cada forma equivalente declarada pelo cenário: número de nós,
    operadores ausentes do conjunto do algoritmo (``expressavel = False`` se faltar algum) e
    ``L = w1·e + w2·s`` (pesos do baseline, mesma formulação da função de perda da candidata),
    calculada nos dados de treino limpos da repetição 0. Candidatas triviais: a constante média,
    cada variável sozinha e ``c·variável`` com ``c`` de mínimos quadrados. Se alguma trivial tem
    ``L`` menor que todas as formas de referência, a busca premia a trivial (a formulação da perda
    impede a convergência, não o algoritmo).
    """
    cand = candidate or Candidate("literal", {})
    design = design_for(
        algorithm,
        scenario,
        cand,
        repetitions=1,
        master_seed=master_seed,
        output_root=OUTPUT_ROOT,
        n_simulations=n_simulations,
        n_time_points=n_time_points,
    )
    cfg = bs.build_config(design)
    sc = get_scenario(scenario, cfg.physics)
    provider = DataProvider(cfg)
    data = provider.clean(scenario, 0)
    objective = build_objective(
        cfg, None, None, algorithm, baseline=True, feature_names=data.feature_names
    )
    X, y = data.X, data.y_observed  # noqa: N806
    allowed = set(cfg.sr.operators[algorithm])  # type: ignore[index]
    forms: list[tuple[str, str]] = [
        (f, "referência" if i == 0 else "equivalente")
        for i, f in enumerate(sc.equivalent_ground_truths())
    ]
    forms += [(f, "extra") for f in extra_forms]
    forms.append((repr(float(np.mean(y))), "trivial: constante média"))
    for j, name in enumerate(data.feature_names):
        forms.append((name, "trivial: uma variável"))
        col = X[:, j]
        denom = float(np.dot(col, col))
        if denom > 0:
            coef = float(np.dot(col, y) / denom)
            forms.append((f"{coef!r}*{name}", "trivial: c·variável"))
    rows = []
    for text, kind in forms:
        expr = parse_infix(text)
        missing = tuple(sorted(_operators(expr) - allowed))
        loss = objective.evaluate_population([expr], X, y)[0]
        rows.append(
            asdict(
                ExpressivenessRow(
                    forma=text,
                    tipo=kind,
                    nos=size(expr),
                    operadores_ausentes=missing,
                    expressavel=not missing and _variables(expr) <= set(data.feature_names),
                    e=float(loss.e),
                    s=float(loss.s),
                    perda_L=float(loss.total),
                )
            )
        )
    table = pd.DataFrame(rows)
    reference = table[table["tipo"].isin(["referência", "equivalente"])]
    best_ref = float(reference["perda_L"].min()) if not reference.empty else math.inf
    table["vence_referencia"] = table["tipo"].str.startswith("trivial") & (
        table["perda_L"] < best_ref
    )
    out: pd.DataFrame = table.sort_values("perda_L", kind="mergesort").reset_index(drop=True)
    return out


def loss_blocks_convergence(diagnosis: pd.DataFrame) -> bool:
    """Verdadeiro se alguma candidata trivial tem ``L`` menor que todas as formas de referência."""
    return bool(diagnosis["vence_referencia"].any())


__all__ = [
    "Candidate",
    "ConvergenceCriterion",
    "ConvergenceSearchError",
    "SearchSpace",
    "StageResult",
    "best_config_dict",
    "candidate_from_row",
    "cheaper_candidates",
    "confirm",
    "deviations_from_base",
    "estimated_cost",
    "export_best",
    "expressiveness_diagnosis",
    "load_best_config",
    "load_stage_table",
    "loss_blocks_convergence",
    "make_candidate",
    "minimal_mode_requested",
    "notebook05_snippet",
    "rank_table",
    "refine",
    "run_stage",
    "split_overrides",
    "successes",
    "summarize",
]
