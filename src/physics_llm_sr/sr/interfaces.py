"""Interface comum dos regressores simbólicos (DEAP, gplearn, PySR) e do objetivo.

Este módulo define apenas contratos. ``CandidateObjective`` é o Protocol que a
camada ``optimization/`` implementará; ele vive aqui para que ``sr`` não dependa
de ``optimization`` (sem ciclo). O avaliador LLM e a perda composta
``L = w1*e + w2*s + w3*c`` ficam por trás desse Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.symbolic.expression import Expr

FloatArray = np.ndarray[Any, Any]

#: ``integrated``: LLM no laço de busca; ``limited``: integração parcial (ex.: só
#: reavaliação por geração); ``baseline_only``: sem LLM (w3 = 0).
IntegrationStatus = Literal["integrated", "limited", "baseline_only"]


@dataclass(frozen=True)
class LossBreakdown:
    """Perda composta de um candidato: ``total`` e componentes ``e``, ``s``, ``c``.

    ``evaluation`` carrega o resultado bruto do avaliador LLM (ou ``None`` no
    baseline); o tipo concreto pertence à camada ``optimization``/``llm``.
    """

    total: float
    e: float
    s: float
    c: float
    evaluation: object | None = None


@runtime_checkable
class CandidateObjective(Protocol):
    """Objetivo que pontua uma população de candidatos (implementado em optimization/)."""

    def evaluate_population(
        self,
        exprs: Sequence[Expr],
        X: FloatArray,  # noqa: N803 - nome do contrato (PRD)
        y: FloatArray,
    ) -> Sequence[LossBreakdown]:
        """Perda composta de cada expressão, na mesma ordem de ``exprs``."""
        ...


@dataclass(frozen=True)
class GenerationRecord:
    """Estado da busca ao fim de uma geração."""

    generation: int
    best_loss: float
    best_e: float
    best_s: float
    best_c: float
    best_expression: str  # infixo na sintaxe da AST
    n_llm_calls: int
    n_cache_hits: int
    elapsed: float  # segundos


@dataclass(frozen=True)
class RegressorMetadata:
    """Metadados reprodutíveis do regressor."""

    algorithm: str
    llm_in_loop: bool
    integration_status: IntegrationStatus
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class FitResult:
    """Resultado de ``fit``: melhor expressão, histórico por geração e metadados."""

    best_expression: Expr
    history: tuple[GenerationRecord, ...]
    metadata: RegressorMetadata


@runtime_checkable
class SymbolicRegressor(Protocol):
    """Contrato comum dos adaptadores de regressão simbólica."""

    def fit(
        self, dataset: Dataset, objective: CandidateObjective, config: ExperimentConfig
    ) -> FitResult:
        """Executa a busca minimizando ``objective`` sobre o treino de ``dataset``."""
        ...

    def predict(self, X: FloatArray) -> FloatArray:  # noqa: N803 - nome do contrato (PRD)
        """Predição da melhor expressão (finita, semântica protegida)."""
        ...

    def get_best_expression(self) -> Expr:
        """Melhor expressão da busca, na AST comum."""
        ...

    def get_search_history(self) -> Sequence[GenerationRecord]:
        """Histórico por geração."""
        ...

    def get_metadata(self) -> RegressorMetadata:
        """Metadados (algoritmo, ``llm_in_loop``, ``integration_status``)."""
        ...
