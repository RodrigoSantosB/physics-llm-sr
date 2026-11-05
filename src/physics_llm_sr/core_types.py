"""Tipos compartilhados entre todos os subpacotes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Mode(str, Enum):
    """Modo de execução: define o tamanho do experimento."""

    SMOKE = "smoke"
    PILOT = "pilot"
    FULL = "full"


class Track(str, Enum):
    """Trilha científica: replicação fiel do artigo ou análise estendida."""

    PAPER_REPLICATION = "paper_replication"
    EXTENDED_ANALYSIS = "extended_analysis"


@dataclass(frozen=True)
class Dataset:
    """Observações de um cenário físico.

    ``X`` tem forma ``(n, d)`` com colunas nomeadas em ``feature_names``.
    ``y_true`` é o alvo analítico e ``y_observed`` o alvo após ruído.
    """

    X: np.ndarray[Any, Any]
    y_true: np.ndarray[Any, Any]
    y_observed: np.ndarray[Any, Any]
    feature_names: tuple[str, ...]
    target_name: str
    scenario: str
    simulation_ids: np.ndarray[Any, Any]
    t: np.ndarray[Any, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalContext:
    """Contexto físico fornecido ao avaliador LLM."""

    scenario: str
    variable_descriptions: dict[str, str]
    experiment_description: str
    ground_truth: str  # texto infixo na sintaxe da AST
    units: dict[str, str]
