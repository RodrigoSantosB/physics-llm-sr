"""Protocolo comum dos cenários físicos, validação e amostragem de instantes.

Funções puras: nenhuma semente é criada aqui; o ``numpy.random.Generator`` é sempre
recebido do chamador. Este pacote não depende de ``llm``, ``sr`` nem ``optimization``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from physics_llm_sr.config.schemas import SamplingRanges
from physics_llm_sr.core_types import PhysicalContext

#: Colunas de X, na ordem; as que não entram na GT são distratores (GAP-03).
FEATURE_NAMES: tuple[str, ...] = ("mass", "length", "height", "damping", "t")
#: Índice da coluna de tempo em X.
TIME_COLUMN = FEATURE_NAMES.index("t")

FloatArray = np.ndarray[Any, Any]
#: Parâmetros amostrados de uma simulação: ``mass``, ``length``, ``height``, ``damping``.
SimulationParameters = dict[str, float]


@dataclass(frozen=True)
class Trajectory:
    """Uma simulação: ``X`` (n, 5) com colunas ``FEATURE_NAMES``, alvo analítico e instantes."""

    X: FloatArray
    y_true: FloatArray
    t: FloatArray
    parameters: SimulationParameters


@runtime_checkable
class PhysicalScenario(Protocol):
    """Interface de um cenário físico consumida pela camada de dados."""

    name: str
    feature_names: tuple[str, ...]
    target_name: str

    def reference(self, features: FloatArray) -> FloatArray:
        """Alvo analítico (GT) a partir de uma matriz com as 5 colunas."""
        ...

    def sample_parameters(
        self, rng: np.random.Generator, ranges: SamplingRanges
    ) -> SimulationParameters:
        """Amostra os parâmetros de uma simulação a partir dos ranges da config."""
        ...

    def time_end(self, parameters: SimulationParameters) -> float:
        """Fim do intervalo físico de observação (GAP-04)."""
        ...

    def trajectory(self, parameters: SimulationParameters, t: FloatArray) -> Trajectory:
        """Constrói ``X`` e ``y_true`` nos instantes ``t``."""
        ...

    def simulate(
        self,
        rng: np.random.Generator,
        ranges: SamplingRanges,
        n_points: int,
        max_steps: int,
    ) -> Trajectory:
        """Amostra parâmetros, instantes ordenados e devolve a trajetória."""
        ...

    def physical_context(self) -> PhysicalContext:
        """Contexto físico (descrições, unidades e GT infixa) para o avaliador LLM."""
        ...

    def equivalent_ground_truths(self) -> tuple[str, ...]:
        """GT do artigo (primeira) e formas fisicamente equivalentes, em infixo.

        Base da métrica secundária ``tree_distance_equivalent_min``; nunca substitui a primária.
        """
        ...


def require_positive(name: str, value: float | FloatArray) -> None:
    """Levanta ``ValueError`` nomeando ``name`` se algum valor não for finito e > 0."""
    arr = np.asarray(value, dtype=float)
    if not (np.all(np.isfinite(arr)) and np.all(arr > 0.0)):
        raise ValueError(f"{name} deve ser finito e > 0; recebido {value!r}")


def require_non_negative(name: str, value: float | FloatArray) -> None:
    """Levanta ``ValueError`` nomeando ``name`` se algum valor não for finito e >= 0."""
    arr = np.asarray(value, dtype=float)
    if not (np.all(np.isfinite(arr)) and np.all(arr >= 0.0)):
        raise ValueError(f"{name} deve ser finito e >= 0; recebido {value!r}")


def require_finite(name: str, value: float | FloatArray) -> None:
    """Levanta ``ValueError`` nomeando ``name`` se algum valor não for finito."""
    if not np.all(np.isfinite(np.asarray(value, dtype=float))):
        raise ValueError(f"{name} deve ser finito; recebido {value!r}")


def sample_uniform_parameters(
    rng: np.random.Generator, ranges: SamplingRanges
) -> SimulationParameters:
    """Amostra ``mass, length, height, damping`` ~ U(low, high), nesta ordem fixa."""
    return {
        "mass": float(rng.uniform(ranges.mass.low, ranges.mass.high)),
        "length": float(rng.uniform(ranges.length.low, ranges.length.high)),
        "height": float(rng.uniform(ranges.height.low, ranges.height.high)),
        "damping": float(rng.uniform(ranges.damping.low, ranges.damping.high)),
    }


def sample_times(
    rng: np.random.Generator, t_end: float, n_points: int, max_steps: int
) -> FloatArray:
    """Amostra ``n_points`` instantes ~ U(0, t_end), ordenados.

    Critério de parada: ``n_points`` não pode exceder o teto ``max_steps`` por simulação.
    """
    if n_points <= 0:
        raise ValueError(f"n_points deve ser > 0; recebido {n_points}")
    if n_points > max_steps:
        raise ValueError(f"n_points ({n_points}) excede max_steps ({max_steps})")
    require_positive("t_end", t_end)
    return np.sort(rng.uniform(0.0, t_end, size=n_points))


def assemble_features(
    parameters: SimulationParameters, t: FloatArray, **overrides: FloatArray
) -> FloatArray:
    """Monta X (n, 5); colunas constantes vêm de ``parameters``, ``overrides`` substituem."""
    n = t.shape[0]
    cols: dict[str, FloatArray] = {
        name: np.full(n, parameters[name], dtype=float) for name in FEATURE_NAMES if name != "t"
    }
    cols.update(overrides)
    cols["t"] = t
    return np.column_stack([cols[name] for name in FEATURE_NAMES])


def simulate_scenario(
    scenario: PhysicalScenario,
    rng: np.random.Generator,
    ranges: SamplingRanges,
    n_points: int,
    max_steps: int,
) -> Trajectory:
    """Implementação comum de ``simulate``: parâmetros, instantes e trajetória."""
    parameters = scenario.sample_parameters(rng, ranges)
    t = sample_times(rng, scenario.time_end(parameters), n_points, max_steps)
    return scenario.trajectory(parameters, t)
