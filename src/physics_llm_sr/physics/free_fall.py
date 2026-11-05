"""Queda livre (GAP-03): ``v = sqrt(2 g h)``, sem arrasto.

``height`` em X é a distância percorrida ``h(t) = g t^2 / 2`` a partir de ``h0 ~ U(1, 100)``;
``t ~ U(0, sqrt(2 h0 / g))`` (queda até o solo, GAP-04). ``mass``, ``length`` e ``damping``
são distratores.
"""

from __future__ import annotations

import numpy as np

from physics_llm_sr.config.schemas import FreeFallConfig, SamplingRanges
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.physics.base import (
    FEATURE_NAMES,
    FloatArray,
    SimulationParameters,
    Trajectory,
    assemble_features,
    require_non_negative,
    require_positive,
    sample_uniform_parameters,
    simulate_scenario,
)

NAME = "free_fall"
TARGET_NAME = "velocity"
HEIGHT_COLUMN = FEATURE_NAMES.index("height")

UNITS = {
    "mass": "kg",
    "length": "m",
    "height": "m",
    "damping": "1/s",
    "t": "s",
    "velocity": "m/s",
}


def free_fall_velocity(height: float | FloatArray, gravity: float = 9.81) -> FloatArray:
    """Calcula a velocidade ``sqrt(2 g h)`` (m/s) após cair ``height`` (m).

    Domínio: ``height >= 0`` e ``gravity > 0``.
    """
    require_non_negative("height", height)
    require_positive("gravity", gravity)
    return np.sqrt(2.0 * gravity * np.asarray(height, dtype=float))


def free_fall_fall_time(initial_height: float, gravity: float = 9.81) -> float:
    """Tempo até o solo ``sqrt(2 h0 / g)`` a partir de ``h0 > 0``."""
    require_positive("height", initial_height)
    require_positive("gravity", gravity)
    return float(np.sqrt(2.0 * initial_height / gravity))


class FreeFallScenario:
    """Cenário de queda livre parametrizado pela config."""

    name = NAME
    feature_names = FEATURE_NAMES
    target_name = TARGET_NAME

    def __init__(self, config: FreeFallConfig | None = None) -> None:
        """Usa ``config.gravity`` (m/s^2)."""
        self.config = config or FreeFallConfig()

    def reference(self, features: FloatArray) -> FloatArray:
        """``sqrt(2 g height)`` sobre a coluna ``height``."""
        return free_fall_velocity(features[:, HEIGHT_COLUMN], self.config.gravity)

    def sample_parameters(
        self, rng: np.random.Generator, ranges: SamplingRanges
    ) -> SimulationParameters:
        """``h0`` (``height``) ~ U(range) e distratores."""
        return sample_uniform_parameters(rng, ranges)

    def time_end(self, parameters: SimulationParameters) -> float:
        """Tempo de queda até o solo."""
        return free_fall_fall_time(parameters["height"], self.config.gravity)

    def trajectory(self, parameters: SimulationParameters, t: FloatArray) -> Trajectory:
        """``height`` = distância percorrida ``g t^2 / 2``; alvo = ``sqrt(2 g height)``."""
        require_non_negative("t", t)
        fallen = 0.5 * self.config.gravity * t**2
        # erro de ponto flutuante pode exceder h0 em ~1 ulp no instante final
        fallen = np.minimum(fallen, parameters["height"])
        features = assemble_features(parameters, t, height=fallen)
        return Trajectory(X=features, y_true=self.reference(features), t=t, parameters=parameters)

    def simulate(
        self, rng: np.random.Generator, ranges: SamplingRanges, n_points: int, max_steps: int
    ) -> Trajectory:
        """Amostra uma simulação completa."""
        return simulate_scenario(self, rng, ranges, n_points, max_steps)

    def equivalent_ground_truths(self) -> tuple[str, ...]:
        """Formas infixas fisicamente equivalentes; a primeira é a GT do artigo."""
        return (
            self.physical_context().ground_truth,
            f"{self.config.gravity!r}*t",
        )

    def physical_context(self) -> PhysicalContext:
        """Contexto físico com GT ``sqrt(2*g*height)``."""
        return PhysicalContext(
            scenario=NAME,
            variable_descriptions={
                "mass": "mass of the object (does not affect the speed without drag)",
                "length": "characteristic length of the object (irrelevant)",
                "height": "distance fallen from rest",
                "damping": "damping coefficient (irrelevant; no drag)",
                "t": "time elapsed since the start of the fall",
                "velocity": "speed of the freely falling object",
            },
            experiment_description=(
                "Object released from rest in free fall without air resistance, under gravitational "
                f"acceleration g = {self.config.gravity} m/s^2. The speed is predicted after "
                "falling the distance 'height'."
            ),
            ground_truth=f"sqrt(2*{self.config.gravity!r}*height)",
            units=dict(UNITS),
        )
