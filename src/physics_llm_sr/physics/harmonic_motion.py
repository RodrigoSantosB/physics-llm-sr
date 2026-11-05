"""Movimento harmônico simples (GAP-03): ``x = A cos(sqrt(k/m) t + phi)``.

``A = length ~ U(0.01, 0.5)`` m, ``m = mass ~ U(0.1, 10)`` kg, ``k`` e ``phi`` da config;
``t ~ U(0, T)`` com ``T = 2 pi sqrt(m/k)`` (um período). ``height`` e ``damping`` são distratores.
"""

from __future__ import annotations

import math

import numpy as np

from physics_llm_sr.config.schemas import HarmonicMotionConfig, SamplingRanges
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.physics.base import (
    FEATURE_NAMES,
    FloatArray,
    SimulationParameters,
    Trajectory,
    assemble_features,
    require_finite,
    require_non_negative,
    require_positive,
    sample_uniform_parameters,
    simulate_scenario,
)

NAME = "harmonic_motion"
TARGET_NAME = "displacement"
MASS_COLUMN = FEATURE_NAMES.index("mass")
LENGTH_COLUMN = FEATURE_NAMES.index("length")
TIME_COLUMN = FEATURE_NAMES.index("t")

UNITS = {
    "mass": "kg",
    "length": "m",
    "height": "m",
    "damping": "1/s",
    "t": "s",
    "displacement": "m",
}


def harmonic_displacement(
    amplitude: float | FloatArray,
    mass: float | FloatArray,
    spring_constant: float,
    t: float | FloatArray,
    phase: float = 0.0,
) -> FloatArray:
    """Calcula ``A cos(sqrt(k/m) t + phi)`` (m).

    Domínio: ``amplitude >= 0``, ``mass > 0``, ``spring_constant > 0``, ``t >= 0``.
    """
    require_non_negative("amplitude", amplitude)
    require_positive("mass", mass)
    require_positive("spring_constant", spring_constant)
    require_non_negative("t", t)
    require_finite("phase", phase)
    omega = np.sqrt(spring_constant / np.asarray(mass, dtype=float))
    out: FloatArray = np.asarray(amplitude, dtype=float) * np.cos(
        omega * np.asarray(t, dtype=float) + phase
    )
    return out


def harmonic_period(mass: float, spring_constant: float) -> float:
    """Período ``2 pi sqrt(m/k)`` (s)."""
    require_positive("mass", mass)
    require_positive("spring_constant", spring_constant)
    return 2.0 * math.pi * math.sqrt(mass / spring_constant)


class HarmonicMotionScenario:
    """Cenário MHS parametrizado pela config."""

    name = NAME
    feature_names = FEATURE_NAMES
    target_name = TARGET_NAME

    def __init__(self, config: HarmonicMotionConfig | None = None) -> None:
        """Usa ``config.spring_constant`` (N/m) e ``config.phase`` (rad)."""
        self.config = config or HarmonicMotionConfig()

    def reference(self, features: FloatArray) -> FloatArray:
        """``length * cos(sqrt(k/mass) * t + phi)``."""
        return harmonic_displacement(
            features[:, LENGTH_COLUMN],
            features[:, MASS_COLUMN],
            self.config.spring_constant,
            features[:, TIME_COLUMN],
            self.config.phase,
        )

    def sample_parameters(
        self, rng: np.random.Generator, ranges: SamplingRanges
    ) -> SimulationParameters:
        """Amostra ``A`` e ``m`` ~ U(range) e os distratores."""
        return sample_uniform_parameters(rng, ranges)

    def time_end(self, parameters: SimulationParameters) -> float:
        """Um período."""
        return harmonic_period(parameters["mass"], self.config.spring_constant)

    def trajectory(self, parameters: SimulationParameters, t: FloatArray) -> Trajectory:
        """Deslocamento nos instantes ``t``."""
        features = assemble_features(parameters, t)
        return Trajectory(X=features, y_true=self.reference(features), t=t, parameters=parameters)

    def simulate(
        self, rng: np.random.Generator, ranges: SamplingRanges, n_points: int, max_steps: int
    ) -> Trajectory:
        """Amostra uma simulação completa."""
        return simulate_scenario(self, rng, ranges, n_points, max_steps)

    def equivalent_ground_truths(self) -> tuple[str, ...]:
        """Formas infixas fisicamente equivalentes; a primeira é a GT do artigo."""
        return (self.physical_context().ground_truth,)

    def physical_context(self) -> PhysicalContext:
        """Contexto físico com GT ``length*cos(sqrt(k/mass)*t)``."""
        k = repr(self.config.spring_constant)
        phase = self.config.phase
        argument = f"sqrt({k}/mass)*t" + (f"+{phase!r}" if phase != 0.0 else "")
        return PhysicalContext(
            scenario=NAME,
            variable_descriptions={
                "mass": "massa presa à mola",
                "length": "amplitude A do movimento (deslocamento inicial)",
                "height": "altura (irrelevante)",
                "damping": "coeficiente de amortecimento (irrelevante; sem dissipação)",
                "t": "tempo decorrido desde que a massa foi solta",
                "displacement": "deslocamento x da massa em relação ao equilíbrio",
            },
            experiment_description=(
                f"Massa presa a uma mola ideal de constante k = {self.config.spring_constant} N/m, "
                "solta do repouso na amplitude A, sem atrito. Prevê-se o deslocamento x(t)."
            ),
            ground_truth=f"length*cos({argument})",
            units=dict(UNITS),
        )
