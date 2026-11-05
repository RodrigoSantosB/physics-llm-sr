"""Onda amortecida (GAP-03): ``E = E0 exp(-alpha t / 2) cos(k x - omega t)``.

``alpha = damping ~ U(0, 1)`` 1/s, ``x = length`` (posição, m); ``E0``, ``k`` e ``omega`` da
config; ``t ~ U(0, T)`` com ``T = 2 pi / omega``. ``mass`` e ``height`` são distratores.
"""

from __future__ import annotations

import math

import numpy as np

from physics_llm_sr.config.schemas import DampedWaveConfig, SamplingRanges
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

NAME = "damped_wave"
TARGET_NAME = "field"
LENGTH_COLUMN = FEATURE_NAMES.index("length")
DAMPING_COLUMN = FEATURE_NAMES.index("damping")
TIME_COLUMN = FEATURE_NAMES.index("t")

UNITS = {
    "mass": "kg",
    "length": "m",
    "height": "m",
    "damping": "1/s",
    "t": "s",
    "field": "V/m",
}

_TWO_PI = 2.0 * math.pi


def damped_wave_field(
    position: float | FloatArray,
    damping: float | FloatArray,
    t: float | FloatArray,
    amplitude: float = 1.0,
    wave_number: float = _TWO_PI,
    angular_frequency: float = _TWO_PI,
) -> FloatArray:
    """Calcula ``E0 exp(-alpha t/2) cos(k x - omega t)`` (V/m).

    Domínio: ``damping >= 0``, ``t >= 0``, ``amplitude``, ``wave_number`` e
    ``angular_frequency`` > 0.
    """
    require_finite("position", position)
    require_non_negative("damping", damping)
    require_non_negative("t", t)
    require_positive("amplitude", amplitude)
    require_positive("wave_number", wave_number)
    require_positive("angular_frequency", angular_frequency)
    x = np.asarray(position, dtype=float)
    a = np.asarray(damping, dtype=float)
    tt = np.asarray(t, dtype=float)
    out: FloatArray = (
        amplitude * np.exp(-a * tt / 2.0) * np.cos(wave_number * x - angular_frequency * tt)
    )
    return out


def damped_wave_period(angular_frequency: float) -> float:
    """Período temporal ``2 pi / omega`` (s)."""
    require_positive("angular_frequency", angular_frequency)
    return _TWO_PI / angular_frequency


class DampedWaveScenario:
    """Cenário de onda amortecida parametrizado pela config."""

    name = NAME
    feature_names = FEATURE_NAMES
    target_name = TARGET_NAME

    def __init__(self, config: DampedWaveConfig | None = None) -> None:
        """Usa ``amplitude`` (V/m), ``wave_number`` (rad/m) e ``angular_frequency`` (rad/s)."""
        self.config = config or DampedWaveConfig()

    def reference(self, features: FloatArray) -> FloatArray:
        """Campo elétrico a partir de ``length`` (x), ``damping`` (alpha) e ``t``."""
        return damped_wave_field(
            features[:, LENGTH_COLUMN],
            features[:, DAMPING_COLUMN],
            features[:, TIME_COLUMN],
            self.config.amplitude,
            self.config.wave_number,
            self.config.angular_frequency,
        )

    def sample_parameters(
        self, rng: np.random.Generator, ranges: SamplingRanges
    ) -> SimulationParameters:
        """``x``, ``alpha`` ~ U(range) e distratores."""
        return sample_uniform_parameters(rng, ranges)

    def time_end(self, parameters: SimulationParameters) -> float:
        """Um período temporal da onda."""
        del parameters  # T independe dos parâmetros amostrados
        return damped_wave_period(self.config.angular_frequency)

    def trajectory(self, parameters: SimulationParameters, t: FloatArray) -> Trajectory:
        """Campo nos instantes ``t``."""
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
        """Contexto físico com GT infixa."""
        c = self.config
        return PhysicalContext(
            scenario=NAME,
            variable_descriptions={
                "mass": "massa (irrelevante)",
                "length": "posição x ao longo da direção de propagação",
                "height": "altura (irrelevante)",
                "damping": "coeficiente de amortecimento alpha da onda",
                "t": "tempo",
                "field": "campo elétrico E(x, t) da onda",
            },
            experiment_description=(
                f"Onda eletromagnética plana amortecida com amplitude E0 = {c.amplitude} V/m, "
                f"número de onda k = {c.wave_number} rad/m e frequência angular "
                f"omega = {c.angular_frequency} rad/s. A amplitude decai como exp(-alpha t/2)."
            ),
            ground_truth=(
                f"{c.amplitude!r}*exp(-damping*t/2)"
                f"*cos({c.wave_number!r}*length-{c.angular_frequency!r}*t)"
            ),
            units=dict(UNITS),
        )
