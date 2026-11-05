"""Registro dos cenários físicos: ``get_scenario`` e ``SCENARIO_NAMES``."""

from __future__ import annotations

from physics_llm_sr.config.schemas import PhysicsConfig
from physics_llm_sr.physics.base import PhysicalScenario
from physics_llm_sr.physics.damped_wave import DampedWaveScenario
from physics_llm_sr.physics.free_fall import FreeFallScenario
from physics_llm_sr.physics.harmonic_motion import HarmonicMotionScenario

SCENARIO_NAMES: tuple[str, ...] = ("free_fall", "harmonic_motion", "damped_wave")


def get_scenario(name: str, config: PhysicsConfig | None = None) -> PhysicalScenario:
    """Instancia o cenário ``name`` com os parâmetros de ``config`` (default do schema)."""
    cfg = config or PhysicsConfig()
    if name == "free_fall":
        return FreeFallScenario(cfg.free_fall)
    if name == "harmonic_motion":
        return HarmonicMotionScenario(cfg.harmonic_motion)
    if name == "damped_wave":
        return DampedWaveScenario(cfg.damped_wave)
    raise ValueError(f"scenario desconhecido: {name!r}; válidos: {SCENARIO_NAMES}")
