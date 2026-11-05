"""Simuladores físicos e equações de referência (ground truth)."""

from physics_llm_sr.physics.base import (
    FEATURE_NAMES,
    PhysicalScenario,
    SimulationParameters,
    Trajectory,
    sample_times,
)
from physics_llm_sr.physics.registry import SCENARIO_NAMES, get_scenario

__all__ = [
    "FEATURE_NAMES",
    "SCENARIO_NAMES",
    "PhysicalScenario",
    "SimulationParameters",
    "Trajectory",
    "get_scenario",
    "sample_times",
]
