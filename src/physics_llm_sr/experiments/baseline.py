"""Experimento 1 — baseline sem LLM (PRD §7; ``w3 = 0``, pesos ``loss.baseline_weights``)."""

from __future__ import annotations

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.experiments.cells import NONE, Cell, default_noise

NAME = "baseline"


def cells(cfg: ExperimentConfig) -> list[Cell]:
    """Cenários × algoritmos × repetições, ruído padrão, sem LLM."""
    noise = default_noise(cfg)
    ex = cfg.experiments
    return [
        Cell(scenario, sr, NONE, NONE, noise, rep)
        for scenario in ex.scenarios
        for sr in ex.algorithms
        for rep in range(ex.repetitions)
    ]
