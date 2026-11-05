"""Experimento 2 — SR integrado com LLM, pareado com o baseline (PRD §7, Tabela 5).

Cada célula LLM vem com a célula baseline pareada: mesmo dataset (mesma semente de dados
e de ruído da repetição), mesmo split e mesma semente de busca. O prompt é
``experiments.benchmark_prompt`` (A por INC-08/GAP-12; o piloto usa E).
"""

from __future__ import annotations

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.experiments.cells import Cell, default_noise, llm_cell, selected_models

NAME = "llm_integration"


def cells(cfg: ExperimentConfig) -> list[Cell]:
    """Cenários × algoritmos × LLMs × repetições (+ baseline pareado)."""
    noise = default_noise(cfg)
    ex = cfg.experiments
    out: list[Cell] = []
    for scenario in ex.scenarios:
        for sr in ex.algorithms:
            for rep in range(ex.repetitions):
                for model in selected_models(cfg):
                    cell = llm_cell(scenario, sr, model, ex.benchmark_prompt, noise, rep)
                    out.extend((cell, cell.baseline()))
    return out
