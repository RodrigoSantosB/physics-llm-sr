"""Experimento 4 — robustez a ruído gaussiano de média zero (PRD §7, Tabela 3).

Níveis × localizações a partir do MESMO dataset limpo de cada repetição; a semente do
ruído é derivada de (cenário, repetição, tipo, localização, nível) — ver
``experiments.data_provider``. Cada condição roda o LLM ``experiments.noise_model`` com o
prompt ``experiments.noise_prompt`` [GAP-12] e o baseline pareado (``w3 = 0``).
"""

from __future__ import annotations

import pandas as pd

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.evaluation.statistics import summarize_groups
from physics_llm_sr.experiments.cells import (
    BIASED,
    NONE,
    ZERO_MEAN,
    Cell,
    NoiseCondition,
    llm_cell,
)
from physics_llm_sr.experiments.storage import empty_frame

NAME = "noise_robustness"
METRICS = ("mae", "mse", "one_minus_r2", "tree_distance")
GROUPS = ("noise_type", "scenario", "sr", "llm", "noise_location", "noise_level")


def noise_cells(
    cfg: ExperimentConfig, noise_type: str, levels: list[float], locations: list[str]
) -> list[Cell]:
    """Cenários × algoritmos × localizações × níveis × repetições, LLM + baseline."""
    ex = cfg.experiments
    out: list[Cell] = []
    for scenario in ex.scenarios:
        for sr in ex.algorithms:
            for location in locations:
                for level in levels:
                    noise = NoiseCondition(noise_type, location, float(level))
                    for rep in range(ex.repetitions):
                        cell = llm_cell(scenario, sr, ex.noise_model, ex.noise_prompt, noise, rep)
                        out.extend((cell, cell.baseline()))
    return out


def cells(cfg: ExperimentConfig) -> list[Cell]:
    """Matriz do Exp4 (``data.exp4``)."""
    e4 = cfg.data.exp4
    return noise_cells(cfg, ZERO_MEAN, list(e4.levels), list(e4.locations))


def analyze(results: pd.DataFrame) -> pd.DataFrame:
    """Resumo por condição de ruído (LLM e baseline lado a lado), Exp4 e Exp5."""
    sub = results[results["noise_type"].isin([ZERO_MEAN, BIASED])]
    if sub.empty:
        return empty_frame()
    summary: pd.DataFrame = summarize_groups(sub, list(GROUPS), list(METRICS))
    summary["arm"] = (summary["llm"] == NONE).map({True: "baseline", False: "llm"})
    return summary
