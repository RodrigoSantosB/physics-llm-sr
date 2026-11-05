"""Experimento 5 — ruído enviesado (PRD §7, Tabela 4).

Média ``data.exp5.mean_fraction × média do atributo`` (não escala com o nível), desvio
``nível × std``. Parametrização isolada em ``data.exp5`` e aplicada por
``data.noise.NoiseSpec(kind='biased')``; mesmo dataset limpo e sementes do Exp4.
"""

from __future__ import annotations

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.experiments.cells import BIASED, Cell
from physics_llm_sr.experiments.noise_robustness import noise_cells

NAME = "biased_noise"


def cells(cfg: ExperimentConfig) -> list[Cell]:
    """Matriz do Exp5 (``data.exp5``)."""
    e5 = cfg.data.exp5
    return noise_cells(cfg, BIASED, list(e5.levels), list(e5.locations))
