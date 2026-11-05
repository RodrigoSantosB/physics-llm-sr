"""Split treino/validação/teste por ``simulation_id`` (GAP-09)."""

from __future__ import annotations

import dataclasses

import numpy as np

from physics_llm_sr.config.schemas import SplitConfig
from physics_llm_sr.core_types import Dataset

PARTITIONS = ("train", "validation", "test")


def split_by_simulation(
    dataset: Dataset, ratios: SplitConfig, rng: np.random.Generator
) -> dict[str, Dataset]:
    """Particiona por simulação: nenhuma simulação aparece em duas partições.

    As simulações são embaralhadas com ``rng``; ``train`` e ``validation`` recebem
    ``round(n*ratio)`` simulações e ``test`` o restante.
    """
    unique = np.unique(dataset.simulation_ids)
    n = unique.size
    order = rng.permutation(unique)
    n_train = round(n * ratios.train)
    n_val = round(n * ratios.validation)
    groups = {
        "train": order[:n_train],
        "validation": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    parts: dict[str, Dataset] = {}
    for name in PARTITIONS:
        mask = np.isin(dataset.simulation_ids, groups[name])
        meta = dict(dataset.metadata)
        meta["partition"] = name
        parts[name] = dataclasses.replace(
            dataset,
            X=dataset.X[mask],
            y_true=dataset.y_true[mask],
            y_observed=dataset.y_observed[mask],
            simulation_ids=dataset.simulation_ids[mask],
            t=dataset.t[mask],
            metadata=meta,
        )
    return parts
