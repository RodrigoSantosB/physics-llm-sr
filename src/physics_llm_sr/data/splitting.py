"""Split treino/validação/teste por ``simulation_id`` (GAP-09)."""

from __future__ import annotations

import dataclasses

import numpy as np

from physics_llm_sr.config.schemas import SplitConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.validation import compute_data_hash

PARTITIONS = ("train", "validation", "test")


def _allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Alocação por maior resto com mínimo de 1 por partição de proporção > 0."""
    active = [k for k in PARTITIONS if ratios[k] > 0]
    if n < len(active):
        raise ValueError(
            f"{n} simulações são insuficientes para {len(active)} partições não vazias"
        )
    quotas = {k: n * ratios[k] for k in PARTITIONS}
    counts = {k: int(np.floor(quotas[k])) for k in PARTITIONS}
    by_remainder = sorted(PARTITIONS, key=lambda k: (-(quotas[k] - counts[k]), PARTITIONS.index(k)))
    for k in by_remainder[: n - sum(counts.values())]:
        counts[k] += 1
    for k in active:
        if counts[k] == 0:
            donor = max(active, key=lambda d: (counts[d], -PARTITIONS.index(d)))
            counts[donor] -= 1
            counts[k] += 1
    return counts


def split_by_simulation(
    dataset: Dataset, ratios: SplitConfig, rng: np.random.Generator
) -> dict[str, Dataset]:
    """Particiona por simulação: nenhuma simulação aparece em duas partições.

    As simulações são embaralhadas com ``rng``. As contagens seguem alocação por maior
    resto (empates resolvidos na ordem train, validation, test) e toda partição com
    proporção > 0 recebe ao menos 1 simulação. Cada partição recebe o próprio
    ``metadata["data_hash"]``.

    Raises:
        ValueError: menos simulações que partições com proporção > 0.
    """
    unique = np.unique(dataset.simulation_ids)
    counts = _allocate(unique.size, {k: getattr(ratios, k) for k in PARTITIONS})
    order = rng.permutation(unique)
    groups: dict[str, np.ndarray] = {}
    start = 0
    for name in PARTITIONS:
        groups[name] = order[start : start + counts[name]]
        start += counts[name]
    parts: dict[str, Dataset] = {}
    for name in PARTITIONS:
        mask = np.isin(dataset.simulation_ids, groups[name])
        meta = dict(dataset.metadata)
        meta["partition"] = name
        part = dataclasses.replace(
            dataset,
            X=dataset.X[mask],
            y_true=dataset.y_true[mask],
            y_observed=dataset.y_observed[mask],
            simulation_ids=dataset.simulation_ids[mask],
            t=dataset.t[mask],
            metadata=meta,
        )
        meta["data_hash"] = compute_data_hash(part)
        parts[name] = part
    return parts
