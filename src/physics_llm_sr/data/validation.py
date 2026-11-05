"""Validação de integridade, unidades e distribuições de um ``Dataset``, e hash de dados."""

from __future__ import annotations

import hashlib

import numpy as np

from physics_llm_sr.config.schemas import PhysicsConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.physics.registry import get_scenario

#: Tolerâncias relativa/absoluta para conferir ``y_true`` contra a GT do cenário.
GT_RTOL = 1e-9
GT_ATOL = 1e-12
#: Colunas de X cujo limite inferior físico é 0 (``height`` é a distância percorrida na queda).
ZERO_BASED_FEATURES = ("height",)
#: Folga relativa nas faixas amostradas (erro de ponto flutuante).
RANGE_SLACK = 1e-9
HASH_ALGORITHM = "sha256"
_FLOAT_DTYPE = "<f8"
_INT_DTYPE = "<i8"


def compute_data_hash(dataset: Dataset) -> str:
    """Hash SHA-256 do conteúdo (arrays, nomes, cenário); ignora ``metadata``."""
    h = hashlib.new(HASH_ALGORITHM)
    h.update(f"{dataset.scenario}|{dataset.target_name}|{','.join(dataset.feature_names)}".encode())
    for arr in (dataset.X, dataset.y_true, dataset.y_observed, dataset.t):
        a = np.ascontiguousarray(arr, dtype=_FLOAT_DTYPE)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    ids = np.ascontiguousarray(dataset.simulation_ids, dtype=_INT_DTYPE)
    h.update(ids.tobytes())
    return h.hexdigest()


def _inputs_perturbed(dataset: Dataset) -> bool:
    noise = dataset.metadata.get("noise")
    return isinstance(noise, dict) and noise.get("location") in ("inputs", "both")


def validate_dataset(dataset: Dataset, physics: PhysicsConfig | None = None) -> list[str]:
    """Devolve a lista de problemas (vazia = válido).

    Sem ``physics`` verifica forma, finitude, ordenação temporal, unidades e variância;
    com ``physics`` confere também as faixas amostradas e ``y_true`` contra a GT do cenário.
    """
    problems: list[str] = []
    n = dataset.X.shape[0]
    shapes_ok = (
        dataset.X.ndim == 2
        and dataset.X.shape[1] == len(dataset.feature_names)
        and all(
            a.shape == (n,)
            for a in (dataset.y_true, dataset.y_observed, dataset.simulation_ids, dataset.t)
        )
    )
    if not shapes_ok:
        return [f"forma inconsistente: X{dataset.X.shape}, features={len(dataset.feature_names)}"]
    if n == 0:
        return ["dataset vazio"]

    for name, arr in (
        ("X", dataset.X),
        ("y_true", dataset.y_true),
        ("y_observed", dataset.y_observed),
        ("t", dataset.t),
    ):
        if not np.all(np.isfinite(arr)):
            problems.append(f"{name} contém valores não finitos")

    ids = dataset.simulation_ids
    same_sim = ids[1:] == ids[:-1]
    if np.any(np.diff(dataset.t)[same_sim] < 0):
        problems.append("t não ordenado dentro de simulation_id")
    if np.any(dataset.t < 0):
        problems.append("t negativo")
    if np.any(ids[1:][~same_sim] < ids[:-1][~same_sim]) or len(set(ids.tolist())) != (
        int(np.count_nonzero(~same_sim)) + 1
    ):
        problems.append("simulation_id não contíguo")

    units = dataset.metadata.get("units")
    if units is not None:
        missing = [k for k in (*dataset.feature_names, dataset.target_name) if k not in units]
        if missing:
            problems.append(f"unidades ausentes para: {missing}")

    if float(np.std(dataset.y_true)) == 0.0:
        problems.append("y_true constante (distribuição degenerada)")

    if physics is not None and not _inputs_perturbed(dataset):
        problems.extend(_check_against_physics(dataset, physics))
    return problems


def _check_against_physics(dataset: Dataset, physics: PhysicsConfig) -> list[str]:
    problems: list[str] = []
    scenario = get_scenario(dataset.scenario, physics)
    for j, name in enumerate(dataset.feature_names):
        rng = getattr(physics.sampling, name, None)
        if rng is None:
            continue
        low = 0.0 if name in ZERO_BASED_FEATURES else rng.low
        col = dataset.X[:, j]
        slack = RANGE_SLACK * max(abs(rng.high), 1.0)
        if col.min() < low - slack or col.max() > rng.high + slack:
            problems.append(
                f"{name} fora da faixa [{low}, {rng.high}]: [{col.min():.6g}, {col.max():.6g}]"
            )
    if np.all(np.isfinite(dataset.X)) and not np.allclose(
        scenario.reference(dataset.X), dataset.y_true, rtol=GT_RTOL, atol=GT_ATOL
    ):
        problems.append("y_true diverge da GT analítica do cenário")
    return problems
