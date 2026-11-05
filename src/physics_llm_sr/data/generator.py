"""Geração de datasets (12 etapas do PRD 3.2) e persistência Parquet + JSON.

Etapas: 1 RNG por semente derivada nomeada; 2 amostrar parâmetros; 3 validar faixas;
4 malha temporal; 5 respostas analíticas; 6 parada (teto ``max_steps``); 7 X; 8 y_true;
9 ruído; 10 y_observed; 11 validar integridade/unidades/distribuições; 12 persistir.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.noise import NoiseSpec, apply_noise
from physics_llm_sr.data.sampling import sample_simulations
from physics_llm_sr.data.validation import compute_data_hash, validate_dataset
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.artifacts import (
    load_json,
    load_parquet,
    save_json,
    save_parquet,
)
from physics_llm_sr.reproducibility.seeds import derive_rng

__all__ = [
    "DATA_FILENAME",
    "METADATA_FILENAME",
    "compute_data_hash",
    "generate_dataset",
    "load_dataset",
    "save_dataset",
]

DATA_FILENAME = "data.parquet"
METADATA_FILENAME = "metadata.json"
FEATURE_PREFIX = "x_"
ID_COLUMNS = ("scenario", "simulation_id", "t")
Y_TRUE_COLUMN = "y_true"
Y_OBSERVED_COLUMN = "y_observed"
NOISE_SEED_PREFIX = "noise/"
DATA_SEED_PREFIX = "data/"


class DatasetValidationError(ValueError):
    """Dataset gerado não passou na validação."""


def generate_dataset(
    cfg: ExperimentConfig,
    scenario_name: str,
    seed_name: str | None = None,
    noisy: bool = True,
) -> Dataset:
    """Gera o dataset de ``scenario_name``.

    A semente é ``derive_rng(cfg.seeds.master, seed_name)`` (default ``data/<cenário>``).
    Com ``noisy=True`` aplica o ruído padrão de ``cfg.data.noise`` (média zero, ``sigma_fraction``
    do desvio, na localização configurada) com semente derivada ``noise/<seed_name>``; com
    ``noisy=False`` devolve o dataset limpo (``y_observed == y_true``), base de Exp4/Exp5.
    """
    name = seed_name or f"{DATA_SEED_PREFIX}{scenario_name}"
    scenario = get_scenario(scenario_name, cfg.physics)
    rng = derive_rng(cfg.seeds.master, name)
    x, y_true, t, ids = sample_simulations(scenario, cfg.physics, cfg.data, rng)
    context = scenario.physical_context()
    clean = Dataset(
        X=x,
        y_true=y_true,
        y_observed=y_true.copy(),
        feature_names=tuple(scenario.feature_names),
        target_name=scenario.target_name,
        scenario=scenario_name,
        simulation_ids=ids,
        t=t,
        metadata={
            "seed_name": name,
            "master_seed": cfg.seeds.master,
            "n_simulations": cfg.data.n_simulations,
            "n_time_points": cfg.data.n_time_points,
            "max_steps": cfg.data.max_steps,
            "units": dict(context.units),
            "ground_truth": context.ground_truth,
            "noise": None,
        },
    )
    problems = validate_dataset(clean, cfg.physics)
    if problems:
        raise DatasetValidationError(f"dataset {scenario_name} inválido: {problems}")
    out = clean
    if noisy:
        spec = NoiseSpec("zero_mean", cfg.data.noise.sigma_fraction, cfg.data.noise.location)
        out = apply_noise(clean, spec, derive_rng(cfg.seeds.master, f"{NOISE_SEED_PREFIX}{name}"))
    out.metadata["data_hash"] = compute_data_hash(out)
    return out


def _to_frame(ds: Dataset) -> pd.DataFrame:
    """Tabela longa: ids, features ``x_*``, ``y_true`` e ``y_observed``."""
    cols: dict[str, np.ndarray] = {
        "scenario": np.full(ds.t.shape[0], ds.scenario),
        "simulation_id": ds.simulation_ids,
        "t": ds.t,
    }
    for j, name in enumerate(ds.feature_names):
        cols[f"{FEATURE_PREFIX}{name}"] = ds.X[:, j]
    cols[Y_TRUE_COLUMN] = ds.y_true
    cols[Y_OBSERVED_COLUMN] = ds.y_observed
    frame: pd.DataFrame = pd.DataFrame(cols)
    return frame


def save_dataset(dataset: Dataset, directory: str | Path) -> Path:
    """Grava ``data.parquet`` e ``metadata.json`` (com ``data_hash``) em ``directory``."""
    out = Path(directory)
    meta = dict(dataset.metadata)
    meta["data_hash"] = compute_data_hash(dataset)
    meta.update(
        scenario=dataset.scenario,
        target_name=dataset.target_name,
        feature_names=list(dataset.feature_names),
    )
    save_parquet(_to_frame(dataset), out / DATA_FILENAME)
    save_json(meta, out / METADATA_FILENAME)
    return out


def load_dataset(directory: str | Path) -> Dataset:
    """Lê um dataset salvo e confere o ``data_hash``; ``ValueError`` se divergir."""
    src = Path(directory)
    meta = load_json(src / METADATA_FILENAME)
    df = load_parquet(src / DATA_FILENAME)
    names = tuple(meta["feature_names"])
    ds = Dataset(
        X=df[[f"{FEATURE_PREFIX}{n}" for n in names]].to_numpy(dtype=float),
        y_true=df[Y_TRUE_COLUMN].to_numpy(dtype=float),
        y_observed=df[Y_OBSERVED_COLUMN].to_numpy(dtype=float),
        feature_names=names,
        target_name=meta["target_name"],
        scenario=meta["scenario"],
        simulation_ids=df["simulation_id"].to_numpy(dtype=np.int64),
        t=df["t"].to_numpy(dtype=float),
        metadata=meta,
    )
    actual = compute_data_hash(ds)
    if actual != meta["data_hash"]:
        raise ValueError(f"hash dos dados diverge: esperado {meta['data_hash']}, obtido {actual}")
    return dataclasses.replace(ds)
