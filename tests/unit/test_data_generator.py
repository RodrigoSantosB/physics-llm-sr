"""C:dados-repro (T:dados): geração reprodutível, colunas, ids e persistência."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.schemas import DataConfig, ExperimentConfig
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.data.generator import (
    compute_data_hash,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from physics_llm_sr.physics.base import FEATURE_NAMES
from physics_llm_sr.physics.registry import SCENARIO_NAMES

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
SMALL_SIMS = 12
SMALL_POINTS = 20


def small_cfg(master: int = 42) -> ExperimentConfig:
    return ExperimentConfig(
        mode=Mode.SMOKE,
        track=Track.PAPER_REPLICATION,
        seeds={"master": master},
        data=DataConfig(n_simulations=SMALL_SIMS, n_time_points=SMALL_POINTS),
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_same_seed_and_config_give_identical_hash(scenario: str) -> None:
    a = generate_dataset(small_cfg(), scenario)
    b = generate_dataset(small_cfg(), scenario)
    assert a.metadata["data_hash"] == b.metadata["data_hash"] == compute_data_hash(a)


def test_different_master_seed_changes_hash() -> None:
    a = generate_dataset(small_cfg(42), "free_fall")
    b = generate_dataset(small_cfg(43), "free_fall")
    assert a.metadata["data_hash"] != b.metadata["data_hash"]


def test_different_seed_name_changes_hash() -> None:
    a = generate_dataset(small_cfg(), "free_fall", seed_name="data/a")
    b = generate_dataset(small_cfg(), "free_fall", seed_name="data/b")
    assert a.metadata["data_hash"] != b.metadata["data_hash"]


def test_columns_ids_and_shapes() -> None:
    ds = generate_dataset(small_cfg(), "harmonic_motion")
    n = SMALL_SIMS * SMALL_POINTS
    assert ds.scenario == "harmonic_motion"
    assert ds.feature_names == FEATURE_NAMES
    assert ds.X.shape == (n, len(FEATURE_NAMES))
    assert ds.y_true.shape == ds.y_observed.shape == ds.simulation_ids.shape == ds.t.shape == (n,)
    assert set(np.unique(ds.simulation_ids)) == set(range(SMALL_SIMS))
    assert ds.metadata["seed_name"] == "data/harmonic_motion"
    assert ds.metadata["n_simulations"] == SMALL_SIMS


def test_default_noise_is_applied_to_target_only() -> None:
    ds = generate_dataset(small_cfg(), "free_fall")
    assert not np.array_equal(ds.y_true, ds.y_observed)
    clean = generate_dataset(small_cfg(), "free_fall", noisy=False)
    assert np.array_equal(clean.y_true, clean.y_observed)
    assert np.array_equal(clean.X, ds.X)
    assert np.array_equal(clean.y_true, ds.y_true)


def test_persistence_roundtrip_keeps_hash(tmp_path: Path) -> None:
    ds = generate_dataset(small_cfg(), "damped_wave")
    save_dataset(ds, tmp_path / "ds")
    back = load_dataset(tmp_path / "ds")
    assert compute_data_hash(back) == compute_data_hash(ds)
    assert back.metadata["data_hash"] == ds.metadata["data_hash"]
    assert np.array_equal(back.X, ds.X)
    assert back.feature_names == ds.feature_names


def test_load_detects_tampering(tmp_path: Path) -> None:
    ds = generate_dataset(small_cfg(), "free_fall")
    path = save_dataset(ds, tmp_path / "ds")
    import pandas as pd

    df = pd.read_parquet(path / "data.parquet")
    df["y_observed"] = df["y_observed"].to_numpy() + 1.0
    df.to_parquet(path / "data.parquet", index=False)
    with pytest.raises(ValueError, match="hash"):
        load_dataset(path)


def test_replication_config_counts_smoke() -> None:
    cfg = load_config(CONFIGS_DIR / "replication.yaml")
    assert cfg.data.n_simulations == 500
    assert cfg.data.max_steps == 1000
    assert cfg.data.n_time_points <= cfg.data.max_steps


def test_replication_config_generates_500_simulations() -> None:
    cfg = load_config(CONFIGS_DIR / "replication.yaml")
    ds = generate_dataset(cfg, "free_fall")
    assert len(np.unique(ds.simulation_ids)) == 500
    counts = np.bincount(ds.simulation_ids.astype(int))
    assert int(np.max(counts)) <= cfg.data.max_steps
    assert ds.X.shape[0] == 500 * cfg.data.n_time_points
