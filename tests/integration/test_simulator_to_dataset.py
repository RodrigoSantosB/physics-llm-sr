"""Integração simulador -> dataset -> ruído -> split -> persistência."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physics_llm_sr.config.schemas import DataConfig, ExperimentConfig
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.data.generator import (
    compute_data_hash,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from physics_llm_sr.data.noise import NoiseSpec, apply_noise
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.data.validation import validate_dataset
from physics_llm_sr.physics.registry import SCENARIO_NAMES, get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng

CFG = ExperimentConfig(
    mode=Mode.SMOKE,
    track=Track.PAPER_REPLICATION,
    data=DataConfig(n_simulations=30, n_time_points=40),
)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_pipeline_end_to_end(scenario: str, tmp_path: Path) -> None:
    clean = generate_dataset(CFG, scenario, noisy=False)
    sc = get_scenario(scenario, CFG.physics)
    assert np.allclose(clean.y_true, sc.reference(clean.X))
    assert validate_dataset(clean, CFG.physics) == []

    noisy = apply_noise(clean, NoiseSpec("zero_mean", 0.02, "both"), derive_rng(42, "noise/exp4"))
    parts = split_by_simulation(noisy, CFG.data.split, derive_rng(42, "split"))
    assert sum(p.X.shape[0] for p in parts.values()) == noisy.X.shape[0]

    save_dataset(noisy, tmp_path / scenario)
    assert compute_data_hash(load_dataset(tmp_path / scenario)) == compute_data_hash(noisy)
