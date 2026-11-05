"""Validação de integridade, unidades e distribuições."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from physics_llm_sr.config.schemas import DataConfig, ExperimentConfig
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.validation import validate_dataset
from physics_llm_sr.physics.registry import SCENARIO_NAMES

CFG = ExperimentConfig(
    mode=Mode.SMOKE,
    track=Track.PAPER_REPLICATION,
    data=DataConfig(n_simulations=20, n_time_points=30),
)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_generated_clean_dataset_is_valid(scenario: str) -> None:
    assert validate_dataset(generate_dataset(CFG, scenario, noisy=False), CFG.physics) == []


def test_detects_nan_shape_and_range_problems() -> None:
    ds = generate_dataset(CFG, "free_fall", noisy=False)
    bad_y = ds.y_true.copy()
    bad_y[0] = np.nan
    assert any("finit" in p for p in validate_dataset(dataclasses.replace(ds, y_true=bad_y)))
    assert any("forma" in p for p in validate_dataset(dataclasses.replace(ds, t=ds.t[:-1])))
    bad_x = ds.X.copy()
    bad_x[:, 0] = 1e6  # mass fora da faixa
    problems = validate_dataset(dataclasses.replace(ds, X=bad_x), CFG.physics)
    assert any("mass" in p for p in problems)


def test_detects_gt_mismatch_and_unsorted_time() -> None:
    ds = generate_dataset(CFG, "free_fall", noisy=False)
    problems = validate_dataset(dataclasses.replace(ds, y_true=ds.y_true * 2.0), CFG.physics)
    assert any("GT" in p for p in problems)
    t = ds.t.copy()
    t[[0, 1]] = t[[1, 0]]
    assert any("ordenad" in p for p in validate_dataset(dataclasses.replace(ds, t=t)))
