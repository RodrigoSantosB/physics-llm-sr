"""C:ruido (T:ruido): ruído zero_mean/biased por localização e nível."""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.config.schemas import DataConfig, ExperimentConfig
from physics_llm_sr.core_types import Dataset, Mode, Track
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.noise import NoiseSpec, apply_noise
from physics_llm_sr.reproducibility.seeds import derive_rng

LEVELS = (0.01, 0.02, 0.03, 0.04, 0.05)
LOCATIONS = ("inputs", "target", "both")
SE_FACTOR = 3.0
BIAS_FRACTION = 0.01


@pytest.fixture(scope="module")
def clean() -> Dataset:
    cfg = ExperimentConfig(
        mode=Mode.SMOKE,
        track=Track.PAPER_REPLICATION,
        data=DataConfig(n_simulations=100, n_time_points=100),
    )
    return generate_dataset(cfg, "free_fall", noisy=False)


def _std_within_3se(noise: np.ndarray, target_sigma: float) -> bool:
    n = noise.size
    se = target_sigma / np.sqrt(2 * (n - 1))
    return bool(abs(float(np.std(noise, ddof=1)) - target_sigma) <= SE_FACTOR * se)


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("level", LEVELS)
def test_zero_mean_empirical_std_matches_target(
    clean: Dataset, location: str, level: float
) -> None:
    out = apply_noise(clean, NoiseSpec("zero_mean", level, location), derive_rng(1, "noise"))  # type: ignore[arg-type]
    if location in ("target", "both"):
        assert _std_within_3se(out.y_observed - clean.y_true, level * np.std(clean.y_true))
    else:
        assert np.array_equal(out.y_observed, clean.y_true)
    if location in ("inputs", "both"):
        for j in range(clean.X.shape[1]):
            assert _std_within_3se(out.X[:, j] - clean.X[:, j], level * np.std(clean.X[:, j]))
    else:
        assert np.array_equal(out.X, clean.X)


def test_identifiers_and_truth_never_perturbed_and_input_not_mutated(clean: Dataset) -> None:
    x_before, y_before = clean.X.copy(), clean.y_observed.copy()
    out = apply_noise(clean, NoiseSpec("zero_mean", 0.05, "both"), derive_rng(1, "noise"))
    assert np.array_equal(out.simulation_ids, clean.simulation_ids)
    assert np.array_equal(out.t, clean.t)
    assert np.array_equal(out.y_true, clean.y_true)
    assert out.scenario == clean.scenario and out.feature_names == clean.feature_names
    assert np.array_equal(clean.X, x_before) and np.array_equal(clean.y_observed, y_before)
    assert out.X is not clean.X
    assert out.metadata["noise"]["kind"] == "zero_mean"


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("level", LEVELS)
def test_biased_mean_and_std(clean: Dataset, location: str, level: float) -> None:
    out = apply_noise(clean, NoiseSpec("biased", level, location), derive_rng(2, "noise"))  # type: ignore[arg-type]
    if location in ("target", "both"):
        d = out.y_observed - clean.y_true
        sigma = level * np.std(clean.y_true)
        assert abs(d.mean() - BIAS_FRACTION * clean.y_true.mean()) <= SE_FACTOR * sigma / np.sqrt(
            d.size
        )
        assert _std_within_3se(d, sigma)
    if location in ("inputs", "both"):
        for j in range(clean.X.shape[1]):
            col = clean.X[:, j]
            d = out.X[:, j] - col
            sigma = level * np.std(col)
            assert abs(d.mean() - BIAS_FRACTION * col.mean()) <= SE_FACTOR * sigma / np.sqrt(d.size)


def test_same_clean_dataset_and_seed_is_deterministic(clean: Dataset) -> None:
    spec = NoiseSpec("zero_mean", 0.03, "both")
    a = apply_noise(clean, spec, derive_rng(5, "n"))
    b = apply_noise(clean, spec, derive_rng(5, "n"))
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y_observed, b.y_observed)
    assert a.metadata["data_hash"] == b.metadata["data_hash"] != clean.metadata["data_hash"]


def test_invalid_spec_rejected() -> None:
    with pytest.raises(ValueError, match="level"):
        NoiseSpec("zero_mean", -0.1, "target")
    with pytest.raises(ValueError, match="kind"):
        NoiseSpec("weird", 0.1, "target")  # type: ignore[arg-type]


def test_default_bias_fraction_comes_from_exp5_config() -> None:
    import inspect

    from physics_llm_sr.config.schemas import Exp5Config
    from physics_llm_sr.data import noise

    spec = noise.NoiseSpec("biased", 0.01, "target")
    assert spec.bias_fraction == Exp5Config().mean_fraction
    assert "0.01" not in inspect.getsource(noise).replace("level*std", "")  # sem duplicata
