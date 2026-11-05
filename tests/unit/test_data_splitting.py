"""C:split (T:split): split por simulation_id 70/15/15."""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.config.schemas import DataConfig, ExperimentConfig, SplitConfig
from physics_llm_sr.core_types import Dataset, Mode, Track
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.reproducibility.seeds import derive_rng

N_SIMS = 100
N_POINTS = 10
PARTS = ("train", "validation", "test")


@pytest.fixture(scope="module")
def ds() -> Dataset:
    cfg = ExperimentConfig(
        mode=Mode.SMOKE,
        track=Track.PAPER_REPLICATION,
        data=DataConfig(n_simulations=N_SIMS, n_time_points=N_POINTS),
    )
    return generate_dataset(cfg, "free_fall")


def test_partitions_have_disjoint_simulation_ids_and_cover_all(ds: Dataset) -> None:
    parts = split_by_simulation(ds, SplitConfig(), derive_rng(42, "split"))
    assert tuple(parts) == PARTS
    sets = [set(np.unique(parts[p].simulation_ids).tolist()) for p in PARTS]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set.union(*sets) == set(range(N_SIMS))
    assert sum(parts[p].X.shape[0] for p in PARTS) == ds.X.shape[0]


def test_proportions_approximate(ds: Dataset) -> None:
    parts = split_by_simulation(ds, SplitConfig(), derive_rng(42, "split"))
    sizes = [len(np.unique(parts[p].simulation_ids)) for p in PARTS]
    assert sizes == [70, 15, 15]


def test_rows_stay_consistent_within_partition(ds: Dataset) -> None:
    parts = split_by_simulation(ds, SplitConfig(), derive_rng(42, "split"))
    tr = parts["train"]
    assert tr.X.shape[0] == tr.y_true.shape[0] == tr.t.shape[0] == tr.simulation_ids.shape[0]
    assert tr.metadata["partition"] == "train"
    assert tr.feature_names == ds.feature_names


def test_deterministic_by_seed_and_varies_with_seed(ds: Dataset) -> None:
    a = split_by_simulation(ds, SplitConfig(), derive_rng(42, "split"))
    b = split_by_simulation(ds, SplitConfig(), derive_rng(42, "split"))
    c = split_by_simulation(ds, SplitConfig(), derive_rng(43, "split"))
    assert np.array_equal(a["test"].simulation_ids, b["test"].simulation_ids)
    assert not np.array_equal(a["test"].simulation_ids, c["test"].simulation_ids)


def _small(n_sims: int) -> Dataset:
    ids = np.repeat(np.arange(n_sims), 3)
    n = ids.size
    x = np.arange(n * 2, dtype=float).reshape(n, 2)
    y = x[:, 0]
    return Dataset(
        X=x,
        y_true=y,
        y_observed=y,
        feature_names=("a", "b"),
        target_name="y",
        scenario="free_fall",
        simulation_ids=ids,
        t=np.arange(n, dtype=float),
    )


def test_five_simulations_leave_no_empty_partition() -> None:
    parts = split_by_simulation(_small(5), SplitConfig(), derive_rng(42, "split"))
    sizes = {k: np.unique(v.simulation_ids).size for k, v in parts.items()}
    assert sizes == {"train": 3, "validation": 1, "test": 1}


@pytest.mark.parametrize("n_sims", [3, 4, 5, 7, 10, 13, 100])
def test_every_partition_nonempty_and_covering(n_sims: int) -> None:
    parts = split_by_simulation(_small(n_sims), SplitConfig(), derive_rng(1, "split"))
    sizes = [np.unique(parts[k].simulation_ids).size for k in PARTS]
    assert min(sizes) >= 1
    assert sum(sizes) == n_sims


def test_too_few_simulations_raises() -> None:
    with pytest.raises(ValueError, match="simula"):
        split_by_simulation(_small(2), SplitConfig(), derive_rng(1, "split"))


def test_zero_ratio_partition_may_be_empty_and_not_counted() -> None:
    cfg = SplitConfig(train=0.8, validation=0.2, test=0.0)
    parts = split_by_simulation(_small(2), cfg, derive_rng(1, "split"))
    assert parts["test"].X.shape[0] == 0
    assert parts["train"].X.shape[0] > 0 and parts["validation"].X.shape[0] > 0


def test_partitions_have_own_data_hash() -> None:
    from physics_llm_sr.data.validation import compute_data_hash

    base = _small(10)
    parts = split_by_simulation(base, SplitConfig(), derive_rng(1, "split"))
    hashes = {k: v.metadata["data_hash"] for k, v in parts.items()}
    assert len(set(hashes.values())) == 3
    for k, v in parts.items():
        assert hashes[k] == compute_data_hash(v)
