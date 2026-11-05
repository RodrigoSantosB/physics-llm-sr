"""Sementes derivadas estáveis por nome."""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np

from physics_llm_sr.reproducibility.seeds import (
    SEED_UPPER_BOUND,
    derive_rng,
    derive_seed,
    seed_everything,
)


def test_derive_seed_is_deterministic_and_in_range() -> None:
    a = derive_seed(42, "data/free_fall")
    assert a == derive_seed(42, "data/free_fall")
    assert 0 <= a < SEED_UPPER_BOUND


def test_derive_seed_depends_on_name_and_master() -> None:
    assert derive_seed(42, "a") != derive_seed(42, "b")
    assert derive_seed(42, "a") != derive_seed(43, "a")


def test_derive_seed_is_stable_across_processes() -> None:
    code = (
        "from physics_llm_sr.reproducibility.seeds import derive_seed;"
        "print(derive_seed(42, 'split'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert int(out) == derive_seed(42, "split")


def test_derive_rng_reproduces_stream() -> None:
    r1 = derive_rng(7, "noise").random(5)
    r2 = derive_rng(7, "noise").random(5)
    np.testing.assert_array_equal(r1, r2)


def test_seed_everything_fixes_random_and_numpy() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.rand()))
    seed_everything(123)
    assert first == (random.random(), float(np.random.rand()))
