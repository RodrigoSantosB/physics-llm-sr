"""T:early-stop: parada por melhoria relativa < rel_tol por patience gerações."""

from __future__ import annotations

import math

import pytest

from physics_llm_sr.optimization.early_stopping import EarlyStopping


def run(es: EarlyStopping, seq: list[float]) -> int | None:
    for i, v in enumerate(seq):
        if es.update(v):
            return i
    return None


def test_stops_at_expected_generation() -> None:
    # gen0 base; gen1 melhora 50%; gen2..4 melhoria < 0.1% -> para na gen 4
    seq = [10.0, 5.0, 4.9999, 4.9998, 4.9997, 4.0]
    assert run(EarlyStopping(0.001, 3), seq) == 4


def test_does_not_stop_when_improvement_at_least_tol() -> None:
    seq = [1000.0]
    for _ in range(10):
        seq.append(seq[-1] * (1 - 0.0011))
    assert run(EarlyStopping(0.001, 3), seq) is None


def test_exact_tolerance_counts_as_improvement() -> None:
    es = EarlyStopping(0.5, 2)
    assert run(es, [8.0, 4.0, 2.0, 1.0, 0.5]) is None


def test_improvement_resets_counter() -> None:
    seq = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0]
    assert run(EarlyStopping(0.001, 3), seq) == 6


def test_worsening_counts_as_stagnation() -> None:
    assert run(EarlyStopping(0.001, 2), [1.0, 2.0, 3.0]) == 2


def test_zero_previous_best() -> None:
    assert run(EarlyStopping(0.001, 2), [0.0, 0.0, 0.0]) == 2
    es = EarlyStopping(0.001, 2)
    es.update(1.0)
    es.update(0.0)  # melhoria 100%
    assert es.stagnant == 0


def test_infinite_to_finite_is_improvement() -> None:
    es = EarlyStopping(0.001, 2)
    es.update(math.inf)
    assert not es.update(3.0)
    assert es.stagnant == 0


def test_disabled_never_stops() -> None:
    assert run(EarlyStopping(0.001, 1, enabled=False), [1.0] * 5) is None


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(0.0, 3)
    with pytest.raises(ValueError):
        EarlyStopping(0.001, 0)
    with pytest.raises(ValueError):
        EarlyStopping().update(float("nan"))
