"""T:metricas — métricas de regressão e estruturais (casos calculados à mão)."""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.evaluation.metrics import (
    MetricInputError,
    exact_recovery,
    mae,
    mse,
    one_minus_r2,
    r2,
    regression_metrics,
    structural_metrics,
)
from physics_llm_sr.symbolic.expression import parse_infix

Y = np.array([1.0, 2.0, 3.0])
YH = np.array([1.0, 2.0, 4.0])


def test_hand_computed_case() -> None:
    assert mae(Y, YH) == pytest.approx(1 / 3)
    assert mse(Y, YH) == pytest.approx(1 / 3)
    assert r2(Y, YH) == pytest.approx(0.5)
    assert one_minus_r2(Y, YH) == pytest.approx(0.5)
    out = regression_metrics(Y, YH)
    assert set(out) == {"mae", "mse", "r2", "one_minus_r2"}
    assert out["r2"] == pytest.approx(0.5)


def test_perfect_prediction() -> None:
    assert r2(Y, Y) == 1.0
    assert one_minus_r2(Y, Y) == 0.0


def test_r2_zero_variance_policy() -> None:
    const = np.array([2.0, 2.0, 2.0])
    assert r2(const, const) == 1.0
    assert r2(const, const + 1) == 0.0


@pytest.mark.parametrize(
    ("y", "yh"),
    [
        (np.array([1.0, np.nan]), np.array([1.0, 2.0])),
        (np.array([1.0, 2.0]), np.array([1.0, np.inf])),
        (np.array([1.0, 2.0]), np.array([1.0])),
        (np.array([]), np.array([])),
    ],
)
def test_invalid_input_raises(y: np.ndarray, yh: np.ndarray) -> None:
    with pytest.raises(MetricInputError):
        mae(y, yh)
    with pytest.raises(MetricInputError):
        regression_metrics(y, yh)


def test_structural_primary_and_equivalent_min() -> None:
    names = ["height", "t"]
    gt = parse_infix("sqrt(2*9.81*height)", names)
    equiv = (parse_infix("9.81*t", names),)
    cand = parse_infix("9.81*t", names)
    out = structural_metrics(cand, gt, equiv)
    assert out["tree_distance"] > 0.0
    assert out["tree_distance_equivalent_min"] == 0.0
    assert exact_recovery(out["tree_distance_equivalent_min"])
    assert not exact_recovery(out["tree_distance"])
    same = structural_metrics(gt, gt)
    assert same == {"tree_distance": 0.0, "tree_distance_equivalent_min": 0.0}


def test_exact_recovery_rejects_invalid() -> None:
    with pytest.raises(MetricInputError):
        exact_recovery(float("nan"))
