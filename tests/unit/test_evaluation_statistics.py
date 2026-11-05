"""Estatística: média, dp, IC95, resumo por grupo e teste pareado."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.evaluation.statistics import (
    StatisticsInputError,
    paired_comparison,
    summarize,
    summarize_groups,
)


def test_summarize_hand() -> None:
    s = summarize([1.0, 2.0, 3.0])
    assert s.mean == 2.0 and s.sd == pytest.approx(1.0)
    half = 4.302652729911275 / math.sqrt(3)
    assert s.ci_low == pytest.approx(2 - half) and s.ci_high == pytest.approx(2 + half)


def test_n1_ci_undefined() -> None:
    s = summarize([5.0])
    assert s.sd is None and s.ci_low is None and not s.ci_defined


def test_invalid_input() -> None:
    with pytest.raises(StatisticsInputError):
        summarize([1.0, float("nan")])
    with pytest.raises(StatisticsInputError):
        summarize([])


def test_summarize_groups() -> None:
    df = pd.DataFrame({"sr": ["a", "a", "b"], "mae": [1.0, 3.0, 2.0]})
    out = summarize_groups(df, ["sr"], ["mae"])
    a = out[out["sr"] == "a"].iloc[0]
    assert a["mean"] == 2.0 and a["n"] == 2 and bool(a["ci_defined"])
    b = out[out["sr"] == "b"].iloc[0]
    assert not bool(b["ci_defined"]) and np.isnan(b["sd"])


def test_paired_llm_better() -> None:
    base = np.arange(10, 20, dtype=float)
    llm = base - np.linspace(1, 2, 10)
    r = paired_comparison(base, llm)
    assert r.method == "wilcoxon_exact" and r.p_value is not None and r.p_value < 0.01
    assert r.rank_biserial == pytest.approx(1.0) and r.cohens_dz is not None and r.cohens_dz > 0


def test_paired_all_equal_and_small() -> None:
    r = paired_comparison([1.0, 2.0], [1.0, 2.0])
    assert r.p_value == 1.0 and r.method == "degenerate"
    assert paired_comparison([1.0, 2.0], [1.0, 1.0]).p_value is None
    with pytest.raises(StatisticsInputError):
        paired_comparison([1.0], [1.0, 2.0])


def test_paired_ties_use_approx() -> None:
    r = paired_comparison([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert r.method == "wilcoxon_approx"
