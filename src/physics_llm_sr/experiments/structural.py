"""Experimento 6 — comparação estrutural (PRD §7, §8.2).

A distância de cada equação descoberta à GT (primária) e o mínimo sobre as formas
equivalentes declaradas (secundária, rotulada) são calculados por célula em
``experiments.execution``. Aqui ficam o cálculo e o resumo (frequência de recuperação).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from physics_llm_sr.config.schemas import SymbolicConfig
from physics_llm_sr.evaluation.metrics import exact_recovery, structural_metrics
from physics_llm_sr.experiments.storage import empty_frame
from physics_llm_sr.symbolic.expression import Expr
from physics_llm_sr.symbolic.tree_distance import TreeDistanceParams

NAME = "structural"
GROUPS = ("scenario", "sr", "llm", "prompt", "noise_type", "noise_location", "noise_level")


def structural_row(
    expr: Expr, ground_truth: Expr, equivalents: Sequence[Expr], symbolic: SymbolicConfig
) -> dict[str, float | bool]:
    """``tree_distance``, ``tree_distance_equivalent_min``, ``exact_recovery`` (+ equivalente)."""
    params = TreeDistanceParams.from_config(symbolic)
    out: dict[str, float | bool] = dict(structural_metrics(expr, ground_truth, equivalents, params))
    out["exact_recovery"] = exact_recovery(float(out["tree_distance"]))
    out["exact_recovery_equivalent"] = exact_recovery(float(out["tree_distance_equivalent_min"]))
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Frequência de recuperação exata e distância média por configuração."""
    if results.empty:
        return empty_frame()
    grouped = results.groupby(list(GROUPS), dropna=False)
    frame: pd.DataFrame = grouped.agg(
        n=("tree_distance", "size"),
        tree_distance_mean=("tree_distance", "mean"),
        tree_distance_equivalent_min_mean=("tree_distance_equivalent_min", "mean"),
        exact_recovery_rate=("exact_recovery", "mean"),
        exact_recovery_equivalent_rate=("exact_recovery_equivalent", "mean"),
        complexity_mean=("complexity", "mean"),
    ).reset_index()
    return frame
