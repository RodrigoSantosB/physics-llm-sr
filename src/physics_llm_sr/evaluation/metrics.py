"""Métricas de regressão e estruturais (PRD §8).

Política de entrada: vetores 1-D finitos, mesmo comprimento, não vazios. Qualquer violação
levanta :class:`MetricInputError`; nunca há NaN silencioso.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from physics_llm_sr.symbolic.expression import Expr
from physics_llm_sr.symbolic.tree_distance import (
    DEFAULT_TREE_DISTANCE_PARAMS,
    TreeDistanceParams,
    tree_distance,
)

EXACT_RECOVERY_TOL = 1e-12
"""Tolerância numérica para tratar distância como 0 (GAP-11: 0 = correspondência estrutural)."""

R2_ZERO_VARIANCE_PERFECT = 1.0
R2_ZERO_VARIANCE_IMPERFECT = 0.0


class MetricInputError(ValueError):
    """Entrada inválida para uma métrica (NaN, inf, forma ou comprimento incompatíveis)."""


def _prepare(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y, dtype=float)
    b = np.asarray(yhat, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise MetricInputError("y e yhat devem ser vetores 1-D")
    if a.shape != b.shape:
        raise MetricInputError(f"formas diferentes: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise MetricInputError("entrada vazia")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise MetricInputError("entrada contém NaN ou infinito")
    return a, b


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Erro absoluto médio."""
    a, b = _prepare(y, yhat)
    return float(np.mean(np.abs(a - b)))


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Erro quadrático médio."""
    a, b = _prepare(y, yhat)
    return float(np.mean((a - b) ** 2))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coeficiente de determinação ``1 - SS_res/SS_tot``.

    Política para variância zero (``SS_tot == 0``), a mesma do scikit-learn: 1.0 se a
    predição é exata (``SS_res == 0``), senão 0.0. O valor é finito para não contaminar
    médias; use ``mse`` junto para distinguir.
    """
    a, b = _prepare(y, yhat)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot == 0.0:
        return R2_ZERO_VARIANCE_PERFECT if ss_res == 0.0 else R2_ZERO_VARIANCE_IMPERFECT
    return 1.0 - ss_res / ss_tot


def one_minus_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """``1 - R²`` (a métrica reportada pelo artigo)."""
    return 1.0 - r2(y, yhat)


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """Todas as métricas de regressão: mae, mse, r2, one_minus_r2."""
    r = r2(y, yhat)
    return {"mae": mae(y, yhat), "mse": mse(y, yhat), "r2": r, "one_minus_r2": 1.0 - r}


def structural_metrics(
    expr: Expr,
    ground_truth: Expr,
    equivalent_forms: Sequence[Expr] = (),
    params: TreeDistanceParams = DEFAULT_TREE_DISTANCE_PARAMS,
) -> dict[str, float]:
    """Distâncias estruturais.

    - ``tree_distance``: PRIMÁRIA, à GT do artigo.
    - ``tree_distance_equivalent_min``: SECUNDÁRIA (rotulada), mínimo sobre a GT e as formas
      fisicamente equivalentes declaradas. Nunca substitui a primária.
    """
    primary = tree_distance(expr, ground_truth, params)
    others = [tree_distance(expr, f, params) for f in equivalent_forms]
    return {
        "tree_distance": primary,
        "tree_distance_equivalent_min": min([primary, *others]),
    }


def exact_recovery(distance: float) -> bool:
    """Recuperação exata: distância 0 (GAP-11)."""
    if not np.isfinite(distance):
        raise MetricInputError(f"distância inválida: {distance}")
    return bool(distance <= EXACT_RECOVERY_TOL)
