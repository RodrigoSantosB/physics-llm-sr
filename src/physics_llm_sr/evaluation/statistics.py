"""Estatística descritiva e inferencial das repetições (PRD §8, §13).

IC95 da média: ``m ± t(0.975, n-1)·sd/√n`` (t de Student, dp amostral ddof=1). Com ``n = 1``
o dp e o IC são indefinidos (``None``), nunca 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

CONFIDENCE_LEVEL = 0.95
MIN_N_FOR_SD = 2


class StatisticsInputError(ValueError):
    """Entrada inválida (vazia, NaN, inf, tamanhos incompatíveis)."""


@dataclass(frozen=True)
class Summary:
    """Resumo de uma amostra. ``sd``/``ci_*`` são ``None`` quando ``n < 2``."""

    n: int
    mean: float
    sd: float | None
    ci_low: float | None
    ci_high: float | None

    @property
    def ci_defined(self) -> bool:
        """Se o IC95 está definido."""
        return self.ci_low is not None


def _clean(values: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise StatisticsInputError("esperado vetor 1-D não vazio")
    if not np.all(np.isfinite(a)):
        raise StatisticsInputError("valores contêm NaN ou infinito")
    return a


def t_quantile(n: int) -> float:
    """Quantil t(0.975, n-1)."""
    return float(stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2, n - 1))


def summarize(values: Sequence[float] | np.ndarray) -> Summary:
    """Média, dp amostral e IC95 (t de Student)."""
    a = _clean(values)
    n = int(a.size)
    mean = float(np.mean(a))
    if n < MIN_N_FOR_SD:
        return Summary(n, mean, None, None, None)
    sd = float(np.std(a, ddof=1))
    half = t_quantile(n) * sd / math.sqrt(n)
    return Summary(n, mean, sd, mean - half, mean + half)


def summarize_groups(
    df: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str]
) -> pd.DataFrame:
    """Resumo por grupo a partir do DataFrame de resultados (uma linha por repetição).

    Retorna formato longo: ``group_cols + metric, n, mean, sd, ci_low, ci_high, ci_defined``
    (NaN nas colunas indefinidas, com ``ci_defined=False``).
    """
    missing = [c for c in [*group_cols, *metrics] if c not in df.columns]
    if missing:
        raise StatisticsInputError(f"colunas ausentes: {missing}")
    rows: list[dict[str, object]] = []
    for key, sub in df.groupby(list(group_cols), sort=True, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            s = summarize(sub[metric].to_numpy(dtype=float))
            row: dict[str, object] = dict(zip(group_cols, keys, strict=True))
            row.update(
                metric=metric,
                n=s.n,
                mean=s.mean,
                sd=np.nan if s.sd is None else s.sd,
                ci_low=np.nan if s.ci_low is None else s.ci_low,
                ci_high=np.nan if s.ci_high is None else s.ci_high,
                ci_defined=s.ci_defined,
            )
            rows.append(row)
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


@dataclass(frozen=True)
class PairedResult:
    """Resultado do teste pareado baseline vs LLM (diferença = baseline − llm)."""

    n: int
    n_nonzero: int
    mean_diff: float
    statistic: float | None
    p_value: float | None
    cohens_dz: float | None
    rank_biserial: float | None
    method: str
    note: str


def paired_comparison(
    baseline: Sequence[float] | np.ndarray,
    llm: Sequence[float] | np.ndarray,
    alternative: str = "two-sided",
) -> PairedResult:
    """Wilcoxon pareado (scipy) de baseline vs LLM, com tamanho de efeito.

    Diferença ``d = baseline − llm`` (positiva = LLM melhor em métricas de erro).
    Tratamento documentado:

    - Zeros (empates baseline = llm) são descartados (``zero_method="wilcox"``); todos zero
      → ``p = 1.0`` e nota explícita.
    - ``method="auto"`` do scipy: exato se n_nonzero ≤ 50 sem empates de |d|; aproximação
      normal (com correção de empates) caso contrário.
    - ``n_nonzero < 2`` → teste não executado (``p_value=None``). Com n_nonzero pequeno o
      p-valor exato bilateral mínimo é ``2/2^n`` (n=5 → 0.0625): sem poder para 0.05.
    - Tamanho de efeito: ``cohens_dz = mean(d)/sd(d)`` (``None`` se sd = 0 ou n < 2) e
      correlação rank-biserial pareada ``(W+ − W−)/(W+ + W−)``.
    """
    a, b = _clean(baseline), _clean(llm)
    if a.shape != b.shape:
        raise StatisticsInputError("baseline e llm devem ter o mesmo tamanho (pareados)")
    d = a - b
    n = int(d.size)
    nz = d[d != 0.0]
    n_nz = int(nz.size)
    mean_diff = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n >= MIN_N_FOR_SD else 0.0
    dz = mean_diff / sd if sd > 0 else None
    if n_nz == 0:
        return PairedResult(
            n, 0, mean_diff, None, 1.0, dz, 0.0, "degenerate", "todas as diferenças são zero"
        )
    if n_nz < MIN_N_FOR_SD:
        return PairedResult(
            n, n_nz, mean_diff, None, None, dz, None, "not_run", "menos de 2 diferenças não nulas"
        )
    res = stats.wilcoxon(d, zero_method="wilcox", alternative=alternative, method="auto")
    ranks = stats.rankdata(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0]))
    w_minus = float(np.sum(ranks[nz < 0]))
    rbc = (w_plus - w_minus) / (w_plus + w_minus)
    ties = len(np.unique(np.abs(nz))) < n_nz
    method = "wilcoxon_approx" if (ties or n_nz > 50) else "wilcoxon_exact"
    note = f"{n - n_nz} empate(s) descartado(s); empates em |d|: {ties}"
    return PairedResult(
        n, n_nz, mean_diff, float(res.statistic), float(res.pvalue), dz, rbc, method, note
    )
