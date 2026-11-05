"""Comparação com os resultados publicados (PRD §13).

Normativo: ``references/scientific_assumptions.md``, seção "Critérios de comparação
pré-registrados". Regras 1-6 aplicadas nesta ordem; sem tolerância arbitrária.

Formato esperado do DataFrame local (uma linha por execução/repetição): ver
:data:`RESULTS_COLUMNS`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.evaluation.statistics import MIN_N_FOR_SD, t_quantile

CLASSES = ("numericamente_proxima", "qualitativa", "parcial", "divergente", "inconclusiva")
MIN_REPETITIONS_FULL = 25
"""H-05 / regra 1(b): repetições mínimas válidas no modo full."""
PRECISION_DECIMALS = {"mae": 3, "mse": 3, "one_minus_r2": 2, "tree_distance": 2}
"""Precisão publicada k por métrica (seção 'Critérios', entradas por célula)."""
AVERAGE_SCENARIO = "average_3_scenarios"
SCENARIOS = ("free_fall", "harmonic_motion", "damped_wave")
UNSPECIFIED = "unspecified"
REFERENCE_PROMPT = "A"
BASELINE_LLM = "none"
LOW_NOISE_LEVEL = 0.01
NOISE_TABLES = (3, 4)
INC01_PROMPTS = ("F", "G", "H")
RESULTS_COLUMNS = (
    "run_id, scenario, sr, llm, prompt, noise_type, noise_location, noise_level, repetition, "
    "seed, mode, track, mae, mse, one_minus_r2, tree_distance, tree_distance_equivalent_min"
)
"""Colunas do DataFrame de resultados (baseline: llm='none', prompt='none')."""
_COND = ["scenario", "sr", "llm", "prompt", "noise_type", "noise_location", "noise_level"]
_PAPER_DIR = Path(__file__).resolve().parents[3] / "references" / "paper_results"


class ComparisonError(ValueError):
    """Entrada inconsistente para a comparação."""


@dataclass(frozen=True)
class CellResult:
    """Resultado da comparação de uma célula (campos do 'Relatório por célula')."""

    published: float
    precision_k: int
    published_low: float
    published_high: float
    n: int
    mean: float | None
    sd: float | None
    ci_low: float | None
    ci_high: float | None
    abs_diff: float | None
    rel_diff: float | None
    ref_published: float | None
    delta_published: float | None
    delta_local: float | None
    delta_ci_low: float | None
    delta_ci_high: float | None
    direction: str
    magnitude_ratio: float | None
    direction_agrees: bool | None
    reduced_run: bool
    indicative_class: str | None
    classe: str
    reason: str | None


def _ci(a: np.ndarray) -> tuple[float, float, float | None, float | None]:
    n = a.size
    m = float(np.mean(a))
    if n < MIN_N_FOR_SD:
        return m, math.nan, None, None
    sd = float(np.std(a, ddof=1))
    half = t_quantile(n) * sd / math.sqrt(n)
    return m, sd, m - half, m + half


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def classify_cell(
    published: float,
    metric: str,
    local: Sequence[float] | np.ndarray,
    ref_published: float | None = None,
    local_ref: Sequence[float] | np.ndarray | None = None,
    reduced_run: bool = False,
    inconclusive_reason: str | None = None,
    n_min: int = MIN_REPETITIONS_FULL,
) -> CellResult:
    """Classifica uma célula pelas regras 1-6 pré-registradas.

    ``local`` e ``local_ref`` são pareados por repetição (mesmo tamanho). Sem referência
    (``ref_published is None``), a regra 2 não exige direção. ``reduced_run`` (smoke, pilot,
    extended) ou ``n < n_min`` => ``classe='inconclusiva'`` (regra 1(b) e nota final da
    seção) com ``reduced_run=True``; a classe que as regras 2-5 dariam fica em
    ``indicative_class`` (apenas informativa). ``inconclusive_reason`` (regra 1(a)/(c))
    => inconclusiva sem classe indicativa (direção ainda reportada).
    """
    k = PRECISION_DECIMALS[metric]
    r = 0.5 * 10 ** (-k)
    p_low, p_high = max(0.0, published - r), published + r
    a = np.asarray(local, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ComparisonError("valores locais vazios ou não finitos")
    n = int(a.size)
    m, sd, lo, hi = _ci(a)
    abs_diff = m - published
    rel_diff = None if published == 0 else abs_diff / published

    d_loc: float | None = None
    d_lo: float | None = None
    d_hi: float | None = None
    d_pub: float | None = None
    if ref_published is not None and local_ref is not None:
        b = np.asarray(local_ref, dtype=float)
        if b.shape != a.shape:
            raise ComparisonError("local e local_ref devem ser pareados (mesmo tamanho)")
        d_loc, _, d_lo, d_hi = _ci(a - b)
        d_pub = published - ref_published
    has_ref = d_loc is not None
    delta_null = d_pub is not None and abs(d_pub) <= 2 * r
    d_ci_ok = d_lo is not None and d_hi is not None
    excludes0 = d_ci_ok and (d_lo > 0 or d_hi < 0)  # type: ignore[operator]
    contains0 = d_ci_ok and not excludes0
    sign_ok = has_ref and d_pub is not None and d_loc is not None and _sign(d_loc) == _sign(d_pub)

    agrees: bool | None = None
    if has_ref and d_ci_ok and d_pub is not None:
        agrees = bool((sign_ok and excludes0 and not delta_null) or (delta_null and contains0))
    if d_loc is None:
        direction = "sem_referencia"
    elif not d_ci_ok or contains0:
        direction = "sem_mudanca_significativa"
    else:
        direction = "aumenta" if d_loc > 0 else "diminui"
    magnitude = None if (d_loc is None or not d_pub) else d_loc / d_pub

    intersects = lo is not None and hi is not None and lo <= p_high and hi >= p_low
    indicative: str | None = None
    if lo is not None:
        if not has_ref:
            direction_ok: bool | None = True
        else:
            direction_ok = agrees
        if intersects and direction_ok:
            indicative = "numericamente_proxima"
        elif not intersects and has_ref and agrees:
            indicative = "qualitativa"
        elif has_ref and (
            (intersects and not agrees) or (sign_ok and contains0 and not delta_null)
        ):
            indicative = "parcial"
        else:
            indicative = "divergente"

    reduced = reduced_run or n < n_min
    reason = inconclusive_reason
    if reason is None and reduced:
        reason = f"execução reduzida (n={n}, n_min={n_min}, reduced_run={reduced_run})"
    if inconclusive_reason is not None:
        indicative = None
    classe = "inconclusiva" if (reason is not None or indicative is None) else indicative
    if classe == "inconclusiva" and reason is None:
        reason = "IC95 indefinido (n < 2)"
    return CellResult(
        published, k, p_low, p_high, n, m, None if math.isnan(sd) else sd, lo, hi, abs_diff,
        rel_diff, ref_published, d_pub, d_loc, d_lo, d_hi, direction, magnitude, agrees,
        reduced, indicative if classe == "inconclusiva" else None, classe, reason,
    )  # fmt: skip


def load_paper_results(directory: Path | None = None) -> pd.DataFrame:
    """Carrega as Tabelas 2-5 (formato longo) de ``references/paper_results``."""
    base = directory or _PAPER_DIR
    frames = [
        pd.read_csv(base / name, dtype={"value": str, "prompt": str, "source_page": str})
        for name in (
            "table2_prompt_sensitivity.csv",
            "table3_noise_zero_mean.csv",
            "table4_noise_biased.csv",
            "table5_benchmark_supplement.csv",
        )
    ]
    df = pd.concat(frames, ignore_index=True)
    df["value"] = df["value"].astype(float)
    return df


def _local_series(
    df: pd.DataFrame, cond: dict[str, Any], metric: str, wildcards: dict[str, str | None]
) -> pd.Series[float] | None:
    """Série por repetição (média sobre cenários se média-de-3); ``None`` se não há dados."""
    sub = df
    for col, val in cond.items():
        if col == "scenario" and val == AVERAGE_SCENARIO:
            continue
        if col in ("llm", "prompt") and val == UNSPECIFIED:
            fixed = wildcards.get(col)
            if fixed is not None:
                sub = sub[sub[col] == fixed]
            elif col == "llm":
                sub = sub[sub[col] != BASELINE_LLM]
            continue
        if col == "noise_level":
            sub = sub[np.isclose(sub[col].astype(float), float(val))]
        else:
            sub = sub[sub[col] == val]
    if sub.empty:
        return None
    if cond["scenario"] == AVERAGE_SCENARIO:
        if set(sub["scenario"]) < set(SCENARIOS):
            return None
        combos = sub[["llm", "prompt"]].drop_duplicates()
    else:
        combos = sub[["llm", "prompt"]].drop_duplicates()
    if len(combos) > 1:
        raise ComparisonError(
            f"condição ambígua ({len(combos)} combinações llm/prompt) para {cond}; "
            "informe unspecified_llm/unspecified_prompt"
        )
    by_rep = sub.groupby("repetition")[metric]
    if cond["scenario"] == AVERAGE_SCENARIO:
        counts = sub.groupby("repetition")["scenario"].nunique()
        valid = counts[counts == len(SCENARIOS)].index
        return by_rep.mean().loc[valid] if len(valid) else None
    if by_rep.size().max() > 1:
        raise ComparisonError(f"repetições duplicadas para {cond}")
    return by_rep.mean()


def _reference(row: pd.Series[Any], paper: pd.DataFrame) -> tuple[dict[str, Any], float] | None:
    """Célula de referência da seção 'Entradas por célula' (condição, valor publicado)."""
    cond = {c: row[c] for c in _COND}
    table = int(row["table"])
    if table == 2:
        if row["prompt"] == REFERENCE_PROMPT:
            return None
        cond["prompt"] = REFERENCE_PROMPT
    elif table == 5:
        if row["llm"] == BASELINE_LLM:
            return None
        cond["llm"], cond["prompt"] = BASELINE_LLM, BASELINE_LLM
    elif table == 3:
        if np.isclose(row["noise_level"], LOW_NOISE_LEVEL):
            return None
        cond["noise_level"] = LOW_NOISE_LEVEL
    elif table == 4:
        cond["noise_type"] = "zero_mean"
    else:
        return None
    ref_table = 3 if table == 4 else table
    m = paper[(paper["table"] == ref_table) & (paper["metric"] == row["metric"])]
    for c, v in cond.items():
        m = m[np.isclose(m[c], v)] if c == "noise_level" else m[m[c] == v]
    if len(m) != 1:
        return None
    return cond, float(m["value"].iloc[0])


def _baseline_published(row: pd.Series[Any], paper: pd.DataFrame) -> float | None:
    m = paper[
        (paper["table"] == 5)
        & (paper["llm"] == BASELINE_LLM)
        & (paper["scenario"] == row["scenario"])
        & (paper["sr"] == row["sr"])
        & (paper["metric"] == row["metric"])
    ]
    return float(m["value"].iloc[0]) if len(m) == 1 else None


def compare_with_paper(
    results: pd.DataFrame,
    paper: pd.DataFrame | None = None,
    tables: Sequence[int] = (2, 3, 4, 5),
    unspecified_llm: str | None = None,
    unspecified_prompt: str | None = None,
    absolute_level_inconclusive_tables: Sequence[int] = NOISE_TABLES,
    n_min: int = MIN_REPETITIONS_FULL,
) -> pd.DataFrame:
    """Compara os resultados locais com cada célula comparável publicada.

    Aplica :func:`classify_cell` (regras da seção "Critérios de comparação
    pré-registrados"). ``reduced_run`` é verdadeiro se qualquer linha local tiver
    ``mode != 'full'`` ou ``track != 'paper_replication'``, ou se ``n < n_min``.
    Células sem dados locais saem como ``inconclusiva`` (``n = 0``, motivo explícito).
    Tabelas 3 e 4: nível absoluto inconclusivo por padrão (regra 1(a), GAP-12);
    a direção continua reportada. Notas: INC-01 para os prompts F, G, H (Tabela 2).
    """
    need = set(_COND) | {"repetition", "mode"}
    missing = need - set(results.columns)
    if missing:
        raise ComparisonError(f"colunas ausentes no DataFrame de resultados: {sorted(missing)}")
    paper_df = load_paper_results() if paper is None else paper
    wild = {"llm": unspecified_llm, "prompt": unspecified_prompt}
    run_reduced = bool((results["mode"] != "full").any())
    if "track" in results.columns:
        run_reduced = run_reduced or bool((results["track"] != "paper_replication").any())
    out: list[dict[str, Any]] = []
    for _, row in paper_df[paper_df["table"].isin(list(tables))].iterrows():
        info: dict[str, Any] = {c: row[c] for c in ["table", *_COND, "metric"]}
        info["notes"] = "INC-01" if (row["table"] == 2 and row["prompt"] in INC01_PROMPTS) else ""
        metric = str(row["metric"])
        cond = {c: row[c] for c in _COND}
        series = _local_series(results, cond, metric, wild)
        base = results[
            (results["llm"] == BASELINE_LLM)
            & (results["sr"] == row["sr"])
            & (results["noise_type"] == row["noise_type"])
            & (results["noise_location"] == row["noise_location"])
            & np.isclose(results["noise_level"].astype(float), float(row["noise_level"]))
            & (results["scenario"] == row["scenario"])
        ]
        info["baseline_published"] = _baseline_published(row, paper_df)
        info["baseline_local_mean"] = float(base[metric].mean()) if len(base) else None
        if series is None or series.empty:
            info.update(published=row["value"], n=0, classe="inconclusiva", reduced_run=run_reduced)
            info["reason"] = "sem dados locais para a condição"
            out.append(info)
            continue
        ref = _reference(row, paper_df)
        local_ref = ref_pub = None
        if ref is not None:
            rs = _local_series(results, ref[0], metric, wild)
            if rs is not None:
                idx = series.index.intersection(rs.index)
                if len(idx):
                    series, local_ref, ref_pub = series.loc[idx], rs.loc[idx], ref[1]
        reason = None
        if int(row["table"]) in absolute_level_inconclusive_tables:
            reason = "regra 1(a): nível absoluto depende de GAP-12 (limitação)"
        cell = classify_cell(
            float(row["value"]), metric, series.to_numpy(dtype=float),
            ref_pub, None if local_ref is None else local_ref.to_numpy(dtype=float),
            reduced_run=run_reduced, inconclusive_reason=reason, n_min=n_min,
        )  # fmt: skip
        info.update(cell.__dict__)
        out.append(info)
    frame: pd.DataFrame = pd.DataFrame(out)
    return frame
