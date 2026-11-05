# mypy: disable-error-code="no-any-return,unused-ignore,no-untyped-call,arg-type,str-unpack"
"""Tabelas de exibição dos notebooks 01 e 02 (sem lógica científica nova).

Reorganizam ``results.parquet`` (e o cache de avaliações) em tabelas legíveis; os números vêm
sempre dos arquivos persistidos e as comparações usam :mod:`physics_llm_sr.evaluation`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.analysis import llm_inspection
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.evaluation.paper_comparison import BASELINE_LLM, compare_with_paper
from physics_llm_sr.experiments.reporting import PAIR_KEYS

STATUS_OK = "ok"
METRICS = ("mae", "mse", "one_minus_r2", "tree_distance", "tree_distance_equivalent_min")
EQUATION_COLUMNS = (
    "scenario",
    "sr",
    "noise_type",
    "noise_location",
    "noise_level",
    "repetition",
    "expression",
    "ground_truth",
    "tree_distance",
    "tree_distance_equivalent_min",
    "exact_recovery",
    "exact_recovery_equivalent",
)
COST_COLUMNS = (
    "llm_requested",
    "llm_unique",
    "llm_calls",
    "llm_cache_hits",
    "llm_invalid_responses",
    "llm_seconds",
    "train_seconds",
)
PUBLISHED_COLUMNS = (
    "table",
    "scenario",
    "sr",
    "noise_type",
    "noise_location",
    "noise_level",
    "metric",
    "published",
    "n",
    "mean",
    "classe",
    "reason",
)
SIDE_BY_SIDE_KEYS = (*PAIR_KEYS, "prompt")


def ok_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Células concluídas com sucesso (``status == 'ok'``); vazio se não há resultados."""
    if results.empty:
        return results
    out: pd.DataFrame = results[results["status"] == STATUS_OK]
    return out


def error_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Células com erro (chave, tipo e mensagem), para exibir sem interromper o notebook."""
    if results.empty:
        return results
    cols = [c for c in ("cell_key", "error_type", "error") if c in results.columns]
    out: pd.DataFrame = results[results["status"] != STATUS_OK][cols]
    return out


def status_table(results: pd.DataFrame) -> pd.DataFrame:
    """Progresso por algoritmo: células ok, com erro e tempo médio de treino (retomada)."""
    if results.empty:
        return pd.DataFrame()
    grouped = results.groupby("sr")
    out = pd.DataFrame(
        {
            "ok": grouped["status"].apply(lambda s: int((s == STATUS_OK).sum())),
            "error": grouped["status"].apply(lambda s: int((s != STATUS_OK).sum())),
            "train_seconds_mean": grouped["train_seconds"].mean(),
        }
    )
    return out.reset_index()


def default_noise_rows(results: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Células do ruído padrão (``data.noise``: média zero, 1% do desvio no alvo): o Exp1."""
    if results.empty:
        return results
    noise = cfg.data.noise
    keep = (
        (results["noise_type"] == "zero_mean")
        & (results["noise_location"] == noise.location)
        & np.isclose(results["noise_level"].astype(float), noise.sigma_fraction)
    )
    out: pd.DataFrame = results[keep]
    return out


def convergence_history(
    results: pd.DataFrame,
    history: pd.DataFrame,
    group: tuple[str, ...] = ("scenario", "sr"),
) -> pd.DataFrame:
    """Histórico por geração das células de ``results`` (já filtradas), com rótulo ``config``."""
    if results.empty or history.empty:
        return pd.DataFrame()
    cols = ["cell_key", *group]
    merged = history.drop(columns=[c for c in group if c in history.columns]).merge(
        ok_rows(results)[cols], on="cell_key"
    )
    merged["config"] = merged[list(group)].astype(str).agg(" / ".join, axis=1)
    return merged


def metrics_summary(
    results: pd.DataFrame, by: tuple[str, ...] = ("scenario", "sr")
) -> pd.DataFrame:
    """Média, desvio-padrão amostral e ``n`` de cada métrica por grupo (só células ok)."""
    ok = ok_rows(results)
    if ok.empty:
        return pd.DataFrame()
    metrics = [m for m in METRICS if m in ok.columns]
    agg = ok.groupby(list(by))[metrics].agg(["mean", "std"])
    agg.columns = pd.Index([f"{a}_{b}" for a, b in agg.columns.tolist()])  # type: ignore[misc]
    agg.insert(0, "n", ok.groupby(list(by)).size())
    return agg.reset_index()


def equations_table(results: pd.DataFrame) -> pd.DataFrame:
    """Equação descoberta de cada célula ok ao lado da GT e das distâncias estruturais."""
    ok = ok_rows(results)
    if ok.empty:
        return pd.DataFrame()
    cols = [c for c in EQUATION_COLUMNS if c in ok.columns]
    out: pd.DataFrame = (
        ok[cols].sort_values(["scenario", "sr", "repetition"]).reset_index(drop=True)
    )
    return out


def baseline_vs_published(results: pd.DataFrame) -> pd.DataFrame:
    """Baseline local (sem LLM) contra as linhas publicadas das Tabelas 3, 4 e 5.

    Tabela 5: linhas ``none`` (baseline publicado). Tabelas 3 e 4: o artigo não informa o LLM
    nem o prompt (GAP-12); a curva publicada é comparada com o baseline local como referência
    de direção. Células sem dado local saem com ``n = 0``. Rótulo ``classe`` vindo dos
    critérios pré-registrados (execução reduzida nunca é reprodução).
    """
    base = ok_rows(results)
    base = base[base["llm"] == BASELINE_LLM] if not base.empty else base
    if base.empty:
        return pd.DataFrame(columns=list(PUBLISHED_COLUMNS))
    frame = compare_with_paper(
        base, tables=(3, 4, 5), unspecified_llm=BASELINE_LLM, unspecified_prompt=BASELINE_LLM
    )
    frame = frame[frame["n"] > 0]
    published = frame[[c for c in PUBLISHED_COLUMNS if c in frame.columns]]
    return published.reset_index(drop=True)


def side_by_side_equations(results: pd.DataFrame) -> pd.DataFrame:
    """Equação sem LLM e com LLM da mesma repetição (mesmos dados e semente de busca)."""
    ok = ok_rows(results)
    if ok.empty:
        return pd.DataFrame()
    base = ok[ok["llm"] == BASELINE_LLM]
    llm = ok[ok["llm"] != BASELINE_LLM]
    merged = llm.merge(base, on=list(PAIR_KEYS), suffixes=("_llm", "_base"))
    out = pd.DataFrame(
        {
            **{k: merged[k] for k in PAIR_KEYS},
            "llm": merged["llm_llm"],
            "prompt": merged["prompt_llm"],
            "ground_truth": merged["ground_truth_llm"],
            "expression_sem_llm": merged["expression_base"],
            "expression_com_llm": merged["expression_llm"],
            "mudou": merged["expression_base"] != merged["expression_llm"],
            "tree_distance_sem_llm": merged["tree_distance_base"],
            "tree_distance_com_llm": merged["tree_distance_llm"],
        }
    )
    return out.sort_values(["sr", "repetition"]).reset_index(drop=True)


def paired_repetitions(
    results: pd.DataFrame,
    metrics: tuple[str, ...] = ("mae", "mse", "one_minus_r2", "tree_distance"),
) -> pd.DataFrame:
    """Métricas sem e com LLM por repetição (mesmos dados e semente) e a diferença.

    ``<metrica>_diff = sem_llm − com_llm`` (positiva = o LLM reduziu o erro/distância).
    """
    ok = ok_rows(results)
    if ok.empty:
        return pd.DataFrame()
    base = ok[ok["llm"] == BASELINE_LLM]
    llm = ok[ok["llm"] != BASELINE_LLM]
    merged = llm.merge(base, on=list(PAIR_KEYS), suffixes=("_llm", "_base"))
    out = pd.DataFrame({**{k: merged[k] for k in PAIR_KEYS}})
    out["llm"] = merged["llm_llm"]
    out["prompt"] = merged["prompt_llm"]
    for metric in metrics:
        out[f"{metric}_sem_llm"] = merged[f"{metric}_base"]
        out[f"{metric}_com_llm"] = merged[f"{metric}_llm"]
        out[f"{metric}_diff"] = merged[f"{metric}_base"] - merged[f"{metric}_llm"]
    return out.sort_values(["sr", "repetition"]).reset_index(drop=True)


def evaluations_of_discovered(
    results: pd.DataFrame,
    records: pd.DataFrame,
    checkpoints: Mapping[str, str],
    *,
    significant_digits: int,
    key_mode: str,
) -> pd.DataFrame:
    """Avaliação do LLM (lida do cache do run) para a equação final de cada célula ok.

    ``checkpoints``: nome do LLM -> checkpoint gravado no cache. Equações do baseline não
    passam pelo LLM durante a busca: saem com ``found = False`` (nada é inventado). Escores
    ``NaN`` também quando o cache não tem o registro.
    """
    ok = ok_rows(results)
    rows: list[pd.DataFrame] = []
    for row in ok.itertuples(index=False):
        llm = str(row.llm)
        prompt = str(row.prompt)
        looked = llm_inspection.lookup_evaluations(
            records,
            [str(row.expression)],
            scenario=str(row.scenario),
            prompt=prompt,
            model_id=checkpoints.get(llm),
            significant_digits=significant_digits,
            key_mode=key_mode,
        )
        looked.insert(0, "sr", row.sr)
        looked.insert(1, "repetition", row.repetition)
        looked.insert(2, "llm", llm)
        rows.append(looked)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[out["llm"] != BASELINE_LLM].reset_index(drop=True)
    return out.drop(columns=["raw_response"])


def cost_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Custo do LLM por algoritmo: chamadas, acertos de cache, tempos e falhas.

    ``sr_seconds`` = tempo de treino sem o tempo de LLM; ``cells_error`` conta células
    com erro (inclusive falha de avaliação do LLM).
    """
    if results.empty:
        return pd.DataFrame()
    llm = results[results["llm"] != BASELINE_LLM]
    if llm.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for sr, group in llm.groupby("sr"):
        done = group[group["status"] == STATUS_OK]
        row: dict[str, Any] = {
            "sr": sr,
            "cells_ok": len(done),
            "cells_error": len(group) - len(done),
        }
        for col in COST_COLUMNS:
            row[col] = float(done[col].astype(float).sum()) if col in done.columns else np.nan
        row["sr_seconds"] = row["train_seconds"] - row["llm_seconds"]
        calls = row["llm_calls"] + row["llm_cache_hits"]
        row["cache_hit_rate"] = row["llm_cache_hits"] / calls if calls else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
