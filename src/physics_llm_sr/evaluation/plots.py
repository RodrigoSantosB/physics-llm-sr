"""Gráficos a partir dos DataFrames de resultados persistidos (backend Agg, sem display).

Cada função salva a figura em ``path`` e devolve o ``Path``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from physics_llm_sr.evaluation.statistics import summarize_groups  # noqa: E402

FIG_DPI = 120
PROMPTS = tuple("ABCDEFGH")


def _save(fig: Figure, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def _err(s: pd.DataFrame) -> np.ndarray:
    """Meia largura do IC95 (0 onde indefinido)."""
    half: np.ndarray = np.nan_to_num((s["ci_high"] - s["mean"]).to_numpy(dtype=float))
    return half


def plot_loss_per_generation(
    history: pd.DataFrame, path: Path | str, group_col: str = "config"
) -> Path:
    """Perda composta por geração (média entre repetições por grupo).

    Colunas: ``generation, best_loss`` e ``group_col`` (ex.: ``config``).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, sub in history.groupby(group_col):
        g = sub.groupby("generation")["best_loss"].mean()
        ax.plot(g.index, g.to_numpy(), label=str(key))
    ax.set_xlabel("geração")
    ax.set_ylabel("melhor perda composta")
    ax.legend(fontsize="small")
    return _save(fig, path)


def plot_metric_bars(
    results: pd.DataFrame, metric: str, path: Path | str, config_cols: Sequence[str] = ("sr", "llm")
) -> Path:
    """Barras da métrica por configuração, com IC95."""
    s = summarize_groups(results, list(config_cols), [metric])
    labels = ["/".join(str(r[c]) for c in config_cols) for _, r in s.iterrows()]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(s)), 4))
    ax.bar(range(len(s)), s["mean"], yerr=_err(s), capsize=3)
    ax.set_xticks(range(len(s)), labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    return _save(fig, path)


def plot_prompt_heatmap(
    results: pd.DataFrame, metric: str, path: Path | str, column: str = "llm"
) -> Path:
    """Heatmap da média da métrica: prompts A-H (linhas) x ``column`` (colunas)."""
    sub = results[results["prompt"].isin(PROMPTS)]
    pivot = sub.pivot_table(index="prompt", columns=column, values=metric, aggfunc="mean")
    pivot = pivot.reindex([p for p in PROMPTS if p in pivot.index])
    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * max(1, pivot.shape[1]), 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [str(i) for i in pivot.index])
    fig.colorbar(im, ax=ax, label=metric)
    return _save(fig, path)


def plot_noise_curves(
    results: pd.DataFrame, metric: str, path: Path | str, line_col: str = "noise_location"
) -> Path:
    """Métrica vs nível de ruído (média e IC95), uma linha por ``line_col``."""
    s = summarize_groups(results, [line_col, "noise_level"], [metric])
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, sub in s.groupby(line_col):
        sub = sub.sort_values("noise_level")
        ax.errorbar(
            sub["noise_level"], sub["mean"], yerr=_err(sub), marker="o", capsize=3, label=str(key)
        )
    ax.set_xlabel("nível de ruído (fração do desvio-padrão)")
    ax.set_ylabel(metric)
    ax.legend()
    return _save(fig, path)
