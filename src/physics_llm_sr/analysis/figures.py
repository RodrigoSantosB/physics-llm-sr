# ruff: noqa: E501
"""Figuras F1-F8: tabelas publicadas lado a lado com os resultados locais (matplotlib, Agg).

Convenção visual: publicado = marcador/barra vazada ou linha tracejada; local = cheio/sólido
com barra de IC95; cor fixa por LLM em todas as figuras (exceto F2, onde a cor é fixa por RS,
conforme o estilo de referência). Painéis sem dados mostram um aviso em vez de falhar. Todo
arquivo (PNG 200 dpi + PDF) leva o rótulo de execução no rodapé.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from physics_llm_sr.analysis.artifacts import Artifact  # noqa: E402
from physics_llm_sr.analysis.loading import BASELINE_LLM, AnalysisData  # noqa: E402
from physics_llm_sr.analysis.tables import (  # noqa: E402
    NOISE_TABLE_OF,
    TableSpec,
    noise_long,
)
from physics_llm_sr.experiments.reporting import PAIR_KEYS  # noqa: E402

DPI = 200
LLM_COLORS = {"mistral": "#0072B2", "llama2": "#E69F00", "falcon": "#009E73", "none": "#555555"}
EXTRA_COLORS = ("#CC79A7", "#D55E00", "#56B4E9", "#F0E442")
SR_COLORS = {"deap": "C0", "pysr": "C1", "gplearn": "C2"}
ARM_COLORS = ("#332288", "#CC6677", "#117733", "#DDCC77")
TERM_COLORS = {"w1e": "#4477AA", "w2s": "#EE6677", "w3c": "#228833"}
SCEN_SHORT = {"free_fall": "ball_drop", "harmonic_motion": "shm", "damped_wave": "wave"}
LOC_SHORT = {"inputs": "features", "target": "target", "both": "both"}
SRS = ("deap", "pysr", "gplearn")
LLMS = ("mistral", "llama2", "falcon")
PROMPTS = tuple("ABCDEFGH")
GT_PROMPTS = ("D", "F", "G", "H")
HEATMAP_METRICS = ("mae", "mse", "one_minus_r2", "tree_distance")
EFFECT_METRICS = ("mae", "one_minus_r2", "tree_distance")
BAR_METRICS = ("mae", "tree_distance")
METRIC_NAME = {
    "mae": "MAE", "mse": "MSE", "one_minus_r2": "1−R²", "tree_distance": "distância",
}  # fmt: skip
NOT_RUN = "não executado"
NOISE_TITLE = {
    "zero_mean": "Tabela 3: distância vs nível de ruído (média zero)",
    "biased": "Tabela 4: distância vs nível de ruído (enviesado)",
}
NOISE_KIND = {"zero_mean": "de média zero", "biased": "enviesado"}


def llm_color(llm: str) -> str:
    """Cor fixa por LLM (as não mapeadas recebem cores extras, estáveis por nome)."""
    if llm in LLM_COLORS:
        return LLM_COLORS[llm]
    return EXTRA_COLORS[sum(map(ord, llm)) % len(EXTRA_COLORS)]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def _warn(ax: Axes, msg: str = "sem dados locais (não executado)") -> None:
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=8,
            color="#8a1c1c", bbox={"fc": "white", "ec": "#8a1c1c", "alpha": 0.9})  # fmt: skip


def _save(
    fig: Figure, out_dir: Path, stem: str, fig_id: str, ref: str, label: str, desc: str,
    caption: str | None = None,
) -> list[Artifact]:  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, 0.004, label, ha="center", va="bottom", fontsize=7, color="#555555")
    if caption:
        fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8)
    arts = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI)
        arts.append(Artifact(fig_id, path, f"figura/{ext}", ref, label, desc))
    plt.close(fig)
    return arts


def _hatch_nan(ax: Axes, mask: np.ndarray) -> None:
    for i, j in zip(*np.where(mask), strict=True):
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor="#eeeeee",
                edgecolor="#999999",
                hatch="///",
                lw=0,
            )  # fmt: skip
        )
        ax.text(j, i, "n/e", ha="center", va="center", fontsize=5, color="#666666")


# ----------------------------------------------------------------------------- F1
def _matrix(df: pd.DataFrame, col: str, cols: list[tuple[str, str]]) -> np.ndarray:
    idx = df.set_index(["prompt", "llm", "sr"])[col]
    out = np.full((len(PROMPTS), len(cols)), np.nan)
    for i, p in enumerate(PROMPTS):
        for j, (llm, sr) in enumerate(cols):
            if (p, llm, sr) in idx.index:
                out[i, j] = float(idx.loc[(p, llm, sr)])
    return out


def _heat(ax: Axes, mat: np.ndarray, title: str, cols: list[tuple[str, str]], cmap: str,
          vmin: float | None = None, vmax: float | None = None) -> None:  # fmt: skip
    mask = np.isnan(mat)
    im = ax.imshow(np.ma.masked_array(mat, mask), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    _hatch_nan(ax, mask)
    ax.set_xticks(range(len(cols)), [f"{m[:4]}\n{s}" for m, s in cols], fontsize=6)
    ax.set_yticks(range(len(PROMPTS)), [p + ("*" if p in GT_PROMPTS else "") for p in PROMPTS],
                  fontsize=7)  # fmt: skip
    for tick in ax.get_yticklabels():
        if tick.get_text().rstrip("*") in GT_PROMPTS:
            tick.set_color("#B22222")
            tick.set_fontweight("bold")
    ax.set_title(title, fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    if mask.all():
        _warn(ax)


def figure1(data: AnalysisData, t2: pd.DataFrame, out: Path) -> list[Artifact]:
    """F1: heatmaps da Tabela 2, publicado | local | diferença, por métrica."""
    arts: list[Artifact] = []
    cols = [(m, s) for m in LLMS for s in SRS]
    for metric in HEATMAP_METRICS:
        sub = t2[t2["metric"] == metric]
        fig, axes = plt.subplots(len(data.runs), 3, figsize=(13, 3.2 * len(data.runs) + 0.8),
                                 squeeze=False)  # fmt: skip
        for r, run in enumerate(data.runs):
            s = sub[sub["arm"] == run.arm]
            pub, loc, diff = (_matrix(s, c, cols) for c in ("published", "local_mean", "abs_diff"))
            vmax = np.nanmax(np.abs(diff)) if (~np.isnan(diff)).any() else 1.0
            both = np.concatenate([pub.ravel(), loc.ravel()])
            lo, hi = (np.nanmin(both), np.nanmax(both)) if (~np.isnan(both)).any() else (0, 1)
            _heat(axes[r, 0], pub, "publicado", cols, "viridis", lo, hi)
            _heat(axes[r, 1], loc, f"local — {run.arm_short}", cols, "viridis", lo, hi)
            _heat(axes[r, 2], diff, "diferença (local − publicado)", cols, "PuOr", -vmax, vmax)
        fig.suptitle(
            f"F1 — Tabela 2 (sensibilidade a prompts), {METRIC_NAME[metric]}; "
            "* = prompts que revelam a GT (D, F, G, H); hachura = não executado",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        arts += _save(fig, out, f"F1_heatmap_t2_{metric}", f"F1_{metric}", "Tabela 2 (artigo, p.9-10)",
                      data.combined_label(), f"Heatmap Tabela 2, {metric}")  # fmt: skip
    return arts


# ----------------------------------------------------------------------------- F2
def figure2(
    data: AnalysisData, noise_type: str, arm: str, out: Path, caption: str | None = None
) -> list[Artifact]:
    """F2: distância vs nível de ruído (grade cenário x localização; 3 séries por RS)."""
    long = noise_long(data, arm, noise_type)
    scen_list = list(SCEN_SHORT)
    locs = list(LOC_SHORT)
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.2), sharey=True, squeeze=False)
    styles = {
        "com_llm": ("-", "o", 1.0, "{} + LLM"),
        "sem_llm": (":", "x", 1.0, "{} sem LLM"),
        "artigo": ("--", "", 0.5, "{} (artigo)"),
    }
    for i, scen in enumerate(scen_list):
        for j, loc in enumerate(locs):
            ax = axes[i, j]
            any_local = False
            for sr in SRS:
                for src, (ls, mk, alpha, lab) in styles.items():
                    s = long[(long["scenario"] == scen) & (long["noise_location"] == loc)]
                    s = s[(s["sr"] == sr) & (s["source"] == src)].sort_values("noise_level")
                    if s.empty:
                        continue
                    any_local = any_local or src != "artigo"
                    x, y = s["noise_level"].to_numpy(), s["mean"].to_numpy()
                    ax.plot(x, y, ls=ls, marker=mk or None, ms=4, alpha=alpha,
                            color=SR_COLORS[sr], label=lab.format(sr), lw=1.2)  # fmt: skip
                    if src != "artigo":
                        err = (s["ci_high"] - s["ci_low"]).to_numpy() / 2
                        if np.isfinite(err).any():
                            ax.errorbar(x, y, yerr=np.nan_to_num(err), fmt="none",
                                        ecolor=SR_COLORS[sr], elinewidth=0.8, capsize=2)  # fmt: skip
            ax.set_title(f"{SCEN_SHORT[scen]} | ruído em {LOC_SHORT[loc]}", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([0.01, 0.02, 0.03, 0.04, 0.05])
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if not any_local:
                ax.text(0.03, 0.04, "local: não executado", transform=ax.transAxes, fontsize=6.5,
                        color="#8a1c1c")  # fmt: skip
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, labels = h, lab
    by = dict(zip(labels, handles, strict=True))
    order = [lab.format(sr) for sr in SRS for (_, _, _, lab) in styles.values()]
    by_present = [k for k in order if k in by]
    if by_present:
        axes[0, 2].legend([by[k] for k in by_present], by_present, fontsize=5.5, ncol=3,
                          loc="center", bbox_to_anchor=(0.5, 0.78))  # fmt: skip
    fig.suptitle(NOISE_TITLE[noise_type], fontsize=11)
    cap = caption or (
        f"Distância entre árvores em função do nível de ruído {NOISE_KIND[noise_type]}. "
        "Mesma organização da Fig. 2."
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    run = data.run_of(arm)
    n = int(data.ok[data.ok["arm"] == arm]["repetition"].nunique())
    ref = f"Tabela {NOISE_TABLE_OF[noise_type]} (artigo, p.10)"
    return _save(fig, out, f"F2_noise_{noise_type}_{_slug(run.arm_short)}",
                 f"F2_{noise_type}", ref, f"{run.label}, n={n}", f"Ruído {noise_type}", cap)  # fmt: skip


# ----------------------------------------------------------------------------- F3
def figure3(
    data: AnalysisData, t5: pd.DataFrame, arm: str, metric: str, out: Path
) -> list[Artifact]:
    """F3: barras agrupadas da Tabela 5 (baseline + 3 LLM), publicado vs local."""
    run = data.run_of(arm)
    sub = t5[(t5["arm"] == arm) & (t5["metric"] == metric)]
    groups = [BASELINE_LLM, *LLMS]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), squeeze=False)
    w = 0.38
    for i, scen in enumerate(SCEN_SHORT):
        for j, sr in enumerate(SRS):
            ax = axes[i, j]
            s = sub[(sub["scenario"] == scen) & (sub["sr"] == sr)]
            any_local = False
            for k, g in enumerate(groups):
                col = llm_color(g)
                if g == BASELINE_LLM:
                    row = s.iloc[0] if len(s) else None
                    pub = None if row is None else row["baseline_published"]
                    loc, lo, hi = (None if row is None else row["baseline_local_mean"]), None, None
                else:
                    r = s[s["llm"] == g]
                    row = r.iloc[0] if len(r) else None
                    pub = None if row is None else row["published"]
                    loc = None if row is None else row["local_mean"]
                    lo = None if row is None else row["local_ci_low"]
                    hi = None if row is None else row["local_ci_high"]
                if pub is not None and not pd.isna(pub):
                    ax.bar(k - w / 2, pub, w, fill=False, edgecolor=col, hatch="//", lw=1.2)
                if loc is not None and not pd.isna(loc):
                    any_local = True
                    ax.bar(k + w / 2, loc, w, color=col)
                    if lo is not None and not pd.isna(lo) and not pd.isna(hi):
                        ax.errorbar(
                            k + w / 2, loc, yerr=[[loc - lo], [hi - loc]], color="k", capsize=2
                        )
                else:
                    ax.text(
                        k + w / 2, 0, "n/e", ha="center", va="bottom", fontsize=6, color="#8a1c1c"
                    )
            ax.set_xticks(range(len(groups)), ["sem LLM", *LLMS], fontsize=7)
            ax.set_title(f"{SCEN_SHORT[scen]} | {sr}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", alpha=0.3)
            if not any_local:
                _warn(ax)
    fig.suptitle(
        f"F3 — Tabela 5 (suplemento), {METRIC_NAME[metric]}: hachurado vazado = publicado, "
        f"cheio = local ({run.arm_short}), barra = IC95",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return _save(fig, out, f"F3_bars_t5_{metric}_{_slug(run.arm_short)}", f"F3_{metric}",
                 "Tabela 5 (suplemento)", run.label, f"Barras Tabela 5, {metric}")  # fmt: skip


# ----------------------------------------------------------------------------- F4
def figure4(data: AnalysisData, t6: pd.DataFrame, arm: str, out: Path) -> list[Artifact]:
    """F4: tempo de SR, tempo de LLM e chamadas (Tabela 6), publicado vs local."""
    run = data.run_of(arm)
    sub = t6[t6["arm"] == arm]
    panels = [
        ("tempo SR (s)", "published_total_sr_s", "local_total_sr_s"),
        ("tempo LLM (s)", "published_total_llm_s", "local_total_llm_s"),
        ("chamadas LLM (pub.: derivado de total/média, impreciso)", "published_calls_implied",
         "local_calls_mean"),
    ]  # fmt: skip
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), squeeze=False)
    combos = [(sc, sr) for sc in SCEN_SHORT for sr in SRS]
    w = 0.12
    for ax, (title, pcol, lcol) in zip(axes[:, 0], panels, strict=True):
        any_local = False
        for gi, (sc, sr) in enumerate(combos):
            for li, llm in enumerate(LLMS):
                r = sub[(sub["scenario"] == sc) & (sub["sr"] == sr) & (sub["llm"] == llm)]
                x = gi + (li - 1) * 2.2 * w
                col = llm_color(llm)
                if r.empty:
                    continue
                p, loc = r.iloc[0][pcol], r.iloc[0][lcol]
                if not pd.isna(p):
                    ax.bar(x - w / 2, p, w, fill=False, edgecolor=col, hatch="//", lw=1.1)
                if not pd.isna(loc):
                    any_local = True
                    ax.bar(x + w / 2, loc, w, color=col)
                else:
                    ax.text(x + w / 2, 0.02, "n/e", transform=ax.get_xaxis_transform(),
                            ha="center", fontsize=5, color="#8a1c1c")  # fmt: skip
        ax.set_xticks(range(len(combos)), [f"{SCEN_SHORT[a]}\n{b}" for a, b in combos], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(axis="y", alpha=0.3)
        if not any_local:
            _warn(ax)
    handles = [Rectangle((0, 0), 1, 1, color=llm_color(m)) for m in LLMS]
    handles += [Rectangle((0, 0), 1, 1, fill=False, edgecolor="k", hatch="//")]
    axes[0, 0].legend(handles, [*LLMS, "publicado (vazado)"], fontsize=7, ncol=4)
    fig.suptitle(f"F4 — Tabela 6 (suplemento): publicado vs local ({run.arm_short})", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return _save(fig, out, f"F4_runtime_t6_{_slug(run.arm_short)}", "F4_runtime",
                 "Tabela 6 (suplemento)", run.label, "Tempo e chamadas")  # fmt: skip


# ----------------------------------------------------------------------------- F5
def figure5(data: AnalysisData, metric: str, out: Path) -> list[Artifact]:
    """F5: pontos baseline e LLM ligados por repetição, por braço x SR."""
    ok = data.ok
    fig, axes = plt.subplots(len(data.runs), 3, figsize=(11, 3.4 * len(data.runs) + 0.8),
                             squeeze=False)  # fmt: skip
    for r, run in enumerate(data.runs):
        arm = ok[ok["arm"] == run.arm]
        for c, sr in enumerate(SRS):
            ax = axes[r, c]
            s = arm[arm["sr"] == sr]
            base, llm = s[s["llm"] == BASELINE_LLM], s[s["llm"] != BASELINE_LLM]
            ax.set_title(f"{run.arm_short} | {sr}", fontsize=8)
            if base.empty or llm.empty:
                _warn(ax)
                continue
            models = sorted(set(llm["llm"]))
            for k, m in enumerate(models, start=1):
                merged = llm[llm["llm"] == m].merge(base, on=list(PAIR_KEYS), suffixes=("_l", "_b"))
                for b, v in zip(merged[f"{metric}_b"], merged[f"{metric}_l"], strict=True):
                    ax.plot([0, k], [b, v], color="#999999", lw=0.8, zorder=1)
                ax.scatter(
                    np.zeros(len(merged)),
                    merged[f"{metric}_b"],
                    color=LLM_COLORS["none"],
                    zorder=2,
                    s=18,
                )
                ax.scatter(
                    np.full(len(merged), k),
                    merged[f"{metric}_l"],
                    color=llm_color(m),
                    zorder=2,
                    s=18,
                )
            ax.set_xticks(range(len(models) + 1), ["sem LLM", *models], fontsize=7)
            vals = pd.concat([base[metric], llm[metric]]).astype(float)
            if (vals > 0).all() and vals.max() / vals.min() > 50:
                ax.set_yscale("log")
            ax.set_ylabel(METRIC_NAME[metric], fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle(f"F5 — efeito pareado do LLM ({METRIC_NAME[metric]}): linhas ligam a mesma repetição",
                 fontsize=9)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, out, f"F5_paired_{metric}", f"F5_{metric}", "sem equivalente publicado",
                 data.combined_label(), f"Efeito pareado, {metric}")  # fmt: skip


# ----------------------------------------------------------------------------- F6
def figure6(data: AnalysisData, effect: pd.DataFrame, out: Path) -> list[Artifact]:
    """F6: efeito do LLM (diferença média ± IC95) por braço, por SR."""
    fig, axes = plt.subplots(1, len(EFFECT_METRICS), figsize=(14, 4.4), squeeze=False)
    arms = list(data.runs)
    for ax, metric in zip(axes[0], EFFECT_METRICS, strict=True):
        s = effect[effect["metric"] == metric]
        keys = sorted({(a, b) for a, b in zip(s["sr"], s["llm"], strict=True)}) if len(s) else []
        for ai, run in enumerate(arms):
            col = ARM_COLORS[ai % len(ARM_COLORS)]
            for ki, (sr, llm) in enumerate(keys):
                r = s[(s["arm"] == run.arm) & (s["sr"] == sr) & (s["llm"] == llm)]
                if r.empty:
                    continue
                x = ki + (ai - (len(arms) - 1) / 2) * 0.25
                y = float(r["mean_diff"].mean())
                lo, hi = float(r["ci_low"].mean()), float(r["ci_high"].mean())
                err = None if math.isnan(lo) else [[y - lo], [hi - y]]
                ax.errorbar(x, y, yerr=err, fmt="o", color=col, capsize=3,
                            label=run.arm_short if ki == 0 else None)  # fmt: skip
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(keys)), [f"{a}\n{b[:4]}" for a, b in keys], fontsize=7)
        ax.set_title(f"{METRIC_NAME[metric]} (baseline − LLM; > 0 = LLM melhor)", fontsize=8)
        ax.grid(alpha=0.3)
        if not keys:
            _warn(ax)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("F6 — efeito do LLM por braço (média ± IC95; sem IC quando n < 2)", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, out, "F6_arms_effect", "F6_arms", "sem equivalente publicado",
                 data.combined_label(), "Efeito do LLM por braço")  # fmt: skip


# ----------------------------------------------------------------------------- F7
def figure7(data: AnalysisData, loss: pd.DataFrame, out: Path) -> list[Artifact]:
    """F7: barras empilhadas w1·e, w2·s, w3·c por braço x SR (evidencia GAP-02)."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if loss.empty:
        _warn(ax)
    else:
        labels = [f"{a}\n{s}" for a, s in zip(loss["arm_short"], loss["sr"], strict=True)]
        bottom = np.zeros(len(loss))
        for term, name in (("w1e", "w1·e"), ("w2s", "w2·s"), ("w3c", "w3·c")):
            vals = loss[f"frac_{term}"].to_numpy(dtype=float)
            ax.bar(range(len(loss)), vals, bottom=bottom, color=TERM_COLORS[term], label=name)
            bottom += np.nan_to_num(vals)
        ax.set_xticks(range(len(loss)), labels, fontsize=7)
        ax.set_ylabel("fração da perda final (células com LLM)")
        ax.legend(fontsize=8)
    ax.set_title(
        "F7 — decomposição da perda: se w3·c é pequeno, o LLM pesa pouco (GAP-02)", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out, "F7_loss_decomposition", "F7_loss", "sem equivalente publicado (GAP-02)",
                 data.combined_label(), "Decomposição da perda")  # fmt: skip


# ----------------------------------------------------------------------------- F8
def figure8(data: AnalysisData, out: Path) -> list[Artifact]:
    """F8: best_loss por geração, baseline vs LLM, por SR e braço."""
    h = data.history
    fig, axes = plt.subplots(len(data.runs), 3, figsize=(12, 3.5 * len(data.runs) + 0.8),
                             squeeze=False)  # fmt: skip
    for r, run in enumerate(data.runs):
        for c, sr in enumerate(SRS):
            ax = axes[r, c]
            ax.set_title(f"{run.arm_short} | {sr}", fontsize=8)
            s = h[(h["arm"] == run.arm) & (h["sr"] == sr)] if not h.empty else h
            if s.empty:
                _warn(ax, "sem history.parquet para este painel")
                continue
            for llm, g in s.groupby("llm"):
                col = llm_color(str(llm))
                label = "sem LLM" if llm == BASELINE_LLM else str(llm)
                for _, gg in g.groupby(["cell_key"]):
                    ax.plot(gg["generation"], gg["best_loss"], color=col, alpha=0.25, lw=0.8)
                mean = g.groupby("generation")["best_loss"].mean()
                ax.plot(mean.index, mean.to_numpy(), color=col, lw=2, label=label)
            vals = s["best_loss"].astype(float)
            if (vals > 0).all() and vals.max() / vals.min() > 20:
                ax.set_yscale("log")
            ax.set_xlabel("geração", fontsize=7)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
    fig.suptitle(
        "F8 — convergência: best_loss por geração (fino = repetição, grosso = média). "
        "Atenção: o baseline usa w3 = 0, então L não é comparável em valor absoluto.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, out, "F8_convergence", "F8_convergence", "sem equivalente publicado",
                 data.combined_label(), "Convergência")  # fmt: skip


# ----------------------------------------------------------------------------- montagem
def build_figures(
    data: AnalysisData,
    built: dict[str, TableSpec],
    out_dir: Path,
    captions: dict[str, str] | None = None,
) -> list[Artifact]:
    """Gera F1-F8 em ``out_dir`` (PNG 200 dpi + PDF) e devolve os artefatos."""
    caps = captions or {}
    arts: list[Any] = []
    arts += figure1(data, built["t2_prompt_sensitivity"].df, out_dir)
    for noise_type in NOISE_TABLE_OF:
        for run in data.runs:
            arts += figure2(data, noise_type, run.arm, out_dir, caps.get(noise_type))
    for run in data.runs:
        for metric in BAR_METRICS:
            arts += figure3(data, built["t5_benchmark"].df, run.arm, metric, out_dir)
        arts += figure4(data, built["t6_runtime"].df, run.arm, out_dir)
    for metric in EFFECT_METRICS:
        arts += figure5(data, metric, out_dir)
    arts += figure6(data, built["effect_paired"].df, out_dir)
    arts += figure7(data, built["loss_decomposition"].df, out_dir)
    arts += figure8(data, out_dir)
    return list(arts)
