# ruff: noqa: E501
# mypy: disable-error-code="no-any-return,unused-ignore,no-untyped-call"
"""Tabelas publicado x local (CSV + Markdown + LaTeX booktabs).

Todo número vem dos CSVs publicados (``references/paper_results``) ou dos arquivos
persistidos das execuções. Célula sem dado local fica ``NaN`` com ``status_local =
"não executado"`` (nunca zero). A regra de classificação é a de
``evaluation.paper_comparison`` (não reimplementada aqui).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.analysis import latex
from physics_llm_sr.analysis.artifacts import Artifact
from physics_llm_sr.analysis.loading import BASELINE_LLM, AnalysisData, RunInfo
from physics_llm_sr.evaluation.paper_comparison import (
    ComparisonError,
    canonical_llm,
    compare_with_paper,
    load_paper_results,
    resolve_references_dir,
)
from physics_llm_sr.evaluation.statistics import (
    StatisticsInputError,
    paired_comparison,
    summarize,
)
from physics_llm_sr.experiments.reporting import (
    PAIR_KEYS,
    PRIMARY_METRICS,
    loss_term_contributions,
)

EXECUTED = "executado"
NOT_EXECUTED = "não executado"
NAN_TEXT = "n/e"
FOOTNOTE = (
    f"`{NAN_TEXT}` = não executado ou indefinido (NaN); nunca zero. Publicado = valores do "
    "artigo/suplemento; local = média das repetições da execução rotulada."
)
PAPER_REF = {
    2: "Tabela 2 (artigo, p.9-10)",
    3: "Tabela 3 (artigo, p.10)",
    4: "Tabela 4 (artigo, p.10)",
    5: "Tabela 5 (suplemento)",
    6: "Tabela 6 (suplemento)",
}
RUNTIME_TABLE = 6
TABLE6_FILE = "table6_runtime_supplement.csv"
PAPER_RESULTS_DIRNAME = "paper_results"
EFFECT_KEYS = [
    "scenario",
    "sr",
    "llm",
    "prompt",
    "noise_type",
    "noise_location",
    "noise_level",
]
COMPARE_COLUMNS = [
    "table",
    "scenario",
    "sr",
    "llm",
    "prompt",
    "noise_type",
    "noise_location",
    "noise_level",
    "metric",
]
LOCAL_RENAME = {
    "mean": "local_mean",
    "sd": "local_sd",
    "ci_low": "local_ci_low",
    "ci_high": "local_ci_high",
}
COMPARE_VALUE_COLUMNS = [
    "published",
    "n",
    "local_mean",
    "local_sd",
    "local_ci_low",
    "local_ci_high",
    "abs_diff",
    "rel_diff",
    "classe",
    "reason",
    "notes",
]
DIFF_CONVENTION = "abs_diff = local − publicado; rel_diff = abs_diff / publicado"
EFFECT_CONVENTION = "diferença = baseline − LLM (positiva = LLM reduz o erro)"


@dataclass(frozen=True)
class TableSpec:
    """Tabela pronta para gravação."""

    name: str
    df: pd.DataFrame
    title: str
    paper_ref: str
    label: str
    display: tuple[str, ...] | None = None
    note: str = ""
    tex: str | None = None


# ----------------------------------------------------------------------------- comparação
def _wildcards(run: RunInfo, ok: pd.DataFrame) -> dict[str, str | None]:
    ex = run.experiments
    llms = sorted(set(ok["llm"]) - {BASELINE_LLM})
    prompts = sorted(set(ok["prompt"]) - {BASELINE_LLM})
    return {
        "noise_model": ex.get("noise_model") or (llms[0] if len(llms) == 1 else None),
        "noise_prompt": ex.get("noise_prompt") or (prompts[0] if len(prompts) == 1 else None),
        "benchmark_prompt": ex.get("benchmark_prompt")
        or (prompts[0] if len(prompts) == 1 else None),
    }


def _unexecuted(paper: pd.DataFrame, tbls: tuple[int, ...], reason: str) -> pd.DataFrame:
    sub = paper[paper["table"].isin(list(tbls))]
    frame = sub[[*COMPARE_COLUMNS, "notes"]].copy()
    frame["published"] = sub["value"].to_numpy()
    frame["n"] = 0
    frame["classe"] = "inconclusiva"
    frame["reason"] = reason
    return frame


def comparison_all(data: AnalysisData, paper: pd.DataFrame | None = None) -> pd.DataFrame:
    """``compare_with_paper`` por braço (Tabelas 2-5), com coringas de LLM/prompt do GAP-12."""
    if paper is None and "comparison" in data.cache:
        cached: pd.DataFrame = data.cache["comparison"]
        return cached
    paper_df = load_paper_results() if paper is None else paper
    frames: list[pd.DataFrame] = []
    for run in data.runs:
        ok = data.ok
        ok = ok[ok["arm"] == run.arm]
        w = _wildcards(run, ok)
        calls: list[tuple[tuple[int, ...], dict[str, Any]]] = [
            ((2,), {}),
            (
                (3, 4),
                {"unspecified_llm": w["noise_model"], "unspecified_prompt": w["noise_prompt"]},
            ),
            ((5,), {"unspecified_prompt": w["benchmark_prompt"]}),
        ]
        for tbls, kw in calls:
            try:
                frame = compare_with_paper(ok, paper=paper_df, tables=tbls, **kw)
            except ComparisonError as exc:
                warnings.warn(f"{run.arm}, tabelas {tbls}: {exc}", stacklevel=2)
                frame = _unexecuted(paper_df, tbls, f"comparação não executada: {exc}")
            frame = frame.rename(columns=LOCAL_RENAME)
            frame.insert(0, "arm", run.arm)
            frame.insert(1, "rotulo", run.label)
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    for col in COMPARE_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["n"] = out["n"].fillna(0).astype(int)
    out["status_local"] = np.where(out["n"] > 0, EXECUTED, NOT_EXECUTED)
    if paper is None:
        data.cache["comparison"] = out
    return out


def _compare_table(
    comp: pd.DataFrame, table: int, drop: tuple[str, ...] = (), extra: tuple[str, ...] = ()
) -> pd.DataFrame:
    sub = comp[comp["table"] == table]
    cols = [
        "arm",
        "rotulo",
        *[c for c in COMPARE_COLUMNS if c not in drop and c != "table"],
        *COMPARE_VALUE_COLUMNS,
        "status_local",
        *extra,
    ]
    return sub[cols].reset_index(drop=True)


def table2(data: AnalysisData, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela 2: sensibilidade a prompts A-H por LLM x SR x métrica (média de 3 cenários)."""
    comp = comparison_all(data) if comp is None else comp
    out = _compare_table(comp, 2, drop=("noise_type", "noise_location", "noise_level"))
    out["gt_revelada"] = out["prompt"].isin(["D", "F", "G", "H"])
    return out


def table3(data: AnalysisData, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela 3: distância sob ruído de média zero (cenário x SR x localização x nível)."""
    comp = comparison_all(data) if comp is None else comp
    return _compare_table(comp, 3, drop=("llm", "prompt", "noise_type", "metric"))


def table4(data: AnalysisData, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela 4: distância sob ruído enviesado."""
    comp = comparison_all(data) if comp is None else comp
    return _compare_table(comp, 4, drop=("llm", "prompt", "noise_type", "metric"))


def table5(data: AnalysisData, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela 5 (suplemento): baseline + 3 LLM x 3 SR x 3 cenários."""
    comp = comparison_all(data) if comp is None else comp
    return _compare_table(
        comp,
        5,
        drop=("noise_type", "noise_location", "noise_level"),
        extra=("baseline_published", "baseline_local_mean"),
    )


# ----------------------------------------------------------------------------- Tabela 6
def load_table6(references_dir: str | Path | None = None) -> pd.DataFrame:
    """Tabela 6 publicada (tempos e razão), nomes de LLM canônicos."""
    path = resolve_references_dir(references_dir) / PAPER_RESULTS_DIRNAME / TABLE6_FILE
    df = pd.read_csv(path)
    df["llm"] = df["llm"].map(canonical_llm)
    return df


def _default_noise(ok: pd.DataFrame) -> pd.DataFrame:
    sub = ok[(ok["noise_type"] == "zero_mean") & (ok["noise_location"] == "target")]
    if sub.empty:
        return sub
    out: pd.DataFrame = sub[np.isclose(sub["noise_level"], sub["noise_level"].min())]
    return out


def table6(data: AnalysisData, paper6: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela 6: tempo de SR/LLM e nº de chamadas locais vs publicado.

    Local: ``total_sr_s = train_seconds − llm_seconds`` (hipótese: ``train_seconds`` inclui o
    tempo do LLM), ``total_llm_s = llm_seconds``; médias por célula na condição padrão
    (ruído média zero, alvo, menor nível), prompts agrupados. O artigo não publica nº de
    chamadas: ``published_calls_implied`` = total LLM / tempo médio por chamada (derivado, impreciso
    por arredondamento; INC-09 para Falcon).
    """
    pub = load_table6() if paper6 is None else paper6
    rows: list[dict[str, Any]] = []
    for run in data.runs:
        ok = _default_noise(data.ok[data.ok["arm"] == run.arm])
        for _, p in pub.iterrows():
            sub = ok[(ok["scenario"] == p["scenario"]) & (ok["sr"] == p["sr"])]
            sub = sub[sub["llm"].map(canonical_llm) == p["llm"]]
            n = len(sub)
            row: dict[str, Any] = {
                "arm": run.arm,
                "rotulo": run.label,
                "scenario": p["scenario"],
                "sr": p["sr"],
                "llm": p["llm"],
                "n": n,
                "status_local": EXECUTED if n else NOT_EXECUTED,
                "published_total_sr_s": p["total_sr_s"],
                "published_total_llm_s": p["total_llm_s"],
                "published_ratio": p["llm_sr_ratio"],
                "published_avg_llm_per_call_s": p["avg_llm_per_call_s"],
                "published_calls_implied": (
                    p["total_llm_s"] / p["avg_llm_per_call_s"]
                    if p["avg_llm_per_call_s"] > 0
                    else math.nan
                ),
                "notes": p["notes"],
            }
            llm_s = sub["llm_seconds"].astype(float)
            sr_s = sub["train_seconds"].astype(float) - llm_s
            calls = sub["llm_calls"].astype(float)
            row["local_total_sr_s"] = float(sr_s.mean()) if n else math.nan
            row["local_total_llm_s"] = float(llm_s.mean()) if n else math.nan
            row["local_ratio"] = (
                float(llm_s.mean() / sr_s.mean()) if n and sr_s.mean() > 0 else math.nan
            )
            row["local_calls_mean"] = float(calls.mean()) if n else math.nan
            row["local_avg_llm_per_call_s"] = (
                float(llm_s.sum() / calls.sum()) if n and calls.sum() > 0 else math.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- efeito do LLM
def _paired_rows(arm: str, label: str, short: str, ok: pd.DataFrame) -> list[dict[str, Any]]:
    base = ok[ok["llm"] == BASELINE_LLM]
    llm = ok[ok["llm"] != BASELINE_LLM]
    rows: list[dict[str, Any]] = []
    if base.empty or llm.empty:
        return rows
    for key, sub in llm.groupby(EFFECT_KEYS, sort=True):
        merged = sub.merge(base, on=list(PAIR_KEYS), suffixes=("_llm", "_base"))
        if merged.empty:
            continue
        changed = int((merged["expression_llm"] != merged["expression_base"]).sum())
        for metric in PRIMARY_METRICS:
            row: dict[str, Any] = {"arm": arm, "arm_short": short, "rotulo": label}
            row.update(dict(zip(EFFECT_KEYS, key, strict=True)))
            b = merged[f"{metric}_base"].to_numpy(dtype=float)
            v = merged[f"{metric}_llm"].to_numpy(dtype=float)
            row.update(metric=metric, n=len(merged), n_expr_changed=changed)
            try:
                s = summarize(b - v)
                res = paired_comparison(b, v)
            except StatisticsInputError as exc:
                row.update(note=str(exc))
            else:
                row.update(
                    mean_base=float(b.mean()),
                    mean_llm=float(v.mean()),
                    mean_diff=s.mean,
                    ci_low=np.nan if s.ci_low is None else s.ci_low,
                    ci_high=np.nan if s.ci_high is None else s.ci_high,
                    p_value=np.nan if res.p_value is None else res.p_value,
                    n_nonzero=res.n_nonzero,
                    method=res.method,
                    cohens_dz=np.nan if res.cohens_dz is None else res.cohens_dz,
                    note=res.note,
                )
            rows.append(row)
    return rows


def paired_effect(data: AnalysisData) -> pd.DataFrame:
    """Efeito pareado baseline vs LLM (mesma repetição) por braço x SR x métrica."""
    rows: list[dict[str, Any]] = []
    for run in data.runs:
        ok = data.ok
        rows.extend(_paired_rows(run.arm, run.label, run.arm_short, ok[ok["arm"] == run.arm]))
    cols = [
        "arm", "arm_short", "rotulo", *EFFECT_KEYS, "metric", "n", "mean_base", "mean_llm",
        "mean_diff", "ci_low", "ci_high", "p_value", "n_nonzero", "method", "cohens_dz",
        "n_expr_changed", "note",
    ]  # fmt: skip
    out = pd.DataFrame(rows).reindex(columns=cols)
    out["convention"] = EFFECT_CONVENTION
    return out


def arm_comparison(effect: pd.DataFrame) -> pd.DataFrame:
    """Efeito do LLM de cada braço lado a lado (ex.: literal vs normalized)."""
    keys = [*EFFECT_KEYS, "metric"]
    merged: pd.DataFrame | None = None
    for short, sub in effect.groupby("arm_short", sort=False):
        part = sub[[*keys, "n", "mean_diff", "ci_low", "ci_high", "p_value"]].copy()
        part = part.rename(
            columns={c: f"{short}__{c}" for c in ("n", "mean_diff", "ci_low", "ci_high", "p_value")}
        )
        merged = part if merged is None else merged.merge(part, on=keys, how="outer")
    if merged is None:
        return pd.DataFrame(columns=[*keys])
    merged["convention"] = EFFECT_CONVENTION
    return merged.reset_index(drop=True)


# ----------------------------------------------------------------------------- perda
def loss_decomposition(data: AnalysisData) -> pd.DataFrame:
    """Contribuição média de w1·e, w2·s, w3·c à perda final por braço x cenário x SR (GAP-02).

    Usa :func:`loss_term_contributions`; ``frac_*`` = termo médio / soma dos termos médios.
    """
    parts: list[pd.DataFrame] = []
    for run in data.runs:
        if run.weights is None:
            continue
        ok = data.ok
        ok = ok[ok["arm"] == run.arm]
        contrib = loss_term_contributions(ok, *run.weights).reset_index()
        if contrib.empty:
            continue
        counts = (
            ok[ok["llm"] != BASELINE_LLM]
            .groupby(["scenario", "sr"])
            .size()
            .rename("n")
            .reset_index()
        )
        contrib = contrib.merge(counts, on=["scenario", "sr"])
        contrib = contrib.rename(
            columns={
                "w1·e": "w1e",
                "w2·s": "w2s",
                "w3·c": "w3c",
                "fração do termo LLM em L": "frac_w3c_mean_rowwise",
            }
        )
        total = contrib["w1e"] + contrib["w2s"] + contrib["w3c"]
        for term in ("w1e", "w2s", "w3c"):
            contrib[f"frac_{term}"] = contrib[term] / total
        contrib.insert(0, "arm", run.arm)
        contrib.insert(1, "arm_short", run.arm_short)
        contrib.insert(2, "rotulo", run.label)
        contrib["weights"] = str(run.weights)
        parts.append(contrib)
    if not parts:
        return pd.DataFrame(
            columns=["arm", "arm_short", "rotulo", "scenario", "sr", "n", "w1e", "w2s", "w3c"]
        )
    return pd.concat(parts, ignore_index=True)


def classification_summary(data: AnalysisData, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Contagem de células por classe, tabela de origem e se houve execução local."""
    comp = comparison_all(data) if comp is None else comp
    g = comp.assign(executado=comp["n"] > 0)
    out = (
        g.groupby(["arm", "rotulo", "table", "classe", "executado"])
        .size()
        .rename("count")
        .reset_index()
    )
    out["origem"] = out["table"].map(PAPER_REF)
    return out


# ----------------------------------------------------------------------------- ruído / custo
NOISE_TABLE_OF = {"zero_mean": 3, "biased": 4}
NOISE_SOURCES = ("artigo", "com_llm", "sem_llm")
COST_REQUIRED = ("llm_invalid_responses", "llm_calls", "llm_seconds", "train_seconds")


def noise_long(
    data: AnalysisData, arm: str, noise_type: str, paper: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Distância vs nível de ruído em formato longo: artigo, com LLM e sem LLM (local).

    Colunas: scenario, sr, noise_location, noise_level, source, mean, ci_low, ci_high, n.
    Local "com LLM" usa o LLM/prompt do GAP-12 declarados na config (``noise_model``/
    ``noise_prompt``) quando existem.
    """
    paper_df = load_paper_results() if paper is None else paper
    pub = paper_df[paper_df["table"] == NOISE_TABLE_OF[noise_type]]
    rows: list[dict[str, Any]] = [
        {
            "scenario": r.scenario, "sr": r.sr, "noise_location": r.noise_location,
            "noise_level": float(r.noise_level), "source": "artigo", "mean": float(r.value),
            "ci_low": math.nan, "ci_high": math.nan, "n": 0,
        }
        for r in pub.itertuples(index=False)
    ]  # fmt: skip
    run = data.run_of(arm)
    ok = data.ok
    ok = ok[(ok["arm"] == arm) & (ok["noise_type"] == noise_type)]
    w = _wildcards(run, ok)
    with_llm = ok[ok["llm"] != BASELINE_LLM]
    if w["noise_model"]:
        with_llm = with_llm[with_llm["llm"] == canonical_llm(str(w["noise_model"]))]
    if w["noise_prompt"]:
        with_llm = with_llm[with_llm["prompt"] == w["noise_prompt"]]
    for source, sub in (("com_llm", with_llm), ("sem_llm", ok[ok["llm"] == BASELINE_LLM])):
        keys = ["scenario", "sr", "noise_location", "noise_level"]
        for key, g in sub.groupby(keys):
            s = summarize(g["tree_distance"].to_numpy(dtype=float))
            rows.append(
                dict(zip(keys, key, strict=True))
                | {
                    "source": source, "mean": s.mean, "n": s.n,
                    "ci_low": np.nan if s.ci_low is None else s.ci_low,
                    "ci_high": np.nan if s.ci_high is None else s.ci_high,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def cost_table(data: AnalysisData) -> pd.DataFrame:
    """Custo do LLM por braço x LLM x SR (formato da 'Tabela V' de estilo).

    ``calls_mean`` = média de chamadas novas por execução; ``failure_rate_pct`` =
    respostas inválidas / chamadas novas; ``median_llm_sr_ratio`` = mediana por execução de
    ``llm_seconds / (train_seconds − llm_seconds)``. Campo não persistido => ``NaN`` ("n/d")
    e listado em ``missing_fields``.
    """
    rows: list[dict[str, Any]] = []
    for run in data.runs:
        ok = data.ok
        sub_all = ok[(ok["arm"] == run.arm) & (ok["llm"] != BASELINE_LLM)]
        missing = [c for c in COST_REQUIRED if c not in sub_all.columns]
        for (llm, sr), g in sub_all.groupby(["llm", "sr"], sort=True):
            row: dict[str, Any] = {
                "arm": run.arm, "rotulo": run.label, "llm": llm, "sr": sr, "n_exec": len(g),
                "calls_mean": math.nan, "failure_rate_pct": math.nan,
                "median_llm_sr_ratio": math.nan, "missing_fields": ", ".join(missing),
            }  # fmt: skip
            if "llm_calls" in g.columns:
                row["calls_mean"] = float(g["llm_calls"].astype(float).mean())
                calls = float(g["llm_calls"].astype(float).sum())
                if "llm_invalid_responses" in g.columns and calls > 0:
                    row["failure_rate_pct"] = (
                        100.0 * float(g["llm_invalid_responses"].astype(float).sum()) / calls
                    )
            if "llm_seconds" in g.columns and "train_seconds" in g.columns:
                sr_s = g["train_seconds"].astype(float) - g["llm_seconds"].astype(float)
                valid = sr_s > 0
                if valid.any():
                    ratio = g["llm_seconds"].astype(float)[valid] / sr_s[valid]
                    row["median_llm_sr_ratio"] = float(ratio.median())
            rows.append(row)
    cols = ["arm", "rotulo", "llm", "sr", "n_exec", "calls_mean", "failure_rate_pct",
            "median_llm_sr_ratio", "missing_fields"]  # fmt: skip
    return pd.DataFrame(rows).reindex(columns=cols)


def this_work(data: AnalysisData, arm: str) -> str:
    """Trecho de legenda 'Este trabalho: n = ...; com LLM = ..., prompt ...' vindo dos dados."""
    ok = data.ok
    ok = ok[ok["arm"] == arm]
    reps = ok.groupby(["scenario", "sr", "llm", "prompt"])["repetition"].nunique()
    n = int(reps.max()) if len(reps) else 0
    llms = sorted(set(ok["llm"]) - {BASELINE_LLM})
    prompts = sorted(set(ok["prompt"]) - {BASELINE_LLM})
    names = ", ".join(latex.LLM_PT.get(m, m) for m in llms) or "n/d"
    return (
        f"Este trabalho: n = {n}; com LLM = {names}, prompt {', '.join(prompts) or 'n/d'}. "
        f"Execução: {data.label_of(arm)}."
    )


# ----------------------------------------------------------------------------- montagem
class _Counter:
    """Numeração romana sequencial das tabelas LaTeX."""

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return self.n


def _per_arm(
    data: AnalysisData, df: pd.DataFrame, counter: _Counter, render: Any, title: str, ref: str
) -> str:
    """Uma tabela LaTeX por braço (legenda com o rótulo e 'Este trabalho')."""
    parts = []
    for run in data.runs:
        sub = df[df["arm"] == run.arm] if "arm" in df.columns else df
        caption = f"{title} ({ref}). {this_work(data, run.arm)}"
        parts.append(render(sub, counter.next(), caption, run.arm))
    return "\n".join(parts)


def build_tables(data: AnalysisData, paper: pd.DataFrame | None = None) -> dict[str, TableSpec]:
    """Todas as tabelas, indexadas pelo nome do arquivo (sem extensão)."""
    comp = comparison_all(data, paper)
    effect = paired_effect(data)
    cnt = _Counter()
    label = data.combined_label()
    t2, t3, t4, t5 = (f(data, comp) for f in (table2, table3, table4, table5))
    t6 = table6(data)
    cost = cost_table(data)
    loss = loss_decomposition(data)
    summ = classification_summary(data, comp)
    arms_df = arm_comparison(effect)
    arm_shorts = list(dict.fromkeys(effect["arm_short"])) if not effect.empty else []

    def noise_tex(nt: str, num: int) -> str:
        parts = []
        for run in data.runs:
            long = noise_long(data, run.arm, nt, paper)
            kind = "média zero" if nt == "zero_mean" else "enviesado"
            cap = (
                f"Robustez a ruído {kind} (1%-5%), medida pela distância entre árvores "
                f"({PAPER_REF[num]}; LLM/prompt do artigo não informados, GAP-12). "
                f"{this_work(data, run.arm)}"
            )
            parts.append(latex.noise_table(long, cnt.next(), cap))
        return "\n".join(parts)

    def pa(df: pd.DataFrame, title: str, ref: str, fn: Any) -> str:
        return _per_arm(data, df, cnt, lambda d, n, c, _a: fn(d, n, c), title, ref)

    tex: dict[str, str] = {
        "t2_prompt_sensitivity": pa(t2, "Sensibilidade a prompts A-H, média de 3 cenários",
                                    PAPER_REF[2], latex.prompt_table),
        "t3_noise_zero_mean": noise_tex("zero_mean", 3),
        "t4_noise_biased": noise_tex("biased", 4),
        "t5_benchmark": pa(t5, "Benchmark baseline e três LLM", PAPER_REF[5],
                           latex.benchmark_table),
        "t6_llm_cost": pa(cost, "Chamadas aos LLMs, taxa de falha de interpretação (chamadas "
                          "novas) e mediana da razão de tempo LLM/RS por execução",
                          "formato da Tabela V; sem equivalente publicado", latex.cost_table),
        "effect_paired": pa(effect, "Efeito pareado do LLM (baseline − LLM)",
                            "sem equivalente publicado", latex.effect_table),
    }  # fmt: skip
    tex["effect_arms"] = latex.arms_table(
        arms_df, cnt.next(), f"Efeito do LLM por braço (baseline − LLM). {label}", arm_shorts
    )
    tex["t6_runtime"] = latex.generic(
        cnt.next(), f"Tempo e chamadas: publicado vs local ({PAPER_REF[6]}). {label}",
        t6.drop(columns=["rotulo", "notes"]), 1,
    )  # fmt: skip
    tex["loss_decomposition"] = latex.generic(
        cnt.next(), f"Decomposição da perda final (GAP-02). {label}",
        loss.drop(columns=[c for c in ("rotulo", "arm_short", "weights") if c in loss.columns]), 3,
    )  # fmt: skip
    tex["classification_summary"] = latex.generic(
        cnt.next(), f"Resumo de classificações pré-registradas. {label}",
        summ.drop(columns=["rotulo"]), 0,
    )  # fmt: skip
    missing = sorted({m for v in cost["missing_fields"].dropna() for m in str(v).split(", ") if m})
    cost_note = (
        f"Campos não persistidos (n/d): {', '.join(missing)}." if missing else "Campos completos."
    )
    spec = [
        ("t2_prompt_sensitivity", t2, "Sensibilidade a prompts (A-H)", PAPER_REF[2], DIFF_CONVENTION),
        ("t3_noise_zero_mean", t3, "Ruído de média zero: distância", PAPER_REF[3], DIFF_CONVENTION),
        ("t4_noise_biased", t4, "Ruído enviesado: distância", PAPER_REF[4], DIFF_CONVENTION),
        ("t5_benchmark", t5, "Benchmark baseline + 3 LLM", PAPER_REF[5], DIFF_CONVENTION),
        ("t6_runtime", t6, "Tempo de execução e chamadas LLM", PAPER_REF[6],
         "local_total_sr_s = train − llm"),
        ("t6_llm_cost", cost, "Custo do LLM (chamadas, falha, razão LLM/RS)",
         "formato 'Tabela V' de estilo; sem equivalente publicado", cost_note),
        ("effect_paired", effect, "Efeito pareado do LLM (baseline vs LLM)",
         "sem equivalente publicado (artigo não reporta pareamento)", EFFECT_CONVENTION),
        ("effect_arms", arms_df, "Efeito do LLM por braço", "sem equivalente publicado",
         EFFECT_CONVENTION),
        ("loss_decomposition", loss, "Decomposição da perda (GAP-02)",
         "sem equivalente publicado (pesos GAP-01)", ""),
        ("classification_summary", summ, "Resumo de classificações pré-registradas",
         "Tabelas 2-5", ""),
    ]  # fmt: skip
    return {
        name: TableSpec(name, df, title, ref, label, note=note, tex=tex[name])
        for name, df, title, ref, note in spec
    }


# ----------------------------------------------------------------------------- escrita
def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return NAN_TEXT
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def to_markdown(spec: TableSpec) -> str:
    """Tabela em Markdown com título, rótulo de execução e rodapé."""
    df = spec.df if spec.display is None else spec.df[list(spec.display)]
    lines = [
        f"### {spec.title} — {spec.paper_ref}",
        "",
        f"**Execução:** {spec.label}",
        "",
        "| " + " | ".join(map(str, df.columns)) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    for rec in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v).replace("|", "\\|") for v in rec) + " |")
    lines += ["", f"_{FOOTNOTE} {spec.note}_", ""]
    return "\n".join(lines)


def write_table(spec: TableSpec, out_dir: Path) -> list[Artifact]:
    """Grava CSV, Markdown e LaTeX; todo CSV traz a coluna ``rotulo``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = spec.df.copy()
    if "rotulo" not in df.columns:
        df.insert(0, "rotulo", spec.label)
    full = TableSpec(**{**spec.__dict__, "df": df})
    tex = spec.tex if spec.tex is not None else latex.generic(1, f"{spec.title}. {spec.label}", df)
    renders = {"csv": df.to_csv(index=False), "md": to_markdown(full), "tex": tex}
    arts: list[Artifact] = []
    for ext, text in renders.items():
        path = out_dir / f"{spec.name}.{ext}"
        path.write_text(text, encoding="utf-8")
        kind = f"tabela/{ext}"
        arts.append(Artifact(spec.name, path, kind, spec.paper_ref, spec.label, spec.title))
    return arts


def write_master_tex(specs: dict[str, TableSpec], out_dir: Path) -> Artifact:
    """``tabelas.tex`` mestre (documentclass article) que inclui todas as tabelas."""
    path = out_dir / "tabelas.tex"
    path.write_text(latex.master([f"{n}.tex" for n in specs]), encoding="utf-8")
    label = " || ".join(dict.fromkeys(s.label for s in specs.values()))
    return Artifact("tabelas", path, "tabela/tex-mestre", "Tabelas 2-6", label, "Mestre LaTeX")
