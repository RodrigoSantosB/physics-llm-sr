# ruff: noqa: E501
# mypy: disable-error-code="no-any-return,unused-ignore,no-untyped-call"
"""Exporta os resultados persistidos no formato de arquivos de uma reprodução anterior.

``python -m physics_llm_sr.analysis.legacy_export --runs <dir...> --out <dir>`` grava
``resultados_benchmark.csv``, ``resultados_prompts.csv``, ``resultados_ruido.csv``,
``llm_cache.jsonl``, ``commit.txt``, ``tabelas_<modo>/`` (tabelas 2-6, ``overhead_llm.csv`` e as
figuras ``fig_tabela{2,3,4}_*.png``) e ``FORMATO_LEGADO.md``. Puramente aditivo: só LÊ os arquivos
persistidos (``results.parquet``, ``run.json``, ``cells/*/manifest.json``, cache LLM) e os valores
publicados (``references/paper_results``). Nenhum número é inventado: campo sem origem = ``nan``.
Use UMA saída por braço (trilha/normalização); execuções do mesmo braço podem ser combinadas.

Arquivo <- experimentos (coluna ``experiments``): benchmark = baseline + llm_integration (Exp1/2);
prompts = prompt_sensitivity (Exp3); ruido = noise_robustness + biased_noise (Exp4/5). Só células
``status == 'ok'``. Arquivo sem linhas = só o cabeçalho.

Coluna-alvo <- origem (colunas de ``results.parquet``)::

    key            <- scenario|sr|llm|prompt|noise_mode|noise_level|biased(0/1)|rep
    scenario       <- scenario (free_fall=ball_drop, harmonic_motion=shm, damped_wave=wave)
    sr, rep        <- sr, repetition
    llm            <- llm (llama2=llama; baseline 'none' mantido)
    prompt         <- prompt (baseline 'none' vira '-')
    noise_mode     <- noise_location (inputs=features, target, both)
    noise_level    <- noise_level
    biased         <- noise_type == 'biased'
    MAE/MSE/one_minus_R2/R2 <- test_mae/test_mse/test_one_minus_r2/test_r2 (métricas de TESTE;
                      se a coluna test_* não existe, mae/mse/one_minus_r2 e R2 = 1 - one_minus_R2)
    distance       <- tree_distance
    size           <- complexity
    plaus          <- 1 - final_c (média dos 3 escores do LLM); nan sem LLM no laço
    loss           <- final_loss
    equation       <- expression (infixa, como persistida)
    raw_equation   <- forma prefixa em notação funcional, via symbolic.to_prefix sobre
                      parse_infix(expression) (sem eval); nan se a expressão não for lida
    n_generations  <- n_generations
    n_evaluated    <- nan: nº de avaliações de fitness não é persistido (llm_requested conta
                      só avaliações pedidas ao LLM e é 0 no baseline)
    sr_seconds     <- train_seconds - llm_seconds
    llm_seconds    <- llm_seconds
    llm_ratio      <- llm_seconds / sr_seconds (nan se sr_seconds <= 0)
    llm_calls      <- llm_calls
    llm_cached     <- llm_cache_hits
    llm_fail_rate  <- llm_invalid_responses / llm_calls (0.0 sem chamadas)
    timestamp      <- created_at do cells/<cell_key>/manifest.json (senão created_at do run.json)

``llm_cache.jsonl``: uma linha por entrada do cache LLM da config (``llm.cache_dir``), com
``key = "<llm>|<hash12 do contexto>|<alvo> = <expressão>"``, ``score`` = média dos 3 escores
(``null`` em erro), ``ok`` e ``raw`` (resposta crua; se ausente, JSON dos escores).
``commit.txt``: ``git_sha`` de cada ``run.json`` (uma linha por execução). Detalhes e rótulo de
execução reduzida em ``FORMATO_LEGADO.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from physics_llm_sr.analysis import figures, loading, tables  # noqa: I001  (figures fixa Agg)
from physics_llm_sr.analysis.loading import BASELINE_LLM, AnalysisData, AnalysisError, RunInfo
from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.evaluation.paper_comparison import load_paper_results
from physics_llm_sr.llm.cache import CACHE_FILENAME
from physics_llm_sr.symbolic.expression import Const, Expr, Op, Var, parse_infix, to_prefix

BENCHMARK_FILE = "resultados_benchmark.csv"
PROMPTS_FILE = "resultados_prompts.csv"
NOISE_FILE = "resultados_ruido.csv"
CACHE_OUT = "llm_cache.jsonl"
COMMIT_FILE = "commit.txt"
MD_FILE = "FORMATO_LEGADO.md"
NAN = "nan"
EXPERIMENT_FILES = {
    BENCHMARK_FILE: ("baseline", "llm_integration"),
    PROMPTS_FILE: ("prompt_sensitivity",),
    NOISE_FILE: ("noise_robustness", "biased_noise"),
}
LEGACY_COLUMNS = [
    "key", "scenario", "sr", "llm", "prompt", "noise_mode", "noise_level", "biased", "rep",
    "MAE", "MSE", "one_minus_R2", "R2", "distance", "size", "plaus", "loss", "equation",
    "raw_equation", "n_generations", "n_evaluated", "sr_seconds", "llm_seconds", "llm_ratio",
    "llm_calls", "llm_cached", "llm_fail_rate", "timestamp",
]  # fmt: skip
INT_COLUMNS = ("size", "n_generations", "llm_calls", "llm_cached", "rep")
METRICS = ("MAE", "MSE", "one_minus_R2", "distance")
PAPER_METRIC = {
    "MAE": "mae", "MSE": "mse", "one_minus_R2": "one_minus_r2", "distance": "tree_distance",
}  # fmt: skip
LLM_LEGACY = {
    "llama2": "llama", "llama2_7b": "llama", "mistral_7b": "mistral", "falcon_7b": "falcon",
}  # fmt: skip
LLM_CANONICAL = {"llama": "llama2"}
LEGACY_LLMS = ("falcon", "llama", "mistral")
SCENARIOS = tuple(figures.SCEN_SHORT.values())
SCENARIO_CANONICAL = {v: k for k, v in figures.SCEN_SHORT.items()}
SRS = ("deap", "gplearn", "pysr")
NOISE_MODES = tuple(figures.LOC_SHORT.values())
NOISE_LEVELS = (0.01, 0.02, 0.03, 0.04, 0.05)
TARGET_NAMES = {"free_fall": "velocity", "harmonic_motion": "displacement", "damped_wave": "field"}
NO_PROMPT = "-"
NO_CALLS_RATE = 0.0
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
T2_ARTIGO = "average_3_scenarios"

#: (coluna-alvo, origem) renderizado em ``FORMATO_LEGADO.md`` (mesmo conteúdo do docstring).
MAPPING: list[tuple[str, str]] = [
    ("key", "`scenario|sr|llm|prompt|noise_mode|noise_level|biased|rep` (biased = 0/1)"),
    ("scenario", "`scenario`: free_fall=ball_drop, harmonic_motion=shm, damped_wave=wave"),
    ("sr", "`sr`"),
    ("llm", "`llm` (llama2 vira `llama`; baseline = `none`)"),
    ("prompt", "`prompt` (baseline `none` vira `-`)"),
    ("noise_mode", "`noise_location` (inputs = features; target; both)"),
    ("noise_level", "`noise_level`"),
    ("biased", "`noise_type == 'biased'`"),
    ("rep", "`repetition`"),
    ("MAE", "`test_mae` (métrica de TESTE; sem `test_*`: `mae`)"),
    ("MSE", "`test_mse` (idem; sem `test_*`: `mse`)"),
    ("one_minus_R2", "`test_one_minus_r2` (idem; sem `test_*`: `one_minus_r2`)"),
    ("R2", "`test_r2` (sem `test_*`: 1 - one_minus_R2)"),
    ("distance", "`tree_distance`"),
    ("size", "`complexity`"),
    ("plaus", "1 - `final_c` (média dos 3 escores do LLM); nan sem LLM no laço"),
    ("loss", "`final_loss`"),
    ("equation", "`expression` (infixa, como persistida; nomes de variável do cenário)"),
    ("raw_equation", "prefixa em notação funcional: `symbolic.to_prefix` sobre `parse_infix(expression)` (sem eval); nan se ilegível"),
    ("n_generations", "`n_generations`"),
    ("n_evaluated", "nan: nº de avaliações de fitness não é persistido (`llm_requested` só conta pedidos ao LLM e é 0 no baseline)"),
    ("sr_seconds", "`train_seconds` - `llm_seconds` (train_seconds inclui o tempo do LLM)"),
    ("llm_seconds", "`llm_seconds`"),
    ("llm_ratio", "`llm_seconds / sr_seconds` (nan se sr_seconds <= 0)"),
    ("llm_calls", "`llm_calls`"),
    ("llm_cached", "`llm_cache_hits`"),
    ("llm_fail_rate", "`llm_invalid_responses / llm_calls` (0.0 sem chamadas)"),
    ("timestamp", "`created_at` de `cells/<cell_key>/manifest.json` (senão do `run.json`), UTC, `AAAA-MM-DD HH:MM:SS`"),
]  # fmt: skip


# ----------------------------------------------------------------------------- utilidades
def legacy_llm(name: str) -> str:
    """Nome da LLM no formato legado (``llama2`` -> ``llama``; demais inalterados)."""
    return LLM_LEGACY.get(name, name)


def functional_prefix(expr: Expr) -> str:
    """Notação funcional (prefixa): ``mul(3.5, x)``; mesma ordem de ``symbolic.to_prefix``."""
    if isinstance(expr, Const):
        return repr(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Op):
        return f"{expr.name}({', '.join(functional_prefix(c) for c in expr.children)})"
    raise TypeError(f"nó desconhecido: {expr!r}")


def raw_equation(expression: Any) -> str:
    """Forma prefixa da expressão infixa via ``parse_infix`` (nunca ``eval``); ``nan`` se falhar."""
    try:
        expr = parse_infix(str(expression))
        tokens = to_prefix(expr)
        text = functional_prefix(expr)
    except Exception:  # noqa: BLE001 - texto não confiável: qualquer falha vira nan
        return NAN
    return text if tokens else NAN


def _col(df: pd.DataFrame, name: str, fallback: str | None = None) -> pd.Series[Any]:
    for candidate in (name, fallback):
        if candidate is not None and candidate in df.columns:
            return df[candidate]
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _fmt_ts(text: Any) -> str:
    try:
        return dt.datetime.fromisoformat(str(text)).strftime(TS_FORMAT)
    except ValueError:
        return NAN


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _timestamps(run: RunInfo, cell_keys: Iterable[str], created_at: str) -> list[str]:
    out = []
    for key in cell_keys:
        manifest = _read_json(run.run_dir / "cells" / str(key) / "manifest.json")
        out.append(_fmt_ts(manifest.get("created_at") or created_at))
    return out


# ----------------------------------------------------------------------------- linhas
def _experiment_tokens(value: Any) -> set[str]:
    return {t.strip() for t in str(value).split(",") if t.strip()}


def legacy_rows(df: pd.DataFrame, timestamps: list[str] | None = None) -> pd.DataFrame:
    """Converta linhas ``ok`` de ``results.parquet`` ao layout legado (``LEGACY_COLUMNS``)."""
    out = pd.DataFrame(index=df.index)
    scenario = df["scenario"].map(lambda s: figures.SCEN_SHORT.get(s, s))
    llm = df["llm"].map(legacy_llm)
    prompt = df["prompt"].where(df["prompt"] != BASELINE_LLM, NO_PROMPT)
    mode = df["noise_location"].map(lambda s: figures.LOC_SHORT.get(s, s))
    level = df["noise_level"].astype(float)
    biased = df["noise_type"] == "biased"
    rep = df["repetition"].astype(int)
    out["key"] = [
        f"{a}|{b}|{c}|{d}|{e}|{f:g}|{int(g)}|{h}"
        for a, b, c, d, e, f, g, h in zip(
            scenario, df["sr"], llm, prompt, mode, level, biased, rep, strict=True
        )
    ]
    out["scenario"], out["sr"], out["llm"], out["prompt"] = scenario, df["sr"], llm, prompt
    out["noise_mode"], out["noise_level"], out["biased"], out["rep"] = mode, level, biased, rep
    out["MAE"] = _col(df, "test_mae", "mae").astype(float)
    out["MSE"] = _col(df, "test_mse", "mse").astype(float)
    out["one_minus_R2"] = _col(df, "test_one_minus_r2", "one_minus_r2").astype(float)
    out["R2"] = (
        _col(df, "test_r2").astype(float) if "test_r2" in df.columns else 1 - out["one_minus_R2"]
    )
    out["distance"] = df["tree_distance"].astype(float)
    out["size"] = _col(df, "complexity")
    in_loop = (
        df["llm_in_loop"].astype(bool) if "llm_in_loop" in df.columns else df["llm"] != BASELINE_LLM
    )
    final_c = _col(df, "final_c").astype(float)
    out["plaus"] = (1.0 - final_c).where(in_loop & np.isfinite(final_c), np.nan)
    out["loss"] = _col(df, "final_loss").astype(float)
    out["equation"] = df["expression"]
    memo: dict[str, str] = {}
    for e in df["expression"].astype(str).unique():
        memo[e] = raw_equation(e)
    out["raw_equation"] = df["expression"].astype(str).map(memo)
    out["n_generations"] = _col(df, "n_generations")
    out["n_evaluated"] = np.nan
    llm_s = df["llm_seconds"].astype(float)
    sr_s = df["train_seconds"].astype(float) - llm_s
    out["sr_seconds"], out["llm_seconds"] = sr_s, llm_s
    out["llm_ratio"] = (llm_s / sr_s).where(sr_s > 0, np.nan)
    calls = df["llm_calls"].astype(float)
    out["llm_calls"] = df["llm_calls"]
    out["llm_cached"] = _col(df, "llm_cache_hits")
    invalid = _col(df, "llm_invalid_responses").astype(float)
    out["llm_fail_rate"] = (invalid / calls).where(calls > 0, NO_CALLS_RATE)
    out["timestamp"] = timestamps if timestamps is not None else NAN
    for name in INT_COLUMNS:
        out[name] = out[name].astype("Int64")
    return out[LEGACY_COLUMNS].reset_index(drop=True)


def _write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    df.to_csv(path, na_rep=NAN, **kwargs)


# ----------------------------------------------------------------------------- tabelas
def _paper_value(paper: pd.DataFrame, **cond: Any) -> float:
    sub = paper
    for col, val in cond.items():
        sub = sub[sub[col] == val]
    return float(sub["value"].iloc[0]) if len(sub) else math.nan


def table2_legacy(prompts: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    """Média por prompt x llm x sr (cenários e repetições agrupados) com ``*_artigo``."""
    cols = ["prompt", "llm", "sr"]
    for m in METRICS:
        cols += [m, f"{m}_artigo"]
    sub = prompts[prompts["llm"] != BASELINE_LLM]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    pub = paper[paper["table"] == 2]
    rows = []
    for (p, llm, sr), g in sub.groupby(["prompt", "llm", "sr"]):
        row: dict[str, Any] = {"prompt": p, "llm": llm, "sr": sr}
        for m in METRICS:
            row[m] = round(float(g[m].mean()), 4)
            row[f"{m}_artigo"] = _paper_value(
                pub, scenario=T2_ARTIGO, sr=sr, llm=LLM_CANONICAL.get(llm, llm), prompt=p,
                metric=PAPER_METRIC[m],
            )  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def table5_legacy(bench: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    """Benchmark por scenario x llm (``baseline`` = sem LLM) x sr com ``*_artigo``."""
    cols = ["scenario", "llm", "sr"]
    for m in METRICS:
        cols += [m, f"{m}_artigo"]
    if bench.empty:
        return pd.DataFrame(columns=cols)
    pub = paper[paper["table"] == 5]
    frame = bench.assign(llm=bench["llm"].where(bench["llm"] != BASELINE_LLM, "baseline"))
    rows = []
    for (scen, llm, sr), g in frame.groupby(["scenario", "llm", "sr"]):
        paper_llm = BASELINE_LLM if llm == "baseline" else LLM_CANONICAL.get(llm, llm)
        row: dict[str, Any] = {"scenario": scen, "llm": llm, "sr": sr}
        for m in METRICS:
            row[m] = round(float(g[m].mean()), 4)
            row[f"{m}_artigo"] = _paper_value(
                pub, scenario=SCENARIO_CANONICAL.get(scen, scen), sr=sr, llm=paper_llm,
                metric=PAPER_METRIC[m],
            )  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


T6_METRICS = (
    "total_sr_s", "total_llm_s", "llm_sr_ratio", "sr_per_call_s", "llm_per_call_s",
    "sr_per_iter_s", "llm_per_iter_s",
)  # fmt: skip
T6_PAPER = dict(
    zip(
        T6_METRICS,
        ("total_sr_s", "total_llm_s", "llm_sr_ratio", "avg_sr_per_call_s", "avg_llm_per_call_s",
         "avg_sr_per_iter_s", "avg_llm_per_iter_s"),
        strict=True,
    )
)  # fmt: skip


def _safe_div(a: float, b: float) -> float:
    return a / b if b and np.isfinite(a) and np.isfinite(b) and b > 0 else math.nan


def table6_legacy(bench: pd.DataFrame, paper6: pd.DataFrame) -> pd.DataFrame:
    """Tempos por scenario x sr x llm (benchmark, só com LLM) com ``*_artigo`` do suplemento.

    Totais = média por execução de ``sr_seconds``/``llm_seconds``; razão = total LLM / total SR;
    por chamada = total / média de ``llm_calls``; por iteração = total / média de
    ``n_generations``. Sem divisor positivo = nan.
    """
    cols = ["scenario", "sr", "llm", *T6_METRICS, *[f"{m}_artigo" for m in T6_METRICS]]
    sub = bench[bench["llm"] != BASELINE_LLM]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    pub = paper6.assign(llm=paper6["llm"].map(legacy_llm))
    rows = []
    for (scen, sr, llm), g in sub.groupby(["scenario", "sr", "llm"]):
        sr_s, llm_s = float(g["sr_seconds"].mean()), float(g["llm_seconds"].mean())
        calls = float(g["llm_calls"].astype(float).mean())
        gens = float(g["n_generations"].astype(float).mean())
        local = {
            "total_sr_s": sr_s, "total_llm_s": llm_s, "llm_sr_ratio": _safe_div(llm_s, sr_s),
            "sr_per_call_s": _safe_div(sr_s, calls), "llm_per_call_s": _safe_div(llm_s, calls),
            "sr_per_iter_s": _safe_div(sr_s, gens), "llm_per_iter_s": _safe_div(llm_s, gens),
        }  # fmt: skip
        match = pub[
            (pub["scenario"] == SCENARIO_CANONICAL.get(scen, scen))
            & (pub["sr"] == sr)
            & (pub["llm"] == llm)
        ]
        row: dict[str, Any] = {"scenario": scen, "sr": sr, "llm": llm}
        for m in T6_METRICS:
            row[m] = round(local[m], 3)
            row[f"{m}_artigo"] = float(match[T6_PAPER[m]].iloc[0]) if len(match) else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def noise_wide(noise: pd.DataFrame, *, biased: bool, baseline: bool) -> pd.DataFrame:
    """Distância média em formato largo (cabeçalho triplo noise_mode / noise_level / scenario,sr)."""
    sub = noise[(noise["biased"] == biased) & ((noise["llm"] == BASELINE_LLM) == baseline)]
    modes = [*NOISE_MODES, *sorted(set(sub["noise_mode"]) - set(NOISE_MODES))]
    levels = sorted({*NOISE_LEVELS, *map(float, sub["noise_level"])})
    scens = [*SCENARIOS, *sorted(set(sub["scenario"]) - set(SCENARIOS))]
    srs = [*SRS, *sorted(set(sub["sr"]) - set(SRS))]
    index = pd.MultiIndex.from_product([scens, srs], names=["scenario", "sr"])
    columns = pd.MultiIndex.from_product([modes, levels], names=["noise_mode", "noise_level"])
    if sub.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    wide = (
        sub.groupby(["scenario", "sr", "noise_mode", "noise_level"])["distance"]
        .mean()
        .round(3)
        .unstack(["noise_mode", "noise_level"])
    )
    return wide.reindex(index=index, columns=columns)


def overhead_legacy(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Por sr x llm (só com LLM): razão LLM/SR mediana, chamadas, falha de parse e execuções."""
    cols = [
        "sr",
        "llm",
        "razao_LLM_SR_mediana",
        "chamadas_LLM_media",
        "falha_parse_media",
        "execucoes",
    ]
    allrows = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else None
    if allrows is None or allrows.empty:
        return pd.DataFrame(columns=cols)
    sub = allrows[allrows["llm"] != BASELINE_LLM].drop_duplicates("key")
    rows = [
        {
            "sr": sr,
            "llm": llm,
            "razao_LLM_SR_mediana": round(float(g["llm_ratio"].astype(float).median()), 3),
            "chamadas_LLM_media": round(float(g["llm_calls"].astype(float).mean()), 3),
            "falha_parse_media": round(float(g["llm_fail_rate"].astype(float).mean()), 3),
            "execucoes": len(g),
        }  # fmt: skip
        for (sr, llm), g in sub.groupby(["sr", "llm"])
    ]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------- cache LLM
def find_cache_files(runs: list[RunInfo], override: Path | None = None) -> list[Path]:
    """Arquivos ``evaluations.jsonl`` das execuções (``llm.cache_dir`` da config).

    Caminho relativo é procurado nos ancestrais do ``run_dir`` e, por fim, na raiz do projeto.
    """
    if override is not None:
        return (
            [Path(override) / CACHE_FILENAME] if (Path(override) / CACHE_FILENAME).exists() else []
        )
    found: list[Path] = []
    for run in runs:
        cfg = (_read_json(run.run_dir / loading.RUN_FILE).get("config") or {}).get("llm") or {}
        cache_dir = cfg.get("cache_dir")
        if not cache_dir:
            continue
        candidates = (
            [Path(cache_dir) / CACHE_FILENAME]
            if Path(cache_dir).is_absolute()
            else [p / cache_dir / CACHE_FILENAME for p in Path(run.run_dir).resolve().parents]
        )
        candidates.append(resolve_project_path(cache_dir) / CACHE_FILENAME)
        for cand in candidates:
            if cand.exists():
                if cand.resolve() not in [f.resolve() for f in found]:
                    found.append(cand)
                break
    return found


def _checkpoint_names(runs: list[RunInfo]) -> dict[str, str]:
    names: dict[str, str] = {}
    for run in runs:
        llm = (_read_json(run.run_dir / loading.RUN_FILE).get("config") or {}).get("llm") or {}
        for m in llm.get("models") or []:
            names[str(m.get("checkpoint"))] = legacy_llm(str(m.get("name")))
    return names


def cache_records(files: list[Path], names: dict[str, str]) -> list[dict[str, Any]]:
    """Registros legados do cache: ``{"key", "score", "ok", "raw"}`` (ordem de 1ª aparição).

    O último registro ``ok`` de cada chave do cache vale; erros só entram se não houver ``ok``
    (``ok=false``, ``score=null``, ``raw`` = respostas cruas). Linhas ilegíveis são ignoradas.
    """
    order: list[str] = []
    best: dict[str, dict[str, Any]] = {}
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    meta = rec.get("metadata") or {}
                    key = str(rec["key"])
                    ok = rec["status"] == "ok"
                    if ok:
                        ev = rec["evaluation"]
                        scores = [float(ev[k]) for k in
                                  ("dimensional_consistency", "simplicity", "physical_realism")]  # fmt: skip
                        score: float | None = sum(scores) / 3
                        raw = str(rec.get("raw_response") or "") or json.dumps(
                            {
                                k: ev[k]
                                for k in (
                                    "dimensional_consistency",
                                    "simplicity",
                                    "physical_realism",
                                )
                            }
                        )
                    else:
                        score, raw = None, "\n".join(map(str, rec.get("raw_responses") or []))
                    model = str(meta.get("model_id"))
                    llm = names.get(model) or legacy_llm(model.split(":")[0])
                    target = TARGET_NAMES.get(str(meta.get("scenario")), "y")
                    legacy_key = f"{llm}|{str(meta.get('context_hash', ''))[:12]}|{target} = {meta['expression']}"
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if key not in best:
                    order.append(key)
                elif ok is False and best[key]["ok"]:
                    continue
                best[key] = {"key": legacy_key, "score": score, "ok": ok, "raw": raw}
    return [best[k] for k in order]


# ----------------------------------------------------------------------------- figuras
def figure_table2(t2: pd.DataFrame, paper: pd.DataFrame, label: str, path: Path) -> None:
    """``fig_tabela2_prompts.png``: distância por prompt (média sobre cenários), nosso x artigo."""
    prompts = list(figures.PROMPTS)
    pub = paper[(paper["table"] == 2) & (paper["metric"] == "tree_distance")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, sr in zip(axes, figures.SRS, strict=True):
        for k, llm in enumerate(LEGACY_LLMS[::-1]):  # mistral, llama, falcon
            color = f"C{k}"
            s = t2[(t2["sr"] == sr) & (t2["llm"] == llm)].set_index("prompt")["distance"]
            xs = [i for i, p in enumerate(prompts) if p in s.index]
            if xs:
                ax.plot(xs, [float(s[prompts[i]]) for i in xs], "-o", color=color, lw=2,
                        label=f"{llm} (nosso)")  # fmt: skip
            else:
                ax.plot([], [], "-o", color=color, label=f"{llm} (nosso)")
            a = pub[(pub["sr"] == sr) & (pub["llm"] == LLM_CANONICAL.get(llm, llm))]
            a = a.set_index("prompt")["value"]
            ax.plot(range(len(prompts)), [float(a.get(p, np.nan)) for p in prompts], "--",
                    color=color, alpha=0.6, label=f"{llm} (artigo)")  # fmt: skip
        ax.set_xticks(range(len(prompts)), prompts)
        ax.set_title(f"SR = {sr}")
        ax.set_xlabel("Prompt")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Distância de árvore")
    axes[0].set_ylim(-0.05, 1.05)
    axes[-1].legend(fontsize=8, loc="center right")
    if t2.empty:
        axes[0].text(0.5, 0.5, "local: não executado", transform=axes[0].transAxes,
                     ha="center", fontsize=8, color="#8a1c1c")  # fmt: skip
    fig.suptitle("Tabela 2: distância por prompt (média sobre cenários)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.text(0.5, 0.004, label, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _single_arm(data: AnalysisData) -> AnalysisData:
    """Une execuções do MESMO braço (modo/trilha/normalização) numa só; recusa braços mistos."""
    kinds = {(r.mode, r.track, r.normalization) for r in data.runs}
    if len(kinds) > 1:
        raise AnalysisError(
            f"execuções de braços distintos {sorted(kinds)}: use uma saída (--out) por braço"
        )
    first = data.runs[0]
    results = data.results.assign(arm=first.arm)
    return AnalysisData(results, data.history, [first])


# ----------------------------------------------------------------------------- orquestração
@dataclass(frozen=True)
class Exported:
    """Resumo de uma exportação."""

    out_dir: Path
    tables_dir: Path
    row_counts: dict[str, int]
    all_nan: dict[str, list[str]]
    cache_entries: int


def _all_nan(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    return [c for c in df.columns if df[c].isna().all() or (df[c].astype(str) == NAN).all()]


def render_md(data: AnalysisData, shas: list[str], exp: Exported, cache_files: list[Path]) -> str:
    """Texto de ``FORMATO_LEGADO.md``."""
    lines = [
        "# Formato legado (exportação para comparação direta)", "",
        "Resultados persistidos exportados nos nomes de arquivo, colunas e layout de uma "
        "reprodução anterior. **Os números são NOSSOS** (nada foi recalculado ou inventado); "
        "campo sem origem é `nan`.", "",
        "## Rótulo de execução", "",
    ]  # fmt: skip
    for r in data.runs:
        lines.append(f"- `{r.run_dir}`: {r.label}")
    lines += [
        "",
        "O rótulo vale para todos os arquivos desta pasta (CSV, tabelas e figuras); as colunas "
        "não o carregam.", "",
        "## Arquivos e experimentos de origem", "",
        "| arquivo | experimentos (coluna `experiments`) | linhas |", "|---|---|---|",
    ]  # fmt: skip
    for fname, exps in EXPERIMENT_FILES.items():
        lines.append(f"| `{fname}` | {', '.join(exps)} | {exp.row_counts[fname]} |")
    lines += [
        "",
        "Só células com `status == 'ok'`. Uma célula pedida por vários experimentos aparece em "
        "cada arquivo correspondente. Arquivo sem linhas contém só o cabeçalho.", "",
        "## Mapeamento coluna-alvo <- origem", "", "| coluna | origem |", "|---|---|",
    ]  # fmt: skip
    lines += [f"| `{c}` | {o} |" for c, o in MAPPING]
    lines += ["", "## Colunas totalmente `nan` nesta exportação", ""]
    for fname, cols in exp.all_nan.items():
        empty = exp.row_counts[fname] == 0
        lines.append(
            f"- `{fname}`: "
            + ("(sem linhas, só cabeçalho)" if empty else ", ".join(cols) or "(nenhuma)")
        )
    lines += [
        "",
        "`n_evaluated` é sempre `nan` (não persistido). `plaus` é `nan` no baseline e quando "
        "`final_c` não é finito; `raw_equation` é `nan` se a expressão não pôde ser lida.", "",
        "## `llm_cache.jsonl`", "",
        f"{exp.cache_entries} entradas de: "
        + (", ".join(f"`{p}`" for p in cache_files) if cache_files else "(nenhum cache encontrado)")
        + ". Formato `{key, score, ok, raw}` com `key = \"<llm>|<hash12 do contexto>|<alvo> = "
        "<expressão>\"` (hash12 = 12 primeiros hex do SHA-256 do contexto físico; alvo = variável "
        "de saída do cenário). `score` = média dos 3 escores (`null` se a avaliação falhou). `raw` = "
        "resposta crua do LLM; se não estiver no cache, o JSON dos escores. A chave legada não "
        "inclui variante de prompt nem identidade do modelo: pode repetir entre linhas (uma linha "
        "por entrada do nosso cache). O cache pode conter entradas de outras execuções.", "",
        "## `commit.txt`", "", "`git_sha` de cada `run.json` (uma linha por execução):", "",
    ]  # fmt: skip
    lines += [f"- `{s}`" for s in shas]
    lines += [
        "",
        f"## `{exp.tables_dir.name}/`", "",
        "- `tabela2_prompts.csv`: média por prompt x llm x sr (cenários e repetições agrupados) "
        "do arquivo de prompts; `*_artigo` = Tabela 2 publicada (média dos 3 cenários).",
        "- `tabela3_*`/`tabela4_*`: distância média, formato largo (cabeçalho de 3 linhas: "
        "noise_mode / noise_level / scenario,sr); `*_ruido_*` = com LLM, `*_baseline_sem_llm` = "
        "sem LLM; vazio = `nan` (não executado). Tabela 3 = média zero, 4 = enviesado.",
        "- `tabela5_benchmark.csv`: scenario x llm (`baseline` = sem LLM) x sr, com `*_artigo` "
        "(suplemento).",
        "- `tabela6_tempos.csv`: totais = média por execução de `sr_seconds`/`llm_seconds`; razão = "
        "total LLM / total SR; por chamada = total / média de `llm_calls`; por iteração = total / "
        "média de `n_generations`; `*_artigo` = Tabela 6 do suplemento (`avg_*`); `nan` sem divisor "
        "positivo ou sem valor publicado.",
        "- `overhead_llm.csv`: por sr x llm (só com LLM, todas as células exportadas, sem "
        "duplicar chave): mediana de `llm_ratio`, média de `llm_calls` e de `llm_fail_rate`, "
        "contagem de execuções.",
        "- `fig_tabela2_prompts.png`, `fig_tabela3_ruido.png`, `fig_tabela4_ruido.png`: mesmo "
        "estilo das figuras de referência (nosso = cheio, artigo = tracejado); as de ruído "
        "reusam `analysis.figures.figure2` (série 'com LLM' = modelo/prompt declarados na config).",
        "- Valores publicados vêm de `references/paper_results/*.csv`; célula publicada ausente = `nan`.",
        "",
    ]  # fmt: skip
    return "\n".join(lines)


def export_legacy(
    run_dirs: list[Path] | list[str], out_dir: Path | str, cache_dir: Path | None = None
) -> Exported:
    """Gera a exportação em ``out_dir`` e devolve o resumo."""
    data = loading.load_runs(run_dirs)
    orig_runs = data.runs
    data = _single_arm(data)
    if "experiments" not in data.results.columns:
        raise AnalysisError("results.parquet sem a coluna 'experiments'")
    out = Path(out_dir)
    ok = data.ok
    modes = sorted({r.mode for r in data.runs})
    tables_dir = out / f"tabelas_{modes[0]}"
    tables_dir.mkdir(parents=True, exist_ok=True)
    first = data.runs[0]

    originals = [r.run_dir for r in orig_runs]
    created = {d: str(_read_json(d / loading.RUN_FILE).get("created_at", "")) for d in originals}
    shas = [
        str(_read_json(d / loading.RUN_FILE).get("git_sha") or "desconhecido") for d in originals
    ]
    run_of_id = {str(_read_json(d / loading.RUN_FILE).get("run_id")): d for d in originals}

    ts = []
    for _, row in ok.iterrows():
        d = run_of_id.get(str(row.get("run_id")), originals[0])
        manifest = _read_json(d / "cells" / str(row["cell_key"]) / "manifest.json")
        ts.append(_fmt_ts(manifest.get("created_at") or created[d]))
    ok = ok.assign(_ts=ts)
    tokens = ok["experiments"].map(_experiment_tokens)

    frames: dict[str, pd.DataFrame] = {}
    for fname, exps in EXPERIMENT_FILES.items():
        mask = tokens.map(lambda t, e=exps: bool(t & set(e)))
        sub = ok[mask]
        frame = legacy_rows(sub, sub["_ts"].tolist())
        if frame["key"].duplicated().any():
            raise AnalysisError(f"chaves duplicadas em {fname}: use uma saída por execução/braço")
        frames[fname] = frame
        _write_csv(frame, out / fname, index=False)

    paper = load_paper_results()
    paper6 = tables.load_table6()
    t2 = table2_legacy(frames[PROMPTS_FILE], paper)
    t5 = table5_legacy(frames[BENCHMARK_FILE], paper)
    t6 = table6_legacy(frames[BENCHMARK_FILE], paper6)
    noise = frames[NOISE_FILE]
    _write_csv(t2, tables_dir / "tabela2_prompts.csv", index=False)
    for num, biased, with_llm_name in ((3, False, "tabela3_ruido_media_zero.csv"),
                                       (4, True, "tabela4_ruido_enviesado.csv")):  # fmt: skip
        _write_csv(noise_wide(noise, biased=biased, baseline=False), tables_dir / with_llm_name)
        _write_csv(
            noise_wide(noise, biased=biased, baseline=True),
            tables_dir / f"tabela{num}_baseline_sem_llm.csv",
        )
    _write_csv(t5, tables_dir / "tabela5_benchmark.csv", index=False)
    _write_csv(t6, tables_dir / "tabela6_tempos.csv", index=False)
    _write_csv(overhead_legacy(list(frames.values())), tables_dir / "overhead_llm.csv", index=False)

    figure_table2(t2, paper, first.label, tables_dir / "fig_tabela2_prompts.png")
    with tempfile.TemporaryDirectory() as tmp:
        for noise_type, fname in (("zero_mean", "fig_tabela3_ruido.png"),
                                  ("biased", "fig_tabela4_ruido.png")):  # fmt: skip
            figures.figure2(data, noise_type, first.arm, Path(tmp))
            png = next(Path(tmp).glob(f"F2_noise_{noise_type}_*.png"))
            shutil.copyfile(png, tables_dir / fname)

    cache_files = find_cache_files(orig_runs, cache_dir)
    records = cache_records(cache_files, _checkpoint_names(orig_runs))
    with (out / CACHE_OUT).open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / COMMIT_FILE).write_text("\n".join(shas) + "\n", encoding="utf-8")

    exported = Exported(
        out_dir=out,
        tables_dir=tables_dir,
        row_counts={k: len(v) for k, v in frames.items()},
        all_nan={k: _all_nan(v) for k, v in frames.items()},
        cache_entries=len(records),
    )
    (out / MD_FILE).write_text(render_md(data, shas, exported, cache_files), encoding="utf-8")
    return exported


def main(argv: list[str] | None = None) -> int:
    """Linha de comando: ``--runs <dirs...> --out <dir> [--cache-dir <dir>]``."""
    parser = argparse.ArgumentParser(
        prog="physics_llm_sr.analysis.legacy_export",
        description=__doc__.split("\n")[0] if __doc__ else None,
    )
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="diretórios de runs")
    parser.add_argument("--out", required=True, type=Path, help="pasta de saída")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="llm.cache_dir (padrão: da config)"
    )
    args = parser.parse_args(argv)
    res = export_legacy(args.runs, args.out, args.cache_dir)
    print(f"exportação legada em {res.out_dir} (veja {res.out_dir / MD_FILE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
