"""Consolidação, análises, comparação com o artigo (Exp7) e figuras de uma execução.

Tudo aqui lê SOMENTE arquivos persistidos no diretório da execução: :func:`replot` e
:func:`report` reconstroem tabelas e figuras sem treinar de novo (PRD §11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from physics_llm_sr.config.loader import parse_config
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.evaluation import plots
from physics_llm_sr.evaluation.paper_comparison import ComparisonError, compare_with_paper
from physics_llm_sr.evaluation.statistics import (
    StatisticsInputError,
    paired_comparison,
    summarize_groups,
)
from physics_llm_sr.experiments import noise_robustness, prompt_sensitivity, structural
from physics_llm_sr.experiments.cells import NONE, ZERO_MEAN
from physics_llm_sr.experiments.storage import (
    FIGURES_DIR,
    HISTORY_FILE,
    RESULTS_FILE,
    RUN_FILE,
    STATUS_ERROR,
    STATUS_OK,
    empty_frame,
    read_histories,
    read_rows,
)
from physics_llm_sr.reproducibility.artifacts import load_json, save_json, save_parquet

PRIMARY_METRICS = ("mae", "mse", "one_minus_r2", "tree_distance")
SUMMARY_METRICS = (*PRIMARY_METRICS, "tree_distance_equivalent_min")
CONDITION = ("scenario", "sr", "llm", "prompt", "noise_type", "noise_location", "noise_level")
PAIR_KEYS = ("scenario", "sr", "noise_type", "noise_location", "noise_level", "repetition")
REPORT_FILE = "report.json"
SUMMARY_FILE = "summary.csv"
PAIRED_FILE = "paired_comparison.csv"
COMPARISON_FILE = "comparison.csv"
PROMPT_FILE = "prompt_sensitivity.csv"
NOISE_FILE = "noise_robustness.csv"
STRUCTURAL_FILE = "structural.csv"
REDUCED_LABEL = (
    "EXECUÇÃO REDUZIDA ({mode}/{track}): não é resultado completo nem reprodução do artigo"
)
#: Colunas que variam entre execuções idênticas (tempo, identidade, custo dependente de cache).
NON_METHODOLOGICAL_COLUMNS = (
    "run_id",
    "train_seconds",
    "llm_seconds",
    "cell_seconds",
    "llm_calls",
    "llm_cache_hits",
    "llm_invalid_responses",
)


def load_run_config(run_dir: Path) -> ExperimentConfig:
    """Config da execução (gravada em ``run.json``)."""
    return parse_config(load_json(Path(run_dir) / RUN_FILE)["config"])


def is_reduced(cfg: ExperimentConfig) -> bool:
    """Smoke, pilot ou extended nunca são resultado completo de reprodução."""
    return cfg.mode is not Mode.FULL or cfg.track is not Track.PAPER_REPLICATION


def consolidate(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Junta as células em ``results.parquet`` e ``history.parquet`` (ordem por chave)."""
    run_dir = Path(run_dir)
    results = pd.DataFrame(read_rows(run_dir))
    if not results.empty:
        results = results.sort_values("cell_key", kind="stable").reset_index(drop=True)
    history = read_histories(run_dir)
    if not history.empty:
        history = history.sort_values(["cell_key", "generation"], kind="stable")
        history = history.reset_index(drop=True)
    save_parquet(results, run_dir / RESULTS_FILE)
    save_parquet(history, run_dir / HISTORY_FILE)
    return results, history


def methodological_view(results: pd.DataFrame) -> pd.DataFrame:
    """Resultados sem colunas de tempo/identidade/custo, ordenados (comparação entre runs)."""
    cols = [c for c in results.columns if c not in NON_METHODOLOGICAL_COLUMNS]
    out: pd.DataFrame = results[cols].sort_values("cell_key", kind="stable").reset_index(drop=True)
    return out


def paired_table(ok: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon pareado baseline × LLM por condição e métrica (diferença = baseline − LLM)."""
    base = ok[ok["llm"] == NONE]
    llm = ok[ok["llm"] != NONE]
    rows: list[dict[str, Any]] = []
    if base.empty or llm.empty:
        return empty_frame()
    for key, sub in llm.groupby(list(CONDITION), sort=True):
        merged = sub.merge(base, on=list(PAIR_KEYS), suffixes=("_llm", "_base"))
        for metric in PRIMARY_METRICS:
            row: dict[str, Any] = dict(zip(CONDITION, key, strict=True))
            row["metric"] = metric
            try:
                res = paired_comparison(
                    merged[f"{metric}_base"].to_numpy(dtype=float),
                    merged[f"{metric}_llm"].to_numpy(dtype=float),
                )
            except StatisticsInputError as exc:
                row.update(n=len(merged), method="not_run", note=str(exc))
            else:
                row.update(res.__dict__)
            rows.append(row)
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


def comparison_table(ok: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Exp7: ``compare_with_paper`` por tabela, com os coringas de LLM/prompt do GAP-12."""
    ex = cfg.experiments
    calls: list[tuple[tuple[int, ...], dict[str, str | None]]] = [
        ((2,), {}),
        ((3, 4), {"unspecified_llm": ex.noise_model, "unspecified_prompt": ex.noise_prompt}),
        ((5,), {"unspecified_prompt": ex.benchmark_prompt}),
    ]
    frames = []
    for tables, wild in calls:
        frames.append(
            compare_with_paper(
                ok,
                tables=tables,
                unspecified_llm=wild.get("unspecified_llm"),
                unspecified_prompt=wild.get("unspecified_prompt"),
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "track", cfg.track.value)
    out.insert(0, "mode", cfg.mode.value)
    return out


def replot(run_dir: Path) -> list[Path]:
    """Regenera as figuras a partir de ``results.parquet`` e ``history.parquet``."""
    run_dir = Path(run_dir)
    results = pd.read_parquet(run_dir / RESULTS_FILE)
    history = pd.read_parquet(run_dir / HISTORY_FILE)
    figs = run_dir / FIGURES_DIR
    out: list[Path] = []
    if not history.empty:
        out.append(plots.plot_loss_per_generation(history, figs / "loss_per_generation.png"))
    if results.empty:
        return out
    ok = results[results["status"] == STATUS_OK]
    default = ok[ok["noise_type"] == ZERO_MEAN]
    if not default.empty:
        level = default["noise_level"].min()
        default = default[default["noise_level"] == level]
        cols = ("scenario", "sr", "llm", "prompt")
        for metric in PRIMARY_METRICS:
            out.append(plots.plot_metric_bars(default, metric, figs / f"{metric}_bars.png", cols))
    llm = ok[ok["prompt"].isin(plots.PROMPTS)]
    if llm["prompt"].nunique() > 1:
        for metric in ("mae", "tree_distance"):
            out.append(plots.plot_prompt_heatmap(llm, metric, figs / f"prompt_{metric}.png"))
    for (noise_type, arm), sub in ok.groupby(["noise_type", ok["llm"] == NONE], sort=True):
        if sub["noise_level"].nunique() > 1:
            name = f"noise_{noise_type}_{'baseline' if arm else 'llm'}_tree_distance.png"
            out.append(plots.plot_noise_curves(sub, "tree_distance", figs / name))
    return out


def report(run_dir: Path) -> dict[str, Any]:
    """Consolida, analisa, compara com o artigo e regenera as figuras; grava ``report.json``."""
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    results, _ = consolidate(run_dir)
    reduced = is_reduced(cfg)
    info: dict[str, Any] = {
        "run_dir": str(run_dir),
        "mode": cfg.mode.value,
        "track": cfg.track.value,
        "reduced_run": reduced,
        "complete_result": not reduced,
        "label": REDUCED_LABEL.format(mode=cfg.mode.value, track=cfg.track.value)
        if reduced
        else "execução completa (full/paper_replication)",
        "n_cells": len(results),
        "n_ok": 0,
        "n_error": 0,
        "errors": [],
        "files": {},
    }
    if results.empty:
        save_json(info, run_dir / REPORT_FILE)
        return info
    ok = results[results["status"] == STATUS_OK].copy()
    errors = results[results["status"] == STATUS_ERROR]
    info["n_ok"], info["n_error"] = len(ok), len(errors)
    info["errors"] = errors[["cell_key", "error"]].to_dict("records")
    tables: dict[str, pd.DataFrame] = {}
    if not ok.empty:
        tables[SUMMARY_FILE] = summarize_groups(ok, list(CONDITION), list(SUMMARY_METRICS))
        tables[PAIRED_FILE] = paired_table(ok)
        tables[PROMPT_FILE] = prompt_sensitivity.analyze(ok)
        tables[NOISE_FILE] = noise_robustness.analyze(ok[ok["experiments"].str.contains("noise")])
        tables[STRUCTURAL_FILE] = structural.summarize(ok)
        try:
            tables[COMPARISON_FILE] = comparison_table(ok, cfg)
        except (ComparisonError, FileNotFoundError) as exc:
            info["comparison_error"] = f"{type(exc).__name__}: {exc}"
    for name, table in tables.items():
        if table.empty:
            continue
        path = run_dir / name
        for col in ("mode", "track"):
            if col not in table.columns:
                table.insert(0, col, getattr(cfg, col).value)
        table.to_csv(path, index=False)
        info["files"][name] = str(path)
    if COMPARISON_FILE in tables:
        comp = tables[COMPARISON_FILE]
        info["comparison_classes"] = comp["classe"].value_counts().to_dict()
    info["figures"] = [str(p) for p in replot(run_dir)]
    save_json(json.loads(json.dumps(info, default=str)), run_dir / REPORT_FILE)
    return info
