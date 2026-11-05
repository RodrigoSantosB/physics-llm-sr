# mypy: disable-error-code="no-any-return,unused-ignore,no-untyped-call"
"""Auxiliares de exibição do notebook 04 (comparação com o artigo; sem lógica científica)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from physics_llm_sr.analysis import pipeline

RUN_DIRS_ENV = "PHYSICS_LLM_SR_RUN_DIRS"
"""Lista de diretórios de runs (separador ``os.pathsep``); sobrepõe ``RUN_DIRS`` do notebook."""


def resolve_run_dirs(run_dirs: list[str]) -> list[Path]:
    """``RUN_DIRS`` do notebook, ou a variável de ambiente se definida."""
    env = os.environ.get(RUN_DIRS_ENV)
    chosen = env.split(os.pathsep) if env else run_dirs
    return [Path(p) for p in chosen if p]


def generate(run_dirs: list[str], out_dir: str | None = None) -> Path:
    """Gera tabelas e figuras e devolve o diretório de saída."""
    return pipeline.run_analysis(resolve_run_dirs(run_dirs), out_dir)


def table(out: Path, name: str, columns: list[str] | None = None, head: int = 12) -> pd.DataFrame:
    """Carrega ``tables/<name>.csv`` (primeiras ``head`` linhas, colunas opcionais)."""
    df = pd.read_csv(out / "tables" / f"{name}.csv")
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    return df.head(head)


def show_figures(out: Path, prefix: str) -> list[str]:
    """Exibe (IPython) as figuras PNG ``figures/<prefix>*.png``; devolve os nomes."""
    from IPython.display import Image, display

    files = sorted((out / "figures").glob(f"{prefix}*.png"))
    for f in files:
        print(f.name)
        display(Image(filename=str(f), width=900))
    return [f.name for f in files]


def facts(out: Path) -> dict[str, Any]:
    """Números para as leituras guiadas, todos lidos das tabelas geradas."""
    t = out / "tables"
    summ = pd.read_csv(t / "classification_summary.csv")
    eff = pd.read_csv(t / "effect_paired.csv")
    loss = pd.read_csv(t / "loss_decomposition.csv")
    t5 = pd.read_csv(t / "t5_benchmark.csv")
    done = summ[summ["executado"]]
    return {
        "cells_total": int(summ["count"].sum()),
        "cells_executed": int(done["count"].sum()),
        "classes": done.groupby("classe")["count"].sum().to_dict(),
        "t5_executed": int((t5["status_local"] == "executado").sum()),
        "effect_rows": len(eff),
        "effect_mae": eff[eff["metric"] == "mae"][["arm_short", "sr", "mean_diff", "p_value"]],
        "expr_changed": int(eff["n_expr_changed"].fillna(0).sum()),
        "loss_frac_w3c": loss[["arm_short", "sr", "frac_w3c"]],
        "labels": sorted(set(summ["rotulo"])),
    }
