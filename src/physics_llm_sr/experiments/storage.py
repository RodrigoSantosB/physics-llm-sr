"""Layout do diretório de execução e gravação atômica por célula (checkpoint).

``<run_dir>/``
  ``run.json`` (identidade + config), ``manifest.json`` (manifesto da execução),
  ``cells/<chave>/{result.json, history.parquet, manifest.json}`` (uma por célula),
  ``results.parquet``, ``history.parquet`` (consolidados), relatórios e ``figures/``.

Cada célula é escrita num diretório temporário e renomeada (``os.replace``) para o
destino: uma célula interrompida no meio nunca aparece como concluída.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from physics_llm_sr.reproducibility.artifacts import load_json, save_json, save_parquet

RUN_FILE = "run.json"
CELLS_DIR = "cells"
RESULT_FILE = "result.json"
CELL_HISTORY_FILE = "history.parquet"
CELL_MANIFEST_FILE = "manifest.json"
RESULTS_FILE = "results.parquet"
HISTORY_FILE = "history.parquet"
FIGURES_DIR = "figures"
STATUS_OK = "ok"
STATUS_ERROR = "error"
_TMP_PREFIX = ".tmp-"
_TRASH_PREFIX = ".old-"


def empty_frame() -> pd.DataFrame:
    """DataFrame vazio tipado."""
    frame: pd.DataFrame = pd.DataFrame()
    return frame


def cell_dir(run_dir: Path, key: str) -> Path:
    """Diretório definitivo da célula."""
    return run_dir / CELLS_DIR / key


def write_cell(
    run_dir: Path,
    key: str,
    row: dict[str, Any],
    history: pd.DataFrame | None,
    manifest: dict[str, Any] | None,
) -> Path:
    """Grava a célula atomicamente (substitui uma tentativa anterior com erro)."""
    cells = run_dir / CELLS_DIR
    cells.mkdir(parents=True, exist_ok=True)
    tmp = cells / f"{_TMP_PREFIX}{key}-{uuid.uuid4().hex}"
    tmp.mkdir()
    if history is not None:
        save_parquet(history, tmp / CELL_HISTORY_FILE)
    if manifest is not None:
        save_json(manifest, tmp / CELL_MANIFEST_FILE)
    save_json(row, tmp / RESULT_FILE)  # por último: marca a célula como completa
    final = cell_dir(run_dir, key)
    if final.exists():
        trash = cells / f"{_TRASH_PREFIX}{key}-{uuid.uuid4().hex}"
        os.replace(final, trash)
        os.replace(tmp, final)
        shutil.rmtree(trash)
    else:
        os.replace(tmp, final)
    return final


def read_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Linhas de resultado de todas as células gravadas (ordem por chave)."""
    cells = run_dir / CELLS_DIR
    if not cells.is_dir():
        return []
    rows = []
    for d in sorted(cells.iterdir()):
        if d.name.startswith((_TMP_PREFIX, _TRASH_PREFIX)) or not (d / RESULT_FILE).is_file():
            continue
        rows.append(load_json(d / RESULT_FILE))
    return rows


def completed_keys(run_dir: Path) -> set[str]:
    """Chaves das células concluídas com sucesso (puladas na retomada)."""
    return {r["cell_key"] for r in read_rows(run_dir) if r.get("status") == STATUS_OK}


def read_histories(run_dir: Path) -> pd.DataFrame:
    """Históricos de todas as células concluídas, concatenados."""
    frames = [
        pd.read_parquet(d / CELL_HISTORY_FILE)
        for d in sorted((run_dir / CELLS_DIR).glob("*"))
        if not d.name.startswith((_TMP_PREFIX, _TRASH_PREFIX))
        and (d / CELL_HISTORY_FILE).is_file()
        and (d / RESULT_FILE).is_file()
    ]
    if not frames:
        return empty_frame()
    out: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return out
