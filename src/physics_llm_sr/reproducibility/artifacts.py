"""Persistência de artefatos: JSON, Parquet e diretório de execução."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.reproducibility.manifests import RunManifest

MANIFEST_FILENAME = "manifest.json"
JSON_INDENT = 2


def _json_default(obj: Any) -> Any:
    """Serializa tipos numpy e Path."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"objeto não serializável em JSON: {type(obj).__name__}")


def save_json(data: Any, path: str | Path) -> Path:
    """Grava ``data`` como JSON (UTF-8), criando diretórios pais."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, indent=JSON_INDENT, default=_json_default, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def load_json(path: str | Path) -> Any:
    """Lê um JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Grava um DataFrame em Parquet."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    return target


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Lê um Parquet."""
    return pd.read_parquet(path)


def create_run_dir(output_dir: str | Path, run_id: str) -> Path:
    """Cria ``output_dir/run_id``; falha se já existir (execuções não se sobrescrevem).

    ``output_dir`` relativo é resolvido contra a raiz de projeto (``config.paths``).
    """
    run_dir = resolve_project_path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_manifest(manifest: RunManifest, run_dir: str | Path) -> Path:
    """Grava o manifesto da execução em ``run_dir/manifest.json``."""
    return save_json(manifest.model_dump(mode="json"), Path(run_dir) / MANIFEST_FILENAME)


def load_manifest(run_dir: str | Path) -> RunManifest:
    """Lê o manifesto de uma execução."""
    return RunManifest.model_validate(load_json(Path(run_dir) / MANIFEST_FILENAME))
