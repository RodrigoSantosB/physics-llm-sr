"""Manifesto de execução: identidade, hashes, ambiente, sementes e modelos (PRD 11)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset

UNKNOWN = "unknown"
RUN_ID_UUID_CHARS = 8
SUBPROCESS_TIMEOUT_S = 10.0
#: Bibliotecas cujas versões entram no manifesto (as ausentes viram ``None``).
TRACKED_LIBRARIES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sympy",
    "pydantic",
    "deap",
    "gplearn",
    "scikit-learn",
    "pysr",
    "transformers",
    "torch",
    "physics-llm-sr",
)


class RunManifest(BaseModel):
    """Registro reconstruível de uma execução."""

    run_id: str
    created_at: str
    mode: str
    track: str
    config_hash: str
    data_hash: str | None
    git_sha: str
    git_dirty: bool | None
    python_version: str
    julia_version: str | None
    library_versions: dict[str, str | None]
    seeds: dict[str, int]
    hardware: dict[str, Any]
    llm_models: list[dict[str, Any]]
    inference: dict[str, Any]
    prompt: dict[str, Any] | None
    timings: dict[str, float]
    interruption: str | None


def new_run_id() -> str:
    """Identificador único: timestamp UTC + sufixo aleatório."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:RUN_ID_UUID_CHARS]}"


def hash_config(config: BaseModel) -> str:
    """SHA-256 do JSON canônico (chaves ordenadas) da configuração."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _update_with_array(h: Any, arr: np.ndarray[Any, Any]) -> None:
    contiguous = np.ascontiguousarray(arr)
    h.update(str(contiguous.dtype).encode())
    h.update(str(contiguous.shape).encode())
    h.update(contiguous.tobytes())


def hash_dataset(dataset: Dataset) -> str:
    """SHA-256 do conteúdo numérico e dos nomes de um ``Dataset``."""
    h = hashlib.sha256()
    for arr in (dataset.X, dataset.y_true, dataset.y_observed, dataset.simulation_ids, dataset.t):
        _update_with_array(h, arr)
    h.update(json.dumps([dataset.feature_names, dataset.target_name, dataset.scenario]).encode())
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_sha(cwd: str | Path | None = None) -> str:
    """SHA do HEAD ou ``"unknown"`` fora de um repositório git."""
    where = Path(cwd) if cwd is not None else None
    return _run(["git", "rev-parse", "HEAD"], where) or UNKNOWN


def _git_dirty(cwd: Path | None) -> bool | None:
    status = _run(["git", "status", "--porcelain"], cwd)
    return None if status is None else bool(status)


def library_versions(names: tuple[str, ...] = TRACKED_LIBRARIES) -> dict[str, str | None]:
    """Versões instaladas das bibliotecas rastreadas (``None`` se ausente)."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def julia_version() -> str | None:
    """Versão do Julia, se o executável existir; nunca é exigido."""
    exe = shutil.which("julia")
    return _run([exe, "--version"]) if exe else None


def hardware_info() -> dict[str, Any]:
    """CPU e GPU (via torch se instalado; senão nvidia-smi opcional)."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpus": [],
    }
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            return info
    smi = shutil.which("nvidia-smi")
    if smi:
        names = _run([smi, "--query-gpu=name", "--format=csv,noheader"])
        if names:
            info["gpus"] = names.splitlines()
    return info


def build_manifest(
    config: ExperimentConfig,
    *,
    dataset: Dataset | None = None,
    seeds: dict[str, int] | None = None,
    prompt: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    interruption: str | None = None,
    run_id: str | None = None,
    repo_dir: str | Path | None = None,
) -> RunManifest:
    """Monta o manifesto de uma execução a partir da config e do ambiente atual."""
    where = Path(repo_dir) if repo_dir is not None else Path(__file__).parent
    selected = set(config.experiments.models)
    models = [
        m.model_dump(mode="json") for m in config.llm.models if not selected or m.name in selected
    ]
    return RunManifest(
        run_id=run_id or new_run_id(),
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=config.mode.value,
        track=config.track.value,
        config_hash=hash_config(config),
        data_hash=hash_dataset(dataset) if dataset is not None else None,
        git_sha=git_sha(where),
        git_dirty=_git_dirty(where),
        python_version=sys.version.split()[0],
        julia_version=julia_version(),
        library_versions=library_versions(),
        seeds=dict(seeds or {"master": config.seeds.master}),
        hardware=hardware_info(),
        llm_models=models,
        inference={
            "backend": config.llm.backend,
            "temperature": config.llm.temperature,
            "max_new_tokens": config.llm.max_new_tokens,
            "max_retries": config.llm.max_retries,
            "failure_policy": config.llm.failure_policy,
        },
        prompt=prompt,
        timings=dict(timings or {}),
        interruption=interruption,
    )
