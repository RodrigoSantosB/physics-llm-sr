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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.validation import compute_data_hash

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
    data_hashes: dict[str, str]
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
    metrics: dict[str, Any]
    discovered_expressions: list[dict[str, Any]]
    history_file: str | None
    stop_conditions: dict[str, Any]


def new_run_id() -> str:
    """Identificador único: timestamp UTC + sufixo aleatório."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:RUN_ID_UUID_CHARS]}"


def hash_config(config: BaseModel) -> str:
    """SHA-256 do JSON canônico (chaves ordenadas) da configuração."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: Único algoritmo de hash de dataset do projeto (``data.validation``).
hash_dataset = compute_data_hash


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


def code_identity(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Identidade do código: ``git_sha``, ``git_dirty`` e ``git_diff_hash``.

    ``git_diff_hash`` = SHA-256 de ``git diff HEAD`` (mudanças rastreadas não commitadas;
    ``None`` se a árvore está limpa ou fora de um repositório). Arquivos não rastreados não
    entram no hash.
    """
    where = Path(repo_dir) if repo_dir is not None else Path(__file__).parent
    dirty = _git_dirty(where)
    diff = _run(["git", "diff", "HEAD"], where) if dirty else None
    return {
        "git_sha": git_sha(where),
        "git_dirty": dirty,
        "git_diff_hash": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }


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
    datasets: Mapping[str, Dataset] | None = None,
    seeds: dict[str, int] | None = None,
    prompt: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    interruption: str | None = None,
    metrics: dict[str, Any] | None = None,
    discovered_expressions: list[dict[str, Any]] | None = None,
    history_file: str | Path | None = None,
    backends: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    repo_dir: str | Path | None = None,
) -> RunManifest:
    """Monta o manifesto de uma execução a partir da config e do ambiente atual.

    ``datasets``: ``{rótulo dataset/condição: Dataset}`` (um hash por entrada, via
    :func:`hash_dataset`). ``backends``: ``{nome do modelo: backend}``; cada entrada de
    ``llm_models`` ganha ``effective`` = ``backend.model_metadata()`` (SHA/digest efetivos,
    após ``resolve_identity()`` quando existir). ``history_file`` é só o ponteiro para o
    histórico de busca; ``stop_conditions`` junta a regra de parada da config e a
    interrupção observada.
    """
    where = Path(repo_dir) if repo_dir is not None else Path(__file__).parent
    selected = set(config.experiments.models)
    models = [
        m.model_dump(mode="json") for m in config.llm.models if not selected or m.name in selected
    ]
    for entry in models:
        backend = (backends or {}).get(entry["name"])
        if backend is not None:
            resolve = getattr(backend, "resolve_identity", None)
            if callable(resolve):
                resolve()
            entry["effective"] = backend.model_metadata()
    return RunManifest(
        run_id=run_id or new_run_id(),
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=config.mode.value,
        track=config.track.value,
        config_hash=hash_config(config),
        data_hashes={k: hash_dataset(d) for k, d in (datasets or {}).items()},
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
        metrics=dict(metrics or {}),
        discovered_expressions=list(discovered_expressions or []),
        history_file=None if history_file is None else str(history_file),
        stop_conditions={
            "early_stopping": config.sr.early_stopping.model_dump(mode="json"),
            "interruption": interruption,
        },
    )
