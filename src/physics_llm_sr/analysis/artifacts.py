"""Registro de artefatos gerados (insumo do ``INDEX.md``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    """Um arquivo gerado, a tabela do artigo correspondente e o rótulo de execução."""

    artifact_id: str
    path: Path
    kind: str
    paper_ref: str
    label: str
    description: str = ""
