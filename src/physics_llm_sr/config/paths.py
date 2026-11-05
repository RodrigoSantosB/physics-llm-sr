"""Raiz de projeto configurável para caminhos relativos (cache, saídas), útil no Colab.

``cache_dir`` e ``output_dir`` relativos são resolvidos contra :func:`project_root`:
variável de ambiente :data:`PROJECT_ROOT_ENV` (ex.: ``/content/drive/MyDrive/physics-llm-sr``
no notebook, para que cache e resultados sobrevivam à sessão) e, na ausência dela, o
diretório de trabalho atual. Caminhos absolutos não são alterados.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "PHYSICS_LLM_SR_PROJECT_ROOT"


def project_root() -> Path:
    """Raiz de projeto: ``$PHYSICS_LLM_SR_PROJECT_ROOT`` ou o diretório atual."""
    env = os.environ.get(PROJECT_ROOT_ENV)
    return Path(env).expanduser().resolve() if env else Path.cwd().resolve()


def resolve_project_path(path: str | Path) -> Path:
    """Caminho absoluto: relativos são ancorados em :func:`project_root`."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else project_root() / p
