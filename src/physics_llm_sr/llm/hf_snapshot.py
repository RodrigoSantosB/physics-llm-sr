"""Pré-download verificado dos pesos Hugging Face (corrige ``header too large``).

Arquivos ``.safetensors`` truncados (cache num Drive cheio/FUSE, disco sem espaço) só
falham tarde, dentro do ``from_pretrained``. Aqui o snapshot é baixado explicitamente, sem
os ``.bin`` duplicados, cada ``.safetensors`` é verificado (tamanho esperado ou cabeçalho) e
um arquivo inválido é apagado e rebaixado uma única vez; a segunda falha é um erro claro.
"""

from __future__ import annotations

import fnmatch
import importlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

from physics_llm_sr.llm.interfaces import BackendError

#: Só o necessário: configs, tokenizer e safetensors (os ``.bin`` duplicados somam ~15 GB).
ALLOW_PATTERNS: tuple[str, ...] = ("*.json", "*.safetensors", "tokenizer*", "*.model")
SAFETENSORS_SUFFIX = ".safetensors"
HEADER_LEN_BYTES = 8
MAX_HEADER_BYTES = 100 * 1024 * 1024
FREE_SPACE_MARGIN_BYTES = 1024**3
HF_HOME_ENV = "HF_HOME"
DEFAULT_HF_HOME = "~/.cache/huggingface"
_LOCAL_HINT = (
    "use o disco local do Colab (HF_HOME_DIR = '/content/hf_home'), libere espaço e, se o "
    "cache antigo estava no Drive, apague-o (rm -rf) e reinicie o ambiente"
)


def hf_home() -> str:
    """HF_HOME em uso (variável de ambiente ou o padrão do Hugging Face)."""
    return os.environ.get(HF_HOME_ENV) or str(Path(DEFAULT_HF_HOME).expanduser())


def header_problem(path: Path) -> str | None:
    """Valida o cabeçalho safetensors (u64 LE N, N < tamanho e < 100 MB, N bytes JSON)."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(HEADER_LEN_BYTES)
        if len(raw) < HEADER_LEN_BYTES:
            return f"arquivo com {size} bytes (menor que o prefixo do cabeçalho)"
        (length,) = struct.unpack("<Q", raw)
        if length >= size or length >= MAX_HEADER_BYTES:
            return f"cabeçalho declara {length} bytes (arquivo com {size}): header too large"
        try:
            json.loads(handle.read(length))
        except ValueError:
            return "cabeçalho não é JSON válido"
    return None


def _problem(path: Path, expected: dict[str, int]) -> str | None:
    want = expected.get(path.name)
    got = path.stat().st_size
    if want is not None and got != want:
        return f"tamanho esperado {want} bytes, obtido {got}"
    problem = header_problem(path)
    return f"tamanho esperado {want}, obtido {got}; {problem}" if problem else None


def _expected_sizes(hub: Any, repo_id: str, revision: str | None) -> dict[str, int]:
    """Tamanhos dos arquivos a baixar (vazio se a consulta falhar, ex.: offline)."""
    try:
        info = hub.HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        return {
            str(s.rfilename): int(s.size)
            for s in info.siblings
            if s.size is not None
            and any(fnmatch.fnmatch(str(s.rfilename), p) for p in ALLOW_PATTERNS)
        }
    except Exception:  # offline/sem rede: cai na validação do cabeçalho
        return {}


def _check_space(hub: Any, repo_id: str, revision: str | None, sizes: dict[str, int]) -> None:
    missing = 0
    for name, size in sizes.items():
        cached = hub.try_to_load_from_cache(repo_id, name, revision=revision)
        if not isinstance(cached, str):
            missing += size
    if not missing:
        return
    probe = Path(hf_home()).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    need = missing + FREE_SPACE_MARGIN_BYTES
    if free < need:
        raise BackendError(
            f"espaço insuficiente em HF_HOME={hf_home()}: livre {free} bytes, necessário "
            f"{need} bytes para {repo_id}; {_LOCAL_HINT}"
        )


def _download(hub: Any, repo_id: str, revision: str | None, **extra: Any) -> Path:
    return Path(
        hub.snapshot_download(
            repo_id, revision=revision, allow_patterns=list(ALLOW_PATTERNS), **extra
        )
    )


def _invalid(snapshot: Path, expected: dict[str, int]) -> dict[Path, str]:
    return {
        p: msg
        for p in sorted(snapshot.rglob(f"*{SAFETENSORS_SUFFIX}"))
        if (msg := _problem(p, expected))
    }


def _remove(path: Path) -> None:
    blob = path.resolve() if path.is_symlink() else path
    path.unlink(missing_ok=True)
    blob.unlink(missing_ok=True)


def download_verified_snapshot(repo_id: str, revision: str | None) -> Path:
    """Baixa e verifica o snapshot; devolve o diretório local para ``from_pretrained``."""
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise BackendError("huggingface_hub indisponível: instale o extra [hf]") from exc
    expected = _expected_sizes(hub, repo_id, revision)
    _check_space(hub, repo_id, revision, expected)
    snapshot = _download(hub, repo_id, revision)
    bad = _invalid(snapshot, expected)
    if not bad:
        return snapshot
    for path in bad:
        _remove(path)
    snapshot = _download(hub, repo_id, revision, force_download=True)
    bad = _invalid(snapshot, expected)
    if bad:
        detail = "; ".join(f"{p.name}: {msg}" for p, msg in bad.items())
        raise BackendError(
            f"pesos de {repo_id} corrompidos após novo download ({detail}). "
            f"HF_HOME em uso: {hf_home()}. Dica: {_LOCAL_HINT}"
        )
    return snapshot
