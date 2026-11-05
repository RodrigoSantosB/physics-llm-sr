"""Carga de configuração YAML com erros que nomeiam o campo inválido."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from physics_llm_sr.config.schemas import ExperimentConfig


class ConfigError(ValueError):
    """Configuração ausente ou inválida."""


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        path = ".".join(str(p) for p in err["loc"]) or "<raiz>"
        lines.append(f"{path}: {err['msg']}")
    return "; ".join(lines)


def parse_config(data: dict[str, Any]) -> ExperimentConfig:
    """Valida um dicionário e devolve a configuração tipada."""
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"configuração inválida: {_format_errors(exc)}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    """Lê um arquivo YAML e devolve a configuração tipada."""
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"arquivo de configuração não encontrado: {file}")
    try:
        with file.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML inválido em {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{file}: o YAML deve ser um mapeamento no nível raiz")
    return parse_config(data)
