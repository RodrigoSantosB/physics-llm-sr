"""Histórico da busca: acumula ``GenerationRecord``, DataFrame e JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pandas as pd

from physics_llm_sr.sr.interfaces import GenerationRecord

RECORD_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(GenerationRecord))
CONFIG_COLUMN = "config"


class SearchHistory:
    """Registros por geração de uma execução, rotulada por ``config``.

    O JSON usa o dialeto do Python (``Infinity``/``NaN`` para perdas inválidas).
    """

    def __init__(self, config: str, metadata: dict[str, Any] | None = None) -> None:
        """``config`` rotula a execução (ex.: ``algoritmo|modelo|prompt``)."""
        self.config = config
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._records: list[GenerationRecord] = []

    def __len__(self) -> int:
        """Número de gerações registradas."""
        return len(self._records)

    @property
    def records(self) -> tuple[GenerationRecord, ...]:
        """Registros em ordem de inserção."""
        return tuple(self._records)

    @property
    def best_loss(self) -> float:
        """Menor ``best_loss`` registrada (``+inf`` se vazio)."""
        return min((r.best_loss for r in self._records), default=float("inf"))

    def add(self, record: GenerationRecord) -> None:
        """Acrescenta o registro de uma geração."""
        self._records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Colunas ``config`` + campos de ``GenerationRecord`` (compatível com o plotter)."""
        rows = [{CONFIG_COLUMN: self.config, **asdict(r)} for r in self._records]
        frame: pd.DataFrame = pd.DataFrame(rows, columns=[CONFIG_COLUMN, *RECORD_COLUMNS])
        return frame

    def to_json(self) -> str:
        """Serialização JSON."""
        return json.dumps(
            {
                "config": self.config,
                "metadata": self.metadata,
                "records": [asdict(r) for r in self._records],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> SearchHistory:
        """Inverso de :meth:`to_json`."""
        data = json.loads(text)
        hist = cls(data["config"], data.get("metadata"))
        for r in data["records"]:
            hist.add(GenerationRecord(**r))
        return hist

    def save(self, path: Path | str) -> Path:
        """Grava o JSON em ``path`` (cria diretórios)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str) -> SearchHistory:
        """Lê um histórico gravado por :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
