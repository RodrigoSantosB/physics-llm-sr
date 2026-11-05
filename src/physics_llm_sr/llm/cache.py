"""Cache de avaliações LLM em memória com persistência JSONL (PRD §6.3, §15, GAP-14).

Chave (:func:`cache_key`): SHA-256 de ``[canonical_key(expr), identidade do modelo,
variante#hash_do_prompt, hash_do_contexto]``. A identidade (:func:`model_identity`) reúne
backend, checkpoint, revisão efetiva (SHA do Hub; digest no Ollama) e quantização; o
backend simulado tem identidade própria que nunca coincide com a de um modelo real.
O escore não depende dos dados, então a mesma entrada vale entre repetições, níveis de ruído e algoritmos.

Persistência: um registro JSON por linha, só acrescentado (``status = "ok"`` com a
avaliação e a resposta crua; ``status = "error"`` com a mensagem e as respostas cruas).
Na releitura, o último registro ``ok`` de cada chave vale; erros ficam só para auditoria e
**nunca** são servidos como acerto (a expressão volta a ser avaliada). Linhas corrompidas
(ex.: escrita interrompida) ou com escores fora do contrato são ignoradas e contadas em
``corrupted_lines``; isso apenas causa nova avaliação.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from physics_llm_sr.config.schemas import LLMConfig
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.interfaces import BackendError, EquationEvaluation, InvalidScoreValueError
from physics_llm_sr.symbolic.canonicalization import canonical_key
from physics_llm_sr.symbolic.expression import Expr

logger = logging.getLogger(__name__)

#: Nome do arquivo JSONL dentro de ``llm.cache_dir``.
CACHE_FILENAME = "evaluations.jsonl"
STATUS_OK = "ok"
STATUS_ERROR = "error"
#: Versão do formato da chave/registro (entra na chave).
CACHE_FORMAT_VERSION = "2"
FAKE_BACKEND = "fake"
OLLAMA_BACKEND = "ollama"


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def context_hash(physical_context: PhysicalContext) -> str:
    """SHA-256 de todos os campos do contexto físico."""
    return _sha256_json(dataclasses.asdict(physical_context))


def model_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Identidade efetiva do modelo a partir de ``backend.model_metadata()``.

    Campos: ``backend``, ``checkpoint``, ``revision`` (``digest`` do Ollama tem precedência
    sobre a revisão configurada) e ``quantization``. O backend simulado recebe o prefixo
    ``fake::`` no checkpoint. Ollama sem digest nem revisão é recusado: o mesmo tag pode
    apontar para pesos diferentes e o cache serviria resultados de outro modelo.
    """
    backend = str(metadata.get("backend"))
    checkpoint = str(metadata.get("checkpoint"))
    revision = metadata.get("digest") or metadata.get("revision")
    if backend == FAKE_BACKEND:
        checkpoint = f"{FAKE_BACKEND}::{checkpoint}"
    if backend == OLLAMA_BACKEND and not revision:
        raise BackendError(
            f"digest do modelo Ollama {checkpoint!r} indisponível (servidor inacessível ou "
            "modelo ausente): sem identidade efetiva o cache não é seguro"
        )
    quantization = metadata.get("quantization")
    return {
        "backend": backend,
        "checkpoint": checkpoint,
        "revision": None if revision is None else str(revision),
        "quantization": None if quantization is None else str(quantization),
    }


def cache_key(
    expression: Expr,
    identity: Mapping[str, Any],
    prompt_variant: str,
    prompt_hash: str,
    context_digest: str,
) -> str:
    """Chave hexadecimal (SHA-256) de uma avaliação (``identity`` de :func:`model_identity`)."""
    return _sha256_json(
        [
            CACHE_FORMAT_VERSION,
            canonical_key(expression),
            dict(identity),
            [prompt_variant, prompt_hash],
            context_digest,
        ]
    )


@dataclass
class CacheStats:
    """Contadores de consulta ao cache."""

    hits: int = 0
    misses: int = 0

    @property
    def calls_avoided(self) -> int:
        """Chamadas ao backend evitadas pelo cache (= acertos)."""
        return self.hits


class EvaluationCache:
    """Cache de :class:`EquationEvaluation` por chave, opcionalmente persistido em JSONL."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Sem ``path``: só memória. Com ``path``: carrega o arquivo existente e acrescenta."""
        self.path = Path(path) if path is not None else None
        self._entries: dict[str, EquationEvaluation] = {}
        self.stats = CacheStats()
        self.error_count = 0
        self.corrupted_lines = 0
        if self.path is not None and self.path.exists():
            self._load(self.path)

    @classmethod
    def from_config(cls, llm_config: LLMConfig) -> EvaluationCache | None:
        """Cache persistido em ``llm.cache_dir``; ``None`` se ``llm.cache_enabled`` é falso."""
        if not llm_config.cache_enabled:
            return None
        return cls(Path(llm_config.cache_dir) / CACHE_FILENAME)

    def _load(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    status = record["status"]
                    if status == STATUS_OK:
                        evaluation = EquationEvaluation.from_dict(record["evaluation"])
                        self._entries[str(record["key"])] = evaluation
                    elif status == STATUS_ERROR:
                        self.error_count += 1
                    else:
                        raise ValueError(f"status desconhecido {status!r}")
                except (ValueError, KeyError, TypeError, InvalidScoreValueError) as exc:
                    self.corrupted_lines += 1
                    logger.warning("cache %s linha %d ignorada: %s", path, lineno, exc)

    def __len__(self) -> int:
        """Número de avaliações válidas em memória."""
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        """Pertinência sem afetar contadores."""
        return key in self._entries

    def get(self, key: str) -> EquationEvaluation | None:
        """Avaliação em cache (conta acerto/erro)."""
        found = self._entries.get(key)
        if found is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return found

    def _append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def put(
        self,
        key: str,
        evaluation: EquationEvaluation,
        *,
        raw_response: str,
        metadata: dict[str, Any],
    ) -> None:
        """Registra uma avaliação válida (memória + disco)."""
        self._entries[key] = evaluation
        self._append(
            {
                "key": key,
                "status": STATUS_OK,
                "evaluation": evaluation.to_dict(),
                "raw_response": raw_response,
                "metadata": metadata,
            }
        )

    def record_error(
        self,
        key: str,
        error: str,
        *,
        raw_responses: list[str],
        metadata: dict[str, Any],
    ) -> None:
        """Registra uma falha de avaliação (auditoria; nunca servida como acerto)."""
        self.error_count += 1
        self._append(
            {
                "key": key,
                "status": STATUS_ERROR,
                "error": error,
                "raw_responses": raw_responses,
                "metadata": metadata,
            }
        )
