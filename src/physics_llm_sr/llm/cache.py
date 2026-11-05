"""Cache de avaliações LLM em memória com persistência JSONL (PRD §6.3, §15, GAP-14).

Chave (:func:`cache_key`): SHA-256 de ``[expression_key(expr), identidade do modelo,
variante#hash_do_prompt, hash_do_contexto]``. :func:`expression_key` é a forma canônica
com constantes arredondadas a ``significant_digits`` algarismos significativos (default 3,
hipótese GAP-14 = precisão publicada; vale para todos os algoritmos) no modo ``exact``, ou
só a estrutura (constantes mascaradas) no modo ``structure`` — **desvio**, permitido apenas
em ``extended_analysis`` (``llm.cache_key_mode``). A identidade (:func:`model_identity`) reúne
backend, checkpoint, revisão efetiva (SHA do Hub; digest no Ollama) e quantização; o
backend simulado tem identidade própria que nunca coincide com a de um modelo real.
O escore não depende dos dados, então a mesma entrada vale entre repetições, níveis de
ruído e algoritmos.

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
from typing import Any, Literal

from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.config.schemas import LLMConfig
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.interfaces import BackendError, EquationEvaluation, InvalidScoreValueError
from physics_llm_sr.symbolic.canonicalization import canonical_key
from physics_llm_sr.symbolic.expression import Const, Expr, Op

logger = logging.getLogger(__name__)

#: Nome do arquivo JSONL dentro de ``llm.cache_dir``.
CACHE_FILENAME = "evaluations.jsonl"
STATUS_OK = "ok"
STATUS_ERROR = "error"
#: Versão do formato da chave/registro (entra na chave).
#: 3: chave com algarismos significativos/modo (GAP-14); 4: identidade com batch_size e
#: prompt_format (lote com padding e template mudam a saída do transformers).
CACHE_FORMAT_VERSION = "4"
FAKE_BACKEND = "fake"
OLLAMA_BACKEND = "ollama"

CacheKeyMode = Literal["exact", "structure"]
CACHE_KEY_MODES: tuple[str, ...] = ("exact", "structure")
#: Algarismos significativos das constantes na chave (GAP-14: precisão publicada).
DEFAULT_CACHE_SIGNIFICANT_DIGITS = 3
#: ``exact``: constantes arredondadas; ``structure``: constantes mascaradas (desvio).
DEFAULT_CACHE_KEY_MODE: CacheKeyMode = "exact"
#: Valor que substitui toda constante no modo ``structure`` (antes de ordenar os filhos).
STRUCTURE_MASK_VALUE = 1.0
#: Prefixo da chave estrutural (nunca colide com uma chave ``exact``).
STRUCTURE_KEY_PREFIX = "structure:"


def _mask_constants(expr: Expr) -> Expr:
    if isinstance(expr, Const):
        return Const(STRUCTURE_MASK_VALUE)
    if isinstance(expr, Op):
        return Op(expr.name, tuple(_mask_constants(c) for c in expr.children))
    return expr


def validate_key_options(significant_digits: int, key_mode: str) -> None:
    """Valida ``significant_digits >= 1`` e ``key_mode`` em :data:`CACHE_KEY_MODES`.

    Raises:
        ValueError: opção inválida.
    """
    if significant_digits < 1:
        raise ValueError("significant_digits deve ser >= 1")
    if key_mode not in CACHE_KEY_MODES:
        raise ValueError(f"key_mode deve ser um de {CACHE_KEY_MODES}, não {key_mode!r}")


def expression_key(
    expression: Expr,
    significant_digits: int = DEFAULT_CACHE_SIGNIFICANT_DIGITS,
    key_mode: str = DEFAULT_CACHE_KEY_MODE,
) -> str:
    """Chave textual da expressão para cache e deduplicação de avaliações LLM.

    ``exact``: ``canonical_key`` com constantes a ``significant_digits`` algarismos.
    ``structure``: toda constante vira :data:`STRUCTURE_MASK_VALUE` antes da canonização
    (a ordem dos filhos comutativos não depende das constantes) e a chave recebe o
    prefixo :data:`STRUCTURE_KEY_PREFIX`.

    Raises:
        ValueError: ``significant_digits < 1`` ou ``key_mode`` desconhecido.
    """
    validate_key_options(significant_digits, key_mode)
    if key_mode == "structure":
        return STRUCTURE_KEY_PREFIX + canonical_key(_mask_constants(expression))
    return canonical_key(expression, significant_digits)


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def context_hash(physical_context: PhysicalContext) -> str:
    """SHA-256 de todos os campos do contexto físico."""
    return _sha256_json(dataclasses.asdict(physical_context))


def model_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Identidade efetiva do modelo a partir de ``backend.model_metadata()``.

    Campos: ``backend``, ``checkpoint``, ``revision`` (``digest`` do Ollama tem precedência
    sobre a revisão configurada), ``quantization``, ``batch_size`` e ``prompt_format``
    (transformers: o lote com padding à esquerda e o formato do prompt — chat template com
    ou sem ``system`` ou texto simples — alteram a saída em fp16; ``None`` nos demais).
    O backend simulado recebe o prefixo ``fake::`` no checkpoint. Ollama sem digest nem
    revisão é recusado: o mesmo tag pode apontar para pesos diferentes e o cache serviria
    resultados de outro modelo.
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
        "batch_size": metadata.get("batch_size"),
        "prompt_format": metadata.get("prompt_format"),
    }


def cache_key(
    expression: Expr,
    identity: Mapping[str, Any],
    prompt_variant: str,
    prompt_hash: str,
    context_digest: str,
    *,
    significant_digits: int = DEFAULT_CACHE_SIGNIFICANT_DIGITS,
    key_mode: str = DEFAULT_CACHE_KEY_MODE,
) -> str:
    """Chave hexadecimal (SHA-256) de uma avaliação (``identity`` de :func:`model_identity`).

    A expressão entra via :func:`expression_key` (``significant_digits``, ``key_mode``).
    """
    return _sha256_json(
        [
            CACHE_FORMAT_VERSION,
            expression_key(expression, significant_digits, key_mode),
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
        """Cache persistido em ``llm.cache_dir``; ``None`` se ``llm.cache_enabled`` é falso.

        ``cache_dir`` relativo é resolvido contra a raiz de projeto (``config.paths``).
        """
        if not llm_config.cache_enabled:
            return None
        return cls(resolve_project_path(llm_config.cache_dir) / CACHE_FILENAME)

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
