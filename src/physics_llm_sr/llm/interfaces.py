"""Contrato público do avaliador LLM (PRD §6.1, §6.3).

Consumidores (função de perda, adaptadores SR) dependem apenas deste módulo:

- :class:`EquationEvaluation`: os três escores em ``[0, 1]`` e o feedback textual.
- :class:`PhysicsEquationEvaluator`: ``evaluate`` (uma expressão) e ``evaluate_many``
  (lote, ordem preservada). Implementações: ``scoring.LLMEquationEvaluator`` (LLM real ou
  backend simulado) e ``scoring.FakeEquationEvaluator`` (tabela/função determinística).
- :class:`TextGenerationBackend`: geração de texto em lote por pares ``(system, user)``.

Nenhuma implementação pode devolver nota substituta: falha de avaliação levanta
:class:`EvaluationFailedError`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.symbolic.expression import Expr

#: Chaves dos escores, na ordem do artigo (dim corr, simp, sim).
SCORE_KEYS: tuple[str, ...] = ("dimensional_consistency", "simplicity", "physical_realism")
#: Chave do feedback textual.
FEEDBACK_KEY = "feedback"
#: Conjunto exato de chaves da resposta (PRD §6.3).
RESPONSE_KEYS: frozenset[str] = frozenset((*SCORE_KEYS, FEEDBACK_KEY))
#: Limites fechados de cada escore.
SCORE_MIN = 0.0
SCORE_MAX = 1.0


class LLMError(RuntimeError):
    """Base dos erros do subpacote ``llm``."""


class InvalidScoreValueError(LLMError, ValueError):
    """Escore fora de ``[0, 1]``, não finito ou de tipo não numérico."""


class EvaluationFailedError(LLMError):
    """A avaliação não produziu resposta válida; nenhuma nota substituta é atribuída."""


class BackendConfigurationError(LLMError, ValueError):
    """Configuração de backend recusada (ex.: ``trust_remote_code``, revisão ausente)."""


class BackendError(LLMError):
    """Falha do mecanismo de inferência (rede, contagem de respostas, dependência)."""


def validate_score(name: str, value: object) -> float:
    """Escore validado: ``value`` como ``float`` real, finito e em ``[0, 1]``.

    ``bool`` é recusado apesar de ser subclasse de ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidScoreValueError(f"{name} deve ser número, recebeu {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidScoreValueError(f"{name} não finito: {number!r}")
    if not SCORE_MIN <= number <= SCORE_MAX:
        raise InvalidScoreValueError(f"{name}={number!r} fora de [{SCORE_MIN}, {SCORE_MAX}]")
    return number


@dataclass(frozen=True)
class EquationEvaluation:
    """Avaliação física de uma equação candidata (escores em ``[0, 1]``)."""

    dimensional_consistency: float
    simplicity: float
    physical_realism: float
    feedback: str

    def __post_init__(self) -> None:
        """Valida os três escores e o tipo do feedback."""
        for key in SCORE_KEYS:
            object.__setattr__(self, key, validate_score(key, getattr(self, key)))
        if not isinstance(self.feedback, str):
            raise InvalidScoreValueError("feedback deve ser str")

    @property
    def scores(self) -> tuple[float, float, float]:
        """Os três escores na ordem de :data:`SCORE_KEYS`."""
        return (self.dimensional_consistency, self.simplicity, self.physical_realism)

    def to_dict(self) -> dict[str, Any]:
        """Representação JSON-serializável."""
        return {
            "dimensional_consistency": self.dimensional_consistency,
            "simplicity": self.simplicity,
            "physical_realism": self.physical_realism,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EquationEvaluation:
        """Reconstrói a partir de :meth:`to_dict` (com validação)."""
        return cls(
            dimensional_consistency=data["dimensional_consistency"],
            simplicity=data["simplicity"],
            physical_realism=data["physical_realism"],
            feedback=data["feedback"],
        )


@runtime_checkable
class PhysicsEquationEvaluator(Protocol):
    """Avaliador físico de equações (PRD §6.1)."""

    def evaluate(
        self, expression: Expr, physical_context: PhysicalContext, prompt_variant: str
    ) -> EquationEvaluation:
        """Avalia uma expressão; levanta :class:`EvaluationFailedError` se falhar."""
        ...

    def evaluate_many(
        self,
        expressions: Sequence[Expr],
        physical_context: PhysicalContext,
        prompt_variant: str,
    ) -> list[EquationEvaluation]:
        """Avalia um lote; o i-ésimo resultado corresponde à i-ésima expressão."""
        ...


@runtime_checkable
class TextGenerationBackend(Protocol):
    """Mecanismo de inferência desacoplado (Ollama, transformers, simulado)."""

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Gera uma resposta por par ``(system, user)``, na mesma ordem."""
        ...

    def model_metadata(self) -> dict[str, Any]:
        """Identificação reprodutível do modelo.

        Chaves mínimas: ``backend``, ``checkpoint``, ``revision``, ``quantization``, ``infra``.
        """
        ...
