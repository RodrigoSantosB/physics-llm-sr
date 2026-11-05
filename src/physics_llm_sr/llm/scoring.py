"""Avaliadores físicos e penalidade ``c`` da perda composta (PRD §5, §6.3, §8.3, §15).

- :func:`penalty`: ``c = 1 - (c1 + c2 + c3) / 3`` (0 = melhor, 1 = pior).
- :class:`LLMEquationEvaluator`: prompt → backend → parser, com cache, deduplicação
  canônica em lote, retry só para resposta inválida e métricas.
- :class:`FakeEquationEvaluator`: determinístico (tabela por ``canonical_key`` com default
  explícito, ou função) para testes das ondas seguintes.

Política de falha (``failure_policy``):

- ``"retry"``: até ``max_retries`` novas tentativas com o mesmo prompt para as expressões
  cuja resposta foi inválida; persistindo, :class:`EvaluationFailedError`.
- ``"abort"``: :class:`EvaluationFailedError` na primeira resposta inválida.

Em ambos os casos não há nota substituta: quem chama decide interromper a execução
primária. Erros do backend (rede, dependência) não são reexecutados e propagam.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.interfaces import (
    BackendError,
    EquationEvaluation,
    EvaluationFailedError,
    TextGenerationBackend,
)
from physics_llm_sr.llm.prompting import PromptLibrary, RenderedPrompt
from physics_llm_sr.llm.response_parser import InvalidResponseError, parse_evaluation
from physics_llm_sr.symbolic.canonicalization import canonical_key
from physics_llm_sr.symbolic.expression import Expr

logger = logging.getLogger(__name__)

FailurePolicy = Literal["retry", "abort"]
FAILURE_POLICIES: tuple[str, ...] = ("retry", "abort")
#: Default de novas tentativas quando não vem da config (``llm.max_retries``).
DEFAULT_MAX_RETRIES = 2
#: Número de escores combinados na penalidade.
N_SCORES = 3


def penalty(evaluation: EquationEvaluation) -> float:
    """Penalidade ``c = 1 - (c1 + c2 + c3) / 3`` em ``[0, 1]`` (GAP-02, modo literal)."""
    return 1.0 - sum(evaluation.scores) / N_SCORES


@dataclass
class EvaluatorMetrics:
    """Contadores de custo (PRD §8.3, §15)."""

    requested: int = 0
    deduplicated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    backend_calls: int = 0
    batches: int = 0
    invalid_responses: int = 0
    failures: int = 0
    inference_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Snapshot serializável."""
        return asdict(self)


@dataclass(frozen=True)
class _Item:
    key: str
    expression: Expr
    prompt: RenderedPrompt


class LLMEquationEvaluator:
    """Implementa ``PhysicsEquationEvaluator`` sobre um ``TextGenerationBackend``."""

    def __init__(
        self,
        backend: TextGenerationBackend,
        prompt_library: PromptLibrary,
        cache: Any = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        failure_policy: str = "retry",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Configura o avaliador.

        Args:
            backend: mecanismo de inferência.
            prompt_library: templates e variantes A-H.
            cache: ``EvaluationCache`` opcional.
            max_retries: novas tentativas por expressão com resposta inválida (``>= 0``).
            failure_policy: ``"retry"`` ou ``"abort"``.
            clock: relógio para medir o tempo de inferência.
        """
        if max_retries < 0:
            raise ValueError("max_retries deve ser >= 0")
        if failure_policy not in FAILURE_POLICIES:
            raise ValueError(f"failure_policy deve ser um de {FAILURE_POLICIES}")
        self.backend = backend
        self.prompt_library = prompt_library
        self.cache = cache
        self.max_retries = max_retries
        self.failure_policy = failure_policy
        self._clock = clock
        self.metrics = EvaluatorMetrics()

    @property
    def max_attempts(self) -> int:
        """Tentativas por expressão (1 em ``abort``; ``1 + max_retries`` em ``retry``)."""
        return 1 if self.failure_policy == "abort" else 1 + self.max_retries

    def evaluate(
        self, expression: Expr, physical_context: PhysicalContext, prompt_variant: str
    ) -> EquationEvaluation:
        """Avalia uma expressão (ver :meth:`evaluate_many`)."""
        return self.evaluate_many([expression], physical_context, prompt_variant)[0]

    def evaluate_many(
        self,
        expressions: Sequence[Expr],
        physical_context: PhysicalContext,
        prompt_variant: str,
    ) -> list[EquationEvaluation]:
        """Avalia um lote, deduplicando por forma canônica; ordem preservada.

        Raises:
            EvaluationFailedError: alguma expressão sem resposta válida após a política.
            BackendError: falha do backend (não reexecutada).
        """
        self.metrics.requested += len(expressions)
        order: list[str] = []
        unique: dict[str, _Item] = {}
        for expr in expressions:
            key = canonical_key(expr)
            order.append(key)
            if key in unique:
                self.metrics.deduplicated += 1
                continue
            prompt = self.prompt_library.render(expr, physical_context, prompt_variant)
            unique[key] = _Item(key, expr, prompt)
        results: dict[str, EquationEvaluation] = {}
        pending = list(unique.values())
        errors: dict[str, list[str]] = {}
        attempts = 0
        while pending and attempts < self.max_attempts:
            attempts += 1
            pending = self._attempt(pending, results, errors)
        if pending:
            self.metrics.failures += len(pending)
            detail = "; ".join(f"{item.key}: {errors[item.key][-1]}" for item in pending)
            raise EvaluationFailedError(
                f"avaliação LLM falhou para {len(pending)} de {len(unique)} expressões únicas "
                f"após {attempts} tentativa(s) (política {self.failure_policy}): {detail}"
            )
        return [results[key] for key in order]

    def _attempt(
        self,
        pending: list[_Item],
        results: dict[str, EquationEvaluation],
        errors: dict[str, list[str]],
    ) -> list[_Item]:
        prompts = [(item.prompt.system, item.prompt.user) for item in pending]
        start = self._clock()
        responses = self.backend.generate(prompts)
        self.metrics.inference_seconds += self._clock() - start
        self.metrics.batches += 1
        self.metrics.backend_calls += len(prompts)
        if len(responses) != len(prompts):
            raise BackendError(
                f"backend devolveu {len(responses)} respostas para {len(prompts)} prompts"
            )
        still: list[_Item] = []
        for item, response in zip(pending, responses, strict=True):
            logger.debug("resposta LLM [%s]: %r", item.key, response)
            try:
                results[item.key] = parse_evaluation(response)
            except InvalidResponseError as exc:
                self.metrics.invalid_responses += 1
                message = f"{type(exc).__name__}: {exc}"
                errors.setdefault(item.key, []).append(message)
                logger.warning("resposta LLM inválida [%s]: %s", item.key, message)
                still.append(item)
        return still


class FakeEquationEvaluator:
    """Avaliador simulado determinístico (sem LLM).

    Modo tabela: ``canonical_key(expr)`` → avaliação; chave ausente usa ``default`` se
    fornecido explicitamente, senão levanta :class:`EvaluationFailedError`.
    Modo função: ``function(expr, ctx, variant)``.
    """

    def __init__(
        self,
        table: Mapping[str | Expr, EquationEvaluation] | None = None,
        default: EquationEvaluation | None = None,
        function: Callable[[Expr, PhysicalContext, str], EquationEvaluation] | None = None,
    ) -> None:
        """Use :meth:`from_table` ou :meth:`from_function`."""
        if (table is None) == (function is None):
            raise ValueError("forneça exatamente um de table ou function")
        self._table = {
            (k if isinstance(k, str) else canonical_key(k)): v for k, v in (table or {}).items()
        }
        self._default = default
        self._function = function
        self.calls = 0

    @classmethod
    def from_table(
        cls,
        table: Mapping[str | Expr, EquationEvaluation],
        default: EquationEvaluation | None = None,
    ) -> FakeEquationEvaluator:
        """Tabela por forma canônica (chaves ``Expr`` ou ``canonical_key``)."""
        return cls(table=table, default=default)

    @classmethod
    def from_function(
        cls, function: Callable[[Expr, PhysicalContext, str], EquationEvaluation]
    ) -> FakeEquationEvaluator:
        """Avaliação calculada por ``function``."""
        return cls(function=function)

    def evaluate(
        self, expression: Expr, physical_context: PhysicalContext, prompt_variant: str
    ) -> EquationEvaluation:
        """Avaliação determinística de uma expressão."""
        self.calls += 1
        if self._function is not None:
            return self._function(expression, physical_context, prompt_variant)
        key = canonical_key(expression)
        if key in self._table:
            return self._table[key]
        if self._default is not None:
            return self._default
        raise EvaluationFailedError(f"FakeEquationEvaluator: sem avaliação para {key}")

    def evaluate_many(
        self,
        expressions: Sequence[Expr],
        physical_context: PhysicalContext,
        prompt_variant: str,
    ) -> list[EquationEvaluation]:
        """Avalia cada expressão em ordem."""
        return [self.evaluate(e, physical_context, prompt_variant) for e in expressions]
