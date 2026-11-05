"""Perda composta ``L = w1*e + w2*s + w3*c`` (PRD §5; GAP-01, GAP-02, GAP-05).

- ``e``: erro (MSE, ou Huber para PySR) entre predição e ``y``.
- ``s``: tamanho da expressão em nós.
- ``c``: ``penalty(avaliação LLM) = 1 - (c1+c2+c3)/3``.

Modos de escala (GAP-02): ``literal`` (e em SI, s = nós) e ``normalized``
(e = MSE/Var(y), s = nós/max_nodes; **desvio**, registrado em ``metadata``).

Convenções explícitas:

- ``w3 == 0``: o avaliador nunca é chamado; ``LossBreakdown.c = 0.0`` e
  ``evaluation = None``, rotulado em ``diagnostics[i]["c_status"] =
  "not_evaluated_w3_zero"``. ``total`` não depende de ``c``.
- ``e`` não finito: candidato inválido (``total = +inf``, ``e = +inf``, ``c = NaN``,
  ``evaluation = None``), ``diagnostics[i]["invalid"] = True``, contado em
  ``counters.invalid`` e **não** enviado ao LLM. ``LossBreakdown`` não tem campo
  ``invalid``; o diagnóstico vive em ``CombinedObjective.diagnostics`` (um dict por
  expressão da última chamada, mesma ordem).
- ``EvaluationFailedError`` do avaliador propaga: nunca há nota substituta.
- Deduplicação do lote por :func:`llm.cache.expression_key` com ``significant_digits``
  (default 3, GAP-14) e ``key_mode`` (``exact`` | ``structure``; ``structure`` mascara as
  constantes e é desvio, só em ``extended_analysis``). Expressões com a mesma chave
  recebem a avaliação do primeiro representante do lote.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from physics_llm_sr.config.schemas import WEIGHT_SUM_TOLERANCE, ExperimentConfig
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.cache import (
    DEFAULT_CACHE_KEY_MODE,
    DEFAULT_CACHE_SIGNIFICANT_DIGITS,
    expression_key,
    validate_key_options,
)
from physics_llm_sr.llm.interfaces import PhysicsEquationEvaluator
from physics_llm_sr.llm.scoring import penalty
from physics_llm_sr.sr.interfaces import FloatArray, LossBreakdown
from physics_llm_sr.symbolic.expression import Expr, evaluate, size
from physics_llm_sr.symbolic.operators import DEFAULT_PROTECTION, ProtectionParams

Normalization = Literal["literal", "normalized"]
ErrorMetric = Literal["mse", "huber"]
NORMALIZATIONS: tuple[str, ...] = ("literal", "normalized")
ERROR_METRICS: tuple[str, ...] = ("mse", "huber")
#: Variância mínima de ``y`` aceita no modo normalized.
MIN_TARGET_VARIANCE = 1e-12
#: Fator do termo quadrático de Huber (``0.5 * r^2``).
HUBER_QUADRATIC_FACTOR = 0.5
#: Prompt padrão (variante E: contexto completo sem GT revelada é decidido nos prompts).
DEFAULT_PROMPT_VARIANT = "E"
C_STATUS_EVALUATED = "evaluated"
C_STATUS_W3_ZERO = "not_evaluated_w3_zero"
C_STATUS_INVALID = "invalid"


@dataclass
class ObjectiveCounters:
    """Contadores de custo acumulados (PRD §8.3, §15)."""

    requested: int = 0  # candidatos recebidos
    unique_llm: int = 0  # formas canônicas únicas enviadas ao avaliador
    invalid: int = 0  # candidatos com e não finito (sem LLM)
    llm_batches: int = 0  # chamadas a evaluate_many
    cache_hits: int = 0  # via evaluator.metrics, se existir
    llm_calls: int = 0  # backend_calls via evaluator.metrics, se existir
    llm_seconds: float = 0.0  # inference_seconds do avaliador; senão tempo de parede

    def as_dict(self) -> dict[str, Any]:
        """Snapshot serializável."""
        return asdict(self)


def huber(residual: FloatArray, delta: float) -> float:
    """Média da perda de Huber: ``0.5 r²`` se ``|r| <= delta``, senão ``delta(|r| - 0.5 delta)``."""
    a = np.abs(residual)
    quad = HUBER_QUADRATIC_FACTOR * a * a
    lin = delta * (a - HUBER_QUADRATIC_FACTOR * delta)
    return float(np.mean(np.where(a <= delta, quad, lin)))


class CombinedObjective:
    """Implementa ``CandidateObjective`` com a perda composta."""

    def __init__(
        self,
        weights: tuple[float, float, float],
        evaluator: PhysicsEquationEvaluator | None,
        physical_context: PhysicalContext | None,
        feature_names: Sequence[str],
        prompt_variant: str = DEFAULT_PROMPT_VARIANT,
        normalization: str = "literal",
        error_metric: str = "mse",
        huber_delta: float = 1.0,
        max_nodes: int | None = None,
        protection: ProtectionParams = DEFAULT_PROTECTION,
        significant_digits: int = DEFAULT_CACHE_SIGNIFICANT_DIGITS,
        key_mode: str = DEFAULT_CACHE_KEY_MODE,
    ) -> None:
        """Valida e guarda a configuração.

        ``significant_digits``/``key_mode``: chave de deduplicação do lote enviado ao LLM
        (``llm.cache_significant_digits``/``llm.cache_key_mode``).

        Raises:
            ValueError: pesos negativos/soma != 1, modo/métrica desconhecidos,
                ``huber_delta <= 0``, ``normalized`` sem ``max_nodes``, ``w3 > 0``
                sem avaliador/contexto, ou opção de chave inválida.
        """
        validate_key_options(significant_digits, key_mode)
        if len(weights) != len(("w1", "w2", "w3")):
            raise ValueError("weights deve ter 3 elementos (w1, w2, w3)")
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError(f"pesos {weights} devem ser finitos e >= 0")
        if abs(sum(weights) - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"pesos {weights} somam {sum(weights)}; esperado 1")
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization deve ser um de {NORMALIZATIONS}")
        if error_metric not in ERROR_METRICS:
            raise ValueError(f"error_metric deve ser um de {ERROR_METRICS}")
        if not huber_delta > 0:
            raise ValueError("huber_delta deve ser > 0")
        if normalization == "normalized" and (max_nodes is None or max_nodes < 1):
            raise ValueError("normalization='normalized' exige max_nodes >= 1")
        if weights[2] > 0 and (evaluator is None or physical_context is None):
            raise ValueError("w3 > 0 exige evaluator e physical_context")
        self.weights = weights
        self.evaluator = evaluator
        self.physical_context = physical_context
        self.feature_names = tuple(feature_names)
        self.prompt_variant = prompt_variant
        self.normalization: str = normalization
        self.error_metric: str = error_metric
        self.huber_delta = huber_delta
        self.max_nodes = max_nodes
        self.protection = protection
        self.significant_digits = significant_digits
        self.key_mode: str = key_mode
        self.counters = ObjectiveCounters()
        self.diagnostics: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {
            "weights": weights,
            "normalization": normalization,
            "deviation": "normalized" if normalization == "normalized" else None,
            "error_metric": error_metric,
            "huber_delta": huber_delta if error_metric == "huber" else None,
            "max_nodes": max_nodes,
            "prompt_variant": prompt_variant,
            "llm_in_loop": weights[2] > 0,
            "cache_significant_digits": significant_digits,
            "cache_key_mode": key_mode,
        }

    # ------------------------------------------------------------------ componentes

    def _error(self, expr: Expr, X: FloatArray, y: FloatArray, var_y: float) -> float:  # noqa: N803
        pred = evaluate(expr, X, self.feature_names, self.protection)
        with np.errstate(all="ignore"):
            if self.error_metric == "huber":
                e = huber(pred - y, self.huber_delta)
            else:
                e = float(np.mean((pred - y) ** 2))
            if self.normalization == "normalized":
                e = e / var_y
        return e

    def _size(self, expr: Expr) -> float:
        n = float(size(expr))
        if self.normalization == "normalized":
            assert self.max_nodes is not None
            return n / self.max_nodes
        return n

    # ------------------------------------------------------------------ API

    def evaluate_population(
        self,
        exprs: Sequence[Expr],
        X: FloatArray,  # noqa: N803 - nome do contrato (PRD)
        y: FloatArray,
    ) -> list[LossBreakdown]:
        """Perda composta de cada expressão (ordem preservada).

        Uma única chamada a ``evaluate_many`` por invocação, com candidatos válidos
        deduplicados por :meth:`llm_key`.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        with np.errstate(all="ignore"):
            var_y = float(np.var(y_arr))
        if self.normalization == "normalized" and not var_y > MIN_TARGET_VARIANCE:
            raise ValueError("modo normalized exige Var(y) > 0")
        w1, w2, w3 = self.weights
        n = len(exprs)
        self.counters.requested += n
        errors = [self._error(e, X, y_arr, var_y) for e in exprs]
        valid = [math.isfinite(e) for e in errors]
        self.counters.invalid += valid.count(False)

        evals: dict[str, Any] = {}
        if w3 > 0:
            unique: dict[str, Expr] = {}
            for ok, expr in zip(valid, exprs, strict=True):
                if ok:
                    unique.setdefault(self.llm_key(expr), expr)
            if unique:
                assert self.evaluator is not None and self.physical_context is not None
                evals = self._call_llm(unique)

        out: list[LossBreakdown] = []
        self.diagnostics = []
        for expr, e, ok in zip(exprs, errors, valid, strict=True):
            s = self._size(expr)
            if not ok:
                out.append(LossBreakdown(math.inf, math.inf, s, math.nan, None))
                self.diagnostics.append({"invalid": True, "c_status": C_STATUS_INVALID})
            elif w3 == 0:
                out.append(LossBreakdown(w1 * e + w2 * s, e, s, 0.0, None))
                self.diagnostics.append({"invalid": False, "c_status": C_STATUS_W3_ZERO})
            else:
                evaluation = evals[self.llm_key(expr)]
                c = penalty(evaluation)
                out.append(LossBreakdown(w1 * e + w2 * s + w3 * c, e, s, c, evaluation))
                self.diagnostics.append({"invalid": False, "c_status": C_STATUS_EVALUATED})
        return out

    def llm_key(self, expr: Expr) -> str:
        """Chave de deduplicação/cache da avaliação LLM (``significant_digits``, ``key_mode``)."""
        return expression_key(expr, self.significant_digits, self.key_mode)

    def _call_llm(self, unique: dict[str, Expr]) -> dict[str, Any]:
        assert self.evaluator is not None and self.physical_context is not None
        metrics = getattr(self.evaluator, "metrics", None)
        before = metrics.as_dict() if metrics is not None else None
        start = time.perf_counter()
        results = self.evaluator.evaluate_many(
            list(unique.values()), self.physical_context, self.prompt_variant
        )
        wall = time.perf_counter() - start
        self.counters.unique_llm += len(unique)
        self.counters.llm_batches += 1
        if before is not None and metrics is not None:
            after = metrics.as_dict()
            self.counters.cache_hits += int(after["cache_hits"] - before["cache_hits"])
            self.counters.llm_calls += int(after["backend_calls"] - before["backend_calls"])
            self.counters.llm_seconds += float(
                after["inference_seconds"] - before["inference_seconds"]
            )
        else:
            self.counters.llm_seconds += wall
        return dict(zip(unique.keys(), results, strict=True))


def build_objective(
    cfg: ExperimentConfig,
    evaluator: PhysicsEquationEvaluator | None,
    physical_context: PhysicalContext | None,
    algorithm: str,
    baseline: bool,
    feature_names: Sequence[str],
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> CombinedObjective:
    """Fábrica a partir da config.

    - ``baseline=True``: pesos ``loss.baseline_weights`` (w3 = 0; avaliador ignorado).
    - ``algorithm == "pysr"`` com ``sr.pysr.loss == "huber"``: erro Huber (GAP-05);
      demais: MSE.
    - ``normalized``: ``max_nodes = 2**(sr.deap.max_depth + 1) - 1`` (nós de uma árvore
      binária completa na profundidade máxima; a config não tem ``loss.max_nodes``).
    - Chave de deduplicação: ``llm.cache_significant_digits`` e ``llm.cache_key_mode``
      quando existirem na config; senão os defaults (3, ``exact``).
    """
    loss = cfg.loss
    weights = loss.baseline_weights if baseline else (loss.w1, loss.w2, loss.w3)
    use_huber = algorithm == "pysr" and cfg.sr.pysr.loss == "huber"
    max_nodes = 2 ** (cfg.sr.deap.max_depth + 1) - 1
    return CombinedObjective(
        weights=weights,
        evaluator=None if baseline else evaluator,
        physical_context=physical_context,
        feature_names=feature_names,
        prompt_variant=prompt_variant,
        normalization=loss.normalization,
        error_metric="huber" if use_huber else "mse",
        huber_delta=cfg.sr.pysr.huber_delta,
        max_nodes=max_nodes if loss.normalization == "normalized" else None,
        protection=ProtectionParams.from_config(cfg.symbolic),
        significant_digits=int(
            getattr(cfg.llm, "cache_significant_digits", DEFAULT_CACHE_SIGNIFICANT_DIGITS)
        ),
        key_mode=str(getattr(cfg.llm, "cache_key_mode", DEFAULT_CACHE_KEY_MODE)),
    )
