"""Adaptador gplearn com o termo LLM participando da seleção (GAP-08).

Problema
--------
No gplearn 0.4.2 a aptidão de cada programa é calculada em
``gplearn.genetic._parallel_evolve`` por ``_Program.raw_fitness(X, y, w)``, que chama
``metric(y, y_pred, w)``. A API pública (``make_fitness(function(y, y_pred, w))``) não
dá acesso à expressão e avalia um programa por vez, sem lote por geração; logo não
permite ``L = w1*e + w2*s + w3*c`` com ``c`` por expressão nem a chamada única ao
LLM por geração.

Leitura do código instalado (``gplearn/genetic.py``, 0.4.2)
----------------------------------------------------------
- ``BaseSymbolic.fit`` itera ``for gen in range(prior_generations, self.generations)``;
  com ``warm_start=True`` retoma de ``len(self._programs)`` e descarta as sementes das
  gerações já feitas, então N chamadas com ``generations = 1..N`` reproduzem a mesma
  sequência aleatória de uma chamada única com ``generations = N``.
- A geração ``g`` recebe ``parents = self._programs[g - 1]`` e o torneio
  (``_tournament`` dentro de ``_parallel_evolve``) lê **``parents[p].fitness_``** e
  escolhe ``argmin`` (métrica ``mse``, ``greater_is_better = False``).
- ``fitness_ = raw_fitness_ - parsimony_coefficient * len(program) * sign`` é
  calculado logo após a geração ser criada e não é relido antes da geração seguinte.

Mecanismo escolhido (opção 1 do briefing)
-----------------------------------------
Laço geracional externo: ``SymbolicRegressor(warm_start=True, n_jobs=1)`` e, a cada
iteração, ``generations += 1`` e ``fit``. Ao fim de cada geração, a população
corrente (``self._programs[-1]``) é convertida para a AST comum e pontuada **em lote**
por ``objective.evaluate_population`` (uma chamada ao LLM por geração, deduplicada pelo
objective); ``raw_fitness_`` e ``fitness_`` de cada programa são sobrescritos com
``L.total`` **antes** da geração seguinte. Assim o torneio nativo do gplearn seleciona
os pais por ``L``. Operadores genéticos, probabilidades, torneio e inicialização são
os do gplearn, sem alteração.

Alternativas rejeitadas:
- (2) subclasse de ``_Program`` sobrescrevendo ``raw_fitness``: ``_parallel_evolve``
  instancia ``gplearn.genetic._Program`` pelo nome do módulo, então exigiria
  *monkeypatch* global; além disso a avaliação é por programa (sem lote, uma chamada
  LLM por indivíduo).
- ``make_fitness``: sem acesso à expressão (ver acima).

Desvios e decisões registrados (``metadata.extra``):
- Uso de API privada (``_programs``, ``raw_fitness_``/``fitness_`` dos programas): a
  versão é verificada (``GPLEARN_SUPPORTED_VERSION``); outra versão gera erro.
- ``parsimony_coefficient = 0``: o termo ``w2*s`` do objective já penaliza o tamanho e,
  como ``fitness_`` é sobrescrito por ``L``, a parcimônia nativa não teria efeito na
  seleção; ``config.sr.gplearn.parsimony_coefficient`` é registrado como ignorado
  para não penalizar o tamanho duas vezes.
- A aptidão nativa (MSE) ainda é calculada pelo gplearn em cada geração (custo extra,
  descartado); o critério nativo ``stopping_criteria`` não tem efeito porque cada
  ``fit`` executa exatamente uma geração. A parada é a do projeto (EarlyStopping).
- A melhor expressão é a de menor ``L`` vista em qualquer geração (registro elitista
  fora da seleção; o gplearn não tem elitismo e isso não altera a busca).
- Hiperparâmetros sem campo em ``config.sr.gplearn`` (``tournament_size``,
  ``p_crossover``, ``p_*_mutation``, ``init_depth``, ``const_range``...) usam os
  defaults da biblioteca e são registrados em ``metadata.extra["gplearn_params"]``.

Conversão ``_Program -> AST``: ``program.program`` é uma lista prefixa de
``_Function``/``int`` (índice de feature)/``float``; os nomes nativos coincidem com
o registro de operadores (``add, sub, mul, div, neg, log, sin, cos, sqrt``) e ``exp``
é uma função customizada (``make_function``) que chama ``apply_operator("exp")``
com os mesmos ``ProtectionParams`` da AST. A divisão e o log nativos do gplearn usam
``0.001`` e ``fallback = 1.0``/``0.0``; se a config divergir disso, ``fit`` recusa.
Avaliação nativa e ``symbolic.evaluate`` coincidem exceto quando um intermediário
nativo é não finito (a AST satura em ``±power_clip``) e na fronteira exata
``|b| == eps`` (gplearn usa ``>``; a AST, ``<``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any, Final

import gplearn
import numpy as np
from gplearn.functions import _Function, make_function
from gplearn.genetic import SymbolicRegressor as _GplearnSymbolicRegressor

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.llm.interfaces import EvaluationFailedError
from physics_llm_sr.optimization.early_stopping import EarlyStopping
from physics_llm_sr.reproducibility.seeds import derive_seed
from physics_llm_sr.sr.interfaces import (
    CandidateObjective,
    FitResult,
    FloatArray,
    GenerationRecord,
    LossBreakdown,
    RegressorMetadata,
    StopReason,
    ensure_feature_alignment,
)
from physics_llm_sr.symbolic.expression import Expr, evaluate, from_prefix, to_infix
from physics_llm_sr.symbolic.operators import (
    DEFAULT_PROTECTION,
    ProtectionParams,
    apply_operator,
)

ALGORITHM: Final = "gplearn"
#: Versão cujo código privado (``_programs``, ``fitness_``) foi lido e testado.
GPLEARN_SUPPORTED_VERSION = "0.4.2"
#: Nome da semente derivada (``reproducibility.seeds.derive_seed``).
SEED_NAME = "sr/gplearn"
#: Métrica nativa (só para o gplearn; a seleção usa ``L``). ``mse`` é minimizada.
NATIVE_METRIC = "mse"
#: Parcimônia nativa desligada: ``w2*s`` do objective já penaliza o tamanho.
NATIVE_PARSIMONY = 0.0
#: Uma única tarefa: o LLM e o cache ficam no processo principal.
N_JOBS = 1
#: Operadores nativos do gplearn com o mesmo nome e semântica do registro da AST.
NATIVE_FUNCTIONS: frozenset[str] = frozenset(
    {"add", "sub", "mul", "div", "neg", "log", "sin", "cos", "sqrt"}
)
#: Operadores implementados via ``make_function`` com a proteção da AST.
CUSTOM_FUNCTIONS: frozenset[str] = frozenset({"exp"})
#: Constantes fixas de ``gplearn.functions._protected_division``/``_protected_log``.
GPLEARN_DIVISION_EPSILON = 0.001
GPLEARN_DIVISION_FALLBACK = 1.0
GPLEARN_LOG_EPSILON = 0.001


class UnsupportedGplearnError(RuntimeError):
    """Versão do gplearn ou configuração incompatível com o mecanismo in-loop."""


# --------------------------------------------------------------------------- funções


def _custom_function(name: str, protection: ProtectionParams) -> _Function:
    """Cria a ``_Function`` do gplearn que aplica o operador protegido da AST."""

    def _apply(x1: FloatArray) -> FloatArray:
        return apply_operator(name, x1, params=protection)

    return make_function(function=_apply, name=name, arity=1)


def make_function_set(
    operators: Sequence[str], protection: ProtectionParams = DEFAULT_PROTECTION
) -> list[str | _Function]:
    """``function_set`` do gplearn para os operadores da config (GAP-06).

    Raises:
        UnsupportedGplearnError: operador sem equivalente no gplearn (ex.: ``pow``) ou
            proteção da config diferente da proteção nativa de ``div``/``log``.
    """
    native_protection_ok = (
        protection.division_epsilon == GPLEARN_DIVISION_EPSILON
        and protection.division_fallback == GPLEARN_DIVISION_FALLBACK
        and protection.log_epsilon == GPLEARN_LOG_EPSILON
    )
    out: list[str | _Function] = []
    for name in operators:
        if name in NATIVE_FUNCTIONS:
            if name in ("div", "log") and not native_protection_ok:
                raise UnsupportedGplearnError(
                    f"{name} nativo do gplearn usa eps={GPLEARN_DIVISION_EPSILON} e "
                    f"fallback={GPLEARN_DIVISION_FALLBACK}; config diverge: {protection}"
                )
            out.append(name)
        elif name in CUSTOM_FUNCTIONS:
            out.append(_custom_function(name, protection))
        else:
            raise UnsupportedGplearnError(f"operador {name!r} sem equivalente no gplearn")
    return out


# --------------------------------------------------------------------------- conversão


def program_to_expr(program: Sequence[Any], feature_names: Sequence[str]) -> Expr:
    """AST comum equivalente a ``_Program.program`` (lista prefixa).

    ``_Function`` -> nome do operador; ``int`` -> ``feature_names[i]``; ``float`` ->
    constante.
    """
    tokens: list[str] = []
    for node in program:
        if isinstance(node, _Function):
            tokens.append(node.name)
        elif isinstance(node, (int, np.integer)):
            tokens.append(feature_names[int(node)])
        else:
            tokens.append(repr(float(node)))
    return from_prefix(tokens, feature_names)


def apply_objective_fitness(
    programs: Sequence[Any],
    objective: CandidateObjective,
    X: FloatArray,  # noqa: N803 - nome do contrato (PRD)
    y: FloatArray,
    feature_names: Sequence[str],
) -> tuple[list[Expr], list[LossBreakdown]]:
    """Pontua a população em lote e grava ``L.total`` em ``raw_fitness_``/``fitness_``.

    É o ponto de integração: o torneio da geração seguinte lê ``fitness_`` dos pais.
    """
    exprs = [program_to_expr(p.program, feature_names) for p in programs]
    losses = list(objective.evaluate_population(exprs, X, y))
    for program, loss in zip(programs, losses, strict=True):
        program.raw_fitness_ = loss.total
        program.fitness_ = loss.total
    return exprs, losses


def build_estimator(
    config: ExperimentConfig,
    feature_names: Sequence[str],
    protection: ProtectionParams,
    seed: int,
) -> _GplearnSymbolicRegressor:
    """``SymbolicRegressor`` configurado para avançar uma geração por ``fit``."""
    return _GplearnSymbolicRegressor(
        population_size=config.sr.population,
        generations=0,
        function_set=make_function_set(config.sr.operators[ALGORITHM], protection),
        metric=NATIVE_METRIC,
        parsimony_coefficient=NATIVE_PARSIMONY,
        feature_names=list(feature_names),
        warm_start=True,
        low_memory=True,
        n_jobs=N_JOBS,
        random_state=seed,
    )


def check_gplearn_version() -> None:
    """Recusa versões cujo código privado não foi verificado."""
    if gplearn.__version__ != GPLEARN_SUPPORTED_VERSION:
        raise UnsupportedGplearnError(
            f"gplearn {gplearn.__version__} != {GPLEARN_SUPPORTED_VERSION}: o mecanismo "
            "in-loop usa API privada; revalide antes de atualizar"
        )


# --------------------------------------------------------------------------- adaptador


def _counter(objective: CandidateObjective, name: str) -> float:
    counters = getattr(objective, "counters", None)
    return float(getattr(counters, name, 0) or 0)


def _counters_dict(objective: CandidateObjective) -> dict[str, float]:
    counters = getattr(objective, "counters", None)
    if counters is None or not hasattr(counters, "as_dict"):
        return {}
    return {k: float(v) for k, v in counters.as_dict().items()}


def _llm_in_loop(objective: CandidateObjective) -> bool:
    metadata = getattr(objective, "metadata", None)
    if isinstance(metadata, dict) and "llm_in_loop" in metadata:
        return bool(metadata["llm_in_loop"])
    weights = getattr(objective, "weights", None)
    return bool(weights is not None and weights[2] > 0)


class GplearnRegressor:
    """``SymbolicRegressor`` (contrato do projeto) sobre ``gplearn.SymbolicRegressor``."""

    def __init__(self) -> None:
        """Estado vazio até ``fit``."""
        self._best: Expr | None = None
        self._best_program: Any = None
        self._feature_names: tuple[str, ...] = ()
        self._protection: ProtectionParams = DEFAULT_PROTECTION
        self._history: list[GenerationRecord] = []
        self._metadata = RegressorMetadata(ALGORITHM, False, "baseline_only")
        self.estimator: _GplearnSymbolicRegressor | None = None
        self.stop_reason: StopReason | None = None

    def fit(
        self,
        train: Dataset,
        objective: CandidateObjective,
        config: ExperimentConfig,
        validation: Dataset | None = None,
    ) -> FitResult:
        """Evolui ``config.sr.generations`` gerações com seleção por ``L`` (ver módulo).

        ``validation`` não participa da seleção (anti-vazamento) e é ignorado.

        Raises:
            EvaluationFailedError: falha do avaliador LLM (propaga; ``stop_reason =
                "error"`` fica registrado no adaptador).
            UnsupportedGplearnError: versão/configuração incompatível.
        """
        ensure_feature_alignment(train, objective)
        check_gplearn_version()
        start = time.perf_counter()
        self._feature_names = tuple(train.feature_names)
        self._protection = ProtectionParams.from_config(config.symbolic)
        self._history = []
        self._best = None
        self._best_program = None
        self.stop_reason = None
        X = np.asarray(train.X, dtype=np.float64)  # noqa: N806
        y = np.asarray(train.y_observed, dtype=np.float64)
        seed = derive_seed(config.seeds.master, SEED_NAME)
        est = build_estimator(config, self._feature_names, self._protection, seed)
        self.estimator = est
        in_loop = _llm_in_loop(objective)
        self._metadata = RegressorMetadata(
            algorithm=ALGORITHM,
            llm_in_loop=in_loop,
            integration_status="integrated" if in_loop else "baseline_only",
            extra={
                "mechanism": "warm_start_generation_loop_fitness_override",
                "gplearn_version": gplearn.__version__,
                "seed": seed,
                "seed_name": SEED_NAME,
                "parsimony_coefficient_native": NATIVE_PARSIMONY,
                "parsimony_coefficient_config_ignored": config.sr.gplearn.parsimony_coefficient,
                "gplearn_params": {
                    k: v
                    for k, v in est.get_params().items()
                    if k not in ("function_set", "feature_names")
                },
                "function_set": list(config.sr.operators[ALGORITHM]),
            },
        )
        stopper = EarlyStopping.from_config(config.sr.early_stopping)
        best_loss: LossBreakdown | None = None
        stop: StopReason = "max_generations"
        try:
            for gen in range(config.sr.generations):
                gen_start = time.perf_counter()
                calls0 = _counter(objective, "llm_calls")
                hits0 = _counter(objective, "cache_hits")
                est.set_params(generations=gen + 1)
                est.fit(X, y)
                population = est._programs[-1]
                exprs, losses = apply_objective_fitness(
                    population, objective, X, y, self._feature_names
                )
                i = int(np.argmin([loss.total for loss in losses]))
                if best_loss is None or losses[i].total < best_loss.total:
                    best_loss = losses[i]
                    self._best = exprs[i]
                    self._best_program = population[i]
                assert best_loss is not None and self._best is not None
                self._history.append(
                    GenerationRecord(
                        generation=gen,
                        best_loss=best_loss.total,
                        best_e=best_loss.e,
                        best_s=best_loss.s,
                        best_c=best_loss.c,
                        best_expression=to_infix(self._best),
                        n_llm_calls=int(_counter(objective, "llm_calls") - calls0),
                        n_cache_hits=int(_counter(objective, "cache_hits") - hits0),
                        elapsed=time.perf_counter() - gen_start,
                    )
                )
                if math.isfinite(best_loss.total) and stopper.update(best_loss.total):
                    stop = "early_stopping"
                    break
        except EvaluationFailedError:
            self.stop_reason = "error"
            raise
        self.stop_reason = stop
        # Coerência com a API nativa: est.predict usa o melhor programa por L.
        est._program = self._best_program
        assert self._best is not None
        return FitResult(
            best_expression=self._best,
            history=tuple(self._history),
            metadata=self._metadata,
            elapsed_seconds=time.perf_counter() - start,
            n_generations=len(self._history),
            stop_reason=stop,
            llm_counters=_counters_dict(objective),
        )

    def predict(self, X: FloatArray) -> FloatArray:  # noqa: N803 - nome do contrato (PRD)
        """``symbolic.evaluate`` da melhor expressão (mesmos ``feature_names`` do fit)."""
        return evaluate(self.get_best_expression(), X, self._feature_names, self._protection)

    def get_best_expression(self) -> Expr:
        """Melhor expressão (menor ``L``) na AST comum."""
        if self._best is None:
            raise RuntimeError("fit ainda não foi executado")
        return self._best

    def get_search_history(self) -> Sequence[GenerationRecord]:
        """Histórico por geração (melhor ``L`` acumulada)."""
        return tuple(self._history)

    def get_metadata(self) -> RegressorMetadata:
        """Metadados do último ``fit``."""
        return self._metadata
