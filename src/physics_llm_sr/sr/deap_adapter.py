"""Adaptador DEAP (programação genética) com o LLM participando da SELEÇÃO.

Mecanismo de integração (``integrated``)
----------------------------------------
Loop geracional explícito no estilo ``deap.algorithms.eaSimple``::

    seleção por torneio (tools.selTournament, tamanho 3) -> clone -> varAnd (cx/mut)
    -> avaliação em LOTE dos indivíduos inválidos via objective.evaluate_population
    -> substituição da população

A aptidão (``FitnessMin``) de cada indivíduo é ``LossBreakdown.total`` da perda
composta; logo, com ``w3 > 0`` as notas do LLM entram na comparação dos torneios de
CADA geração (não há reordenação posterior). A avaliação é em lote (uma chamada a
``evaluate_population`` por geração, que deduplica e chama o LLM uma vez).

Hiperparâmetros (PRD 4.1; ``config.sr`` e ``config.sr.deap``, usados literalmente):
população, gerações, probabilidade de crossover/mutação, torneio, profundidade máxima
(``gp.staticLimit`` de altura em ``mate`` e ``mutate``) e profundidade inicial
(``genHalfAndHalf``); todos são registrados em ``RegressorMetadata.extra``.

Segurança (PRD: sem ``eval`` irrestrito): ``gp.compile`` usa ``eval`` internamente e
NÃO é usado no caminho de produção. Indivíduos são convertidos para a AST comum via
notação prefixa (``symbolic.from_prefix``) e avaliados por ``symbolic.evaluate``. As
funções do ``PrimitiveSet`` são as protegidas de ``symbolic.operators`` (nenhuma
semântica reimplementada) e só servem para o ``gp.compile`` dos testes, sobre árvores
geradas pelo próprio DEAP.

Constantes efêmeras: ``U(-S, S)`` com ``S = RANDOM_CONSTANT_SCALE`` (escala já usada
por ``symbolic.random_expression``; não há campo na config). A aleatoriedade do DEAP
vem do módulo ``random``, semeado a cada ``fit`` com a semente derivada.
Determinismo: sem ``toolbox.map`` paralelo.

Contagem de gerações: a população inicial é a geração 0 (também registrada);
``n_generations`` é o número de gerações evoluídas e ``history`` tem
``n_generations + 1`` registros. O melhor indivíduo é o do ``HallOfFame(1)`` (melhor
já visto, pois ``eaSimple`` não tem elitismo); ``best_loss`` por geração e a parada
antecipada usam a melhor perda acumulada.
"""

from __future__ import annotations

import functools
import operator
import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from deap import algorithms, base, creator, gp, tools

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
from physics_llm_sr.symbolic.expression import (
    RANDOM_CONSTANT_SCALE,
    Const,
    Expr,
    Op,
    Var,
    evaluate,
    from_prefix,
    to_infix,
)
from physics_llm_sr.symbolic.operators import ProtectionParams, apply_operator, get_operator

ALGORITHM: Literal["deap"] = "deap"
DEFAULT_SEED_NAME = "sr/deap"
EPHEMERAL_NAME = "deap_const"
#: Profundidade das subárvores geradas pela mutação uniforme (``genFull``).
MUTATION_MIN_DEPTH = 0
MUTATION_MAX_DEPTH = 2
HOF_SIZE = 1
_FITNESS_CLASS = "PhysLLMFitnessMin"
_INDIVIDUAL_CLASS = "PhysLLMIndividual"


def _ensure_creator_classes() -> None:
    """Cria (uma vez) as classes globais de fitness/indivíduo do DEAP."""
    if not hasattr(creator, _FITNESS_CLASS):
        creator.create(_FITNESS_CLASS, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, _INDIVIDUAL_CLASS):
        creator.create(
            _INDIVIDUAL_CLASS, gp.PrimitiveTree, fitness=getattr(creator, _FITNESS_CLASS)
        )


def _protected(name: str, protection: ProtectionParams) -> Callable[..., FloatArray]:
    def _fn(*args: Any) -> FloatArray:
        return apply_operator(name, *args, params=protection)

    _fn.__name__ = name
    return _fn


def build_primitive_set(
    feature_names: Sequence[str],
    operators: Sequence[str],
    protection: ProtectionParams,
) -> gp.PrimitiveSet:
    """``PrimitiveSet`` com operadores protegidos de ``symbolic.operators`` e terminais.

    Terminais: uma variável por ``feature_names`` (argumento renomeado) e a constante
    efêmera ``U(-S, S)``. Os nomes das primitivas são os tokens da AST.
    """
    pset = gp.PrimitiveSet("MAIN", len(feature_names))
    for name in operators:
        pset.addPrimitive(_protected(name, protection), get_operator(name).arity, name=name)
    pset.addEphemeralConstant(
        EPHEMERAL_NAME,
        functools.partial(random.uniform, -RANDOM_CONSTANT_SCALE, RANDOM_CONSTANT_SCALE),
    )
    pset.renameArguments(**{f"ARG{i}": n for i, n in enumerate(feature_names)})
    return pset


def individual_to_expr(individual: Sequence[Any]) -> Expr:
    """Converta um ``PrimitiveTree`` (ordem prefixa) na AST comum via ``from_prefix``."""
    tokens: list[str] = []
    for node in individual:
        if isinstance(node, gp.Primitive):
            tokens.append(node.name)
        elif isinstance(node.value, str):  # argumento renomeado = feature
            tokens.append(node.value)
        else:  # constante efêmera
            tokens.append(repr(float(node.value)))
    return from_prefix(tokens)


def expr_to_individual(expr: Expr, pset: gp.PrimitiveSet) -> Any:
    """Constrói um indivíduo DEAP a partir de uma AST (usado em provas controladas)."""
    _ensure_creator_classes()
    nodes: list[Any] = []
    ephemeral = pset.terminals[object][-1]  # classe efêmera registrada por último

    def _walk(e: Expr) -> None:
        if isinstance(e, Op):
            nodes.append(pset.mapping[e.name])
            for child in e.children:
                _walk(child)
        elif isinstance(e, Var):
            nodes.append(pset.mapping[e.name])
        elif isinstance(e, Const):
            node = ephemeral()
            node.value = e.value
            nodes.append(node)

    _walk(expr)
    return getattr(creator, _INDIVIDUAL_CLASS)(nodes)


def evaluate_individuals(
    individuals: Sequence[Any],
    objective: CandidateObjective,
    X: FloatArray,  # noqa: N803 - nome do contrato (PRD)
    y: FloatArray,
) -> int:
    """Avalia em LOTE os indivíduos sem fitness válida (uma chamada ao objective).

    Define ``fitness.values = (total,)`` e ``individual.loss`` (``LossBreakdown``).
    Devolve quantos foram avaliados.
    """
    invalid = [ind for ind in individuals if not ind.fitness.valid]
    if not invalid:
        return 0
    losses = objective.evaluate_population([individual_to_expr(i) for i in invalid], X, y)
    for ind, loss in zip(invalid, losses, strict=True):
        ind.fitness.values = (float(loss.total),)
        ind.loss = loss
    return len(invalid)


class DeapRegressor:
    """Regressor simbólico DEAP (``SymbolicRegressor``) com LLM na seleção."""

    def __init__(self, seed_name: str = DEFAULT_SEED_NAME) -> None:
        """``seed_name`` deriva a semente a partir de ``config.seeds.master``."""
        self.seed_name = seed_name
        self._best: Expr | None = None
        self._feature_names: tuple[str, ...] = ()
        self._protection: ProtectionParams | None = None
        self._history: list[GenerationRecord] = []
        self._final_population: list[Expr] = []
        self._metadata = RegressorMetadata(ALGORITHM, False, "baseline_only", None)
        self.stop_reason: StopReason | None = None

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        train: Dataset,
        objective: CandidateObjective,
        config: ExperimentConfig,
        validation: Dataset | None = None,
    ) -> FitResult:
        """Executa o loop geracional (ver docstring do módulo)."""
        ensure_feature_alignment(train, objective)
        sr, dp = config.sr, config.sr.deap
        protection = ProtectionParams.from_config(config.symbolic)
        seed = derive_seed(config.seeds.master, self.seed_name)
        random.seed(seed)
        np.random.seed(seed % (2**32))  # noqa: NPY002 - DEAP/numpy legado

        feature_names = tuple(train.feature_names)
        X, y = train.X, train.y_observed  # noqa: N806
        _ensure_creator_classes()
        pset = build_primitive_set(feature_names, sr.operators[ALGORITHM], protection)
        height = operator.attrgetter("height")
        toolbox = base.Toolbox()
        toolbox.register(
            "expr", gp.genHalfAndHalf, pset=pset, min_=dp.init_min_depth, max_=dp.init_max_depth
        )
        toolbox.register(
            "individual", tools.initIterate, getattr(creator, _INDIVIDUAL_CLASS), toolbox.expr
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selTournament, tournsize=dp.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=MUTATION_MIN_DEPTH, max_=MUTATION_MAX_DEPTH)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=height, max_value=dp.max_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=height, max_value=dp.max_depth))

        counters = getattr(objective, "counters", None)
        llm_in_loop = bool(getattr(objective, "weights", (0, 0, 0))[2] > 0)
        self._history = []
        self._best = None
        self._feature_names = feature_names
        self._protection = protection
        self.stop_reason = None
        self._metadata = RegressorMetadata(
            ALGORITHM,
            llm_in_loop,
            "integrated" if llm_in_loop else "baseline_only",
            self._extra(config, seed),
        )

        stopper = EarlyStopping.from_config(sr.early_stopping)
        hof = tools.HallOfFame(HOF_SIZE)
        start = time.perf_counter()
        population = toolbox.population(n=sr.population)
        generation = 0
        stop: StopReason = "max_generations"
        try:
            while True:
                gen_start = time.perf_counter()
                before = _snapshot(counters)
                evaluate_individuals(population, objective, X, y)
                hof.update(population)
                self._record(generation, hof[0], before, counters, time.perf_counter() - gen_start)
                self._final_population = [individual_to_expr(i) for i in population]
                self._best = individual_to_expr(hof[0])
                if stopper.update(hof[0].fitness.values[0]):
                    stop = "early_stopping"
                    break
                if generation >= sr.generations:
                    break
                generation += 1
                offspring = toolbox.select(population, len(population))
                offspring = [toolbox.clone(i) for i in offspring]
                offspring = _var_and(offspring, toolbox, dp.crossover_prob, dp.mutation_prob)
                population[:] = offspring
        except EvaluationFailedError:
            self.stop_reason = "error"
            raise
        self.stop_reason = stop
        assert self._best is not None
        return FitResult(
            best_expression=self._best,
            history=tuple(self._history),
            metadata=self._metadata,
            elapsed_seconds=time.perf_counter() - start,
            n_generations=generation,
            stop_reason=stop,
            llm_counters=_numeric(counters),
        )

    def _record(
        self,
        generation: int,
        best: Any,
        before: Mapping[str, float],
        counters: Any,
        elapsed: float,
    ) -> None:
        loss: LossBreakdown = best.loss
        now = _snapshot(counters)
        self._history.append(
            GenerationRecord(
                generation=generation,
                best_loss=float(best.fitness.values[0]),
                best_e=float(loss.e),
                best_s=float(loss.s),
                best_c=float(loss.c),
                best_expression=to_infix(individual_to_expr(best)),
                n_llm_calls=int(now["llm_calls"] - before["llm_calls"]),
                n_cache_hits=int(now["cache_hits"] - before["cache_hits"]),
                elapsed=elapsed,
            )
        )

    @staticmethod
    def _extra(config: ExperimentConfig, seed: int) -> dict[str, Any]:
        sr, dp = config.sr, config.sr.deap
        return {
            "population": sr.population,
            "generations": sr.generations,
            "crossover_prob": dp.crossover_prob,
            "mutation_prob": dp.mutation_prob,
            "tournament_size": dp.tournament_size,
            "max_depth": dp.max_depth,
            "init_min_depth": dp.init_min_depth,
            "init_max_depth": dp.init_max_depth,
            "operators": list(sr.operators[ALGORITHM]),
            "ephemeral_constant_range": [-RANDOM_CONSTANT_SCALE, RANDOM_CONSTANT_SCALE],
            "seed": seed,
            "error_metric": "mse",  # GAP-05
            "gp_compile_used": False,  # eval interno do DEAP fora do caminho de produção
            "evaluation": "symbolic.evaluate",
            "selection": "tools.selTournament",
            "early_stopping": sr.early_stopping.model_dump(),
        }

    # ------------------------------------------------------------------ API

    def predict(self, X: FloatArray) -> FloatArray:  # noqa: N803 - nome do contrato (PRD)
        """``symbolic.evaluate`` da melhor AST (mesmos ``feature_names`` do fit)."""
        if self._best is None or self._protection is None:
            raise RuntimeError("fit deve ser chamado antes de predict")
        return evaluate(self._best, X, self._feature_names, self._protection)

    def get_best_expression(self) -> Expr:
        """Melhor expressão (HallOfFame) na AST comum."""
        if self._best is None:
            raise RuntimeError("fit deve ser chamado antes")
        return self._best

    def get_search_history(self) -> Sequence[GenerationRecord]:
        """Um registro por geração (a 0 é a população inicial)."""
        return tuple(self._history)

    def get_metadata(self) -> RegressorMetadata:
        """Metadados reprodutíveis (hiperparâmetros literais em ``extra``)."""
        return self._metadata

    def final_population_expressions(self) -> list[Expr]:
        """População final convertida para a AST (para análises da prova (b))."""
        return list(self._final_population)


def _var_and(offspring: list[Any], toolbox: base.Toolbox, cxpb: float, mutpb: float) -> list[Any]:
    """``algorithms.varAnd`` do DEAP (cx e mut independentes, invalida fitness)."""
    out: list[Any] = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
    return out


def _snapshot(counters: Any) -> dict[str, float]:
    if counters is None:
        return {"llm_calls": 0.0, "cache_hits": 0.0}
    return {"llm_calls": float(counters.llm_calls), "cache_hits": float(counters.cache_hits)}


def _numeric(counters: Any) -> dict[str, float]:
    if counters is None:
        return {}
    return {k: float(v) for k, v in counters.as_dict().items()}
