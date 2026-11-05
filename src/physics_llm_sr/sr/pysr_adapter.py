"""Adaptador PySR com o termo LLM dentro da seleção (GAP-08, decisão pós-spike).

Mecanismo (``integrated``)
--------------------------
O objetivo completo ``L = w1*e + w2*s + w3*c`` é a ``loss_function`` Julia do PySR
(:data:`JULIA_PRELUDE`, função ``plsr_loss``), portanto é a perda que o PySR usa em
torneio, substituição, migração, otimização de constantes e hall of fame:

- ``e``: Huber(δ = ``objective.huber_delta``) ou MSE, conforme ``objective.error_metric``
  (GAP-05: PySR usa Huber); no modo ``normalized`` dividido por ``Var(y_train)``
  (``np.var`` do ``y`` passado, como no :class:`CombinedObjective`).
- ``s``: ``count_nodes`` da árvore Julia (= ``symbolic.size`` da AST); ``/ max_nodes``
  no modo ``normalized``.
- ``c``: chamada síncrona a um callable Python (:class:`PenaltyCallback`) com o texto
  ``string_tree`` da árvore, sob ``PythonCall.GIL.@lock`` (o PySR libera o GIL durante a
  busca; sem o lock há segfault). O callback converte o texto na AST
  (``parse_infix``, parser com whitelist), aplica ``symbolic.canonicalization.simplify``
  (constantes dobradas: a nota não pode depender de ``cos(constante)``, que o PySR dobra
  depois — armadilha do spike) e obtém ``c`` de ``objective.evaluate_population([expr])``
  (avaliador + cache do projeto). Não há lote: a perda é chamada por árvore. Um memo por
  :func:`llm.cache.expression_key` (``significant_digits``/``key_mode`` do objective)
  evita repetir a chamada ao objective para a mesma chave.
- ``w3 == 0``: o callback não é registrado e o Julia não chama o Python (avaliador nunca
  chamado).

Pesos, escala, ``max_nodes``, métrica e δ são lidos do próprio objective (mesmos
parâmetros do :class:`CombinedObjective`). Paridade: ao fim, cada linha do hall of fame é
reavaliada por ``objective.evaluate_population`` e comparada com a perda do Julia
(``metadata.extra["audit"]``; tolerância relativa :data:`AUDIT_REL_TOL`).

Histórico: ``warm_start=True`` e ``niterations=1`` por ``fit`` do PySR. A seleção já é
in-loop (``loss_function``); o laço externo de uma iteração por vez existe só para ler o
hall of fame após cada iteração, registrar o :class:`GenerationRecord` e aplicar o early
stopping do projeto (0,1 %/3). Cada iteração ``i`` usa ``random_state =
derive_seed(semente_do_adaptador, f"it{i}")`` (:func:`iteration_seed`): sementes distintas
por iteração (sem re-semear o RNG com o mesmo valor a cada ``fit`` do warm start) e
determinísticas; a lista vai para ``metadata.extra["iteration_seeds"]``.

Registro e auditoria usam a mesma forma do callback: ``e`` e ``s`` da árvore crua (como
``plsr_loss`` no Julia: ``count_nodes`` da árvore otimizada) e ``c`` da forma
``simplify``-ada obtida pelo memo do callback (:meth:`PenaltyCallback.penalty`), logo
nenhuma chamada LLM nova quando a forma já foi avaliada na busca. Auditoria fora da
tolerância NÃO é descartada: ``metadata.extra["audit_failed"] = True`` e
``audit_reason``; o runner marca a célula (em full + paper_replication vira erro).

Falha do LLM (CONVENCOES, pós-spike): o callback registra a exceção, passa a devolver
``+inf`` (a árvore não é selecionada e nenhuma nota substituta é usada) e não chama mais
o avaliador; após a iteração corrente o adaptador levanta a exceção
(``EvaluationFailedError``), ``stop_reason = "error"`` e o resultado é descartado.

Determinismo: ``parallelism="serial"``, ``deterministic=True``, ``random_state`` =
semente derivada (``reproducibility.seeds``), ``batching=False``, ``procs`` ausente.
``multithreading`` não é usado (spike: segfault/não determinismo com o callback).

Operadores (GAP-06) e semântica protegida
------------------------------------------
Os nomes da config são mapeados para os operadores NATIVOS do PySR
(:data:`PYSR_OPERATORS`). Diferenças em relação a ``symbolic.operators``:

- ``/``: nativo é divisão IEEE (``b = 0`` -> ``Inf``/``NaN`` -> árvore inválida, ``L = Inf``);
  a AST devolve ``division_fallback`` se ``|b| < division_epsilon``.
- ``log``: nativo ``safe_log`` devolve ``NaN`` para ``x <= 0`` (árvore inválida); a AST usa
  ``log|x|`` e ``0`` se ``|x| < log_epsilon``.
- ``^``: nativo ``safe_pow`` (``NaN`` para base negativa com expoente não inteiro e para
  ``0^negativo``); a AST usa ``|a|^b`` nesse caso e recorta em ``±power_clip``.
- ``exp``: nativo sem recorte (overflow -> ``Inf`` -> inválida); a AST recorta o argumento
  em ``±exp_clip``. ``sqrt`` (fora do GAP-06): nativo ``NaN`` para ``x < 0``; AST ``sqrt|x|``.

Qualquer ``NaN``/``Inf`` nativo torna a árvore inválida (``L = Inf``), então as árvores do
hall of fame são finitas no treino. Nos pontos de domínio válido (denominador
``|b| >= division_epsilon``, ``log`` com ``x >= log_epsilon``, ``|a^b| <= power_clip``,
``|arg exp| <= exp_clip``) ``predict`` via AST coincide com o ``predict`` nativo até
arredondamento de ponto flutuante (libm Julia vs numpy; tolerância relativa
:data:`PREDICT_REL_TOL` no teste). A auditoria de paridade detecta qualquer árvore em que a
diferença de semântica altere ``L``.

Conversão: a melhor expressão é o texto ``string_tree`` da árvore (coluna ``equation``)
lido por ``parse_infix`` — espelho exato da árvore otimizada (mesmo nº de nós, logo mesmo
``s``). A forma sympy do PySR é convertida por ``symbolic.from_sympy`` (sem ``sympify`` de
string) e registrada em ``metadata.extra["best_expression_sympy"]``; ela não é a principal
porque o sympy reescreve a árvore (``x^0.5 -> sqrt(x)``, dobra produtos) e mudaria ``s``.

Importar este módulo não inicializa o Julia: ``pysr``/``juliacall`` são importados só em
``fit``. Falha ao subir o Julia/juliacall em ``fit`` vira :class:`AdapterRuntimeError`
(célula ``error`` no runner). Versão do PySR fora de :data:`PYSR_SUPPORTED_SERIES` (a partir
de :data:`PYSR_MIN_VERSION`) é recusada (:class:`UnsupportedPysrError`), como no gplearn.
"""

from __future__ import annotations

import math
import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.llm.cache import (
    DEFAULT_CACHE_KEY_MODE,
    DEFAULT_CACHE_SIGNIFICANT_DIGITS,
    expression_key,
)
from physics_llm_sr.optimization.early_stopping import EarlyStopping
from physics_llm_sr.reproducibility.seeds import derive_seed
from physics_llm_sr.sr.interfaces import (
    AdapterRuntimeError,
    CandidateObjective,
    FitResult,
    FloatArray,
    GenerationRecord,
    LossBreakdown,
    RegressorMetadata,
    StopReason,
    ensure_feature_alignment,
)
from physics_llm_sr.symbolic.canonicalization import simplify
from physics_llm_sr.symbolic.expression import (
    Expr,
    ExpressionError,
    Op,
    Var,
    evaluate,
    from_sympy,
    parse_infix,
    to_infix,
)
from physics_llm_sr.symbolic.operators import DEFAULT_PROTECTION, OPERATORS, ProtectionParams

ALGORITHM: Final = "pysr"
#: Série do PySR validada (API ``julia_import``, ``loss_function``, ``warm_start``).
PYSR_SUPPORTED_SERIES = (2, 7)
#: Versão mínima validada dentro da série (pyproject: ``pysr>=2.7.2,<2.8``).
PYSR_MIN_VERSION = (2, 7, 2)
#: Nome da semente derivada (``reproducibility.seeds.derive_seed``).
SEED_NAME = "sr/pysr"
#: Operador do registro (GAP-06) -> operador nativo do PySR.
PYSR_OPERATORS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "^",
    "neg": "neg",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
}
#: Defaults dos hiperparâmetros ausentes em ``config.sr.pysr`` (registrados em metadata).
#: ``populations = 1``: hipótese GAP-08 (uma população de ``sr.population`` indivíduos).
DEFAULT_POPULATIONS = 1
#: ``maxsize`` padrão do PySR 2.x (nós).
DEFAULT_MAXSIZE = 30
#: ``None``: usa o default do PySR (380 ciclos por iteração no 2.x).
DEFAULT_NCYCLES_PER_ITERATION: int | None = None
#: ``tournament_selection_n`` padrão do PySR 2.x; recortado em ``population_size - 1``
#: quando a população é menor (o PySR exige ``n < population_size``; ex.: smoke).
DEFAULT_TOURNAMENT_SELECTION_N = 15
#: Precisão de ponto flutuante no Julia (paridade com a avaliação numpy em float64).
PYSR_PRECISION = 64
#: Parcimônia nativa desligada: ``w2*s`` já está em ``L``.
NATIVE_PARSIMONY = 0.0
#: Tolerância relativa da auditoria ``|loss_julia - L_python| / max(1, |L_python|)``.
AUDIT_REL_TOL = 1e-9
#: Tolerância relativa documentada para ``predict`` AST vs nativo (domínio válido).
PREDICT_REL_TOL = 1e-9
#: Prefixo dos nomes de variável passados ao PySR (``x1..xn``, 1-based como no Julia).
JULIA_VARIABLE_PREFIX = "x"

JULIA_PRELUDE = r"""
using PythonCall
const PLSR_CB = Ref{Any}(nothing)
const PLSR_P = Ref((w1=0.0, w2=0.0, w3=0.0, huber=true, delta=1.0,
                    normalized=false, var_y=1.0, max_nodes=1.0))
const PLSR_NCALLS = Ref(0)
plsr_set_callback!(f) = (PLSR_CB[] = f; nothing)
plsr_clear_callback!() = (PLSR_CB[] = nothing; nothing)
function plsr_set_params!(w1, w2, w3, huber, delta, normalized, var_y, max_nodes)
    PLSR_P[] = (w1=Float64(w1), w2=Float64(w2), w3=Float64(w3), huber=Bool(huber),
                delta=Float64(delta), normalized=Bool(normalized), var_y=Float64(var_y),
                max_nodes=Float64(max_nodes))
    PLSR_NCALLS[] = 0
    nothing
end
function plsr_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    pred, ok = eval_tree_array(tree, dataset.X, options)
    !ok && return L(Inf)
    p = PLSR_P[]
    e = 0.0
    @inbounds for i in eachindex(pred)
        r = Float64(pred[i]) - Float64(dataset.y[i])
        if p.huber
            a = abs(r)
            e += a <= p.delta ? 0.5 * a * a : p.delta * (a - 0.5 * p.delta)
        else
            e += r * r
        end
    end
    e /= length(pred)
    isfinite(e) || return L(Inf)
    s = Float64(count_nodes(tree))
    if p.normalized
        e /= p.var_y
        s /= p.max_nodes
    end
    total = p.w1 * e + p.w2 * s
    if p.w3 > 0
        PLSR_NCALLS[] += 1
        txt = string_tree(tree, options)
        # O PySR chama equation_search sem o GIL: reaquisição obrigatória.
        c = PythonCall.GIL.@lock pyconvert(Float64, PLSR_CB[](txt))
        isfinite(c) || return L(Inf)
        total += p.w3 * c
    end
    return L(total)
end
"""
#: Início da mensagem de aviso do PySR sobre ``constraints`` para ``^`` (filtrada).
PYSR_POW_WARNING = "You are using the `\\^` operator"
#: Nome da função Julia passada como ``loss_function`` (definida uma vez no prelúdio).
JULIA_LOSS_NAME = "plsr_loss"

_JULIA_READY = False


class UnsupportedPysrError(RuntimeError):
    """Configuração/objective incompatível com o mecanismo in-loop do PySR."""


# --------------------------------------------------------------------------- utilitários


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits or digits != piece:
            if digits:
                parts.append(int(digits))
            break
        parts.append(int(digits))
    return tuple(parts)


def check_pysr_version(version: str) -> None:
    """Recusa versões do PySR fora da série validada.

    Raises:
        UnsupportedPysrError: versão fora de ``PYSR_SUPPORTED_SERIES`` ou abaixo do mínimo.
    """
    parsed = _version_tuple(version)
    series = PYSR_SUPPORTED_SERIES
    if parsed[: len(series)] != series or parsed < PYSR_MIN_VERSION:
        raise UnsupportedPysrError(
            f"pysr {version} fora da série validada "
            f"{'.'.join(map(str, PYSR_MIN_VERSION))}..<{series[0]}.{series[1] + 1}: o "
            "mecanismo in-loop (loss_function + callback) precisa ser revalidado"
        )


def iteration_seed(seed: int, iteration: int) -> int:
    """``random_state`` da iteração ``iteration`` do laço warm start (determinística)."""
    return derive_seed(seed, f"it{iteration}")


def julia_variable_names(n_features: int) -> list[str]:
    """Nomes ``x1..xn`` usados no PySR/Julia (1-based, como o ``string_tree`` padrão)."""
    return [f"{JULIA_VARIABLE_PREFIX}{i + 1}" for i in range(n_features)]


def _rename(expr: Expr, mapping: dict[str, str]) -> Expr:
    if isinstance(expr, Var):
        return Var(mapping[expr.name])
    if isinstance(expr, Op):
        return Op(expr.name, tuple(_rename(c, mapping) for c in expr.children))
    return expr


def julia_text_to_expr(text: str, feature_names: Sequence[str]) -> Expr:
    """Texto ``string_tree`` do SymbolicRegression.jl -> AST comum (``^`` -> ``**``).

    As variáveis ``x1..xn`` viram ``feature_names[i]``. Usa ``parse_infix`` (whitelist).

    Raises:
        ExpressionError: texto fora da gramática (ex.: constante ``Inf``/``NaN``).
    """
    jl_names = julia_variable_names(len(feature_names))
    expr = parse_infix(text.replace("^", "**"), variables=jl_names)
    return _rename(expr, dict(zip(jl_names, feature_names, strict=True)))


def sympy_to_expr(obj: Any, feature_names: Sequence[str]) -> Expr:
    """Forma sympy do PySR -> AST via ``symbolic.from_sympy`` (sem ``sympify`` de string)."""
    jl_names = julia_variable_names(len(feature_names))
    expr = from_sympy(obj, variables=jl_names)
    return _rename(expr, dict(zip(jl_names, feature_names, strict=True)))


def pysr_operators(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(binary_operators, unary_operators)`` nativos do PySR para os operadores da config.

    Raises:
        UnsupportedPysrError: operador sem equivalente no mapeamento.
    """
    binary: list[str] = []
    unary: list[str] = []
    for name in names:
        if name not in PYSR_OPERATORS:
            raise UnsupportedPysrError(f"operador {name!r} sem equivalente no PySR")
        (binary if OPERATORS[name].arity == 2 else unary).append(PYSR_OPERATORS[name])
    return binary, unary


@dataclass(frozen=True)
class ObjectiveScale:
    """Parâmetros do ``CombinedObjective`` replicados na perda Julia."""

    weights: tuple[float, float, float]
    huber: bool
    huber_delta: float
    normalized: bool
    var_y: float
    max_nodes: float
    protection: ProtectionParams
    significant_digits: int
    key_mode: str

    @classmethod
    def from_objective(cls, objective: CandidateObjective, y: FloatArray) -> ObjectiveScale:
        """Lê pesos, métrica, δ, normalização e ``max_nodes`` do objective.

        Raises:
            UnsupportedPysrError: objective sem esses atributos (não replicável no Julia).
        """
        required = ("weights", "error_metric", "huber_delta", "normalization", "max_nodes")
        missing = [a for a in required if not hasattr(objective, a)]
        if missing:
            raise UnsupportedPysrError(
                f"objective sem {missing}: a perda Julia precisa replicar o CombinedObjective"
            )
        attrs: Any = objective
        w1, w2, w3 = (float(w) for w in attrs.weights)
        normalized = attrs.normalization == "normalized"
        with np.errstate(all="ignore"):
            var_y = float(np.var(np.asarray(y, dtype=np.float64)))
        return cls(
            weights=(w1, w2, w3),
            huber=attrs.error_metric == "huber",
            huber_delta=float(attrs.huber_delta),
            normalized=normalized,
            var_y=var_y,
            max_nodes=float(attrs.max_nodes) if normalized else 1.0,
            protection=getattr(objective, "protection", DEFAULT_PROTECTION),
            significant_digits=int(
                getattr(objective, "significant_digits", DEFAULT_CACHE_SIGNIFICANT_DIGITS)
            ),
            key_mode=str(getattr(objective, "key_mode", DEFAULT_CACHE_KEY_MODE)),
        )


@dataclass
class PenaltyCallback:
    """Callable chamado pelo Julia: texto da árvore -> ``c`` (ou ``+inf``).

    Nunca deixa exceção atravessar para o Julia: falhas viram ``+inf`` e ficam em
    ``failure`` (o adaptador levanta após a iteração).
    """

    objective: CandidateObjective
    X: FloatArray
    y: FloatArray
    feature_names: tuple[str, ...]
    scale: ObjectiveScale
    calls: int = 0
    memo_hits: int = 0
    parse_errors: int = 0
    non_finite: int = 0
    failure: BaseException | None = None
    memo: dict[str, float] = field(default_factory=dict)

    def __call__(self, text: Any) -> float:
        """Penalidade ``c`` da forma simplificada da árvore (texto ``string_tree``)."""
        self.calls += 1
        if self.failure is not None:
            return math.inf
        try:
            raw = julia_text_to_expr(str(text), self.feature_names)
        except ExpressionError:
            self.parse_errors += 1
            return math.inf
        return self.penalty(raw)

    def penalty(self, raw: Expr) -> float:
        """``c`` de ``simplify(raw)`` reutilizando o memo (0 chamadas LLM se já avaliada).

        Usado pelo callback e pelo registro/auditoria do adaptador (mesma forma).
        """
        if self.failure is not None:
            return math.inf
        try:
            expr = simplify(raw, self.scale.protection)
        except ExpressionError:
            self.parse_errors += 1
            return math.inf
        key = expression_key(expr, self.scale.significant_digits, self.scale.key_mode)
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached
        try:
            loss = self.objective.evaluate_population([expr], self.X, self.y)[0]
        except Exception as exc:  # noqa: BLE001 - exceção não pode atravessar o Julia
            self.failure = exc
            return math.inf
        c = loss.c if math.isfinite(loss.c) else math.inf
        if not math.isfinite(c):
            self.non_finite += 1
        self.memo[key] = c
        return c

    def stats(self) -> dict[str, Any]:
        """Contadores do callback (custo GAP-14)."""
        return {
            "callback_calls": self.calls,
            "unique_keys": len(self.memo),
            "memo_hits": self.memo_hits,
            "parse_errors": self.parse_errors,
            "non_finite_c": self.non_finite,
        }


def _julia() -> Any:
    """``juliacall.Main`` com o prelúdio carregado (importa ``pysr`` antes do juliacall)."""
    global _JULIA_READY
    # ``pysr.julia_import`` configura threads/sinais e devolve o ``Main`` do juliacall.
    from pysr.julia_import import jl

    if not _JULIA_READY:
        jl.seval("using SymbolicRegression")
        jl.seval(JULIA_PRELUDE)
        _JULIA_READY = True
    return jl


def _cfg(section: Any, name: str, default: Any, defaults_used: dict[str, Any]) -> Any:
    value = getattr(section, name, None)
    if value is None:
        defaults_used[name] = default
        return default
    return value


def _counter(objective: CandidateObjective, name: str) -> float:
    counters = getattr(objective, "counters", None)
    return float(getattr(counters, name, 0) or 0)


def _llm_calls(objective: CandidateObjective) -> float:
    """Chamadas ao backend LLM; sem métricas no avaliador (ex.: Fake), formas únicas avaliadas."""
    evaluator = getattr(objective, "evaluator", None)
    if evaluator is not None and getattr(evaluator, "metrics", None) is not None:
        return _counter(objective, "llm_calls")
    return _counter(objective, "unique_llm")


def _counters_dict(objective: CandidateObjective) -> dict[str, float]:
    counters = getattr(objective, "counters", None)
    if counters is None or not hasattr(counters, "as_dict"):
        return {}
    return {k: float(v) for k, v in counters.as_dict().items()}


def _rel_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(1.0, abs(b))


# --------------------------------------------------------------------------- adaptador


class PySRRegressor:
    """``SymbolicRegressor`` do projeto sobre ``pysr.PySRRegressor`` (LLM in-loop)."""

    def __init__(self) -> None:
        """Estado vazio até ``fit``."""
        self._best: Expr | None = None
        self._best_index: int | None = None
        self._feature_names: tuple[str, ...] = ()
        self._protection: ProtectionParams = DEFAULT_PROTECTION
        self._history: list[GenerationRecord] = []
        self._metadata = RegressorMetadata(ALGORITHM, False, "baseline_only")
        self.model: Any = None
        self.callback: PenaltyCallback | None = None
        self.stop_reason: StopReason | None = None
        self.audit: dict[str, Any] = {}

    # ----------------------------------------------------------------- construção

    def build_params(
        self, config: ExperimentConfig, seed: int, defaults_used: dict[str, Any]
    ) -> dict[str, Any]:
        """Parâmetros do ``pysr.PySRRegressor`` (config + defaults registrados)."""
        pysr_cfg = config.sr.pysr
        binary, unary = pysr_operators(config.sr.operators[ALGORITHM])
        params: dict[str, Any] = {
            "binary_operators": binary,
            "unary_operators": unary,
            "loss_function": JULIA_LOSS_NAME,
            "parsimony": NATIVE_PARSIMONY,
            "niterations": 1,
            "warm_start": True,
            "populations": int(_cfg(pysr_cfg, "populations", DEFAULT_POPULATIONS, defaults_used)),
            "population_size": int(
                _cfg(pysr_cfg, "population_size", config.sr.population, defaults_used)
            ),
            "maxsize": int(_cfg(pysr_cfg, "maxsize", DEFAULT_MAXSIZE, defaults_used)),
            "batching": False,
            "parallelism": "serial",
            "deterministic": True,
            "random_state": seed,
            "precision": PYSR_PRECISION,
            "model_selection": "accuracy",
            "progress": False,
            "verbosity": 0,
            "temp_equation_file": True,
        }
        tournament_n = int(
            _cfg(pysr_cfg, "tournament_selection_n", DEFAULT_TOURNAMENT_SELECTION_N, defaults_used)
        )
        if tournament_n >= params["population_size"]:
            tournament_n = params["population_size"] - 1
            defaults_used["tournament_selection_n_clamped_to"] = tournament_n
        params["tournament_selection_n"] = tournament_n
        ncycles = _cfg(
            pysr_cfg, "ncycles_per_iteration", DEFAULT_NCYCLES_PER_ITERATION, defaults_used
        )
        if ncycles is not None:
            params["ncycles_per_iteration"] = int(ncycles)
        return params

    # ----------------------------------------------------------------- fit

    def fit(
        self,
        train: Dataset,
        objective: CandidateObjective,
        config: ExperimentConfig,
        validation: Dataset | None = None,
    ) -> FitResult:
        """Busca do PySR com ``L`` completa como perda (ver módulo).

        ``validation`` não participa da seleção (anti-vazamento) e é ignorado.

        Raises:
            EvaluationFailedError: falha do avaliador LLM durante a busca (resultado
                descartado; ``stop_reason = "error"``).
            UnsupportedPysrError: objective/configuração incompatível.
        """
        ensure_feature_alignment(train, objective)
        start = time.perf_counter()
        self._feature_names = tuple(train.feature_names)
        self._history = []
        self._best = None
        self._best_index = None
        self.stop_reason = None
        self.audit = {}
        X = np.asarray(train.X, dtype=np.float64)  # noqa: N806
        y = np.asarray(train.y_observed, dtype=np.float64)
        scale = ObjectiveScale.from_objective(objective, y)
        if scale.normalized and not scale.var_y > 0:
            raise UnsupportedPysrError("modo normalized exige Var(y_train) > 0")
        self._protection = scale.protection
        in_loop = scale.weights[2] > 0
        seed = derive_seed(config.seeds.master, SEED_NAME)
        defaults_used: dict[str, Any] = {}
        niterations = int(_cfg(config.sr.pysr, "niterations", config.sr.generations, defaults_used))
        params = self.build_params(config, seed, defaults_used)

        try:
            jl = _julia()
            import pysr
        except Exception as exc:  # juliacall/Julia podem levantar qualquer tipo ao subir
            raise AdapterRuntimeError(
                f"Julia/juliacall indisponível ao iniciar o PySR em fit: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        check_pysr_version(str(pysr.__version__))
        _NativePySRRegressor = pysr.PySRRegressor  # noqa: N806 - classe nativa

        callback: PenaltyCallback | None = None
        if in_loop:
            callback = PenaltyCallback(objective, X, y, self._feature_names, scale)
            jl.plsr_set_callback_b(callback)
        else:
            jl.plsr_clear_callback_b()
        self.callback = callback
        w1, w2, w3 = scale.weights
        jl.plsr_set_params_b(
            w1, w2, w3, scale.huber, scale.huber_delta, scale.normalized, scale.var_y,
            scale.max_nodes,
        )  # fmt: skip

        stopper = EarlyStopping.from_config(config.sr.early_stopping)
        stop: StopReason = "max_generations"
        jl_names = julia_variable_names(len(self._feature_names))
        equations: Any = None
        best_loss = math.inf
        iteration_seeds: list[int] = []
        with tempfile.TemporaryDirectory(prefix="plsr_pysr_") as tmp:
            model = _NativePySRRegressor(tempdir=tmp, **params)
            self.model = model
            try:
                for iteration in range(niterations):
                    it_start = time.perf_counter()
                    calls0 = _llm_calls(objective)
                    hits0 = _counter(objective, "cache_hits")
                    memo0 = callback.memo_hits if callback is not None else 0
                    iteration_seeds.append(iteration_seed(seed, iteration))
                    model.set_params(random_state=iteration_seeds[-1])
                    with warnings.catch_warnings():
                        # Aviso do PySR sobre ``constraints`` de ``^``: o GAP-06 não as define.
                        warnings.filterwarnings("ignore", message=PYSR_POW_WARNING)
                        model.fit(X, y, variable_names=jl_names)
                    if callback is not None and callback.failure is not None:
                        raise callback.failure
                    equations = model.equations_
                    i = int(np.argmin(np.asarray(equations["loss"], dtype=np.float64)))
                    best_loss = float(equations["loss"].iloc[i])
                    best = julia_text_to_expr(
                        str(equations["equation"].iloc[i]), self._feature_names
                    )
                    parts = self._evaluate([best], objective, X, y, callback)[0]
                    if callback is not None and callback.failure is not None:
                        raise callback.failure
                    memo1 = callback.memo_hits if callback is not None else 0
                    self._history.append(
                        GenerationRecord(
                            generation=iteration,
                            best_loss=best_loss,
                            best_e=parts.e,
                            best_s=parts.s,
                            best_c=parts.c,
                            best_expression=to_infix(best),
                            n_llm_calls=int(_llm_calls(objective) - calls0),
                            n_cache_hits=int(
                                _counter(objective, "cache_hits") - hits0 + memo1 - memo0
                            ),
                            elapsed=time.perf_counter() - it_start,
                        )
                    )
                    self._best, self._best_index = best, i
                    if math.isfinite(best_loss) and stopper.update(best_loss):
                        stop = "early_stopping"
                        break
            except BaseException:
                self.stop_reason = "error"
                self._best = None
                self._best_index = None
                raise
            finally:
                jl.plsr_clear_callback_b()
        assert self._best is not None and equations is not None
        self.stop_reason = stop
        self.audit = self._audit(equations, objective, X, y, callback)
        audit_failed = not self.audit["ok"]
        julia_calls = int(jl.seval("PLSR_NCALLS[]"))
        try:
            sympy_best: str | None = to_infix(
                sympy_to_expr(equations["sympy_format"].iloc[self._best_index], self._feature_names)
            )
        except (ExpressionError, ValueError) as exc:
            sympy_best = f"<não convertida: {exc}>"
        self._metadata = RegressorMetadata(
            algorithm=ALGORITHM,
            llm_in_loop=in_loop,
            integration_status="integrated" if in_loop else "baseline_only",
            extra={
                "mechanism": "julia_loss_function_python_callback_gil_lock",
                "history_mechanism": "warm_start_one_iteration_per_fit",
                "pysr_version": pysr.__version__,
                "seed": seed,
                "seed_name": SEED_NAME,
                "iteration_seeds": iteration_seeds,
                "iteration_seed_rule": 'derive_seed(seed, f"it{i}")',
                "niterations": niterations,
                "pysr_params": {
                    k: v for k, v in params.items() if k not in ("loss_function", "warm_start")
                },
                "defaults_used": defaults_used,
                "objective_scale": {
                    "weights": scale.weights,
                    "error_metric": "huber" if scale.huber else "mse",
                    "huber_delta": scale.huber_delta,
                    "normalized": scale.normalized,
                    "var_y_train": scale.var_y,
                    "max_nodes": scale.max_nodes,
                },
                "cache_significant_digits": scale.significant_digits,
                "cache_key_mode": scale.key_mode,
                "julia_loss_llm_calls": julia_calls,
                "callback": callback.stats() if callback is not None else None,
                "audit": self.audit,
                "audit_failed": audit_failed,
                "audit_reason": (
                    f"auditoria de paridade Julia x Python fora da tolerância "
                    f"{AUDIT_REL_TOL:g}: max_rel_diff={self.audit['max_rel_diff']:.3g} em "
                    f"{self.audit['n_rows_out_of_tolerance']} linha(s) do hall of fame"
                )
                if audit_failed
                else None,
                "best_loss_julia": best_loss,
                "best_expression_sympy": sympy_best,
                "operator_semantics": "nativa do PySR (ver docstring de sr.pysr_adapter)",
            },
        )
        return FitResult(
            best_expression=self._best,
            history=tuple(self._history),
            metadata=self._metadata,
            elapsed_seconds=time.perf_counter() - start,
            n_generations=len(self._history),
            stop_reason=stop,
            llm_counters=_counters_dict(objective),
        )

    @staticmethod
    def _evaluate(
        exprs: Sequence[Expr],
        objective: CandidateObjective,
        X: FloatArray,  # noqa: N803
        y: FloatArray,
        callback: PenaltyCallback | None,
    ) -> list[LossBreakdown]:
        """``L`` como na perda Julia: ``e``/``s`` da árvore crua, ``c`` do memo do callback.

        Sem callback (``w3 = 0``) o objective não chama o avaliador. Com callback, ``e``
        e ``s`` vêm de ``objective.components`` (sem LLM) e ``c`` de
        :meth:`PenaltyCallback.penalty` (forma simplificada, memo compartilhado).
        """
        components = getattr(objective, "components", None)
        if callback is None or not callable(components):
            return list(objective.evaluate_population(list(exprs), X, y))
        w1, w2, w3 = callback.scale.weights
        out: list[LossBreakdown] = []
        for expr in exprs:
            e, s = components(expr, X, y)
            if not math.isfinite(e):
                out.append(LossBreakdown(math.inf, math.inf, s, math.nan, None))
                continue
            c = callback.penalty(expr)
            if callback.failure is not None:
                raise callback.failure
            out.append(LossBreakdown(w1 * e + w2 * s + w3 * c, e, s, c, None))
        return out

    def _audit(
        self,
        equations: Any,
        objective: CandidateObjective,
        X: FloatArray,  # noqa: N803
        y: FloatArray,
        callback: PenaltyCallback | None,
    ) -> dict[str, Any]:
        """Paridade ``loss`` Julia vs ``L`` Python (mesma forma do callback) no hall of fame."""
        exprs = [julia_text_to_expr(str(t), self._feature_names) for t in equations["equation"]]
        losses = self._evaluate(exprs, objective, X, y, callback)
        julia = [float(v) for v in equations["loss"]]
        diffs = [_rel_diff(j, lb.total) for j, lb in zip(julia, losses, strict=True)]
        assert self._best_index is not None
        rows = [
            {
                "equation": to_infix(e),
                "loss_julia": j,
                "L_python": lb.total,
                "e": lb.e,
                "s": lb.s,
                "c": lb.c,
                "rel_diff": d,
            }
            for e, j, lb, d in zip(exprs, julia, losses, diffs, strict=True)
        ]
        best_diff = diffs[self._best_index]
        return {
            "tolerance_rel": AUDIT_REL_TOL,
            "best_rel_diff": best_diff,
            "best_ok": best_diff <= AUDIT_REL_TOL,
            "max_rel_diff": max(diffs),
            "n_rows_out_of_tolerance": sum(d > AUDIT_REL_TOL for d in diffs),
            "ok": max(diffs) <= AUDIT_REL_TOL,
            "rows": rows,
        }

    # ----------------------------------------------------------------- consultas

    def predict(self, X: FloatArray) -> FloatArray:  # noqa: N803 - nome do contrato (PRD)
        """``symbolic.evaluate`` da melhor expressão (mesmos ``feature_names`` do fit)."""
        return evaluate(self.get_best_expression(), X, self._feature_names, self._protection)

    def predict_native(self, X: FloatArray) -> FloatArray:  # noqa: N803
        """Predição nativa do PySR para a melhor linha do hall of fame (semântica nativa)."""
        if self.model is None or self._best_index is None:
            raise RuntimeError("fit ainda não foi executado")
        return np.asarray(self.model.predict(X, index=self._best_index), dtype=np.float64)

    def get_best_expression(self) -> Expr:
        """Melhor expressão (menor ``L`` no hall of fame) na AST comum."""
        if self._best is None:
            raise RuntimeError("fit ainda não foi executado (ou falhou)")
        return self._best

    def get_search_history(self) -> Sequence[GenerationRecord]:
        """Histórico por iteração do PySR."""
        return tuple(self._history)

    def get_metadata(self) -> RegressorMetadata:
        """Metadados do último ``fit``."""
        return self._metadata
