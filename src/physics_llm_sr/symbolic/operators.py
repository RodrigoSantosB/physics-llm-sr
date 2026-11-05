"""Registro de operadores e semântica numérica protegida.

Contrato (consumido por adaptadores DEAP/gplearn/PySR, avaliador e métricas)
-----------------------------------------------------------------------------
Cada operador é um :class:`OperatorSpec` com ``name`` (token usado na AST e na
notação prefixa), ``arity``, ``commutative``, ``func`` (função numpy protegida,
vetorizada), ``infix`` (símbolo infixo para operadores binários, ou ``None`` para
operadores escritos como chamada ``nome(...)``) e ``sympy_func`` (construtor
sympy com a semântica matemática *não protegida*).

O registro aqui é o universo de operadores; o subconjunto efetivamente usado por
cada algoritmo vem da configuração (GAP-06), não deste módulo.

Semântica protegida (parâmetros em :class:`ProtectionParams`, que espelha
``config.schemas.SymbolicConfig``; nenhum número mágico local):

- ``add``, ``sub``, ``mul``, ``neg``, ``sin``, ``cos``: operação usual.
- ``div(a, b)``: ``a / b`` se ``|b| >= division_epsilon``; caso contrário devolve
  ``division_fallback`` (=1.0, mesmo comportamento da divisão protegida do
  gplearn, que devolve 1 quando o denominador é ~0).
- ``log(x)``: ``log(|x|)`` se ``|x| >= log_epsilon``; caso contrário ``0.0``
  (padrão gplearn; GAP-06, eps = 0.001). Fronteira: o texto do GAP-06 usa
  ``|x| > 0.001`` / ``|den| <= 0.001``; aqui a comparação é estrita (``<``),
  diferença só em ``|x| == eps`` exatamente.
- ``sqrt(x)``: ``sqrt(|x|)``.
- ``exp(x)``: ``exp(clip(x, -exp_clip, exp_clip))``.
- ``pow(a, b)``: se ``b`` é inteiro, potência real com sinal ``a**b``; se ``b``
  não é inteiro, ``|a|**b`` (evita resultados complexos para base negativa).
  ``0**negativo`` (infinito) e overflow são saturados; o resultado final é
  recortado em ``[-power_clip, power_clip]``.

Saneamento final (todos os operadores, ver :func:`sanitize`): ``NaN`` vira
``NAN_REPLACEMENT`` (0.0) e ``±inf`` vira ``±power_clip``. Assim nenhum operador
devolve valor não finito, inclusive quando recebe entradas não finitas.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import sympy

from physics_llm_sr.config.schemas import SymbolicConfig

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

NAN_REPLACEMENT = 0.0
"""Valor que substitui ``NaN`` na saída de qualquer operador."""

LOG_UNDERFLOW_VALUE = 0.0
"""Valor de ``log(x)`` quando ``|x| < log_epsilon`` (padrão gplearn)."""


class UnknownOperatorError(ValueError):
    """Operador fora do registro."""


@dataclass(frozen=True)
class ProtectionParams:
    """Parâmetros da semântica protegida (espelho de ``SymbolicConfig``)."""

    division_epsilon: float
    division_fallback: float
    log_epsilon: float
    exp_clip: float
    power_clip: float

    @classmethod
    def from_config(cls, cfg: SymbolicConfig) -> ProtectionParams:
        """Constrói os parâmetros a partir da seção ``symbolic`` da configuração."""
        return cls(
            division_epsilon=cfg.division_epsilon,
            division_fallback=cfg.division_fallback,
            log_epsilon=cfg.log_epsilon,
            exp_clip=cfg.exp_clip,
            power_clip=cfg.power_clip,
        )


DEFAULT_PROTECTION = ProtectionParams.from_config(SymbolicConfig())
"""Parâmetros padrão, derivados dos defaults de ``SymbolicConfig``."""


def sanitize(values: FloatArray, params: ProtectionParams) -> FloatArray:
    """Substitui não finitos: ``NaN -> NAN_REPLACEMENT``, ``±inf -> ±power_clip``."""
    return np.nan_to_num(
        values, nan=NAN_REPLACEMENT, posinf=params.power_clip, neginf=-params.power_clip
    )


def _add(p: ProtectionParams, a: FloatArray, b: FloatArray) -> FloatArray:
    return np.add(a, b)


def _sub(p: ProtectionParams, a: FloatArray, b: FloatArray) -> FloatArray:
    return np.subtract(a, b)


def _mul(p: ProtectionParams, a: FloatArray, b: FloatArray) -> FloatArray:
    return np.multiply(a, b)


def _div(p: ProtectionParams, a: FloatArray, b: FloatArray) -> FloatArray:
    small = np.abs(b) < p.division_epsilon
    safe_b = np.where(small, 1.0, b)
    return np.where(small, p.division_fallback, np.divide(a, safe_b))


def _neg(p: ProtectionParams, a: FloatArray) -> FloatArray:
    return np.negative(a)


def _pow(p: ProtectionParams, a: FloatArray, b: FloatArray) -> FloatArray:
    a, b = np.broadcast_arrays(a, b)
    integer_exp = np.isfinite(b) & (b == np.round(b))
    base = np.where(integer_exp, a, np.abs(a))
    out = np.power(base, b)
    out = sanitize(out, p)
    return np.clip(out, -p.power_clip, p.power_clip)


def _exp(p: ProtectionParams, a: FloatArray) -> FloatArray:
    return np.exp(np.clip(a, -p.exp_clip, p.exp_clip))


def _log(p: ProtectionParams, a: FloatArray) -> FloatArray:
    mag = np.abs(a)
    small = mag < p.log_epsilon
    return np.where(small, LOG_UNDERFLOW_VALUE, np.log(np.where(small, 1.0, mag)))


def _sqrt(p: ProtectionParams, a: FloatArray) -> FloatArray:
    return np.sqrt(np.abs(a))


def _sin(p: ProtectionParams, a: FloatArray) -> FloatArray:
    return np.sin(a)


def _cos(p: ProtectionParams, a: FloatArray) -> FloatArray:
    return np.cos(a)


@dataclass(frozen=True)
class OperatorSpec:
    """Especificação de um operador do registro.

    Attributes:
        name: token canônico (AST, notação prefixa e chamada infixa ``name(...)``).
        arity: número de filhos (1 ou 2).
        commutative: se a ordem dos filhos é irrelevante (usado na canonização e
            na distância entre árvores).
        func: função numpy protegida ``func(params, *arrays)`` (sem saneamento;
            use :func:`apply_operator`).
        infix: símbolo infixo para binários (``+ - * / **``) ou ``None``.
        sympy_func: construtor sympy com semântica matemática não protegida.
    """

    name: str
    arity: int
    commutative: bool
    func: Callable[..., FloatArray]
    infix: str | None
    sympy_func: Callable[..., Any]


def _sympy_sub(a: Any, b: Any) -> Any:
    return a - b


def _sympy_div(a: Any, b: Any) -> Any:
    return a / b


_SPECS: tuple[OperatorSpec, ...] = (
    OperatorSpec("add", 2, True, _add, "+", lambda a, b: a + b),
    OperatorSpec("sub", 2, False, _sub, "-", _sympy_sub),
    OperatorSpec("mul", 2, True, _mul, "*", lambda a, b: a * b),
    OperatorSpec("div", 2, False, _div, "/", _sympy_div),
    OperatorSpec("neg", 1, False, _neg, None, lambda a: -a),
    OperatorSpec("pow", 2, False, _pow, "**", lambda a, b: a**b),
    OperatorSpec("exp", 1, False, _exp, None, sympy.exp),
    OperatorSpec("log", 1, False, _log, None, sympy.log),
    OperatorSpec("sqrt", 1, False, _sqrt, None, sympy.sqrt),
    OperatorSpec("sin", 1, False, _sin, None, sympy.sin),
    OperatorSpec("cos", 1, False, _cos, None, sympy.cos),
)

OPERATORS: Mapping[str, OperatorSpec] = MappingProxyType({s.name: s for s in _SPECS})
"""Registro imutável ``nome -> OperatorSpec``."""

INFIX_TO_NAME: Mapping[str, str] = MappingProxyType(
    {s.infix: s.name for s in _SPECS if s.infix is not None}
)
"""Símbolo infixo binário -> nome do operador."""


def get_operator(name: str) -> OperatorSpec:
    """Devolve a especificação do operador ou levanta :class:`UnknownOperatorError`."""
    try:
        return OPERATORS[name]
    except KeyError:
        raise UnknownOperatorError(f"operador desconhecido: {name!r}") from None


def apply_operator(
    name: str, *args: FloatArray, params: ProtectionParams = DEFAULT_PROTECTION
) -> FloatArray:
    """Aplica o operador protegido e saneia o resultado (sempre finito)."""
    spec = get_operator(name)
    if len(args) != spec.arity:
        raise ValueError(f"{name} espera {spec.arity} argumento(s), recebeu {len(args)}")
    arrays = [np.asarray(a, dtype=np.float64) for a in args]
    with np.errstate(all="ignore"):
        out = spec.func(params, *arrays)
    return sanitize(np.asarray(out, dtype=np.float64), params)
