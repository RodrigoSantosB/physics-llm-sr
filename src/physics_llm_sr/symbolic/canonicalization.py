"""Formas canônicas da AST, com três finalidades separadas de propósito.

1. :func:`canonicalize` / :func:`canonical_key` — chave determinística para
   cache (PRD §15): filhos de operadores comutativos ordenados e constantes
   arredondadas a ``significant_digits`` dígitos significativos. NÃO simplifica
   (``x + 0`` e ``x`` têm chaves diferentes), para que duas expressões com a mesma
   chave tenham também a mesma estrutura para a distância entre árvores (a menos
   de ordem de filhos comutativos, que a distância já ignora, e do arredondamento
   de constantes).
2. :func:`simplify` — simplificações triviais opcionais (identidades neutras e
   dobra de subárvores só de constantes). Preserva exatamente a avaliação
   protegida. Nunca é aplicada implicitamente.
3. :func:`structural_normal_form` — canonização mínima exigida pelo GAP-10
   antes da distância (``references/scientific_assumptions.md``): (i) dobra
   subárvores só de constantes (``2*9.81 -> 19.62``); (ii) ``sqrt(u) ->
   pow(u, 0.5)``; (iii) ``exp(c*log(u))`` e ``exp(log(u)*c) -> pow(u, c)``. Sem
   simplificação algébrica: ``x + 0`` continua diferente de ``x``.
"""

from __future__ import annotations

import numpy as np

from physics_llm_sr.symbolic.expression import Const, Expr, ExpressionError, Op, Var, to_prefix
from physics_llm_sr.symbolic.operators import (
    DEFAULT_PROTECTION,
    OPERATORS,
    ProtectionParams,
    apply_operator,
)

DEFAULT_SIGNIFICANT_DIGITS = 6
"""Dígitos significativos das constantes na chave canônica de cache."""

SQRT_EXPONENT = 0.5
"""Expoente usado na reescrita ``sqrt(u) -> pow(u, 0.5)`` (GAP-10)."""

_ZERO = 0.0
_ONE = 1.0


def round_significant(value: float, digits: int) -> float:
    """Arredonda ``value`` para ``digits`` dígitos significativos (``-0.0 -> 0.0``)."""
    if digits < 1:
        raise ValueError("digits deve ser >= 1")
    if value == 0.0:
        return 0.0
    rounded = float(f"{value:.{digits - 1}e}")
    return rounded + 0.0  # normaliza -0.0


def _sort_key(expr: Expr) -> str:
    return " ".join(to_prefix(expr))


def _check(expr: Expr) -> None:
    if not isinstance(expr, (Const, Var, Op)):
        raise ExpressionError(f"nó desconhecido: {expr!r}")


def canonicalize(expr: Expr, significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS) -> Expr:
    """Ordena filhos comutativos e arredonda constantes; não simplifica."""
    _check(expr)
    if isinstance(expr, Const):
        return Const(round_significant(expr.value, significant_digits))
    if isinstance(expr, Var):
        return expr
    assert isinstance(expr, Op)
    children = tuple(canonicalize(c, significant_digits) for c in expr.children)
    if OPERATORS[expr.name].commutative:
        children = tuple(sorted(children, key=_sort_key))
    return Op(expr.name, children)


def canonical_key(expr: Expr, significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS) -> str:
    """Chave textual determinística (prefixa) da forma canônica, para cache."""
    return _sort_key(canonicalize(expr, significant_digits))


def _fold(op: Op, params: ProtectionParams) -> Const:
    args = [np.array([c.value]) for c in op.children if isinstance(c, Const)]
    return Const(float(apply_operator(op.name, *args, params=params)[0]))


def _is_const(expr: Expr, value: float) -> bool:
    return isinstance(expr, Const) and expr.value == value


def simplify(expr: Expr, params: ProtectionParams = DEFAULT_PROTECTION) -> Expr:
    """Aplica simplificações triviais que preservam exatamente a avaliação protegida.

    Regras: dobra de subárvores só de constantes; ``x+0``, ``0+x``, ``x-0``,
    ``x*1``, ``1*x``, ``x/1 -> x``; ``x*0``, ``0*x -> 0``; ``-(-x) -> x``.
    ``pow(x, 1)`` não é simplificado porque ``pow`` recorta em ``power_clip``.
    """
    _check(expr)
    if not isinstance(expr, Op):
        return expr
    children = tuple(simplify(c, params) for c in expr.children)
    node = Op(expr.name, children)
    if all(isinstance(c, Const) for c in children):
        return _fold(node, params)
    name = node.name
    if name == "neg":
        (child,) = children
        if isinstance(child, Op) and child.name == "neg":
            return child.children[0]
        return node
    left, right = children if len(children) == 2 else (children[0], children[0])
    if name == "add":
        if _is_const(right, _ZERO):
            return left
        if _is_const(left, _ZERO):
            return right
    elif name == "sub" and _is_const(right, _ZERO):
        return left
    elif name == "mul":
        if _is_const(left, _ZERO) or _is_const(right, _ZERO):
            return Const(_ZERO)
        if _is_const(right, _ONE):
            return left
        if _is_const(left, _ONE):
            return right
    elif name == "div" and _is_const(right, _ONE):
        return left
    return node


def _exp_log_power(arg: Expr) -> Op | None:
    """Reconhece ``c*log(u)`` ou ``log(u)*c`` e devolve ``pow(u, c)``."""
    if not (isinstance(arg, Op) and arg.name == "mul"):
        return None
    left, right = arg.children
    for coeff, other in ((left, right), (right, left)):
        if isinstance(coeff, Const) and isinstance(other, Op) and other.name == "log":
            return Op("pow", (other.children[0], coeff))
    return None


def structural_normal_form(expr: Expr, params: ProtectionParams = DEFAULT_PROTECTION) -> Expr:
    """Aplica a canonização mínima do GAP-10 (dobra de constantes, sqrt e exp-log)."""
    _check(expr)
    if not isinstance(expr, Op):
        return expr
    children = tuple(structural_normal_form(c, params) for c in expr.children)
    node = Op(expr.name, children)
    if all(isinstance(c, Const) for c in children):
        return _fold(node, params)
    if node.name == "sqrt":
        return Op("pow", (children[0], Const(SQRT_EXPONENT)))
    if node.name == "exp":
        rewritten = _exp_log_power(children[0])
        if rewritten is not None:
            return rewritten
    return node
