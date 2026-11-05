"""AST comum de expressões simbólicas, avaliação vetorizada e conversões seguras.

Contrato (consumido por adaptadores SR, avaliador LLM, métricas e cache)
------------------------------------------------------------------------
- Nós imutáveis e hasheáveis: :class:`Const` (``value: float`` finito),
  :class:`Var` (``name: str``) e :class:`Op` (``name`` do registro de
  :mod:`physics_llm_sr.symbolic.operators`, ``children: tuple[Expr, ...]`` com
  aridade validada). Todos herdam de :class:`Expr`.
- :func:`evaluate` avalia de forma vetorizada com a semântica protegida
  (resultado sempre finito, forma ``(n,)``).
- :func:`size` = número de nós; :func:`depth` = altura com folha = 0 (mesma
  convenção do ``height`` do DEAP).
- Notação prefixa genérica (nome + aridade do registro), usada pelos adaptadores
  DEAP/gplearn: :func:`to_prefix` / :func:`from_prefix`. Constantes são
  serializadas com ``repr(float)`` (roundtrip exato).
- Notação infixa: :func:`to_infix` produz texto com parênteses mínimos que
  :func:`parse_infix` relê na *mesma* árvore. :func:`parse_infix` é o único
  ponto de entrada de texto não confiável: usa ``ast.parse(mode="eval")`` com
  whitelist estrita de nós (``BinOp`` com ``+ - * / **``, ``UnaryOp`` com
  ``-``/``+``, ``Call`` de nomes do registro sem keywords, ``Name`` de variáveis
  permitidas, ``Constant`` numérico real). Qualquer outro nó levanta
  :class:`UnsafeExpressionError`. Nada é executado: não há ``eval``/``exec``
  nem ``sympify`` de string neste pacote.
- sympy (PySR): :func:`to_sympy` / :func:`from_sympy` operam apenas sobre
  objetos sympy (nunca strings). O lado sympy usa semântica matemática não
  protegida; a equivalência numérica vale no domínio válido das funções.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy

from physics_llm_sr.symbolic.operators import (
    DEFAULT_PROTECTION,
    INFIX_TO_NAME,
    OPERATORS,
    FloatArray,
    ProtectionParams,
    apply_operator,
    get_operator,
)

MAX_INFIX_LENGTH = 10_000
"""Tamanho máximo (caracteres) aceito por :func:`parse_infix`."""

DEFAULT_LEAF_PROBABILITY = 0.3
"""Probabilidade de folha em cada nível de :func:`random_expression`."""

RANDOM_VARIABLE_LEAF_PROBABILITY = 0.5
"""Probabilidade de uma folha sorteada ser variável (senão, constante)."""

RANDOM_CONSTANT_SCALE = 10.0
"""Desvio-padrão das constantes sorteadas por :func:`random_expression`."""

_VAR_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


class ExpressionError(ValueError):
    """Expressão malformada ou incompatível com os dados."""


class UnsafeExpressionError(ExpressionError):
    """Texto infixo contém construção fora da whitelist."""


class Expr:
    """Base dos nós da AST (não instanciar diretamente)."""

    __slots__ = ()


@dataclass(frozen=True)
class Const(Expr):
    """Constante numérica finita."""

    value: float

    def __post_init__(self) -> None:
        """Normaliza para ``float`` e rejeita valores não finitos."""
        try:
            value = float(self.value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExpressionError(f"constante inválida: {self.value!r}") from exc
        if not math.isfinite(value):
            raise ExpressionError(f"constante não finita: {value!r}")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class Var(Expr):
    """Variável de entrada (nome de feature)."""

    name: str

    def __post_init__(self) -> None:
        """Valida o identificador (sem ``_`` inicial e sem colidir com operadores)."""
        if not isinstance(self.name, str) or not _VAR_NAME.match(self.name):
            raise ExpressionError(f"nome de variável inválido: {self.name!r}")
        if self.name in OPERATORS:
            raise ExpressionError(f"nome de variável colide com operador: {self.name!r}")


@dataclass(frozen=True)
class Op(Expr):
    """Aplicação de um operador do registro a ``children``."""

    name: str
    children: tuple[Expr, ...]

    def __post_init__(self) -> None:
        """Valida operador, aridade e tipo dos filhos."""
        spec = get_operator(self.name)
        children = tuple(self.children)
        if len(children) != spec.arity:
            raise ExpressionError(
                f"{self.name} espera {spec.arity} filho(s), recebeu {len(children)}"
            )
        if not all(isinstance(c, Expr) for c in children):
            raise ExpressionError("filhos de Op devem ser nós Expr")
        object.__setattr__(self, "children", children)


# --------------------------------------------------------------------------- métricas


def size(expr: Expr) -> int:
    """Número de nós da árvore."""
    if isinstance(expr, Op):
        return 1 + sum(size(c) for c in expr.children)
    return 1


def depth(expr: Expr) -> int:
    """Altura da árvore (folha = 0)."""
    if isinstance(expr, Op):
        return 1 + max(depth(c) for c in expr.children)
    return 0


def variables(expr: Expr) -> frozenset[str]:
    """Conjunto de nomes de variáveis usados."""
    if isinstance(expr, Var):
        return frozenset({expr.name})
    if isinstance(expr, Op):
        return frozenset().union(*(variables(c) for c in expr.children))
    return frozenset()


# --------------------------------------------------------------------------- avaliação


def evaluate(
    expr: Expr,
    X: np.ndarray[Any, Any],  # noqa: N803 - nome do contrato (PRD)
    feature_names: Sequence[str],
    params: ProtectionParams = DEFAULT_PROTECTION,
) -> FloatArray:
    """Avalia ``expr`` nas linhas de ``X`` (forma ``(n, d)``); devolve ``(n,)`` finito."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(feature_names):
        raise ExpressionError(
            f"X com forma {data.shape} incompatível com {len(feature_names)} feature(s)"
        )
    index = {name: i for i, name in enumerate(feature_names)}
    n = data.shape[0]

    def _eval(node: Expr) -> FloatArray:
        if isinstance(node, Const):
            return np.full(n, node.value, dtype=np.float64)
        if isinstance(node, Var):
            if node.name not in index:
                raise ExpressionError(f"variável {node.name!r} ausente em feature_names")
            column: FloatArray = data[:, index[node.name]].copy()
            return column
        if isinstance(node, Op):
            return apply_operator(node.name, *(_eval(c) for c in node.children), params=params)
        raise ExpressionError(f"nó desconhecido: {node!r}")

    return _eval(expr)


# --------------------------------------------------------------------------- prefixa


def to_prefix(expr: Expr) -> tuple[str, ...]:
    """Serializa em notação prefixa: nomes de operador, variáveis e ``repr(float)``."""
    if isinstance(expr, Const):
        return (repr(expr.value),)
    if isinstance(expr, Var):
        return (expr.name,)
    if isinstance(expr, Op):
        out: list[str] = [expr.name]
        for child in expr.children:
            out.extend(to_prefix(child))
        return tuple(out)
    raise ExpressionError(f"nó desconhecido: {expr!r}")


def from_prefix(tokens: Iterable[str], variables: Iterable[str] | None = None) -> Expr:
    """Reconstrói a árvore a partir de tokens prefixos (aridade vinda do registro)."""
    toks = list(tokens)
    allowed = None if variables is None else frozenset(variables)
    pos = 0

    def _read() -> Expr:
        nonlocal pos
        if pos >= len(toks):
            raise ExpressionError("sequência prefixa incompleta")
        tok = toks[pos]
        pos += 1
        if tok in OPERATORS:
            arity = OPERATORS[tok].arity
            return Op(tok, tuple(_read() for _ in range(arity)))
        if _NUMBER.match(tok):
            return Const(float(tok))
        return _make_var(tok, allowed)

    expr = _read()
    if pos != len(toks):
        raise ExpressionError(f"tokens excedentes após posição {pos}: {toks[pos:]}")
    return expr


def _make_var(name: str, allowed: frozenset[str] | None) -> Var:
    if allowed is not None and name not in allowed:
        raise UnsafeExpressionError(f"variável não permitida: {name!r}")
    return Var(name)


# --------------------------------------------------------------------------- infixa

_PREC_ADD = 1
_PREC_MUL = 2
_PREC_UNARY = 3
_PREC_POW = 4
_PREC_ATOM = 5
_BINARY_PREC = {"add": _PREC_ADD, "sub": _PREC_ADD, "mul": _PREC_MUL, "div": _PREC_MUL}


def _precedence(expr: Expr) -> int:
    if isinstance(expr, Op):
        if expr.name in _BINARY_PREC:
            return _BINARY_PREC[expr.name]
        if expr.name == "pow":
            return _PREC_POW
        if expr.name == "neg" and not isinstance(expr.children[0], Const):
            return _PREC_UNARY
    return _PREC_ATOM


def _wrap(text: str, needed: bool) -> str:
    return f"({text})" if needed else text


def to_infix(expr: Expr) -> str:
    """Texto infixo com parênteses mínimos; :func:`parse_infix` relê a mesma árvore.

    Constantes negativas saem entre parênteses (``(-1.5)``) e ``neg`` de uma
    constante sai como chamada ``neg(2.0)`` para não ser confundido com uma
    constante negativa na releitura.
    """
    if isinstance(expr, Const):
        text = repr(expr.value)
        return f"({text})" if expr.value < 0 or text.startswith("-") else text
    if isinstance(expr, Var):
        return expr.name
    if not isinstance(expr, Op):
        raise ExpressionError(f"nó desconhecido: {expr!r}")
    spec = OPERATORS[expr.name]
    if expr.name == "neg":
        child = expr.children[0]
        if isinstance(child, Const):
            return f"neg({to_infix(child)})"
        return "-" + _wrap(to_infix(child), _precedence(child) < _PREC_UNARY)
    if spec.infix is not None:
        left, right = expr.children
        prec = _precedence(expr)
        if expr.name == "pow":
            lp = _wrap(to_infix(left), _precedence(left) <= _PREC_POW)
            rp = _wrap(to_infix(right), _precedence(right) < _PREC_ATOM)
        else:
            lp = _wrap(to_infix(left), _precedence(left) < prec)
            rp = _wrap(to_infix(right), _precedence(right) <= prec)
        return f"{lp} {spec.infix} {rp}"
    return f"{expr.name}(" + ", ".join(to_infix(c) for c in expr.children) + ")"


_AST_BINOPS: dict[type[ast.operator], str] = {
    ast.Add: INFIX_TO_NAME["+"],
    ast.Sub: INFIX_TO_NAME["-"],
    ast.Mult: INFIX_TO_NAME["*"],
    ast.Div: INFIX_TO_NAME["/"],
    ast.Pow: INFIX_TO_NAME["**"],
}


def parse_infix(text: str, variables: Iterable[str] | None = None) -> Expr:
    """Lê texto infixo NÃO CONFIÁVEL como AST, sem executar nada.

    Args:
        text: expressão infixa (sintaxe Python restrita).
        variables: nomes de variáveis permitidos; ``None`` aceita qualquer
            identificador válido para :class:`Var`.

    Raises:
        UnsafeExpressionError: construção fora da whitelist (atributo, subscript,
            lambda, string, nome não permitido, chamada não registrada...).
        ExpressionError: sintaxe inválida, texto longo/profundo demais.
    """
    if not isinstance(text, str):
        raise UnsafeExpressionError("parse_infix aceita apenas str")
    if len(text) > MAX_INFIX_LENGTH:
        raise ExpressionError(f"expressão maior que {MAX_INFIX_LENGTH} caracteres")
    allowed = None if variables is None else frozenset(variables)
    try:
        tree = ast.parse(text.strip(), mode="eval")
        return _convert(tree.body, allowed)
    except SyntaxError as exc:
        raise ExpressionError(f"sintaxe inválida: {exc.msg}") from None
    except (RecursionError, MemoryError):
        raise ExpressionError("expressão profunda demais") from None


def _convert(node: ast.AST, allowed: frozenset[str] | None) -> Expr:
    if isinstance(node, ast.BinOp):
        op_name = _AST_BINOPS.get(type(node.op))
        if op_name is None:
            raise UnsafeExpressionError(f"operador binário não permitido: {type(node.op).__name__}")
        return Op(op_name, (_convert(node.left, allowed), _convert(node.right, allowed)))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return _convert(node.operand, allowed)
        if not isinstance(node.op, ast.USub):
            raise UnsafeExpressionError(f"operador unário não permitido: {type(node.op).__name__}")
        operand = node.operand
        if isinstance(operand, ast.Constant):
            return Const(-_numeric_literal(operand))
        return Op("neg", (_convert(operand, allowed),))
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in OPERATORS:
            raise UnsafeExpressionError("chamada a função não registrada")
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise UnsafeExpressionError("chamada com keywords/starred não permitida")
        return Op(func.id, tuple(_convert(a, allowed) for a in node.args))
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load) or node.id in OPERATORS:
            raise UnsafeExpressionError(f"nome não permitido: {node.id!r}")
        try:
            return _make_var(node.id, allowed)
        except UnsafeExpressionError:
            raise
        except ExpressionError as exc:
            raise UnsafeExpressionError(str(exc)) from None
    if isinstance(node, ast.Constant):
        return Const(_numeric_literal(node))
    raise UnsafeExpressionError(f"construção não permitida: {type(node).__name__}")


def _numeric_literal(node: ast.Constant) -> float:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsafeExpressionError(f"literal não numérico: {type(value).__name__}")
    try:
        result = float(value)
    except OverflowError:
        raise ExpressionError("literal numérico grande demais") from None
    if not math.isfinite(result):
        raise ExpressionError("literal numérico não finito")
    return result


# --------------------------------------------------------------------------- sympy

_SYMPY_FUNCS: dict[Any, str] = {
    sympy.exp: "exp",
    sympy.log: "log",
    sympy.sin: "sin",
    sympy.cos: "cos",
}
_HALF = sympy.Rational(1, 2)


def to_sympy(expr: Expr) -> Any:
    """Gera objeto sympy equivalente (semântica matemática não protegida)."""
    if isinstance(expr, Const):
        return sympy.Float(expr.value)
    if isinstance(expr, Var):
        return sympy.Symbol(expr.name)
    if isinstance(expr, Op):
        return OPERATORS[expr.name].sympy_func(*(to_sympy(c) for c in expr.children))
    raise ExpressionError(f"nó desconhecido: {expr!r}")


def from_sympy(obj: Any, variables: Iterable[str] | None = None) -> Expr:
    """Reconstrói a AST a partir de um OBJETO sympy (ex.: saída do PySR).

    Strings são recusadas: texto deve passar por :func:`parse_infix`. ``Add`` e
    ``Mul`` n-ários viram cadeias binárias à esquerda; termos com coeficiente
    negativo em ``Add`` viram ``sub``; fatores ``b**(-e)`` em ``Mul`` viram
    denominador de ``div``; ``b**(1/2)`` vira ``sqrt``.

    Raises:
        UnknownOperatorError: função sympy fora do registro.
        ExpressionError: objeto não sympy ou número não real/finito.
    """
    if not isinstance(obj, sympy.Basic):
        raise ExpressionError(f"from_sympy aceita apenas objetos sympy, não {type(obj).__name__}")
    allowed = None if variables is None else frozenset(variables)
    return _from_sympy(obj, allowed)


def _fold(name: str, items: list[Expr]) -> Expr:
    acc = items[0]
    for item in items[1:]:
        acc = Op(name, (acc, item))
    return acc


def _is_negative_term(term: Any) -> bool:
    coeff, _ = term.as_coeff_Mul()
    return bool(coeff.is_negative)


def _from_sympy(obj: Any, allowed: frozenset[str] | None) -> Expr:
    if isinstance(obj, sympy.Symbol):
        return _make_var(obj.name, allowed)
    if isinstance(obj, (sympy.Number, sympy.NumberSymbol)):
        if not obj.is_real:
            raise ExpressionError(f"número não real: {obj}")
        return Const(float(obj))
    if isinstance(obj, sympy.Add):
        terms = list(obj.args)
        acc = _from_sympy(terms[0], allowed)
        for term in terms[1:]:
            if _is_negative_term(term):
                acc = Op("sub", (acc, _from_sympy(-term, allowed)))
            else:
                acc = Op("add", (acc, _from_sympy(term, allowed)))
        return acc
    if isinstance(obj, sympy.Mul):
        coeff, factors = obj.as_coeff_mul()
        num: list[Expr] = []
        den: list[Expr] = []
        for factor in factors:
            if isinstance(factor, sympy.Pow) and factor.exp.is_number and factor.exp.is_negative:
                den.append(_from_sympy(factor.base ** (-factor.exp), allowed))
            else:
                num.append(_from_sympy(factor, allowed))
        body = _fold("mul", num) if num else Const(1.0)
        if den:
            body = Op("div", (body, _fold("mul", den)))
        if coeff == 1:
            return body
        if coeff == -1:
            return Op("neg", (body,))
        return Op("mul", (_from_sympy(coeff, allowed), body))
    if isinstance(obj, sympy.Pow):
        base, exponent = obj.args
        if exponent == _HALF:
            return Op("sqrt", (_from_sympy(base, allowed),))
        if exponent.is_number and exponent.is_negative:
            return Op("div", (Const(1.0), _from_sympy(base ** (-exponent), allowed)))
        return Op("pow", (_from_sympy(base, allowed), _from_sympy(exponent, allowed)))
    func = getattr(obj, "func", None)
    if func in _SYMPY_FUNCS and len(obj.args) == 1:
        return Op(_SYMPY_FUNCS[func], (_from_sympy(obj.args[0], allowed),))
    name = getattr(func, "__name__", type(obj).__name__)
    get_operator(str(name).lower())  # levanta UnknownOperatorError
    raise ExpressionError(f"forma sympy não suportada: {obj}")  # pragma: no cover


# --------------------------------------------------------------------------- aleatório


def random_expression(
    rng: np.random.Generator,
    feature_names: Sequence[str],
    max_depth: int,
    operators: Sequence[str] | None = None,
    leaf_probability: float = DEFAULT_LEAF_PROBABILITY,
) -> Expr:
    """Gera árvore aleatória com profundidade ``<= max_depth`` (testes de propriedade)."""
    ops = sorted(OPERATORS) if operators is None else list(operators)
    for name in ops:
        get_operator(name)

    def _gen(level: int) -> Expr:
        if level == 0 or rng.random() < leaf_probability:
            if rng.random() < RANDOM_VARIABLE_LEAF_PROBABILITY:
                return Var(feature_names[int(rng.integers(len(feature_names)))])
            return Const(float(rng.normal(0.0, RANDOM_CONSTANT_SCALE)))
        name = ops[int(rng.integers(len(ops)))]
        return Op(name, tuple(_gen(level - 1) for _ in range(OPERATORS[name].arity)))

    return _gen(max_depth)
