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


# --------------------------------------------------------------------------- forma algébrica

ALGEBRAIC_MAX_ITERATIONS = 8
"""Limite de passadas de :func:`algebraic_normal_form` até o ponto fixo."""

ALGEBRAIC_SIZE_FACTOR = 4
"""Tamanho máximo da forma algébrica: ``FACTOR * size(expr) + SLACK``."""

ALGEBRAIC_SIZE_SLACK = 8
"""Folga aditiva do limite de tamanho da forma algébrica."""

_Factor = tuple[Expr, int]
_Monomial = tuple[_Factor, ...]
_Poly = dict[_Monomial, float]
_CONSTANT_MONOMIAL: _Monomial = ()


class _AlgebraicLimitError(ArithmeticError):
    """Coeficiente não finito ou forma grande demais: a forma algébrica desiste."""


def _finite(value: float) -> float:
    if not np.isfinite(value):
        raise _AlgebraicLimitError("coeficiente não finito")
    return float(value) + 0.0


def _is_integer(value: float) -> bool:
    return float(value).is_integer()


def _factor_order(factor: _Factor) -> str:
    return _sort_key(factor[0])


def _monomial_order(key: _Monomial) -> tuple[bool, str]:
    """Ordem dos termos: monômios não constantes por chave prefixa; o termo constante por último."""
    return (not key, " ".join(f"{_sort_key(a)}^{k}" for a, k in key))


def _const_poly(value: float) -> _Poly:
    value = _finite(value)
    return {} if value == 0.0 else {_CONSTANT_MONOMIAL: value}


def _const_value(poly: _Poly) -> float | None:
    if not poly:
        return 0.0
    if len(poly) == 1 and _CONSTANT_MONOMIAL in poly:
        return poly[_CONSTANT_MONOMIAL]
    return None


def _add_term(poly: _Poly, key: _Monomial, coef: float) -> None:
    total = _finite(poly.get(key, 0.0) + coef)
    if total == 0.0:
        poly.pop(key, None)
    else:
        poly[key] = total


def _poly_add(a: _Poly, b: _Poly, sign: float = 1.0) -> _Poly:
    out = dict(a)
    for key, coef in b.items():
        _add_term(out, key, sign * coef)
    return out


def _poly_scale(poly: _Poly, factor: float) -> _Poly:
    if factor == 0.0:
        return {}
    out: _Poly = {}
    for key, coef in poly.items():
        _add_term(out, key, coef * factor)
    return out


def _leading(poly: _Poly) -> tuple[_Monomial, float]:
    key = min(poly, key=_monomial_order)
    return key, poly[key]


def _single(poly: _Poly) -> tuple[float, _Monomial] | None:
    if len(poly) != 1:
        return None
    ((key, coef),) = poly.items()
    return coef, key


def _sum_atom(poly: _Poly) -> tuple[float, Expr]:
    """Soma com 2+ termos como fator: extrai o coeficiente do termo líder (líder vira 1)."""
    _, content = _leading(poly)
    return content, _rebuild({key: _finite(coef / content) for key, coef in poly.items()})


def _as_monomial(poly: _Poly) -> tuple[float, _Monomial]:
    single = _single(poly)
    if single is not None:
        return single
    content, atom = _sum_atom(poly)
    return content, ((atom, 1),)


def _is_const_pow(atom: Expr) -> bool:
    return (
        isinstance(atom, Op)
        and atom.name == "pow"
        and isinstance(atom.children[1], Const)
        and not _is_integer(atom.children[1].value)
    )


def _normalize_monomial(coef: float, factors: dict[Expr, int], params: ProtectionParams) -> _Poly:
    """Junta expoentes, funde ``exp`` (``exp(a)^i * exp(b)^j -> exp(i*a + j*b)``) e potências.

    ``pow(u, e)^k -> pow(u, k*e)`` quando ``k*e`` não é inteiro (``|u|^e`` elevado a inteiro).
    """
    pending = {a: k for a, k in factors.items() if k != 0}
    for _ in range(ALGEBRAIC_MAX_ITERATIONS):
        exp_arg: _Poly = {}
        has_exp = False
        rest: dict[Expr, int] = {}
        for atom, k in pending.items():
            if isinstance(atom, Op) and atom.name == "exp":
                has_exp = True
                exp_arg = _poly_add(exp_arg, _poly(atom.children[0], params), float(k))
                continue
            if _is_const_pow(atom) and k != 1:
                assert isinstance(atom, Op) and isinstance(atom.children[1], Const)
                new_e = atom.children[1].value * k
                if not _is_integer(new_e):
                    atom, k = Op("pow", (atom.children[0], Const(_finite(new_e)))), 1
            rest[atom] = rest.get(atom, 0) + k
        if has_exp:
            exp_poly = _func_poly("exp", exp_arg, params)
            value = _const_value(exp_poly)
            if value is not None:
                coef = _finite(coef * value)
            else:
                e_coef, e_key = _as_monomial(exp_poly)
                coef = _finite(coef * e_coef)
                for atom, k in e_key:
                    rest[atom] = rest.get(atom, 0) + k
        new_pending = {a: k for a, k in rest.items() if k != 0}
        if new_pending == pending:
            break
        pending = new_pending
    else:
        raise _AlgebraicLimitError("fatores sem ponto fixo")
    if coef == 0.0:
        return {}
    key = tuple(sorted(pending.items(), key=_factor_order))
    return {key: _finite(coef)}


def _poly_mul(a: _Poly, b: _Poly, params: ProtectionParams) -> _Poly:
    va, vb = _const_value(a), _const_value(b)
    if va is not None:
        return _poly_scale(b, va)
    if vb is not None:
        return _poly_scale(a, vb)
    ca, ka = _as_monomial(a)
    cb, kb = _as_monomial(b)
    factors: dict[Expr, int] = {}
    for atom, k in (*ka, *kb):
        factors[atom] = factors.get(atom, 0) + k
    return _normalize_monomial(_finite(ca * cb), factors, params)


def _poly_div(num: _Poly, den: _Poly, params: ProtectionParams) -> _Poly:
    value = _const_value(den)
    if value is not None:
        if abs(value) < params.division_epsilon:
            return _const_poly(params.division_fallback)  # divisão protegida: idêntica
        return {key: _finite(coef / value) for key, coef in num.items()}
    coef, key = _as_monomial(den)
    inverse = _normalize_monomial(_finite(1.0 / coef), {a: -k for a, k in key}, params)
    return _poly_mul(num, inverse, params)


def _atom_poly(atom: Expr) -> _Poly:
    return {((atom, 1),): 1.0}


def _fold_value(name: str, *values: float, params: ProtectionParams) -> float:
    args = [np.array([v]) for v in values]
    return float(apply_operator(name, *args, params=params)[0])


def _func_poly(name: str, arg: _Poly, params: ProtectionParams) -> _Poly:
    """``exp``/``log``/``sin``/``cos`` de um polinômio normalizado."""
    value = _const_value(arg)
    if value is not None:
        return _const_poly(_fold_value(name, value, params=params))
    if name == "exp":
        single = _single(arg)
        if single is not None:
            c, key = single
            if (
                len(key) == 1
                and key[0][1] == 1
                and isinstance(key[0][0], Op)
                and key[0][0].name == "log"
                and not _is_integer(c)
            ):
                # exp(c*log|u|) = |u|^c = pow(u, c) para c não inteiro
                base = _poly(key[0][0].children[0], params)
                return _pow_poly(base, _const_poly(c), params)
        return _atom_poly(Op("exp", (_rebuild(arg),)))
    sign = 1.0
    if name in ("cos", "sin", "log"):
        _, lead = _leading(arg)
        if lead < 0:
            arg = _poly_scale(arg, -1.0)
            if name == "sin":
                sign = -1.0  # sin(-x) = -sin(x); cos e log|.| são pares
    return _poly_scale(_atom_poly(Op(name, (_rebuild(arg),))), sign)


def _pow_poly(base: _Poly, expo: _Poly, params: ProtectionParams) -> _Poly:
    vb, ve = _const_value(base), _const_value(expo)
    if vb is not None and ve is not None:
        return _const_poly(_fold_value("pow", vb, ve, params=params))
    if ve is None:
        return _atom_poly(Op("pow", (_rebuild(base), _rebuild(expo))))
    if _is_integer(ve):
        k = int(ve)
        if k == 0:
            return _const_poly(_ONE)
        coef, key = _as_monomial(base)
        factors = {a: e * k for a, e in key}
        return _normalize_monomial(_finite(coef**k), factors, params)
    # expoente não inteiro: pow(a, e) = |a|^e (semântica protegida)
    coef, key = _as_monomial(base)
    scale = _finite(abs(coef) ** ve)
    if len(key) == 1:
        atom, k = key[0]
        exponent = k * ve
        if _is_const_pow(atom):
            assert isinstance(atom, Op) and isinstance(atom.children[1], Const)
            inner = atom.children[1].value * exponent
            if not _is_integer(inner):
                return _poly_scale(_atom_poly(Op("pow", (atom.children[0], Const(inner)))), scale)
        if not _is_integer(exponent):
            return _poly_scale(_atom_poly(Op("pow", (atom, Const(_finite(exponent))))), scale)
    inner_expr = _rebuild({key: 1.0})
    return _poly_scale(_atom_poly(Op("pow", (inner_expr, Const(ve)))), scale)


def _poly(expr: Expr, params: ProtectionParams) -> _Poly:
    """Polinômio (soma de monômios com coeficiente) equivalente a ``expr``."""
    if isinstance(expr, Const):
        return _const_poly(expr.value)
    if isinstance(expr, Var):
        return _atom_poly(expr)
    if not isinstance(expr, Op):
        raise ExpressionError(f"nó desconhecido: {expr!r}")
    name = expr.name
    kids = [_poly(c, params) for c in expr.children]
    if name == "add":
        return _poly_add(kids[0], kids[1])
    if name == "sub":
        return _poly_add(kids[0], kids[1], -1.0)
    if name == "neg":
        return _poly_scale(kids[0], -1.0)
    if name == "mul":
        return _poly_mul(kids[0], kids[1], params)
    if name == "div":
        return _poly_div(kids[0], kids[1], params)
    if name == "pow":
        return _pow_poly(kids[0], kids[1], params)
    if name == "sqrt":
        return _pow_poly(kids[0], _const_poly(SQRT_EXPONENT), params)
    if name in ("exp", "log", "sin", "cos"):
        return _func_poly(name, kids[0], params)
    raise ExpressionError(f"operador sem regra algébrica: {name}")  # pragma: no cover


def _factor_expr(atom: Expr, k: int) -> Expr:
    return atom if abs(k) == 1 else Op("pow", (atom, Const(float(abs(k)))))


def _chain(items: list[Expr]) -> Expr:
    acc = items[0]
    for item in items[1:]:
        acc = Op("mul", (acc, item))
    return acc


def _term_expr(coef: float, key: _Monomial) -> Expr:
    """Monômio ``coef * Π num / Π den`` (coeficiente à esquerda; denominador num ``div``)."""
    num = [_factor_expr(a, k) for a, k in key if k > 0]
    den = [_factor_expr(a, k) for a, k in key if k < 0]
    if num:
        top = _chain(num)
        if coef != _ONE:
            top = Op("mul", (Const(coef), top))
    else:
        top = Const(coef)
    return Op("div", (top, _chain(den))) if den else top


def _rebuild(poly: _Poly) -> Expr:
    if not poly:
        return Const(_ZERO)
    terms = sorted(poly.items(), key=lambda item: _monomial_order(item[0]))
    key, coef = terms[0]
    if not key:
        acc: Expr = Const(coef)
    elif coef == -_ONE:
        acc = Op("neg", (_term_expr(_ONE, key),))
    else:
        acc = _term_expr(coef, key)
    for key, coef in terms[1:]:
        if coef > 0:
            acc = Op("add", (acc, _term_expr(coef, key)))
        else:
            acc = Op("sub", (acc, _term_expr(-coef, key)))
    return acc


def algebraic_normal_form(expr: Expr, params: ProtectionParams = DEFAULT_PROTECTION) -> Expr:
    """Forma normal algébrica (base da métrica SECUNDÁRIA ``tree_distance_algebraic``).

    Separada de :func:`structural_normal_form` (GAP-10, normativa, que não muda). A expressão é
    lida como soma de monômios ``coef * Π fatorᵏ`` (fatores = variáveis, ``exp``, ``log``,
    ``sin``, ``cos``, ``pow`` com expoente não inteiro ou não constante, e somas usadas como
    fator) e reconstruída em ordem canônica. Identidades aplicadas (todas exatas na matemática
    real; domínio de validade entre parênteses):

    - dobra de constantes, inclusive dentro de funções (semântica protegida, idêntica);
    - ``u + 0``, ``u * 1``, ``1.0 * u -> u``; ``u * 0 -> 0``; ``u - u -> 0``; ``u + u -> 2*u``;
      ``neg(neg(u)) -> u``; ``c1*(c2*u) -> (c1*c2)*u``; ``neg(u)*c -> (-c)*u``;
    - ``u / c -> (1/c)*u`` (``|c| ≥ division_epsilon``; senão a divisão protegida devolve
      ``division_fallback``, e a forma devolve essa constante, idêntica);
    - constantes distribuem sobre somas: ``c*(u - v) -> c*u - c*v`` (só constantes; ``x*(a-b)``
      com ``x`` não constante fica fatorado); uma soma usada como fator tem o coeficiente do termo
      líder extraído (``(2a + 2b)*x -> 2*(a + b)*x``);
    - ``u / v`` com ``v`` monômio -> ``u * v⁻¹``; ``u*u -> pow(u, 2)``; ``u*v/u -> v``
      (denominadores com ``|den| ≥ division_epsilon``);
    - ``u / exp(v) -> u*exp(-v)`` e ``exp(a)*exp(b) -> exp(a + b)``
      (``|a|, |b|, |a+b| ≤ exp_clip``);
    - ``cos(-x) = cos(x)``, ``log|-x| = log|x|`` e ``sin(-x) = -sin(x)``: o argumento fica com o
      termo líder positivo (``cos(a - b) -> cos(b - a)`` pela ordem canônica);
    - ``sqrt(u) -> pow(u, 0.5)``; potência inteira de monômio ``(c*u)^k -> c^k * u^k``; potência
      não inteira ``pow(c*M, e) -> |c|^e * pow(M, e)`` e ``pow(u^k, e) -> pow(u, k*e)`` quando
      ``k*e`` não é inteiro (``pow`` não inteiro usa ``|base|``); ``exp(c*log(u)) -> pow(u, c)``
      só para ``c`` não inteiro (``|u|^c``); resultados com ``|valor| ≤ power_clip``.

    Fora desses domínios a semântica protegida (divisão, ``log``, recortes) pode diferir da
    forma reescrita; os testes de propriedade ignoram esses pontos. Aplica-se até o ponto fixo
    (no máximo :data:`ALGEBRAIC_MAX_ITERATIONS` passadas). Se um coeficiente ficar não finito ou
    a forma passar do limite de tamanho, devolve :func:`structural_normal_form` (sem reescrita
    algébrica).
    """
    _check(expr)
    limit = ALGEBRAIC_SIZE_FACTOR * _size(expr) + ALGEBRAIC_SIZE_SLACK
    current = expr
    try:
        for _ in range(ALGEBRAIC_MAX_ITERATIONS):
            rebuilt = _rebuild(_poly(current, params))
            if _size(rebuilt) > limit:
                raise _AlgebraicLimitError("forma algébrica grande demais")
            if rebuilt == current:
                break
            current = rebuilt
    except (_AlgebraicLimitError, ExpressionError, OverflowError, ZeroDivisionError):
        return structural_normal_form(expr, params)
    return current


def _size(expr: Expr) -> int:
    if isinstance(expr, Op):
        return 1 + sum(_size(c) for c in expr.children)
    return 1
