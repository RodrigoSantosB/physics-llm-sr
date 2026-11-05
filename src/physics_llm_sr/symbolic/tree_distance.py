"""Distância estrutural normalizada entre árvores (GAP-10, PRD §8.2).

Definição (``references/scientific_assumptions.md``, GAP-10). ``pair(a, b)``
devolve ``(S, N)``: soma dos custos locais e número de comparações de nós. A
distância é ``d = S / N`` na raiz, logo ``d ∈ [0, 1]``; ``d = 0`` significa
correspondência estrutural (GAP-11).

1. Folha × folha (``N = 1``): variável×variável ``S = 0`` se nomes iguais,
   senão 1; constante×constante ``S = min(1, α·|v1 − v2|)``; variável×constante
   ``S = 1``.
2. Folha × nó interno: ``S = 1``, ``N = 1``, sem descida.
3. Nós internos com o mesmo operador: unário/não comutativo pareia filhos por
   posição, ``S = ΣS_i``, ``N = 1 + ΣN_i``. Comutativo binário: pareamento
   direto ou cruzado de menor ``S``; empate → maior ``N``; persistindo → direto.
4. Operadores diferentes: mesma aridade e ``descend_on_operator_mismatch`` →
   ``S = 1 + ΣS_i``, ``N = 1 + ΣN_i`` (filhos por posição); caso contrário
   ``S = 1``, ``N = 1``.

Antes da comparação, se ``canonicalize`` (padrão), ambas as árvores passam por
:func:`~physics_llm_sr.symbolic.canonicalization.structural_normal_form`.
Equivalência numérica não implica distância 0 (``x + 0`` vs ``x`` → 1).

``α`` vem de ``SymbolicConfig.tree_distance_constant_penalty``; as opções
``descend_on_operator_mismatch`` e ``canonicalize`` de ``SymbolicConfig.tree_distance``.
"""

from __future__ import annotations

from dataclasses import dataclass

from physics_llm_sr.config.schemas import SymbolicConfig
from physics_llm_sr.symbolic.canonicalization import structural_normal_form
from physics_llm_sr.symbolic.expression import Const, Expr, ExpressionError, Op, Var
from physics_llm_sr.symbolic.operators import OPERATORS, get_operator

DEFAULT_DESCEND_ON_OPERATOR_MISMATCH = True
"""GAP-10: operadores diferentes de mesma aridade continuam a descida."""

DEFAULT_CANONICALIZE = True
"""GAP-10: aplica ``structural_normal_form`` antes de comparar."""

LOCAL_COST_MAX = 1.0
"""Custo local máximo de uma comparação de nós."""

_DEFAULT_ALPHA = SymbolicConfig().tree_distance_constant_penalty


@dataclass(frozen=True)
class TreeDistanceParams:
    """Parâmetros da distância (``alpha`` = penalidade por unidade de |Δconstante|)."""

    alpha: float = _DEFAULT_ALPHA
    descend_on_operator_mismatch: bool = DEFAULT_DESCEND_ON_OPERATOR_MISMATCH
    canonicalize: bool = DEFAULT_CANONICALIZE

    def __post_init__(self) -> None:
        """Valida ``alpha ∈ [0, 1]``."""
        if not 0.0 <= self.alpha <= LOCAL_COST_MAX:
            raise ValueError(f"alpha deve estar em [0, 1], recebeu {self.alpha}")

    @classmethod
    def from_config(cls, cfg: SymbolicConfig) -> TreeDistanceParams:
        """Constrói a partir da seção ``symbolic`` da configuração."""
        return cls(
            alpha=cfg.tree_distance_constant_penalty,
            descend_on_operator_mismatch=cfg.tree_distance.descend_on_operator_mismatch,
            canonicalize=cfg.tree_distance.canonicalize,
        )


DEFAULT_TREE_DISTANCE_PARAMS = TreeDistanceParams()


def _validate(expr: object) -> None:
    if isinstance(expr, (Const, Var)):
        return
    if not isinstance(expr, Op):
        raise ExpressionError(f"nó desconhecido: {expr!r}")
    spec = get_operator(expr.name)
    if len(expr.children) != spec.arity:
        raise ExpressionError(f"aridade inválida em {expr.name}")
    for child in expr.children:
        _validate(child)


def _pair(a: Expr, b: Expr, p: TreeDistanceParams) -> tuple[float, int]:
    a_leaf = not isinstance(a, Op)
    b_leaf = not isinstance(b, Op)
    if a_leaf and b_leaf:
        if isinstance(a, Var) and isinstance(b, Var):
            return (0.0 if a.name == b.name else LOCAL_COST_MAX), 1
        if isinstance(a, Const) and isinstance(b, Const):
            return min(LOCAL_COST_MAX, p.alpha * abs(a.value - b.value)), 1
        return LOCAL_COST_MAX, 1
    if a_leaf or b_leaf:
        return LOCAL_COST_MAX, 1
    assert isinstance(a, Op) and isinstance(b, Op)
    if a.name == b.name:
        s, n = _children(a.children, b.children, p)
        if OPERATORS[a.name].commutative and len(a.children) == 2:
            cross = _children(a.children, b.children[::-1], p)
            if (cross[0], -cross[1]) < (s, -n):
                s, n = cross
        return s, 1 + n
    if len(a.children) == len(b.children) and p.descend_on_operator_mismatch:
        s, n = _children(a.children, b.children, p)
        return LOCAL_COST_MAX + s, 1 + n
    return LOCAL_COST_MAX, 1


def _children(
    left: tuple[Expr, ...], right: tuple[Expr, ...], p: TreeDistanceParams
) -> tuple[float, int]:
    s_total, n_total = 0.0, 0
    for x, y in zip(left, right, strict=True):
        s, n = _pair(x, y, p)
        s_total += s
        n_total += n
    return s_total, n_total


def tree_distance_terms(
    a: Expr, b: Expr, params: TreeDistanceParams = DEFAULT_TREE_DISTANCE_PARAMS
) -> tuple[float, int]:
    """Devolve ``(S, N)`` do GAP-10 (soma de custos locais, nº de comparações).

    Raises:
        ExpressionError: nó que não é ``Const``/``Var``/``Op``.
        UnknownOperatorError: operador fora do registro.
    """
    _validate(a)
    _validate(b)
    if params.canonicalize:
        a, b = structural_normal_form(a), structural_normal_form(b)
    return _pair(a, b, params)


def tree_distance(
    a: Expr, b: Expr, params: TreeDistanceParams = DEFAULT_TREE_DISTANCE_PARAMS
) -> float:
    """Distância estrutural normalizada ``S / N`` em ``[0, 1]`` (0 = mesma estrutura)."""
    s, n = tree_distance_terms(a, b, params)
    return min(LOCAL_COST_MAX, s / n)
