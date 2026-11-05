"""T:inloop-deap (PRD 12.3): o LLM participa da SELECAO do DEAP (a) e da BUSCA (b)."""

from __future__ import annotations

import random

import numpy as np
import pytest
from deap import gp, tools

from physics_llm_sr.config.schemas import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    SRConfig,
)
from physics_llm_sr.core_types import Mode, PhysicalContext, Track
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import CombinedObjective, build_objective
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng
from physics_llm_sr.sr.deap_adapter import (
    DeapRegressor,
    build_primitive_set,
    evaluate_individuals,
    expr_to_individual,
    individual_to_expr,
)
from physics_llm_sr.symbolic.expression import (
    Expr,
    Op,
    depth,
    evaluate,
    parse_infix,
)
from physics_llm_sr.symbolic.operators import DEFAULT_PROTECTION

FEATURES = ("x1", "x2", "x3")
CTX = PhysicalContext("s", {f: f for f in FEATURES}, "exp", "x1", {f: "m" for f in FEATURES})
XCTRL = np.tile(np.linspace(1.0, 5.0, 20).reshape(-1, 1), (1, 3))  # colunas identicas
YCTRL = 2.0 * XCTRL[:, 0]
OPS = ["add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos"]


def _controlled() -> list[Expr]:
    # add(xi, xj), i<=j: mesmo erro, mesmo tamanho (3), formas canonicas distintas
    return [
        parse_infix(f"{FEATURES[i]} + {FEATURES[j]}", FEATURES)
        for i in range(3)
        for j in range(i, 3)
    ]


def _table(order: list[int]) -> dict[str | Expr, EquationEvaluation]:
    exprs = _controlled()
    return {
        exprs[k]: EquationEvaluation(v, v, v, "x")
        for k, v in zip(order, np.linspace(0.0, 1.0, len(exprs)), strict=True)
    }


def _winner(weights: tuple[float, float, float], order: list[int]) -> tuple[int, int]:
    pset = build_primitive_set(FEATURES, OPS, DEFAULT_PROTECTION)
    pop = [expr_to_individual(e, pset) for e in _controlled()]
    fake = FakeEquationEvaluator.from_table(_table(order))
    obj = CombinedObjective(weights, fake, CTX, FEATURES)
    evaluate_individuals(pop, obj, XCTRL, YCTRL)
    errs = {ind.loss.e for ind in pop}
    sizes = {ind.loss.s for ind in pop}
    assert len(errs) == 1 and len(sizes) == 1  # mesmo erro e mesmo tamanho
    random.seed(7)
    (chosen,) = tools.selTournament(pop, 1, 10 * len(pop))  # real, DEAP
    return pop.index(chosen), fake.calls


def test_inloop_a_selection_tournament_changes_with_w3_only() -> None:
    order_a = [0, 1, 2, 3, 4, 5]  # indice 0 = melhor nota (c baixo? nota alta = c baixo)
    order_b = [5, 4, 3, 2, 1, 0]
    # w3>0: vencedor = individuo de maior nota e muda quando as notas mudam
    w_a, _ = _winner((0.4, 0.2, 0.4), order_a)
    w_b, _ = _winner((0.4, 0.2, 0.4), order_b)
    assert w_a != w_b
    assert w_a == order_a.index(5) and w_b == order_b.index(5)
    # w3=0: seleciona igual independentemente das notas; avaliador nunca chamado
    z_a, calls_a = _winner((0.5, 0.5, 0.0), order_a)
    z_b, calls_b = _winner((0.5, 0.5, 0.0), order_b)
    assert z_a == z_b and calls_a == calls_b == 0


def test_gp_compile_and_symbolic_evaluate_agree_on_deap_trees() -> None:
    # gp.compile so aqui, sobre arvores geradas pelo proprio DEAP (nunca texto externo)
    names = ("height", "t", "mass")
    pset = build_primitive_set(names, [*OPS, "pow", "sqrt"], DEFAULT_PROTECTION)
    random.seed(3)
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(30, 3))  # noqa: N806
    for _ in range(40):
        tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset, 1, 4))
        func = gp.compile(tree, pset)
        native = np.asarray([func(*row) for row in X], dtype=float)
        expr = individual_to_expr(tree)
        np.testing.assert_allclose(
            evaluate(expr, X, names), np.nan_to_num(native), rtol=1e-9, atol=1e-9
        )
        # ida e volta
        assert individual_to_expr(expr_to_individual(expr, pset)) == expr


def _free_fall(
    cfg: ExperimentConfig,
) -> tuple[ExperimentConfig, object, PhysicalContext]:
    data = generate_dataset(cfg, "free_fall")
    train = split_by_simulation(data, cfg.data.split, derive_rng(42, "split"))["train"]
    return cfg, train, get_scenario("free_fall", cfg.physics).physical_context()


def _contains_cos(expr: Expr) -> bool:
    if isinstance(expr, Op):
        return expr.name == "cos" or any(_contains_cos(c) for c in expr.children)
    return False


def _run(cfg: ExperimentConfig, train: object, ctx: PhysicalContext, baseline: bool, seed: int):  # type: ignore[no-untyped-def]
    fav = FakeEquationEvaluator.from_function(
        lambda e, _c, _v: (
            EquationEvaluation(1.0, 1.0, 1.0, "cos")
            if _contains_cos(e)
            else EquationEvaluation(0.0, 0.0, 0.0, "no")
        )
    )
    objective = build_objective(cfg, fav, ctx, "deap", baseline, train.feature_names)  # type: ignore[attr-defined]
    reg = DeapRegressor(seed_name=f"sr/deap/test{seed}")
    result = reg.fit(train, objective, cfg)  # type: ignore[arg-type]
    pop_exprs = reg.final_population_expressions()
    frac = sum(_contains_cos(e) for e in pop_exprs) / len(pop_exprs)
    return frac, result, fav.calls


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_inloop_b_search_prefers_llm_favoured_form(seed: int) -> None:
    cfg = ExperimentConfig(
        mode=Mode.SMOKE,
        track=Track.EXTENDED_ANALYSIS,
        data=DataConfig(n_simulations=10, n_time_points=20),
        sr=SRConfig(population=60, generations=12),
        loss=LossConfig(w1=0.2, w2=0.1, w3=0.7, normalization="normalized"),
    )
    cfg, train, ctx = _free_fall(cfg)
    f_llm, res_llm, calls_llm = _run(cfg, train, ctx, False, seed)
    f_base, res_base, calls_base = _run(cfg, train, ctx, True, seed)
    print(f"seed={seed} cos_fraction w3>0={f_llm:.3f} w3=0={f_base:.3f}")
    assert calls_base == 0 and calls_llm > 0
    assert f_llm > f_base
    assert res_llm.llm_counters["llm_batches"] >= 1
    assert res_llm.history[-1].n_llm_calls >= 0


def test_deap_deterministic_depth_limited_and_error_propagates() -> None:
    cfg = ExperimentConfig(
        mode=Mode.SMOKE,
        track=Track.PAPER_REPLICATION,
        data=DataConfig(n_simulations=10, n_time_points=20),
        sr=SRConfig(population=30, generations=6),
    )
    cfg, train, ctx = _free_fall(cfg)

    def go() -> DeapRegressor:
        obj = build_objective(cfg, None, ctx, "deap", True, train.feature_names)  # type: ignore[attr-defined]
        reg = DeapRegressor()
        reg.fit(train, obj, cfg)  # type: ignore[arg-type]
        return reg

    a, b = go(), go()
    assert a.get_best_expression() == b.get_best_expression()
    assert [r.best_loss for r in a.get_search_history()] == [
        r.best_loss for r in b.get_search_history()
    ]
    assert all(depth(e) <= cfg.sr.deap.max_depth for e in a.final_population_expressions())

    from physics_llm_sr.llm.interfaces import EvaluationFailedError

    failing = FakeEquationEvaluator.from_table({})  # sem default: levanta
    obj = build_objective(cfg, failing, ctx, "deap", False, train.feature_names)  # type: ignore[attr-defined]
    reg = DeapRegressor()
    with pytest.raises(EvaluationFailedError):
        reg.fit(train, obj, cfg)  # type: ignore[arg-type]
    assert reg.stop_reason == "error"
