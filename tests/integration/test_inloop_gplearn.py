"""C:inloop-gplearn (T:inloop-gplearn, PRD §12.3): o termo LLM participa da seleção.

(a) Seleção: população controlada (mesmo ``e``, mesmo ``s``, notas LLM diferentes); o
    torneio NATIVO do gplearn (``_parallel_evolve._tournament``, que lê ``fitness_`` dos
    pais) escolhe outro indivíduo quando a nota muda e ``w3 > 0``, e o mesmo quando
    ``w3 = 0``.
(b) Busca: mesma semente, avaliador que favorece ``cos``; ``w3 > 0`` aumenta a fração
    da população final com ``cos`` em relação a ``w3 = 0``. Usa a escala ``normalized``
    (GAP-02): na escala ``literal`` da queda livre o MSE (dezenas a milhares de m²/s²)
    domina ``c`` em [0, 1] e, com o torneio padrão do gplearn (20), a população final
    medida foi a mesma (fração com cos 0.0 vs 0.0, pop 60 x 8 gerações). Isso é
    propriedade da escala do objetivo, não do mecanismo, que é o mesmo nos dois modos.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from gplearn.functions import _function_map

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset, PhysicalContext
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.llm.interfaces import EquationEvaluation, EvaluationFailedError
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import CombinedObjective, build_objective
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng
from physics_llm_sr.sr.gplearn_adapter import (
    GplearnRegressor,
    apply_objective_fitness,
    build_estimator,
    program_to_expr,
)
from physics_llm_sr.symbolic.expression import Expr, parse_infix, to_infix, to_prefix
from physics_llm_sr.symbolic.operators import DEFAULT_PROTECTION

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
GOOD = EquationEvaluation(1.0, 1.0, 1.0, "bom")
BAD = EquationEvaluation(0.0, 0.0, 0.0, "ruim")
W_LLM = (1 / 3, 1 / 3, 1 / 3)
W_BASE = (0.5, 0.5, 0.0)
#: x/x == 1 em todas as linhas (div protegida devolve 1 se |x| < eps): mesmo e, s = 3.
CONTROLLED = ("t", "mass", "height", "length", "damping")
COPIES = 2
SEED = 11
#: Busca (b): população/gerações pequenas (tempo < 60 s), cópia da config smoke.
SEARCH_POPULATION = 60
SEARCH_GENERATIONS = 8


def _setup() -> tuple[ExperimentConfig, Dataset, PhysicalContext]:
    cfg = load_config(SMOKE)
    data = generate_dataset(cfg, "free_fall")
    train = split_by_simulation(data, cfg.data.split, derive_rng(cfg.seeds.master, "split"))[
        "train"
    ]
    return cfg, train, get_scenario("free_fall", cfg.physics).physical_context()


def _selected(
    weights: tuple[float, float, float], favored: str, cfg: ExperimentConfig, train: Dataset
) -> tuple[Counter[str], int]:
    """Offspring do torneio nativo a partir da população controlada."""
    ctx = get_scenario("free_fall", cfg.physics).physical_context()
    fake = FakeEquationEvaluator.from_table(
        {parse_infix(f"{favored}/{favored}", train.feature_names): GOOD}, default=BAD
    )
    objective = CombinedObjective(weights, fake, ctx, train.feature_names)
    est = build_estimator(cfg, train.feature_names, DEFAULT_PROTECTION, SEED)
    # Só reprodução: o offspring é cópia do vencedor do torneio nativo. Torneio = população
    # inteira (independe de sr.gplearn.tournament_size, H-02 = 3): o vencedor é sempre o de
    # menor L, isolando o mecanismo de seleção.
    est.set_params(
        population_size=len(CONTROLLED) * COPIES,
        tournament_size=len(CONTROLLED) * COPIES,
        p_crossover=0.0,
        p_subtree_mutation=0.0,
        p_hoist_mutation=0.0,
        p_point_mutation=0.0,
        generations=1,
    )
    est.fit(train.X, train.y_observed)
    parents: list[Any] = est._programs[-1]
    names = list(train.feature_names)
    for k, program in enumerate(parents):
        var = names.index(CONTROLLED[k % len(CONTROLLED)])
        program.program = [_function_map["div"], var, var]
    _, losses = apply_objective_fitness(
        parents, objective, train.X, train.y_observed, train.feature_names
    )
    assert len({(loss.e, loss.s) for loss in losses}) == 1  # mesmo erro e tamanho
    est.set_params(generations=2)
    est.fit(train.X, train.y_observed)
    offspring = est._programs[-1]
    return Counter(
        to_infix(program_to_expr(p.program, train.feature_names)) for p in offspring
    ), fake.calls


def test_inloop_selection_follows_llm_score_only_when_w3_positive() -> None:
    cfg, train, _ = _setup()
    llm_a, calls_a = _selected(W_LLM, "t", cfg, train)
    llm_b, _ = _selected(W_LLM, "mass", cfg, train)
    base_a, base_calls = _selected(W_BASE, "t", cfg, train)
    base_b, _ = _selected(W_BASE, "mass", cfg, train)
    print(f"w3>0 favor t: {dict(llm_a)}; favor mass: {dict(llm_b)}")
    print(f"w3=0 favor t: {dict(base_a)}; favor mass: {dict(base_b)}")
    assert calls_a > 0 and base_calls == 0
    assert llm_a.most_common(1)[0][0] == "t / t"
    assert llm_b.most_common(1)[0][0] == "mass / mass"
    assert llm_a != llm_b  # seleção muda com a nota LLM (w3 > 0)
    assert base_a == base_b  # w3 = 0: nota irrelevante, mesma seleção


def _cos_fraction(exprs: list[Expr]) -> float:
    return sum("cos" in to_prefix(e) for e in exprs) / len(exprs)


def _favor_cos(expr: Expr, ctx: PhysicalContext, variant: str) -> EquationEvaluation:
    return GOOD if "cos" in to_prefix(expr) else BAD


def test_inloop_search_shifts_population_toward_llm_preference() -> None:
    cfg, train, ctx = _setup()
    cfg = cfg.model_copy(
        update={
            "sr": cfg.sr.model_copy(
                update={
                    "population": SEARCH_POPULATION,
                    "generations": SEARCH_GENERATIONS,
                    "early_stopping": cfg.sr.early_stopping.model_copy(update={"enabled": False}),
                }
            )
        }
    )
    fractions: dict[bool, float] = {}
    best: dict[bool, str] = {}
    max_nodes = 2 ** (cfg.sr.deap.max_depth + 1) - 1  # mesma regra de build_objective
    for baseline in (False, True):
        fake = FakeEquationEvaluator.from_function(_favor_cos)
        objective = CombinedObjective(
            W_BASE if baseline else W_LLM,
            fake,
            ctx,
            train.feature_names,
            normalization="normalized",
            max_nodes=max_nodes,
        )
        reg = GplearnRegressor()
        result = reg.fit(train, objective, cfg)
        assert reg.estimator is not None
        final = [
            program_to_expr(p.program, train.feature_names) for p in reg.estimator._programs[-1]
        ]
        fractions[baseline] = _cos_fraction(final)
        best[baseline] = to_infix(result.best_expression)
        assert result.metadata.integration_status == ("baseline_only" if baseline else "integrated")
        assert (fake.calls > 0) is (not baseline)
    print(f"fração com cos: w3>0={fractions[False]:.3f} w3=0={fractions[True]:.3f}")
    print(f"melhor: w3>0={best[False]} | w3=0={best[True]}")
    assert fractions[False] > fractions[True]


def test_llm_failure_propagates_with_error_stop_reason() -> None:
    cfg, train, ctx = _setup()
    fake = FakeEquationEvaluator.from_table({})  # sem default: EvaluationFailedError
    objective = build_objective(cfg, fake, ctx, "gplearn", False, train.feature_names)
    reg = GplearnRegressor()
    with pytest.raises(EvaluationFailedError):
        reg.fit(train, objective, cfg)
    assert reg.stop_reason == "error"
