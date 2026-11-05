"""C:inloop-pysr (T:inloop-pysr, PRD §12.3, GAP-08): o termo LLM está DENTRO da perda do PySR.

(a) Influência na seleção/busca: mesma semente, avaliador que favorece ``cos``; com
    ``w3 > 0`` a fração do hall of fame com ``cos`` (e a melhor expressão) muda em relação
    a ``w3 = 0``; com ``w3 = 0`` trocar o avaliador não muda nada (avaliador nunca chamado).
    Usa a escala ``normalized`` (GAP-02): na ``literal`` da queda livre o Huber em SI e
    ``s`` em nós dominam ``c`` em [0, 1] (spike: com pesos 1/3 o ``cos`` não entra no HoF).
(b) Paridade: a perda calculada no Julia (``equations_.loss``) == ``L`` do
    ``CombinedObjective.evaluate_population`` para a melhor equação e para todo o HoF.
(c) Falha do LLM: ``EvaluationFailedError`` após a busca, ``stop_reason = "error"``,
    resultado descartado.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

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
from physics_llm_sr.sr.interfaces import FitResult
from physics_llm_sr.sr.pysr_adapter import AUDIT_REL_TOL, PySRRegressor
from physics_llm_sr.symbolic.expression import Expr, to_prefix

pytestmark = pytest.mark.julia

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
GOOD = EquationEvaluation(0.9, 0.9, 0.9, "bom")
BAD = EquationEvaluation(0.1, 0.1, 0.1, "ruim")
W_LLM = (1 / 3, 1 / 3, 1 / 3)
W_BASE = (0.5, 0.5, 0.0)
#: Busca pequena (cópia da smoke): tempo < 60 s por teste após a compilação do Julia.
SEARCH_POPULATION = 30
SEARCH_ITERATIONS = 4


def _setup() -> tuple[ExperimentConfig, Dataset, PhysicalContext]:
    cfg = load_config(SMOKE)
    cfg = cfg.model_copy(
        update={
            "sr": cfg.sr.model_copy(
                update={
                    "population": SEARCH_POPULATION,
                    "generations": SEARCH_ITERATIONS,
                    "early_stopping": cfg.sr.early_stopping.model_copy(update={"enabled": False}),
                }
            )
        }
    )
    data = generate_dataset(cfg, "free_fall")
    train = split_by_simulation(data, cfg.data.split, derive_rng(cfg.seeds.master, "split"))[
        "train"
    ]
    return cfg, train, get_scenario("free_fall", cfg.physics).physical_context()


def _favor_cos(expr: Expr, ctx: PhysicalContext, variant: str) -> EquationEvaluation:
    return GOOD if "cos" in to_prefix(expr) else BAD


def _favor_nothing(expr: Expr, ctx: PhysicalContext, variant: str) -> EquationEvaluation:
    return BAD


def _run(
    weights: tuple[float, float, float], fn: object, cfg: ExperimentConfig, train: Dataset,
    ctx: PhysicalContext,
) -> tuple[FitResult, FakeEquationEvaluator, PySRRegressor]:  # fmt: skip
    fake = FakeEquationEvaluator.from_function(fn)  # type: ignore[arg-type]
    max_nodes = 2 ** (cfg.sr.deap.max_depth + 1) - 1  # mesma regra de build_objective
    objective = CombinedObjective(
        weights, fake, ctx, train.feature_names, normalization="normalized",
        max_nodes=max_nodes, error_metric="huber", huber_delta=cfg.sr.pysr.huber_delta,
    )  # fmt: skip
    reg = PySRRegressor()
    return reg.fit(train, objective, cfg), fake, reg


def _hof(reg: PySRRegressor) -> list[str]:
    assert reg.model is not None
    return [str(e) for e in reg.model.equations_["equation"]]


def _cos_fraction(hof: list[str]) -> float:
    return sum("cos(" in e for e in hof) / len(hof)


def test_inloop_llm_term_changes_search_only_when_w3_positive() -> None:
    cfg, train, ctx = _setup()
    base_a, fake_base_a, reg_base_a = _run(W_BASE, _favor_cos, cfg, train, ctx)
    base_b, fake_base_b, reg_base_b = _run(W_BASE, _favor_nothing, cfg, train, ctx)
    llm_a, fake_llm_a, reg_llm_a = _run(W_LLM, _favor_cos, cfg, train, ctx)
    llm_b, _, reg_llm_b = _run(W_LLM, _favor_nothing, cfg, train, ctx)
    hof = {k: _hof(r) for k, r in {"base_a": reg_base_a, "base_b": reg_base_b,
                                    "llm_a": reg_llm_a, "llm_b": reg_llm_b}.items()}  # fmt: skip
    frac = {k: _cos_fraction(v) for k, v in hof.items()}
    print(f"fração do HoF com cos: {frac}")
    print(f"melhor: w3>0 cos={llm_a.history[-1].best_expression} | "
          f"w3>0 neutro={llm_b.history[-1].best_expression} | "
          f"w3=0={base_a.history[-1].best_expression}")  # fmt: skip
    assert reg_llm_a.callback is not None
    print(f"callback w3>0: {reg_llm_a.callback.stats()}")
    # w3 = 0: avaliador nunca chamado e a busca não depende dele
    assert fake_base_a.calls == 0 and fake_base_b.calls == 0
    assert hof["base_a"] == hof["base_b"]
    # w3 > 0: o avaliador participa e muda a busca
    assert fake_llm_a.calls > 0
    assert llm_a.metadata.integration_status == "integrated"
    assert hof["llm_a"] != hof["base_a"]
    assert hof["llm_a"] != hof["llm_b"]
    assert frac["llm_a"] > frac["base_a"]
    assert "cos" in llm_a.history[-1].best_expression


def test_julia_loss_matches_python_objective() -> None:
    cfg, train, ctx = _setup()
    for weights, normalization in ((W_LLM, "literal"), (W_LLM, "normalized")):
        fake = FakeEquationEvaluator.from_function(_favor_cos)
        objective = build_objective(cfg, fake, ctx, "pysr", False, train.feature_names)
        if normalization == "normalized":
            objective = CombinedObjective(
                weights, fake, ctx, train.feature_names, normalization="normalized",
                max_nodes=2 ** (cfg.sr.deap.max_depth + 1) - 1, error_metric="huber",
            )  # fmt: skip
        reg = PySRRegressor()
        result = reg.fit(train, objective, cfg)
        assert reg.model is not None
        eqs = reg.model.equations_
        i = int(eqs["loss"].astype(float).idxmin())
        julia_loss = float(eqs["loss"].loc[i])
        python = objective.evaluate_population([result.best_expression], train.X, train.y_observed)[
            0
        ]
        print(f"{normalization}: julia={julia_loss!r} python={python.total!r} "
              f"(e={python.e:.6g}, s={python.s:.6g}, c={python.c:.3g}) "
              f"best={result.history[-1].best_expression}")  # fmt: skip
        assert math.isclose(julia_loss, python.total, rel_tol=AUDIT_REL_TOL)
        audit = result.metadata.extra["audit"] if result.metadata.extra else {}
        assert audit["ok"], audit
        print(f"{normalization}: auditoria HoF max_rel_diff={audit['max_rel_diff']:.3g}")


def test_llm_failure_raises_after_search_and_discards_result() -> None:
    cfg, train, ctx = _setup()
    fake = FakeEquationEvaluator.from_table({})  # sem default: EvaluationFailedError
    objective = build_objective(cfg, fake, ctx, "pysr", False, train.feature_names)
    reg = PySRRegressor()
    with pytest.raises(EvaluationFailedError):
        reg.fit(train, objective, cfg)
    assert reg.stop_reason == "error"
    assert fake.calls == 1  # após a falha o avaliador não é mais chamado
    with pytest.raises(RuntimeError):
        reg.get_best_expression()
