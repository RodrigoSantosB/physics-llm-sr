"""T:perda: perda composta L = w1*e + w2*s + w3*c com números à mão."""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.interfaces import EquationEvaluation, EvaluationFailedError
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import CombinedObjective
from physics_llm_sr.symbolic.expression import parse_infix

FEATURES = ("x",)
CTX = PhysicalContext("s", {"x": "x"}, "exp", "x", {"x": "m"})
X = np.array([[1.0], [2.0], [3.0]])
Y = np.array([1.0, 2.0, 3.0])


def ev(a: float, b: float, c: float) -> EquationEvaluation:
    return EquationEvaluation(a, b, c, "fb")


def objective(
    weights: tuple[float, float, float], evaluator: object | None, **kw: object
) -> CombinedObjective:
    return CombinedObjective(
        weights=weights,
        evaluator=evaluator,  # type: ignore[arg-type]
        physical_context=CTX,
        prompt_variant="E",
        feature_names=FEATURES,
        **kw,  # type: ignore[arg-type]
    )


def test_penalty_zero_when_scores_one_and_one_when_zero() -> None:
    perfect, awful = parse_infix("x", FEATURES), parse_infix("x + 1", FEATURES)
    fake = FakeEquationEvaluator.from_table({perfect: ev(1, 1, 1), awful: ev(0, 0, 0)})
    out = objective((0.2, 0.3, 0.5), fake).evaluate_population([perfect, awful], X, Y)
    assert out[0].c == 0.0
    assert out[1].c == 1.0


def test_total_by_hand_literal() -> None:
    # x + 1: MSE = 1, nós = 3, escores (1, 0.5, 0) -> c = 0.5
    expr = parse_infix("x + 1", FEATURES)
    fake = FakeEquationEvaluator.from_table({expr: ev(1.0, 0.5, 0.0)})
    (lb,) = objective((0.5, 0.25, 0.25), fake).evaluate_population([expr], X, Y)
    assert lb.e == pytest.approx(1.0)
    assert lb.s == 3.0
    assert lb.c == pytest.approx(0.5)
    assert lb.total == pytest.approx(0.5 * 1.0 + 0.25 * 3.0 + 0.25 * 0.5)
    assert lb.evaluation is not None


@pytest.mark.parametrize("w", [(-0.1, 0.6, 0.5), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)])
def test_invalid_weights_rejected(w: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        objective(w, FakeEquationEvaluator.from_table({}))


def test_w3_positive_requires_evaluator() -> None:
    with pytest.raises(ValueError):
        objective((0.5, 0.25, 0.25), None)


def test_invalid_options_rejected() -> None:
    with pytest.raises(ValueError):
        objective((0.5, 0.5, 0.0), None, normalization="bogus")
    with pytest.raises(ValueError):
        objective((0.5, 0.5, 0.0), None, error_metric="mae")
    with pytest.raises(ValueError):
        objective((0.5, 0.5, 0.0), None, error_metric="huber", huber_delta=0.0)
    with pytest.raises(ValueError):
        objective((0.5, 0.5, 0.0), None, normalization="normalized")  # sem max_nodes


def test_w3_zero_never_calls_evaluator() -> None:
    fake = FakeEquationEvaluator.from_function(lambda e, c, v: ev(1, 1, 1))
    obj = objective((0.5, 0.5, 0.0), fake)
    expr = parse_infix("x + 1", FEATURES)
    (lb,) = obj.evaluate_population([expr], X, Y)
    assert fake.calls == 0
    assert lb.evaluation is None
    assert lb.c == 0.0
    assert obj.diagnostics[0]["c_status"] == "not_evaluated_w3_zero"
    assert lb.total == pytest.approx(0.5 * 1.0 + 0.5 * 3.0)


def test_normalized_mode() -> None:
    expr = parse_infix("x + 1", FEATURES)
    obj = objective((0.5, 0.5, 0.0), None, normalization="normalized", max_nodes=10)
    (lb,) = obj.evaluate_population([expr], X, Y)
    var = float(np.var(Y))
    assert lb.e == pytest.approx(1.0 / var)
    assert lb.s == pytest.approx(0.3)
    assert obj.metadata["deviation"] == "normalized"


def test_normalized_constant_target_rejected() -> None:
    obj = objective((0.5, 0.5, 0.0), None, normalization="normalized", max_nodes=10)
    with pytest.raises(ValueError):
        obj.evaluate_population([parse_infix("x", FEATURES)], X, np.ones(3))


def test_huber_by_hand() -> None:
    # resíduos = 1 em todos; delta=0.5 -> 0.5*(1-0.25) = 0.375; delta=2 -> 0.5
    expr = parse_infix("x + 1", FEATURES)
    small = objective((1.0, 0.0, 0.0), None, error_metric="huber", huber_delta=0.5)
    big = objective((1.0, 0.0, 0.0), None, error_metric="huber", huber_delta=2.0)
    assert small.evaluate_population([expr], X, Y)[0].e == pytest.approx(0.375)
    assert big.evaluate_population([expr], X, Y)[0].e == pytest.approx(0.5)
    assert small.metadata["error_metric"] == "huber"


def test_non_finite_error_marked_invalid_without_llm() -> None:
    good, bad = parse_infix("x", FEATURES), parse_infix("x + 1", FEATURES)
    fake = FakeEquationEvaluator.from_function(lambda e, c, v: ev(1, 1, 1))
    obj = objective((0.4, 0.2, 0.4), fake)
    y = np.array([1.0, 2.0, np.inf])
    out = obj.evaluate_population([good, bad], X, y)
    assert all(lb.total == np.inf for lb in out)
    assert fake.calls == 0
    assert all(d["invalid"] for d in obj.diagnostics)
    assert obj.counters.invalid == 2


def test_failure_propagates_without_substitute_score() -> None:
    expr = parse_infix("x", FEATURES)
    fake = FakeEquationEvaluator.from_table({})
    with pytest.raises(EvaluationFailedError):
        objective((0.4, 0.2, 0.4), fake).evaluate_population([expr], X, Y)


def test_batch_dedup_single_call_order_preserved() -> None:
    a, b = parse_infix("x", FEATURES), parse_infix("x + 1", FEATURES)
    a2 = parse_infix("x", FEATURES)
    fake = FakeEquationEvaluator.from_table({a: ev(1, 1, 1), b: ev(0, 0, 0)})
    calls: list[int] = []
    orig = fake.evaluate_many

    def spy(exprs, ctx, variant):  # type: ignore[no-untyped-def]
        calls.append(len(exprs))
        return orig(exprs, ctx, variant)

    fake.evaluate_many = spy  # type: ignore[method-assign]
    obj = objective((0.4, 0.2, 0.4), fake)
    out = obj.evaluate_population([a, b, a2, b], X, Y)
    assert calls == [2]
    assert [lb.c for lb in out] == [0.0, 1.0, 0.0, 1.0]
    assert obj.counters.requested == 4
    assert obj.counters.unique_llm == 2


def test_build_objective_from_config() -> None:
    from physics_llm_sr.config.schemas import ExperimentConfig
    from physics_llm_sr.core_types import Mode, Track
    from physics_llm_sr.optimization.objective import build_objective

    cfg = ExperimentConfig(mode=Mode.SMOKE, track=Track.PAPER_REPLICATION)
    fake = FakeEquationEvaluator.from_function(lambda e, c, v: ev(1, 1, 1))
    base = build_objective(cfg, fake, CTX, "deap", True, FEATURES)
    assert base.weights == (0.5, 0.5, 0.0) and base.evaluator is None
    pysr = build_objective(cfg, fake, CTX, "pysr", False, FEATURES)
    assert pysr.error_metric == "huber" and pysr.evaluator is fake
    assert build_objective(cfg, fake, CTX, "gplearn", False, FEATURES).error_metric == "mse"


def test_combined_objective_satisfies_candidate_objective_protocol() -> None:
    from physics_llm_sr.optimization.objective import CombinedObjective
    from physics_llm_sr.sr.interfaces import CandidateObjective

    obj = CombinedObjective((0.5, 0.5, 0.0), None, None, feature_names=["a", "b"])
    assert isinstance(obj, CandidateObjective)
    assert obj.feature_names == ("a", "b")


def test_dedup_uses_three_significant_digits_by_default() -> None:
    a, b = parse_infix("x * 2.0001", FEATURES), parse_infix("x * 2.0002", FEATURES)
    fake = FakeEquationEvaluator.from_function(lambda e, c, v: ev(1, 1, 1))
    obj = objective((0.4, 0.2, 0.4), fake)
    assert obj.significant_digits == 3 and obj.key_mode == "exact"
    obj.evaluate_population([a, b], X, Y)
    assert obj.counters.unique_llm == 1 and fake.calls == 1
    exact6 = objective((0.4, 0.2, 0.4), fake, significant_digits=6)
    exact6.evaluate_population([a, b], X, Y)
    assert exact6.counters.unique_llm == 2


def test_dedup_structure_mode_masks_constants() -> None:
    a, b = parse_infix("x * 2.0", FEATURES), parse_infix("x * 5.0", FEATURES)
    fake = FakeEquationEvaluator.from_function(lambda e, c, v: ev(1, 1, 1))
    obj = objective((0.4, 0.2, 0.4), fake, key_mode="structure")
    out = obj.evaluate_population([a, b], X, Y)
    assert obj.counters.unique_llm == 1 and fake.calls == 1
    assert out[0].c == out[1].c
    assert obj.metadata["cache_key_mode"] == "structure"
    with pytest.raises(ValueError):
        objective((0.4, 0.2, 0.4), fake, key_mode="loose")
    with pytest.raises(ValueError):
        objective((0.4, 0.2, 0.4), fake, significant_digits=0)
