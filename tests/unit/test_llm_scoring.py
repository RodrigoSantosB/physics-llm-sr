"""C:llm-contrato - retry limitado, política de falha, penalidade e avaliador simulado."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physics_llm_sr.llm.inference import FakeBackend
from physics_llm_sr.llm.interfaces import (
    BackendError,
    EquationEvaluation,
    EvaluationFailedError,
    PhysicsEquationEvaluator,
)
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import FakeEquationEvaluator, LLMEquationEvaluator, penalty
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.canonicalization import canonical_key
from physics_llm_sr.symbolic.expression import Expr, parse_infix, to_infix

REPO_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
VALID = json.dumps(
    {"dimensional_consistency": 1.0, "simplicity": 0.5, "physical_realism": 0.25, "feedback": "f"}
)
INVALID = '{"dimensional_consistency": 1.5, "simplicity": 0.5, "physical_realism": 0.2}'
CTX = get_scenario("free_fall").physical_context()
GT = parse_infix("sqrt(2*9.81*height)")
OTHER = parse_infix("mass + t")


@pytest.fixture(scope="module")
def library() -> PromptLibrary:
    return PromptLibrary.load(REPO_PROMPTS)


def _evaluator(
    library: PromptLibrary, backend: FakeBackend, **kwargs: object
) -> LLMEquationEvaluator:
    return LLMEquationEvaluator(backend, library, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- penalidade


def test_penalty_bounds() -> None:
    assert penalty(EquationEvaluation(1, 1, 1, "")) == 0.0
    assert penalty(EquationEvaluation(0, 0, 0, "")) == 1.0
    assert penalty(EquationEvaluation(1.0, 0.5, 0.0, "")) == pytest.approx(0.5)


# ---------------------------------------------------------------------- avaliador LLM


def test_valid_response_parsed(library: PromptLibrary) -> None:
    backend = FakeBackend(VALID)
    ev = _evaluator(library, backend).evaluate(GT, CTX, "E")
    assert ev.scores == (1.0, 0.5, 0.25)
    assert backend.prompt_count == 1
    system, user = backend.calls[0]
    assert "### ROLE" in system and to_infix(GT) in user


def test_retry_is_limited_and_no_substitute_score(library: PromptLibrary) -> None:
    backend = FakeBackend(INVALID)
    ev = _evaluator(library, backend, max_retries=2, failure_policy="retry")
    with pytest.raises(EvaluationFailedError, match="1 de 1"):
        ev.evaluate(GT, CTX, "E")
    assert backend.prompt_count == 3  # 1 tentativa + 2 retries
    m = ev.metrics
    assert m.backend_calls == 3 and m.invalid_responses == 3 and m.failures == 1


def test_retry_recovers_after_invalid(library: PromptLibrary) -> None:
    backend = FakeBackend([INVALID, "texto livre", VALID])
    ev = _evaluator(library, backend, max_retries=2)
    assert ev.evaluate(GT, CTX, "E").simplicity == 0.5
    assert backend.prompt_count == 3
    assert ev.metrics.invalid_responses == 2 and ev.metrics.failures == 0


def test_abort_policy_raises_on_first_invalid(library: PromptLibrary) -> None:
    backend = FakeBackend(INVALID)
    ev = _evaluator(library, backend, max_retries=5, failure_policy="abort")
    with pytest.raises(EvaluationFailedError):
        ev.evaluate(GT, CTX, "E")
    assert backend.prompt_count == 1


def test_injection_response_rejected_not_executed(library: PromptLibrary) -> None:
    sentinel = Path("/tmp/pwned")
    if sentinel.exists():
        sentinel.unlink()
    backend = FakeBackend("__import__('os').system('touch /tmp/pwned')")
    with pytest.raises(EvaluationFailedError):
        _evaluator(library, backend, max_retries=1).evaluate(GT, CTX, "E")
    assert not sentinel.exists()


def test_backend_errors_are_not_retried(library: PromptLibrary) -> None:
    def boom(system: str, user: str) -> str:
        raise BackendError("rede caiu")

    backend = FakeBackend(boom)
    with pytest.raises(BackendError):
        _evaluator(library, backend, max_retries=3).evaluate(GT, CTX, "E")
    assert backend.prompt_count == 1


def test_wrong_response_count_from_backend_rejected(library: PromptLibrary) -> None:
    class Short(FakeBackend):
        def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
            return []

    with pytest.raises(BackendError):
        _evaluator(library, Short(VALID)).evaluate(GT, CTX, "E")


def test_evaluate_many_preserves_order_and_batches(library: PromptLibrary) -> None:
    def by_expr(system: str, user: str) -> str:
        score = 1.0 if "height" in user.split("### CANDIDATE EQUATION")[1].split("###")[0] else 0.0
        return json.dumps(
            {
                "dimensional_consistency": score,
                "simplicity": 0.5,
                "physical_realism": score,
                "feedback": "",
            }
        )

    backend = FakeBackend(by_expr)
    ev = _evaluator(library, backend)
    exprs = [OTHER, GT, OTHER, parse_infix("t + mass")]
    results = ev.evaluate_many(exprs, CTX, "A")
    assert [r.dimensional_consistency for r in results] == [0.0, 1.0, 0.0, 0.0]
    assert backend.generate_calls == 1  # um lote
    assert backend.prompt_count == 2  # duplicatas canônicas deduplicadas
    assert ev.metrics.requested == 4 and ev.metrics.deduplicated == 2
    assert ev.metrics.inference_seconds >= 0.0


def test_invalid_config_rejected(library: PromptLibrary) -> None:
    with pytest.raises(ValueError):
        _evaluator(library, FakeBackend(VALID), max_retries=-1)
    with pytest.raises(ValueError):
        _evaluator(library, FakeBackend(VALID), failure_policy="ignore")


def test_satisfies_protocol(library: PromptLibrary) -> None:
    assert isinstance(_evaluator(library, FakeBackend(VALID)), PhysicsEquationEvaluator)
    assert isinstance(FakeEquationEvaluator.from_table({}), PhysicsEquationEvaluator)


# ------------------------------------------------------------------ avaliador simulado


def test_fake_evaluator_table_by_canonical_key() -> None:
    good = EquationEvaluation(1, 1, 1, "gt")
    fake = FakeEquationEvaluator.from_table({GT: good, canonical_key(OTHER): good})
    assert fake.evaluate(parse_infix("sqrt(height*(9.81*2))"), CTX, "E") == good
    assert fake.evaluate(parse_infix("t + mass"), CTX, "E") == good  # comutativo
    with pytest.raises(EvaluationFailedError):
        fake.evaluate(parse_infix("cos(t)"), CTX, "E")


def test_fake_evaluator_explicit_default_and_function() -> None:
    default = EquationEvaluation(0, 0, 0, "default")
    fake = FakeEquationEvaluator.from_table({}, default=default)
    assert fake.evaluate_many([GT, OTHER], CTX, "E") == [default, default]
    assert fake.calls == 2

    def fn(expr: Expr, ctx: object, variant: str) -> EquationEvaluation:
        v = 1.0 if variant == "D" else 0.5
        return EquationEvaluation(v, v, v, variant)

    fake_fn = FakeEquationEvaluator.from_function(fn)
    assert fake_fn.evaluate(GT, CTX, "D").simplicity == 1.0
    assert fake_fn.evaluate(GT, CTX, "E").simplicity == 0.5
