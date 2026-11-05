"""T:llm-real - avaliação com LLM real via Ollama local (opt-in: ``-m llm_real``).

Verifica só o contrato (escores válidos em [0, 1]), não valores. Requer servidor Ollama
com o modelo :data:`MODEL` (``ollama pull mistral:7b-instruct``); sem ele, o teste é
pulado com motivo explícito.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from physics_llm_sr.llm.cache import EvaluationCache
from physics_llm_sr.llm.inference import OllamaBackend
from physics_llm_sr.llm.interfaces import BackendError
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import LLMEquationEvaluator, penalty
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import parse_infix

MODEL = os.environ.get("PHYSICS_LLM_SR_OLLAMA_MODEL", "mistral:7b-instruct")
ABSURD = "sin(sin(sin(mass))) * exp(damping) + log(length) / t"
MAX_RETRIES = 2

pytestmark = pytest.mark.llm_real


@pytest.fixture(scope="module")
def backend() -> OllamaBackend:
    backend = OllamaBackend(MODEL, quantization="q4", infra="local-ollama")
    try:
        digest = backend.fetch_digest()
    except BackendError as exc:
        pytest.skip(f"Ollama indisponível em {backend.base_url}: {exc}")
    if digest is None:
        pytest.skip(f"modelo {MODEL} não instalado no Ollama")
    backend.revision = digest
    return backend


def test_real_llm_returns_valid_contract(backend: OllamaBackend, tmp_path: Path) -> None:
    ctx = get_scenario("free_fall").physical_context()
    evaluator = LLMEquationEvaluator(
        backend,
        PromptLibrary.load(),
        cache=EvaluationCache(tmp_path / "llm_real.jsonl"),
        max_retries=MAX_RETRIES,
        failure_policy="retry",
    )
    gt = parse_infix(ctx.ground_truth)
    absurd = parse_infix(ABSURD)
    results = evaluator.evaluate_many([gt, absurd], ctx, "E")
    for ev in results:
        assert all(0.0 <= s <= 1.0 for s in ev.scores)
        assert 0.0 <= penalty(ev) <= 1.0
    print(
        "\nLLM real:",
        backend.model_metadata(),
        "\n GT:",
        results[0],
        "\n absurda:",
        results[1],
        "\n métricas:",
        evaluator.metrics.as_dict(),
    )
    assert evaluator.metrics.backend_calls >= 2
    # segunda chamada idêntica vem do cache
    assert evaluator.evaluate(gt, ctx, "E") == results[0]
    assert evaluator.metrics.cache_hits == 1
