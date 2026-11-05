"""T:llm-cache - cache por (expressão canônica, modelo+revisão, variante+prompt, contexto)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from physics_llm_sr.llm.cache import EvaluationCache, cache_key, context_hash
from physics_llm_sr.llm.inference import FakeBackend
from physics_llm_sr.llm.interfaces import EquationEvaluation, EvaluationFailedError
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import LLMEquationEvaluator
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import parse_infix

REPO_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
VALID = json.dumps(
    {"dimensional_consistency": 0.9, "simplicity": 0.8, "physical_realism": 0.7, "feedback": "x"}
)
CTX = get_scenario("free_fall").physical_context()
GT = parse_infix("sqrt(2*9.81*height)")


@pytest.fixture(scope="module")
def library() -> PromptLibrary:
    return PromptLibrary.load(REPO_PROMPTS)


def _make(
    library: PromptLibrary, cache: EvaluationCache, checkpoint: str = "m", revision: str = "r1"
) -> tuple[LLMEquationEvaluator, FakeBackend]:
    backend = FakeBackend(VALID, checkpoint=checkpoint, revision=revision)
    return LLMEquationEvaluator(backend, library, cache=cache), backend


def test_second_identical_evaluation_is_hit(library: PromptLibrary) -> None:
    cache = EvaluationCache()
    ev, backend = _make(library, cache)
    first = ev.evaluate(GT, CTX, "E")
    second = ev.evaluate(parse_infix("sqrt(height*(2*9.81))"), CTX, "E")  # mesma canônica
    assert first == second
    assert backend.prompt_count == 1
    assert cache.stats.hits == 1 and cache.stats.misses == 1 and cache.stats.calls_avoided == 1
    assert ev.metrics.cache_hits == 1 and ev.metrics.cache_misses == 1


def test_changing_model_revision_variant_or_context_misses(library: PromptLibrary) -> None:
    cache = EvaluationCache()
    ev, backend = _make(library, cache)
    ev.evaluate(GT, CTX, "E")
    ev.evaluate(GT, CTX, "B")  # variante
    other_ctx = dataclasses.replace(CTX, experiment_description="outro experimento")
    ev.evaluate(GT, other_ctx, "E")  # contexto
    assert backend.prompt_count == 3
    ev_model, backend_model = _make(library, cache, checkpoint="other-model")
    ev_model.evaluate(GT, CTX, "E")  # modelo
    ev_rev, backend_rev = _make(library, cache, revision="r2")
    ev_rev.evaluate(GT, CTX, "E")  # revisão
    assert backend_model.prompt_count == 1 and backend_rev.prompt_count == 1
    assert cache.stats.hits == 0 and cache.stats.misses == 5


def test_key_depends_on_every_component() -> None:
    base = {
        "expression": GT,
        "model_id": "m",
        "model_revision": "r",
        "prompt_variant": "E",
        "prompt_hash": "h",
        "context_digest": context_hash(CTX),
    }
    k0 = cache_key(**base)  # type: ignore[arg-type]
    for field, value in [
        ("expression", parse_infix("t")),
        ("model_id", "m2"),
        ("model_revision", None),
        ("prompt_variant", "D"),
        ("prompt_hash", "h2"),
        ("context_digest", "c2"),
    ]:
        assert cache_key(**{**base, field: value}) != k0  # type: ignore[arg-type]
    assert len(k0) == 64


def test_context_hash_sensitive_to_units_and_gt() -> None:
    h = context_hash(CTX)
    assert context_hash(dataclasses.replace(CTX, ground_truth="t")) != h
    assert context_hash(dataclasses.replace(CTX, units={})) != h
    assert context_hash(get_scenario("free_fall").physical_context()) == h


def test_persistence_reloaded(tmp_path: Path, library: PromptLibrary) -> None:
    path = tmp_path / "cache" / "llm.jsonl"
    ev, backend = _make(library, EvaluationCache(path))
    original = ev.evaluate(GT, CTX, "E")
    assert backend.prompt_count == 1

    reloaded = EvaluationCache(path)
    ev2, backend2 = _make(library, reloaded)
    assert ev2.evaluate(GT, CTX, "E") == original
    assert backend2.prompt_count == 0
    assert reloaded.stats.hits == 1 and len(reloaded) == 1

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "ok" and record["raw_response"] == VALID
    assert record["metadata"]["prompt_variant"] == "E"
    assert record["metadata"]["model_id"] == "m"


def test_errors_are_persisted_but_never_served(tmp_path: Path, library: PromptLibrary) -> None:
    path = tmp_path / "llm.jsonl"
    cache = EvaluationCache(path)
    backend = FakeBackend("sem json", checkpoint="m", revision="r1")
    ev = LLMEquationEvaluator(backend, library, cache=cache, max_retries=1)
    with pytest.raises(EvaluationFailedError):
        ev.evaluate(GT, CTX, "E")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in records] == ["error"]
    assert "NoJSONObjectError" in records[0]["error"]
    assert records[0]["raw_responses"] == ["sem json", "sem json"]

    reloaded = EvaluationCache(path)
    assert len(reloaded) == 0 and reloaded.error_count == 1
    ev_ok, backend_ok = _make(library, reloaded)
    ev_ok.evaluate(GT, CTX, "E")
    assert backend_ok.prompt_count == 1


def test_corrupted_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "llm.jsonl"
    good = EquationEvaluation(1, 1, 1, "ok")
    cache = EvaluationCache(path)
    cache.put("k1", good, raw_response="{}", metadata={})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "k2", "status": "ok", "evaluation": {"dimensional_c')  # truncada
    reloaded = EvaluationCache(path)
    assert reloaded.get("k1") == good
    assert reloaded.corrupted_lines == 1


def test_out_of_range_record_on_disk_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "llm.jsonl"
    bad = {
        "key": "k",
        "status": "ok",
        "evaluation": {
            "dimensional_consistency": 2.0,
            "simplicity": 1,
            "physical_realism": 1,
            "feedback": "",
        },
    }
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    reloaded = EvaluationCache(path)
    assert reloaded.get("k") is None and reloaded.corrupted_lines == 1
