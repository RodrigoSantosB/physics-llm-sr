"""Integração Expressão → LLM (PRD §12.2): parser seguro → prompt → backend → avaliação."""

from __future__ import annotations

import json
from pathlib import Path

from physics_llm_sr.config.schemas import LLMConfig, LLMModelConfig
from physics_llm_sr.llm.cache import EvaluationCache
from physics_llm_sr.llm.inference import build_backend
from physics_llm_sr.llm.interfaces import PhysicsEquationEvaluator
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import LLMEquationEvaluator, penalty
from physics_llm_sr.physics.registry import SCENARIO_NAMES, get_scenario
from physics_llm_sr.symbolic.expression import parse_infix, to_infix

FEATURES = ("mass", "length", "height", "damping", "t")


def test_expression_to_llm_end_to_end(tmp_path: Path) -> None:
    config = LLMConfig(
        backend="fake",
        models=[LLMModelConfig(name="mistral", checkpoint="fake-mistral", revision="r")],
        max_retries=1,
    )
    backend = build_backend(config, "mistral")
    cache_path = tmp_path / "llm.jsonl"
    evaluator: PhysicsEquationEvaluator = LLMEquationEvaluator(
        backend,
        PromptLibrary.load(),
        cache=EvaluationCache(cache_path),
        max_retries=config.max_retries,
        failure_policy=config.failure_policy,
    )
    for name in SCENARIO_NAMES:
        ctx = get_scenario(name).physical_context()
        gt = parse_infix(ctx.ground_truth, FEATURES)
        candidates = [gt, parse_infix("mass + t", FEATURES)]
        results = evaluator.evaluate_many(candidates, ctx, "E")
        assert len(results) == 2
        for ev in results:
            assert all(0.0 <= s <= 1.0 for s in ev.scores)
            assert 0.0 <= penalty(ev) <= 1.0
        # determinismo: reavaliar devolve o mesmo (cache) para cada cenário
        assert evaluator.evaluate(gt, ctx, "E") == results[0]

    records = [json.loads(x) for x in cache_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2 * len(SCENARIO_NAMES)
    assert {r["metadata"]["scenario"] for r in records} == set(SCENARIO_NAMES)
    first_gt = get_scenario("free_fall").physical_context().ground_truth
    assert records[0]["metadata"]["expression"] == to_infix(parse_infix(first_gt))
    assert all(r["metadata"]["gt_disclosed"] is False for r in records)
