"""Leitura do cache de avaliações e sonda de expressões (notebook 02), com backend simulado."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physics_llm_sr.analysis import llm_inspection as li
from physics_llm_sr.config.loader import load_config
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.cache import CACHE_FILENAME
from physics_llm_sr.llm.inference import FakeBackend
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import LLMEquationEvaluator
from physics_llm_sr.physics.base import FEATURE_NAMES
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import parse_infix

REPO = Path(__file__).resolve().parents[2]
GT = "sqrt(2*9.81*height)"


def _evaluator(cache_path: Path | None) -> LLMEquationEvaluator:
    from physics_llm_sr.llm.cache import EvaluationCache

    cache = EvaluationCache(cache_path) if cache_path else None
    return LLMEquationEvaluator(FakeBackend(), PromptLibrary.load(), cache=cache)


def _ctx() -> PhysicalContext:
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    return get_scenario("free_fall", cfg.physics).physical_context()


def _fill_cache(path: Path, texts: list[str]) -> None:
    ev = _evaluator(path)
    exprs = [parse_infix(t, FEATURE_NAMES) for t in texts]
    ev.evaluate_many(exprs, _ctx(), "E")


def test_read_cache_records_returns_scores_penalty_and_raw(tmp_path: Path) -> None:
    path = tmp_path / CACHE_FILENAME
    _fill_cache(path, [GT, "height + length"])
    with path.open("a", encoding="utf-8") as fh:  # erro e linha corrompida não viram avaliação
        fh.write(json.dumps({"key": "k", "status": "error", "error": "x"}) + "\n{quebrada\n")
    rec = li.read_cache_records(path)
    assert len(rec) == 2 and set(rec["status"]) == {"ok"}
    row = rec[rec["expression"].str.contains("height") & rec["expression"].str.contains("sqrt")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["penalty"] == pytest.approx(1 - (r["c1"] + r["c2"] + r["c3"]) / 3)
    assert isinstance(r["raw_response"], str) and r["raw_response"]
    assert r["scenario"] == "free_fall" and r["prompt_variant"] == "E"


def test_read_cache_records_missing_file_is_empty_with_columns(tmp_path: Path) -> None:
    rec = li.read_cache_records(tmp_path / "nao_existe.jsonl")
    assert rec.empty and {"expression", "c1", "c2", "c3", "penalty"} <= set(rec.columns)


def test_lookup_uses_canonical_form_and_never_invents_scores(tmp_path: Path) -> None:
    path = tmp_path / CACHE_FILENAME
    _fill_cache(path, [GT])
    rec = li.read_cache_records(path)
    found = li.lookup_evaluations(
        rec,
        ["sqrt(height*(2*9.81))", "exp(height)"],  # forma comutada igual; outra ausente
        scenario="free_fall",
        prompt="E",
    )
    assert list(found["found"]) == [True, False]
    assert found.loc[0, "penalty"] == pytest.approx(rec.iloc[0]["penalty"])
    assert found["penalty"].isna()[1] and found["c1"].isna()[1]


def test_cache_file_follows_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSICS_LLM_SR_PROJECT_ROOT", str(tmp_path))
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    assert li.cache_file(cfg) == tmp_path / cfg.llm.cache_dir / CACHE_FILENAME


def test_probe_expressions_shows_raw_scores_feedback_and_penalty() -> None:
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    ctx = get_scenario("free_fall", cfg.physics).physical_context()
    out = li.probe_expressions(
        _evaluator(None), [GT, "9.81*t", "exp(height*height)"], ctx, "E", FEATURE_NAMES
    )
    assert list(out["rotulo"]) == ["GT", "equivalente", "absurda"]
    assert (out["valida"]).all() and out["resposta_crua"].str.len().min() > 0
    assert out["penalty"].between(0, 1).all()
    assert {"c1", "c2", "c3", "feedback", "prompt_user"} <= set(out.columns)


def test_probe_marks_invalid_response_without_substitute_score() -> None:
    class Garbage:
        def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
            return ["isto não é JSON"] * len(prompts)

        def model_metadata(self) -> dict[str, object]:
            return {"backend": "fake", "checkpoint": "x", "revision": None, "quantization": None}

    ev = LLMEquationEvaluator(Garbage(), PromptLibrary.load())
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    ctx = get_scenario("free_fall", cfg.physics).physical_context()
    out = li.probe_expressions(ev, [GT], ctx, "E", FEATURE_NAMES)
    assert not out.loc[0, "valida"] and out["penalty"].isna()[0]
    assert out.loc[0, "resposta_crua"] == "isto não é JSON" and out.loc[0, "erro"]


def test_probe_set_has_gt_equivalent_and_absurd_for_each_scenario() -> None:
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    ff = li.probe_set(get_scenario("free_fall", cfg.physics), absurd="exp(height*height)")
    assert ff == (GT, "9.81*t", "exp(height*height)")
    hm = li.probe_set(get_scenario("harmonic_motion", cfg.physics), absurd="exp(t)")
    assert (
        hm[0] == "length*cos(sqrt(10.0/mass)*t)" and hm[1] == f"1.0*({hm[0]})" and hm[2] == "exp(t)"
    )
    custom = li.probe_set(
        get_scenario("free_fall", cfg.physics), equivalent="sqrt(height*19.62)", absurd="t"
    )
    assert custom[1] == "sqrt(height*19.62)"


def test_close_backend_calls_close_when_available_and_is_silent_otherwise() -> None:
    class Closable(FakeBackend):
        closed = False

        def close(self) -> None:
            self.closed = True

    backend = Closable()
    li.close_backend(LLMEquationEvaluator(backend, PromptLibrary.load()))
    assert backend.closed
    li.close_backend(_evaluator(None))  # FakeBackend sem close: não falha
