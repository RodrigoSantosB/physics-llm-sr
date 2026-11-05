"""Ligação config -> componentes (onda 6): mudar a config muda o parâmetro efetivo.

- ``llm.cache_significant_digits``/``llm.cache_key_mode`` -> dedup e chave de cache do
  ``LLMEquationEvaluator.from_config``.
- ``sr.gplearn.*`` (H-02/H-03) -> ``gplearn.SymbolicRegressor``.
- ``sr.pysr.*`` -> parâmetros do ``pysr.PySRRegressor`` (sem Julia: só ``build_params``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.llm.inference import FakeBackend
from physics_llm_sr.llm.scoring import LLMEquationEvaluator
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.sr.gplearn_adapter import build_estimator
from physics_llm_sr.sr.pysr_adapter import PySRRegressor
from physics_llm_sr.symbolic.expression import parse_infix
from physics_llm_sr.symbolic.operators import DEFAULT_PROTECTION

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"


def _cfg(**sections: dict[str, Any]) -> ExperimentConfig:
    data = load_config(SMOKE).model_dump(mode="json")
    for name, values in sections.items():
        if name == "track":
            data["track"] = values
            continue
        target = data
        for part in name.split("__"):
            target = target[part]
        target.update(values)
    return parse_config(data)


EXPRS = [parse_infix("1.2345*height"), parse_infix("1.2349*height"), parse_infix("2.5*height")]


def _backend_prompts(cfg: ExperimentConfig, tmp_path: Path) -> int:
    llm = cfg.llm.model_copy(update={"cache_dir": str(tmp_path / "cache")})
    backend = FakeBackend()
    evaluator = LLMEquationEvaluator.from_config(llm, backend)
    ctx = get_scenario("free_fall").physical_context()
    evaluator.evaluate_many(EXPRS, ctx, "E")
    return backend.prompt_count


@pytest.mark.parametrize(
    ("sections", "expected_prompts"),
    [
        ({"llm": {"cache_significant_digits": 3}}, 2),  # 1.23*h dedup; 2.5*h
        ({"llm": {"cache_significant_digits": 4}}, 3),  # 1.234 vs 1.235
        ({"llm": {"cache_key_mode": "structure"}, "track": "extended_analysis"}, 1),
    ],
)
def test_evaluator_key_follows_config(
    sections: dict[str, Any], expected_prompts: int, tmp_path: Path
) -> None:
    cfg = _cfg(**sections)
    assert _backend_prompts(cfg, tmp_path) == expected_prompts


def test_gplearn_estimator_reads_config() -> None:
    values = {
        "tournament_size": 7,
        "p_crossover": 0.5,
        "p_subtree_mutation": 0.1,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.2,
        "init_depth": [2, 5],
        "const_range": [-3.0, 4.0],
    }
    cfg = _cfg(sr__gplearn=values)
    params = build_estimator(cfg, ("a", "b"), DEFAULT_PROTECTION, seed=1).get_params()
    for key, value in values.items():
        expected = tuple(value) if isinstance(value, list) else value
        assert params[key] == expected, key
    default = build_estimator(_cfg(), ("a", "b"), DEFAULT_PROTECTION, seed=1).get_params()
    assert (default["tournament_size"], default["init_depth"]) == (3, (1, 8))  # H-02


def test_pysr_params_read_config() -> None:
    values = {
        "populations": 2,
        "population_size": 40,
        "maxsize": 20,
        "ncycles_per_iteration": 77,
        "tournament_selection_n": 9,
    }
    cfg = _cfg(sr__pysr=values)
    used: dict[str, Any] = {}
    params = PySRRegressor().build_params(cfg, seed=1, defaults_used=used)
    for key, value in values.items():
        assert params[key] == value, key
    assert not used  # nenhum default interno usado
