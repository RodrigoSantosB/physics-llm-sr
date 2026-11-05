"""Matriz: ``experiments.baseline_only`` (sem LLM) e filtro ``arm`` (baseline | llm)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.experiments.matrix import build_matrix

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "configs" / "smoke.yaml"


def _data(**experiments: object) -> dict[str, object]:
    data = load_config(BASE).model_dump(mode="json")
    data["experiments"].update(
        run=["baseline", "llm_integration", "noise_robustness", "biased_noise"],
        scenarios=["free_fall"],
        algorithms=["deap", "gplearn"],
        repetitions=2,
        **experiments,
    )
    data["data"]["exp4"].update(levels=[0.01, 0.05], locations=["target"])
    data["data"]["exp5"].update(levels=[0.01], locations=["target", "inputs"])
    return data


def test_baseline_only_keeps_only_cells_without_llm() -> None:
    cfg = parse_config(_data(baseline_only=True))
    cells = [p.cell for p in build_matrix(cfg)]
    assert cells and all(c.is_baseline for c in cells)
    # Exp1: 4; llm_integration: 0 novas (só o baseline pareado, que coincide com o Exp1);
    # Exp4: 0.01/target é o ruído padrão (= Exp1, deduplicada) e sobra 0.05: 1 x 2 sr x 2 rep = 4;
    # Exp5: 1 nível x 2 localizações x 2 sr x 2 rep = 8
    assert len(cells) == 4 + 4 + 8
    kinds = {c.noise.noise_type for c in cells}
    assert kinds == {"zero_mean", "biased"}


def test_baseline_only_does_not_require_declared_noise_model() -> None:
    data = _data(baseline_only=True, noise_model="inexistente")
    parse_config(data)  # não levanta
    with pytest.raises(Exception, match="noise_model"):
        parse_config(_data(baseline_only=False, noise_model="inexistente"))


def test_arm_filter_splits_baseline_and_llm_without_overlap() -> None:
    cfg = parse_config(_data())
    base = build_matrix(cfg, arm="baseline")
    llm = build_matrix(cfg, arm="llm")
    everything = build_matrix(cfg)
    assert all(p.cell.is_baseline for p in base)
    assert all(not p.cell.is_baseline for p in llm)
    keys = {p.cell.key() for p in base} | {p.cell.key() for p in llm}
    assert keys == {p.cell.key() for p in everything}
    assert not ({p.cell.key() for p in base} & {p.cell.key() for p in llm})


def test_arm_filter_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="arm"):
        build_matrix(parse_config(_data()), arm="outro")
