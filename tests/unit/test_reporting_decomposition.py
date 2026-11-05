"""Decomposição dos termos da perda (etapa 18 do notebook) com DataFrame sintético."""

from __future__ import annotations

import pandas as pd
import pytest

from physics_llm_sr.experiments.reporting import loss_term_contributions


def _frame() -> pd.DataFrame:
    frame: pd.DataFrame = pd.DataFrame(
        {
            "scenario": ["ff", "ff", "ff", "ff"],
            "sr": ["deap", "deap", "deap", "gplearn"],
            "llm": ["m1", "m1", "none", "m1"],
            "final_e": [1.0, 3.0, 9.0, 2.0],
            "final_s": [10.0, 10.0, 10.0, 4.0],
            "final_c": [0.5, 0.5, 0.0, 1.0],
        }
    )
    return frame


def test_decomposition_means_and_fraction() -> None:
    out = loss_term_contributions(_frame(), w1=1.0, w2=0.1, w3=2.0)
    assert list(out.columns) == ["w1·e", "w2·s", "w3·c", "fração do termo LLM em L"]
    deap = out.to_dict("index")[("ff", "deap")]
    assert deap["w1·e"] == pytest.approx(2.0)  # média de 1 e 3; baseline excluído
    assert deap["w2·s"] == pytest.approx(1.0)
    assert deap["w3·c"] == pytest.approx(1.0)
    assert deap["fração do termo LLM em L"] == pytest.approx(
        (1 / 3 + 1 / 5) / 2
    )  # fração por linha, depois média
    assert ("ff", "gplearn") in out.index


def test_decomposition_empty_without_llm_rows() -> None:
    only_base = _frame().assign(llm="none")
    assert loss_term_contributions(only_base, w1=1.0, w2=1.0, w3=1.0).empty
