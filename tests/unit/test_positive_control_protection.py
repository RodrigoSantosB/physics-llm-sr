"""Controle positivo: formas equivalentes ignoram pontos em que a proteção dos operadores ativa.

Bug reportado: ``2*height/t`` (exata para t > 0) era reprovada porque em t < 0,001 a divisão
protegida devolve 1 em vez do limite 0. A forma primária continua avaliada em todos os pontos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physics_llm_sr.symbolic.expression import parse_infix, protected_points
from physics_llm_sr.workflows import baseline_study as bs

FEATURES = ("mass", "length", "height", "damping", "t")


def _design(tmp_path: Path) -> bs.StudyDesign:
    # dados suficientes para conter instantes com t < division_epsilon (como no notebook 05)
    return bs.StudyDesign(
        algorithm="gplearn",
        scenarios=("free_fall",),
        repetitions=1,
        master_seed=42,
        n_simulations=500,
        output_root=tmp_path / "estudo",
    )


def test_protected_points_marks_only_protected_region() -> None:
    x = np.array([[1.0, 1.0, 2.0, 0.0, t] for t in (0.0, 0.0005, 0.002, 1.0, -0.0001)])
    mask = protected_points(parse_infix("2*height/t"), x, FEATURES)
    assert mask.tolist() == [True, True, False, False, True]
    assert not protected_points(parse_infix("9.81*t"), x, FEATURES).any()
    assert protected_points(parse_infix("log(t)"), x, FEATURES).tolist()[:3] == [True, True, False]
    assert protected_points(parse_infix("exp(1000*t)"), x, FEATURES).tolist()[3]


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("pc")
    return bs.positive_control(
        "free_fall", _design(tmp), extra_forms=("2.1*height/t", "2*height/t + 0.01")
    )


def test_positive_control_free_fall_approves_all_declared_forms(control) -> None:  # type: ignore[no-untyped-def]
    declared = control[~control["forma"].isin(["2.1*height/t", "2*height/t + 0.01"])]
    assert len(declared) == 4
    assert declared["aprovado"].all()
    row = declared[declared["forma"] == "2*height/t"].iloc[0]
    assert row["pontos_protegidos"] > 0 and row["pontos_protegidos_grid"] > 0
    assert row["erro_max_abs_grid"] < 1e-9
    primary = declared[declared["forma_primaria"]].iloc[0]
    assert primary["pontos_protegidos"] == 0 and primary["pontos_protegidos_grid"] == 0


def test_wrong_forms_stay_rejected_and_mask_does_not_hide_errors(control) -> None:  # type: ignore[no-untyped-def]
    wrong = control.set_index("forma")
    assert not wrong.loc["2.1*height/t", "aprovado"]
    shifted = wrong.loc["2*height/t + 0.01"]
    assert not shifted["aprovado"]
    # fora da região protegida o erro de 0,01 aparece inteiro
    assert shifted["erro_max_abs_grid"] == pytest.approx(0.01, rel=1e-6)
    assert shifted["mae"] == pytest.approx(0.01, rel=1e-6)
