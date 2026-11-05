"""T:comparacao — classificação pré-registrada (scientific_assumptions.md, 'Critérios de comparação')."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.evaluation.paper_comparison import (
    CLASSES,
    classify_cell,
    compare_with_paper,
    load_paper_results,
)

N = 25
P, P_REF = 0.30, 0.50  # distância (2 casas): P = [0.295, 0.305]; Δp = -0.20


def sample(mean: float, sd: float, n: int = N) -> np.ndarray:
    """Amostra determinística com média e dp amostral exatos."""
    z = np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + sd * z


def make(local_mean: float, delta_mean: float, delta_sd: float = 0.02, n: int = N, **kw: object):  # type: ignore[no-untyped-def]
    local = sample(local_mean, 0.01, n)
    diff = sample(delta_mean, delta_sd, n)[::-1]
    return classify_cell(
        P, "tree_distance", local, ref_published=P_REF, local_ref=local - diff, **kw  # type: ignore[arg-type]
    )


def test_numericamente_proxima() -> None:
    r = make(0.30, -0.20)
    assert r.classe == "numericamente_proxima" and not r.reduced_run
    assert r.abs_diff == pytest.approx(0.0, abs=1e-12)
    assert r.rel_diff == pytest.approx(0.0, abs=1e-12)


def test_qualitativa() -> None:
    assert make(0.40, -0.20).classe == "qualitativa"


def test_parcial_ci_intersects_direction_opposite() -> None:
    assert make(0.30, +0.20).classe == "parcial"


def test_parcial_sign_agrees_ci_contains_zero() -> None:
    assert make(0.40, -0.01, delta_sd=0.1).classe == "parcial"


def test_divergente() -> None:
    assert make(0.40, +0.20).classe == "divergente"


def test_inconclusiva_n_menor_que_25() -> None:
    r = make(0.30, -0.20, n=10)
    assert r.classe == "inconclusiva" and r.reduced_run


def test_inconclusiva_gap_reason() -> None:
    r = make(0.30, -0.20, inconclusive_reason="GAP-12")
    assert r.classe == "inconclusiva" and r.indicative_class is None and r.reason == "GAP-12"


def test_no_reference_close() -> None:
    r = classify_cell(P, "tree_distance", sample(0.30, 0.01))
    assert r.classe == "numericamente_proxima" and r.delta_local is None
    assert classify_cell(P, "tree_distance", sample(0.40, 0.01)).classe == "divergente"


def test_relative_none_when_published_zero() -> None:
    r = classify_cell(0.0, "tree_distance", sample(0.0, 0.001))
    assert r.rel_diff is None and r.published_low == 0.0  # truncado em 0
    assert r.classe == "numericamente_proxima"


def test_reduced_run_labeled_never_proxima() -> None:
    r = make(0.30, -0.20, reduced_run=True)
    assert r.reduced_run and r.classe == "inconclusiva"
    assert r.indicative_class == "numericamente_proxima"
    assert r.classe in CLASSES


def _results(mode: str, reps: int = N) -> pd.DataFrame:
    rows = []
    for llm, prompt, mean in [("none", "none", 0.25), ("mistral_7b", "A", 0.10)]:
        for i, v in enumerate(sample(mean, 0.01, reps)):
            rows.append(
                dict(
                    scenario="free_fall", sr="deap", llm=llm, prompt=prompt,
                    noise_type="zero_mean", noise_location="target", noise_level=0.01,
                    repetition=i, mode=mode, track="paper_replication",
                    mae=v, mse=v, one_minus_r2=v, tree_distance=v,
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


def _cell(out: pd.DataFrame, llm: str, metric: str) -> pd.Series:
    sub = out[(out["llm"] == llm) & (out["metric"] == metric)]
    assert len(sub) == 1
    return sub.iloc[0]


def test_compare_with_paper_table5_cells() -> None:
    out = compare_with_paper(_results("full"), paper=load_paper_results(), tables=(5,))
    assert len(out) == 144  # todas as células publicadas; sem dados locais => inconclusiva
    out = out[out["n"] > 0]
    assert len(out) == 8  # 4 métricas x (baseline, mistral) para free_fall/deap
    m = _cell(out, "mistral_7b", "tree_distance")
    assert m["published"] == 0.03 and m["baseline_published"] == 0.25
    assert m["baseline_local_mean"] == pytest.approx(0.25)
    assert m["classe"] == "qualitativa" and not m["reduced_run"]
    assert _cell(out, "none", "tree_distance")["classe"] == "numericamente_proxima"


def test_compare_with_paper_pilot_is_labeled() -> None:
    out = compare_with_paper(_results("pilot", reps=3), paper=load_paper_results(), tables=(5,))
    out = out[out["n"] > 0]
    assert out["reduced_run"].all()
    assert set(out["classe"]) == {"inconclusiva"}
    assert "numericamente_proxima" not in set(out["classe"])


def test_compare_missing_local_is_inconclusive() -> None:
    out = compare_with_paper(_results("full"), paper=load_paper_results(), tables=(2,))
    assert (out["classe"] == "inconclusiva").all()  # Tabela 2 = média de 3 cenários; só há 1
