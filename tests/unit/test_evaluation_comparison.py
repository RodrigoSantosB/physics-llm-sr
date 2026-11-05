"""T:comparacao: classificação pré-registrada (scientific_assumptions.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.evaluation.paper_comparison import (
    CLASSES,
    LLM_ALIASES,
    ComparisonError,
    canonical_llm,
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
        P,
        "tree_distance",
        local,
        ref_published=P_REF,
        local_ref=local - diff,
        **kw,  # type: ignore[arg-type]
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
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


def _cell(out: pd.DataFrame, llm: str, metric: str) -> pd.Series:
    sub = out[(out["llm"] == llm) & (out["metric"] == metric)]
    assert len(sub) == 1
    row: pd.Series = sub.iloc[0]
    return row


def test_compare_with_paper_table5_cells() -> None:
    out = compare_with_paper(_results("full"), paper=load_paper_results(), tables=(5,))
    assert len(out) == 144  # todas as células publicadas; sem dados locais => inconclusiva
    out = out[out["n"] > 0]
    assert len(out) == 8  # 4 métricas x (baseline, mistral) para free_fall/deap
    m = _cell(out, "mistral", "tree_distance")
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


def test_llm_aliases_cover_real_config_names() -> None:
    """Nomes das configs (mistral/llama2/falcon) casam com o CSV (mistral_7b/...)."""
    cfg = load_config("configs/replication.yaml")
    paper = load_paper_results()
    csv_llms = set(paper["llm"]) - {"none", "unspecified"}
    raw = pd.read_csv("references/paper_results/table5_benchmark_supplement.csv")
    assert {"mistral_7b", "llama2_7b", "falcon_7b"} <= set(raw["llm"])  # CSV bruto usa sufixo
    for model in cfg.llm.models:
        assert model.name in csv_llms
        assert canonical_llm(f"{model.name}_7b") == model.name
    assert set(LLM_ALIASES.values()) == {m.name for m in cfg.llm.models}


def test_compare_matches_config_llm_names() -> None:
    res = _results("full")
    res["llm"] = res["llm"].replace({"mistral_7b": "mistral"})
    out = compare_with_paper(res, paper=load_paper_results(), tables=(5,))
    out = out[out["n"] > 0]
    assert len(out) == 8


def test_missing_track_column_is_error() -> None:
    with pytest.raises(ComparisonError, match="track"):
        compare_with_paper(_results("full").drop(columns="track"), tables=(5,))


def test_extended_track_never_reproduction() -> None:
    res = _results("full")
    res["track"] = "extended_analysis"
    out = compare_with_paper(res, paper=load_paper_results(), tables=(5,))
    out = out[out["n"] > 0]
    assert out["reduced_run"].all() and set(out["classe"]) == {"inconclusiva"}


def test_published_ref_without_local_ref_is_inconclusive_rule_1c() -> None:
    res = _results("full")
    res = res[res["llm"] != "none"]  # sem série local do baseline (referência da Tabela 5)
    out = compare_with_paper(res, paper=load_paper_results(), tables=(5,))
    cell = _cell(out[out["n"] > 0], "mistral", "tree_distance")
    assert cell["classe"] == "inconclusiva"
    assert "1(c)" in cell["reason"]
    assert cell["delta_local"] is None or pd.isna(cell["delta_local"])


def _noise_results(levels: tuple[float, ...], noise_type: str, mean_by_level: dict[float, float]):  # type: ignore[no-untyped-def]
    rows = []
    for lvl in levels:
        for i, v in enumerate(sample(mean_by_level[lvl], 0.01, N)):
            rows.append(
                dict(
                    scenario="free_fall", sr="deap", llm="mistral", prompt="A",
                    noise_type=noise_type, noise_location="inputs", noise_level=lvl,
                    repetition=i, mode="full", track="paper_replication",
                    mae=v, mse=v, one_minus_r2=v, tree_distance=v,
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_table4_references_level_1pct_and_table3_cell() -> None:
    paper = load_paper_results()
    t4 = paper[(paper["table"] == 4) & (paper["scenario"] == "free_fall") & (paper["sr"] == "deap")]
    t4 = t4[(t4["noise_location"] == "inputs") & (t4["metric"] == "tree_distance")]
    pub = {float(r.noise_level): float(r.value) for r in t4.itertuples()}
    t3 = paper[(paper["table"] == 3) & (paper["scenario"] == "free_fall") & (paper["sr"] == "deap")]
    t3 = t3[(t3["noise_location"] == "inputs") & (t3["metric"] == "tree_distance")]
    pub3 = {float(r.noise_level): float(r.value) for r in t3.itertuples()}
    res = pd.concat(
        [
            _noise_results((0.01, 0.02), "biased", {0.01: pub[0.01], 0.02: pub[0.02]}),
            _noise_results((0.01, 0.02), "zero_mean", {0.01: pub3[0.01], 0.02: pub3[0.02]}),
        ]
    )
    out = compare_with_paper(
        res, paper=paper, tables=(4,), absolute_level_inconclusive_tables=(),
        unspecified_llm="mistral", unspecified_prompt="A",
    )  # fmt: skip
    sel = out[(out["metric"] == "tree_distance") & (out["noise_location"] == "inputs")]
    lvl2 = sel[np.isclose(sel["noise_level"], 0.02)].iloc[0]
    lvl1 = sel[np.isclose(sel["noise_level"], 0.01)].iloc[0]
    # nível > 1%: referência = nível 1% do mesmo tipo (Tabela 4)
    assert lvl2["ref_published"] == pytest.approx(pub[0.01])
    # nível 1%: referência = célula correspondente da Tabela 3
    assert lvl1["ref_published"] == pytest.approx(pub3[0.01])
