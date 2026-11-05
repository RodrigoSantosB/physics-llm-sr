# mypy: disable-error-code="no-any-return,unused-ignore"
# ruff: noqa: E501
"""T:comparativo: tabelas e figuras artigo x experimentos locais (pacote ``analysis``)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.analysis import figures, loading, tables
from physics_llm_sr.analysis.pipeline import run_analysis
from physics_llm_sr.evaluation.paper_comparison import load_paper_results

REPS = 3
BASE_MAE = {"deap": [1.0, 2.0, 3.0], "gplearn": [4.0, 5.0, 6.0]}
LLM_MAE = {"deap": [0.5, 2.5, 2.0], "gplearn": [4.0, 5.0, 5.0]}


def _rows(track: str, mode: str, norm: str, shift: float) -> pd.DataFrame:
    rows = []
    for sr in ("deap", "gplearn"):
        for llm, prompt, vals in (
            ("none", "none", BASE_MAE[sr]),
            ("mistral", "E", LLM_MAE[sr]),
        ):
            for rep in range(REPS):
                mae = vals[rep] + shift
                rows.append(
                    {
                        "run_id": "r",
                        "cell_key": f"free_fall__{sr}__{llm}__{prompt}__r{rep}",
                        "scenario": "free_fall",
                        "sr": sr,
                        "llm": llm,
                        "prompt": prompt,
                        "noise_type": "zero_mean",
                        "noise_location": "target",
                        "noise_level": 0.01,
                        "repetition": rep,
                        "seed": rep,
                        "mode": mode,
                        "track": track,
                        "normalization": norm,
                        "mae": mae,
                        "mse": mae**2,
                        "one_minus_r2": 0.1 * mae,
                        "tree_distance": 0.5 + 0.1 * rep,
                        "tree_distance_equivalent_min": 0.5,
                        "expression": "x" if llm == "none" else ("x" if rep == 0 else "y"),
                        "final_e": 2.0,
                        "final_s": 4.0,
                        "final_c": 1.0 if llm != "none" else 0.0,
                        "train_seconds": 10.0 if llm == "none" else 30.0,
                        "llm_seconds": 0.0 if llm == "none" else 20.0,
                        "llm_calls": 0 if llm == "none" else 10,
                        "n_generations": 3,
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def _make_run(root: Path, name: str, track: str, norm: str, shift: float) -> Path:
    run = root / name
    run.mkdir(parents=True)
    res = _rows(track, "pilot", norm, shift)
    res.to_parquet(run / "results.parquet")
    hist = res[["cell_key", "scenario", "sr", "llm", "prompt", "repetition"]].copy()
    hist = hist.loc[hist.index.repeat(3)].reset_index(drop=True)
    hist["generation"] = np.tile(np.arange(3), len(hist) // 3)
    hist["best_loss"] = 10.0 / (1 + hist["generation"])
    hist["mode"], hist["track"] = "pilot", track
    hist.to_parquet(run / "history.parquet")
    cfg = {
        "mode": "pilot",
        "track": track,
        "loss": {"w1": 0.5, "w2": 0.25, "w3": 0.25, "normalization": norm},
        "experiments": {"benchmark_prompt": "E", "noise_model": "mistral", "noise_prompt": "E"},
    }
    (run / "run.json").write_text(json.dumps({"mode": "pilot", "track": track, "config": cfg}))
    return run


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    tmp_path = tmp_path_factory.mktemp("runs")
    return [
        _make_run(tmp_path, "lit", "paper_replication", "literal", 0.0),
        _make_run(tmp_path, "norm", "extended_analysis", "normalized", 0.1),
    ]


@pytest.fixture(scope="module")
def data(runs: list[Path]) -> loading.AnalysisData:
    return loading.load_runs(runs)


def test_loading_arms_and_labels(data: loading.AnalysisData) -> None:
    assert set(data.results["arm"]) == {
        "paper_replication/pilot/lit",
        "extended_analysis/pilot/norm",
    }
    labels = {r.arm: r.label for r in data.runs}
    assert "PILOTO" in labels["paper_replication/pilot/lit"]
    assert "não é reprodução" in labels["paper_replication/pilot/lit"]
    assert "desvio declarado" in labels["extended_analysis/pilot/norm"]
    assert not data.history.empty and "arm" in data.history.columns


def test_loading_validates_columns(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "bad", "paper_replication", "literal", 0.0)
    pd.read_parquet(run / "results.parquet").drop(columns=["mae"]).to_parquet(
        run / "results.parquet"
    )
    with pytest.raises(loading.AnalysisError, match="mae"):
        loading.load_runs([run])


def test_t5_diff_is_local_minus_published(data: loading.AnalysisData) -> None:
    t5 = tables.table5(data)
    paper = load_paper_results()
    pub = paper[
        (paper["table"] == 5)
        & (paper["scenario"] == "free_fall")
        & (paper["sr"] == "deap")
        & (paper["llm"] == "mistral")
        & (paper["metric"] == "mae")
    ]["value"].iloc[0]
    row = t5[
        (t5["arm"] == "paper_replication/pilot/lit")
        & (t5["sr"] == "deap")
        & (t5["llm"] == "mistral")
        & (t5["metric"] == "mae")
    ].iloc[0]
    assert row["published"] == pub
    assert row["n"] == 3
    assert row["local_mean"] == pytest.approx(5.0 / 3.0)
    assert row["abs_diff"] == pytest.approx(row["local_mean"] - pub)
    assert row["status_local"] == "executado"
    assert "PILOTO" in row["rotulo"]


def test_missing_cells_are_labeled_nan(data: loading.AnalysisData) -> None:
    t2 = tables.table2(data)
    assert len(t2) > 0
    assert (t2["status_local"] == "não executado").all()
    assert t2["local_mean"].isna().all()
    assert (t2["n"] == 0).all()
    assert not (t2["local_mean"] == 0).any()
    t3 = tables.table3(data)
    done = t3[t3["status_local"] == "executado"]
    assert set(done["noise_location"]) == {"target"}
    assert (t3["status_local"] == "não executado").any()


def test_paired_effect(data: loading.AnalysisData) -> None:
    eff = tables.paired_effect(data)
    row = eff[
        (eff["arm"] == "paper_replication/pilot/lit")
        & (eff["sr"] == "deap")
        & (eff["metric"] == "mae")
    ].iloc[0]
    assert row["mean_diff"] == pytest.approx(np.mean([0.5, -0.5, 1.0]))  # baseline - llm
    assert row["n"] == 3
    assert row["n_expr_changed"] == 2
    assert 0 <= row["p_value"] <= 1
    assert row["ci_low"] < row["mean_diff"] < row["ci_high"]
    cmp = tables.arm_comparison(eff)
    assert any(c.endswith("mean_diff") for c in cmp.columns)
    assert len(cmp) == len(eff) // 2


def test_loss_decomposition_fractions(data: loading.AnalysisData) -> None:
    dec = tables.loss_decomposition(data)
    row = dec.iloc[0]
    total = row["frac_w1e"] + row["frac_w2s"] + row["frac_w3c"]
    assert total == pytest.approx(1.0)
    assert row["w1e"] == pytest.approx(0.5 * 2.0)
    assert row["w3c"] == pytest.approx(0.25 * 1.0)


def test_runtime_table(data: loading.AnalysisData) -> None:
    t6 = tables.table6(data)
    row = t6[(t6["sr"] == "deap") & (t6["arm"] == "paper_replication/pilot/lit")].iloc[0]
    assert row["local_total_llm_s"] == pytest.approx(20.0)
    assert row["local_total_sr_s"] == pytest.approx(10.0)
    assert row["local_calls_mean"] == pytest.approx(10.0)
    assert row["status_local"] == "executado"
    assert (t6["status_local"] == "não executado").any()


def test_classification_summary(data: loading.AnalysisData) -> None:
    summ = tables.classification_summary(data)
    assert {"arm", "table", "classe", "executado", "count", "origem"} <= set(summ.columns)
    assert summ["count"].sum() > 0


def test_write_table_formats(data: loading.AnalysisData, tmp_path: Path) -> None:
    spec = tables.build_tables(data)["t5_benchmark"]
    arts = tables.write_table(spec, tmp_path)
    exts = {a.path.suffix for a in arts}
    assert exts == {".csv", ".md", ".tex"}
    tex = (tmp_path / "t5_benchmark.tex").read_text()
    assert "\\toprule" in tex and "PILOTO" in tex
    assert "PILOTO" in (tmp_path / "t5_benchmark.md").read_text()


def test_all_figures_created(data: loading.AnalysisData, tmp_path: Path) -> None:
    built = tables.build_tables(data)
    arts = figures.build_figures(data, built, tmp_path)
    ids = {a.artifact_id.split("_")[0] for a in arts}
    assert {f"F{i}" for i in range(1, 9)} <= ids
    for a in arts:
        assert a.path.exists() and a.path.stat().st_size > 0
    assert any(a.path.suffix == ".pdf" for a in arts)
    assert all("PILOTO" in a.label for a in arts)


def test_figures_without_data_do_not_fail(data: loading.AnalysisData, tmp_path: Path) -> None:
    only_base = loading.AnalysisData(
        results=data.results[data.results["llm"] == "none"].copy(),
        history=data.history.iloc[0:0].copy(),
        runs=data.runs,
    )
    arts = figures.build_figures(only_base, tables.build_tables(only_base), tmp_path)
    assert len(arts) >= 8


def test_latex_noise_table_style(data: loading.AnalysisData, tmp_path: Path) -> None:
    built = tables.build_tables(data)
    tex = built["t3_noise_zero_mean"].tex
    assert tex is not None
    for token in ("Tabela I", r"\textsc", r"\toprule", r"\cmidrule", "Ruído nas entradas",
                  "Ruído no alvo", "Ruído em ambos", "Este trabalho: n = 3", "Mistral", "prompt E"):  # fmt: skip
        assert token in tex, token
    # ordem das linhas do bloco: Artigo, Com LLM, Sem LLM; Cenário só na primeira linha
    pos = [tex.index(x) for x in ("Artigo &", "Com LLM &", "Sem LLM &")]
    assert pos == sorted(pos)
    assert "0,17" in tex  # vírgula decimal (DEAP, queda livre, entradas, 1%: artigo)
    assert not re.search(r"\d\.\d\d \\\\", tex)
    assert tex.count("Queda livre") == tex.count(r"\begin{tabular}")  # cenário só na 1ª linha
    assert "n/e" in tex  # células locais não executadas


def test_latex_cost_table_reports_missing_field(data: loading.AnalysisData) -> None:
    cost = tables.build_tables(data)["t6_llm_cost"]
    assert (cost.df["missing_fields"] == "llm_invalid_responses").all()
    assert cost.df["failure_rate_pct"].isna().all()
    assert "n/d" in (cost.tex or "") and "llm_invalid_responses" in cost.note
    assert cost.df["median_llm_sr_ratio"].iloc[0] == pytest.approx(2.0)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex ausente")
def test_master_tex_compiles(data: loading.AnalysisData, tmp_path: Path) -> None:
    built = tables.build_tables(data)
    for spec in built.values():
        tables.write_table(spec, tmp_path)
    tables.write_master_tex(built, tmp_path)
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "tabelas.tex"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert (tmp_path / "tabelas.pdf").exists(), proc.stdout[-800:]


def test_f2_three_series_per_sr_and_panel_titles(
    data: loading.AnalysisData, tmp_path: Path
) -> None:
    import matplotlib.pyplot as plt

    captured: list[Any] = []
    orig = plt.close
    plt.close = lambda fig=None: captured.append(fig)  # type: ignore[assignment]
    try:
        arm = data.arms[0]
        figures.figure2(data, "zero_mean", arm, tmp_path)
    finally:
        plt.close = orig  # type: ignore[assignment]
    fig = captured[0]
    axes = fig.axes
    titles = [a.get_title() for a in axes]
    assert "ball_drop | ruído em features" in titles
    assert "wave | ruído em both" in titles
    assert fig._suptitle.get_text() == "Tabela 3: distância vs nível de ruído (média zero)"
    ax = axes[1]  # ball_drop | target (fixture: free_fall, target, 1%)
    labels = {ln.get_label() for ln in ax.get_lines()}
    for sr in ("deap", "gplearn"):
        assert {f"{sr} + LLM", f"{sr} sem LLM", f"{sr} (artigo)"} <= labels
    styles = {ln.get_label(): ln.get_linestyle() for ln in ax.get_lines()}
    assert styles["deap + LLM"] == "-" and styles["deap sem LLM"] == ":"
    assert styles["deap (artigo)"] == "--"
    colors = {ln.get_color() for ln in ax.get_lines() if ln.get_label().startswith("deap")}
    assert colors == {"C0"}
    assert any("PILOTO" in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_cli_end_to_end(runs: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "physics_llm_sr.analysis", "--runs", *map(str, runs),
         "--out", str(out)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    index = (out / "INDEX.md").read_text()
    assert "Tabela 5" in index and "F1" in index and "PILOTO" in index
    assert (out / "tables" / "t2_prompt_sensitivity.csv").exists()


def test_run_analysis_returns_index(runs: list[Path], tmp_path: Path) -> None:
    out = run_analysis(runs, tmp_path / "o")
    assert (out / "INDEX.md").exists()


SMOKE = Path(__file__).resolve().parents[2] / "results" / "smoke"


def _find_smoke() -> Path | None:
    cands = sorted(SMOKE.glob("*/results.parquet")) if SMOKE.exists() else []
    return cands[-1].parent if cands else None


@pytest.mark.skipif(_find_smoke() is None, reason="run smoke real inexistente")
def test_cli_on_real_smoke(tmp_path: Path) -> None:
    smoke = _find_smoke()
    assert smoke is not None
    run_analysis([smoke], tmp_path / "o")
    assert (tmp_path / "o" / "INDEX.md").exists()


def test_notebook_run_dirs_resolve_relative_to_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_llm_sr.analysis import notebook

    monkeypatch.setenv("PHYSICS_LLM_SR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv(notebook.RUN_DIRS_ENV, raising=False)
    got = notebook.resolve_run_dirs(["results/nb01_baseline/x", str(tmp_path / "abs")])
    assert got == [tmp_path / "results/nb01_baseline/x", tmp_path / "abs"]
