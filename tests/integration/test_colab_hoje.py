# ruff: noqa: E501
"""Perfil ``colab_hoje``: configs da execução de hoje (queda livre, ~4 h no A100).

Confere as duas configs, a matriz por etapa P1-P4, o perfil, a estimativa de tempo e (com
backend ``fake`` e tamanhos mínimos, sem Julia: PySR fora da execução) P1-P4 + análise +
exportação legada de ponta a ponta.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr import colab
from physics_llm_sr.analysis.legacy_export import export_legacy
from physics_llm_sr.analysis.pipeline import run_analysis
from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.core_types import Track
from physics_llm_sr.experiments.matrix import build_matrix

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO / "configs" / "colab_hoje_queda_livre.yaml"
LITERAL = REPO / "configs" / "colab_hoje_queda_livre_literal.yaml"
#: células totais por experimento e NOVAS por etapa (deduplicadas contra as anteriores).
TOTAL_BY_EXP = {"exp1": 9, "exp2": 18, "exp3": 45, "exp4": 54, "exp5": 36}
NEW_BY_STAGE = {"P1": 18, "P2": 27, "P3": 36, "P4": 36}
FAKE_SR = ["deap", "gplearn"]


def test_configs_load_and_validate() -> None:
    main = load_config(MAIN)
    assert main.track is Track.EXTENDED_ANALYSIS and main.loss.normalization == "normalized"
    assert main.llm.backend == "transformers" and main.llm.batch_size == 16
    assert main.llm.models[0].checkpoint == "mistralai/Mistral-7B-Instruct-v0.1"
    assert main.llm.models[0].revision == "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca"
    assert main.experiments.max_cell_seconds == 1800
    assert main.llm.prompt_variants == ["A", "C", "E", "H"]
    assert (main.data.exp4.levels, main.data.exp4.locations) == ([0.01, 0.03, 0.05], ["target"])
    assert (main.data.exp5.levels, main.data.exp5.locations) == ([0.01, 0.05], ["target"])
    pysr = main.sr.pysr
    assert (pysr.populations, pysr.population_size) == (1, 50)
    assert (pysr.ncycles_per_iteration, pysr.niterations) == (50, 10)
    assert (main.sr.population, main.sr.generations) == (100, 50)
    lit = load_config(LITERAL)
    assert lit.track is Track.PAPER_REPLICATION and lit.loss.normalization == "literal"
    assert lit.experiments.run == ["baseline", "llm_integration"]


def test_matrix_counts_per_experiment_and_stage_only_free_fall() -> None:
    cfg = load_config(MAIN)
    for exp, total in TOTAL_BY_EXP.items():
        assert len(build_matrix(cfg, only=[exp])) == total, exp
    everything = build_matrix(cfg)
    assert len(everything) == sum(NEW_BY_STAGE.values()) == 117
    assert {p.cell.scenario for p in everything} == {"free_fall"}
    assert {p.cell.sr for p in everything} == {"deap", "gplearn", "pysr"}
    assert {p.cell.llm for p in everything} == {"mistral", "none"}
    noise = {(p.cell.noise.noise_type, p.cell.noise.noise_location) for p in everything}
    assert noise == {("zero_mean", "target"), ("biased", "target")}
    prompts = Counter(p.cell.prompt for p in everything if not p.cell.is_baseline)
    assert prompts["A"] == prompts["C"] == prompts["H"] == 9 and set(prompts) == {
        "A",
        "C",
        "E",
        "H",
    }
    assert len(build_matrix(load_config(LITERAL))) == 18


def test_profile_colab_hoje_resolved() -> None:
    assert "colab_hoje" in colab.PROFILES
    assert colab.CONFIG_FILES[("colab_hoje", "extended")] == "colab_hoje_queda_livre.yaml"
    assert (
        colab.CONFIG_FILES[("colab_hoje", "replication")] == "colab_hoje_queda_livre_literal.yaml"
    )
    cfg = colab.notebook_config("colab_hoje", "extended", repo_dir=REPO)
    assert cfg.output_dir == "results/colab_hoje_queda_livre"
    assert colab.notebook_config("colab_hoje", "replication", repo_dir=REPO).track is (
        Track.PAPER_REPLICATION
    )


def test_estimate_runtime_is_deterministic_and_correct() -> None:
    a = colab.estimate_runtime(MAIN, 0.3, 2.5)
    b = colab.estimate_runtime(MAIN, 0.3, 2.5)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["stage"]) == ["P1", "P2", "P3", "P4"]
    assert list(a["cells"]) == [18, 27, 36, 36]
    assert list(a["llm_cells"]) == [9, 27, 18, 18]
    # P1: 3 reps x (deap 300 + gplearn 1000 em lote; pysr 200 em série)
    assert a.loc[0, "llm_calls"] == 3 * (300 + 1000 + 200)
    assert a.loc[0, "llm_seconds"] == pytest.approx(3 * (1300 * 0.3 + 200 * 2.5))
    custom = colab.estimate_runtime(MAIN, 1.0, 1.0, {"deap": 10, "gplearn": 10, "pysr": 10})
    assert custom.loc[0, "llm_seconds"] == 90.0


def _fake_cfg(tmp_path: Path):  # type: ignore[no-untyped-def]
    data = load_config(MAIN).model_dump(mode="json")
    data["llm"]["backend"] = "fake"
    data["llm"]["batch_size"] = 1
    data["data"].update(n_simulations=5, n_time_points=20)
    data["sr"].update(population=10, generations=3)
    data["experiments"]["repetitions"] = 1
    data["experiments"]["max_cell_seconds"] = None
    data["experiments"]["algorithms"] = FAKE_SR  # sem Julia: PySR fora da fixture
    data["output_dir"] = str(tmp_path / "res")
    return parse_config(data)


def test_stages_analysis_and_legacy_export_end_to_end_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    cfg = _fake_cfg(tmp_path)
    run_dir = tmp_path / "run"
    for stage, exps in colab.HOJE_STAGES.items():
        _, summary = colab.run_notebook_stage(cfg, list(exps), run_dir)
        assert summary is not None and summary.failed == 0, stage
    results = colab.load_results(run_dir)
    assert set(results["status"]) == {"ok"} and set(results["scenario"]) == {"free_fall"}
    assert set(results["prompt"]) == {"none", "A", "C", "E", "H"}
    analysis = run_analysis([run_dir], tmp_path / "analysis")
    assert (analysis / "INDEX.md").is_file() and list(analysis.rglob("*.png"))
    legacy = export_legacy([run_dir], tmp_path / "legado")
    assert legacy.out_dir.is_dir() and any(legacy.out_dir.iterdir())
    # retomada: nada a recalcular
    _, again = colab.run_notebook_stage(cfg, ["exp1", "exp2"], run_dir)
    assert again is not None and again.executed == 0


# ------------------------------------------------------------------ presets e config derivada

from physics_llm_sr.config.loader import ConfigError  # noqa: E402

#: contagem de células da matriz completa por preset (build_matrix) e total de horas de LLM
#: (hipóteses 0,3 s/chamada em lote, 2,5 s em série).
PRESET_CELLS = {
    "hoje": 117,
    "hoje_literal": 18,
    "3_modelos_queda_livre": 189,
    "completo": 18675,
    "completo_extended": 18675,
}


@pytest.mark.parametrize(("preset", "cells"), list(PRESET_CELLS.items()))
def test_presets_resolve_and_count_cells(preset: str, cells: int, tmp_path: Path) -> None:
    path = colab.build_run_config(preset, out_dir=tmp_path)
    assert path.parent == tmp_path / "configs_derivadas"
    cfg = load_config(path)
    assert len(build_matrix(cfg)) == cells
    assert {m.name for m in cfg.llm.models} == {"mistral", "llama2", "falcon"}


def test_preset_content(tmp_path: Path) -> None:
    three = load_config(colab.build_run_config("3_modelos_queda_livre", out_dir=tmp_path))
    assert three.experiments.models == ["mistral", "llama2", "falcon"]
    full = load_config(colab.build_run_config("completo", out_dir=tmp_path))
    assert full.experiments.repetitions == 25 and full.track is Track.PAPER_REPLICATION
    assert len(full.experiments.scenarios) == 3 and len(full.llm.prompt_variants) == 8
    ext = load_config(colab.build_run_config("completo_extended", out_dir=tmp_path))
    assert ext.track is Track.EXTENDED_ANALYSIS and ext.loss.normalization == "normalized"
    assert colab.preset_profile("hoje") == ("colab_hoje", "extended")
    with pytest.raises(ValueError, match="PRESET"):
        colab.preset_profile("nada")


def test_build_run_config_each_override(tmp_path: Path) -> None:
    def build(**kw: object) -> object:
        return load_config(colab.build_run_config("hoje", out_dir=tmp_path, **kw))  # type: ignore[arg-type]

    assert build(models=["mistral", "falcon"]).experiments.models == ["mistral", "falcon"]  # type: ignore[attr-defined]
    assert build(scenarios=["harmonic_motion"]).experiments.scenarios == ["harmonic_motion"]  # type: ignore[attr-defined]
    assert build(experiments=["exp1", "exp2"]).experiments.run == ["baseline", "llm_integration"]  # type: ignore[attr-defined]
    assert build(algorithms=["deap"]).experiments.algorithms == ["deap"]  # type: ignore[attr-defined]
    assert build(repetitions=25).experiments.repetitions == 25  # type: ignore[attr-defined]
    assert build(prompts=list("ABCDEFGH")).llm.prompt_variants == list("ABCDEFGH")  # type: ignore[attr-defined]
    nl = build(noise_levels=[0.02])
    assert nl.data.exp4.levels == [0.02] == nl.data.exp5.levels  # type: ignore[attr-defined]
    rep = build(arm="replication")
    assert rep.track is Track.PAPER_REPLICATION and rep.loss.normalization == "literal"  # type: ignore[attr-defined]
    only_falcon = build(models=["falcon"])
    assert only_falcon.experiments.noise_model == "falcon"  # type: ignore[attr-defined]
    # sobrescrita separa o diretório de resultados; sem sobrescrita vale o do base
    assert build(repetitions=25).output_dir.startswith("results/colab_hoje_queda_livre_")  # type: ignore[attr-defined]
    assert build().output_dir == "results/colab_hoje_queda_livre"  # type: ignore[attr-defined]


def test_build_run_config_rejects_invalid_with_clear_message(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="models"):
        colab.build_run_config("hoje", models=["gpt5"], out_dir=tmp_path)
    with pytest.raises(ConfigError, match="repetitions"):
        colab.build_run_config("hoje", repetitions=0, out_dir=tmp_path)
    with pytest.raises(ValueError, match="exp9|desconhecido"):
        colab.build_run_config("hoje", experiments=["exp9"], out_dir=tmp_path)
    with pytest.raises(ValueError, match="arm"):
        colab.build_run_config("hoje", arm="x", out_dir=tmp_path)


def test_build_run_config_hash_is_deterministic(tmp_path: Path) -> None:
    a = colab.build_run_config("hoje", repetitions=5, out_dir=tmp_path / "a")
    b = colab.build_run_config("hoje", repetitions=5, out_dir=tmp_path / "b")
    c = colab.build_run_config("hoje", repetitions=6, out_dir=tmp_path / "c")
    assert a.name == b.name != c.name
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_plan_table_and_estimate_for_three_models(tmp_path: Path) -> None:
    cfg = load_config(colab.build_run_config("3_modelos_queda_livre", out_dir=tmp_path))
    plan = colab.plan_table(cfg).set_index("experimento")
    assert plan.loc["llm_integration", "modelo:llama2"] == 9
    assert plan.loc["baseline", "modelo:none"] == 9
    est = colab.estimate_runtime(cfg, 0.3, 2.5)
    assert list(est["cells"]) == [36, 81, 36, 36]
    only = load_config(colab.build_run_config("hoje", experiments=["exp3"], out_dir=tmp_path))
    assert list(colab.estimate_runtime(only, 0.3, 2.5)["stage"]) == ["P2"]
    assert colab.stage_experiments(only, "P1") == []


def test_ensure_hf_access_requires_token_only_for_llama2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(colab, "login_huggingface", lambda: False)
    ok = load_config(colab.build_run_config("hoje", out_dir=tmp_path))
    assert colab.ensure_hf_access(ok) == []
    falcon = load_config(colab.build_run_config("hoje", models=["falcon"], out_dir=tmp_path))
    assert colab.ensure_hf_access(falcon) == []
    llama = load_config(
        colab.build_run_config("hoje", models=["mistral", "llama2"], out_dir=tmp_path)
    )
    with pytest.raises(RuntimeError, match="HF_TOKEN.*meta-llama/Llama-2-7b-chat-hf"):
        colab.ensure_hf_access(llama)
    monkeypatch.setattr(colab, "login_huggingface", lambda: True)
    assert colab.ensure_hf_access(llama) == ["llama2"]
