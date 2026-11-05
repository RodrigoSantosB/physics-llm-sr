"""Configs derivadas dos notebooks 01 (baseline sem LLM), 02 (um LLM, um caso) e 03 (pipeline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_llm_sr import colab
from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.experiments.matrix import build_matrix

REPO = Path(__file__).resolve().parents[2]
NB01_DEFAULTS = {
    "scenarios": ["free_fall", "harmonic_motion", "damped_wave"],
    "algorithms": ["deap", "gplearn", "pysr"],
    "repetitions": 3,
    "noise_levels": [0.01, 0.03, 0.05],
    "noise_locations": ["target"],
    "braco": "literal",
}


def _nb01(tmp_path: Path, **over: object) -> Path:
    return colab.build_baseline_config(out_dir=tmp_path, **{**NB01_DEFAULTS, **over})  # type: ignore[arg-type]


def _nb02(tmp_path: Path, **over: object) -> Path:
    args: dict[str, object] = {
        "scenario": "free_fall",
        "experiment": "benchmark",
        "llm": "mistral",
        "algorithms": ["deap", "gplearn", "pysr"],
        "repetitions": 3,
        "prompt": "E",
        "prompts": ["A", "C", "E", "H"],
        "braco": "normalized",
        "noise_levels": [0.01, 0.05],
        "noise_locations": ["target"],
    }
    args.update(over)
    return colab.build_single_llm_config(out_dir=tmp_path, **args)  # type: ignore[arg-type]


def test_nb01_config_has_no_llm_and_only_baseline_cells(tmp_path: Path) -> None:
    cfg = load_config(_nb01(tmp_path))
    assert cfg.experiments.baseline_only and cfg.llm.backend == "fake"
    assert cfg.track is Track.PAPER_REPLICATION and cfg.loss.normalization == "literal"
    assert cfg.output_dir.startswith("results/nb01_baseline/")
    cells = [p.cell for p in build_matrix(cfg)]
    assert cells and all(c.is_baseline for c in cells)
    # 3 cenários x 3 SR x 3 rep x (Exp1 + Exp4 níveis 0.03/0.05 + Exp5 níveis 0.01/0.03/0.05)
    assert len(cells) == 27 * (1 + 2 + 3)
    assert load_config(_nb01(tmp_path, braco="normalized")).track is Track.EXTENDED_ANALYSIS


def test_nb01_validation_is_clear(tmp_path: Path) -> None:
    with pytest.raises(colab.NotebookParameterError, match="CENARIOS"):
        _nb01(tmp_path, scenarios=["lua"])
    with pytest.raises(colab.NotebookParameterError, match="ALGORITMOS"):
        _nb01(tmp_path, algorithms=["gp"])
    with pytest.raises(colab.NotebookParameterError, match="REPETICOES"):
        _nb01(tmp_path, repetitions=0)
    with pytest.raises(colab.NotebookParameterError, match="NIVEIS_RUIDO"):
        _nb01(tmp_path, noise_levels=[-0.1])
    with pytest.raises(colab.NotebookParameterError, match="LOCALIZACOES"):
        _nb01(tmp_path, noise_locations=["lado"])
    with pytest.raises(colab.NotebookParameterError, match="BRACO"):
        _nb01(tmp_path, braco="x")


@pytest.mark.parametrize(
    ("experiment", "run", "cells"),
    [
        ("benchmark", ["llm_integration"], 2 * 3 * 2),  # (LLM + baseline) x 2 SR x 3 rep
        ("prompts", ["prompt_sensitivity"], 3 * 2 * (1 + 4)),  # baseline + 4 prompts
        (
            "ruido",
            ["noise_robustness"],
            2 * 2 * 3 * 2,
        ),  # 2 níveis x 2 SR x 3 rep x (LLM + baseline)
        ("ruido_enviesado", ["biased_noise"], 3 * 2 * 2 * 2),
    ],
)
def test_nb02_experiments_map_to_matrix(
    tmp_path: Path, experiment: str, run: list[str], cells: int
) -> None:
    cfg = load_config(_nb02(tmp_path, experiment=experiment, algorithms=["deap", "gplearn"]))
    assert cfg.experiments.run == run
    assert cfg.experiments.scenarios == ["free_fall"] and cfg.experiments.models == ["mistral"]
    assert cfg.experiments.noise_model == "mistral" and cfg.experiments.noise_prompt == "E"
    assert cfg.output_dir.startswith(f"results/nb02_free_fall_{experiment}_mistral/")
    assert cfg.track is Track.EXTENDED_ANALYSIS and cfg.loss.normalization == "normalized"
    assert len(build_matrix(cfg)) == cells
    base = build_matrix(cfg, arm="baseline")
    assert base and all(p.cell.is_baseline for p in base)


def test_nb02_prompt_and_braco_and_llm(tmp_path: Path) -> None:
    cfg = load_config(_nb02(tmp_path, prompt="A", llm="falcon", scenario="damped_wave"))
    assert cfg.experiments.benchmark_prompt == "A" and cfg.experiments.models == ["falcon"]
    assert cfg.experiments.scenarios == ["damped_wave"]
    lit = load_config(_nb02(tmp_path, braco="literal"))
    assert lit.track is Track.PAPER_REPLICATION and lit.loss.normalization == "literal"
    assert lit.llm.evaluation_scope == "all"
    assert lit.llm.models[0].revision  # revisões fixadas (GAP-07)
    assert lit.llm.batch_size == 16 and lit.llm.backend == "transformers"


def test_nb02_validation_is_clear(tmp_path: Path) -> None:
    with pytest.raises(colab.NotebookParameterError, match="CENARIO"):
        _nb02(tmp_path, scenario="lua")
    with pytest.raises(colab.NotebookParameterError, match="CENARIO"):
        _nb02(tmp_path, scenario=["free_fall", "damped_wave"])
    with pytest.raises(colab.NotebookParameterError, match="EXPERIMENTO"):
        _nb02(tmp_path, experiment="exp9")
    with pytest.raises(colab.NotebookParameterError, match="LLM"):
        _nb02(tmp_path, llm="gpt5")
    with pytest.raises(colab.NotebookParameterError, match="PROMPT"):
        _nb02(tmp_path, prompt="Z")
    with pytest.raises(colab.NotebookParameterError, match="PROMPTS"):
        _nb02(tmp_path, experiment="prompts", prompts=["A", "Z"])
    with pytest.raises(colab.NotebookParameterError, match="BRACO"):
        _nb02(tmp_path, braco="x")


def test_nb02_llama2_requires_hf_token_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(colab, "login_huggingface", lambda: False)
    cfg = load_config(_nb02(tmp_path, llm="llama2"))
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        colab.ensure_hf_access(cfg)


def test_smoke_shrinks_everything_and_uses_fake_backend(tmp_path: Path) -> None:
    cfg = load_config(_nb02(tmp_path, smoke=True))
    assert cfg.llm.backend == "fake" and cfg.mode is Mode.SMOKE
    assert "pysr" not in cfg.experiments.algorithms and cfg.experiments.repetitions == 2
    assert cfg.sr.generations <= 5 and cfg.data.n_simulations <= 10
    assert cfg.llm.cache_dir.endswith("_smoke") and cfg.output_dir.startswith("results/nb02_")
    one = load_config(_nb01(tmp_path, smoke=True))
    assert one.mode is Mode.SMOKE and "pysr" not in one.experiments.algorithms
    assert colab.ensure_hf_access(load_config(_nb02(tmp_path, llm="llama2", smoke=True))) == []


def test_smoke_flag_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(colab.NOTEBOOK_SMOKE_ENV, raising=False)
    assert colab.smoke_requested() is False
    monkeypatch.setenv(colab.NOTEBOOK_SMOKE_ENV, "1")
    assert colab.smoke_requested() is True


def test_derived_config_is_deterministic_and_in_notebook_subdir(tmp_path: Path) -> None:
    a, b = _nb01(tmp_path / "a"), _nb01(tmp_path / "b")
    assert a.name == b.name and a.parent == tmp_path / "a" / "configs_derivadas" / "nb01"
    assert _nb01(tmp_path / "c", repetitions=4).name != a.name
    assert _nb02(tmp_path).parent == tmp_path / "configs_derivadas" / "nb02"


def test_nb03_prefix_and_cache_do_not_collide(tmp_path: Path) -> None:
    path = colab.build_run_config(
        "hoje",
        out_dir=tmp_path,
        output_prefix="nb03_",
        derived_dir_name="configs_derivadas/nb03",
        patch={"llm": {"cache_dir": ".cache/llm_nb03"}},
    )
    cfg = load_config(path)
    assert cfg.output_dir == "results/nb03_colab_hoje_queda_livre"
    assert cfg.llm.cache_dir == ".cache/llm_nb03"
    assert path.parent == tmp_path / "configs_derivadas" / "nb03"
    smoke = load_config(colab.build_run_config("smoke", out_dir=tmp_path, output_prefix="nb03_"))
    assert smoke.output_dir == "results/nb03_notebook_smoke/replication"


def test_cpu_estimate_counts_cells_and_applies_cap(tmp_path: Path) -> None:
    cfg = load_config(_nb01(tmp_path))
    est = colab.estimate_cpu_runtime(cfg).set_index("sr")
    assert est["cells"].sum() == 27 * 6 and set(est.index) == {"deap", "gplearn", "pysr"}
    assert est.loc["pysr", "seconds"] == 54 * colab.DEFAULT_CPU_SECONDS_PER_CELL["pysr"]
    custom = colab.estimate_cpu_runtime(cfg, {"deap": 1.0}).set_index("sr")
    assert custom.loc["deap", "seconds"] == 54.0
    capped = colab.estimate_cpu_runtime(cfg, {"pysr": 1e9}).set_index("sr")
    assert capped.loc["pysr", "seconds"] == 54 * cfg.experiments.max_cell_seconds  # type: ignore[operator]


def test_environment_report_can_skip_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    sys.modules.pop("torch", None)
    report = colab.environment_report(include_cuda=False)
    assert report["cuda_disponível"] is None and "torch" not in sys.modules


def test_ensure_project_root_defaults_to_repo_and_respects_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    assert colab.ensure_project_root() == REPO.resolve()
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    assert colab.ensure_project_root() == tmp_path.resolve()


def test_plan_table_of_baseline_only_has_no_llm_column(tmp_path: Path) -> None:
    plan = colab.plan_table(load_config(_nb01(tmp_path)))
    assert list(plan.columns) == ["experimento", "modelo:none"]
    assert int(plan["modelo:none"].sum()) > 0


def test_require_gpu_blocks_transformers_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(_nb02(tmp_path))
    monkeypatch.setattr(
        colab, "environment_report", lambda include_cuda=True: {"cuda_disponível": False}
    )
    with pytest.raises(RuntimeError, match="A100"):
        colab.require_gpu(cfg)
    monkeypatch.setattr(
        colab, "environment_report", lambda include_cuda=True: {"cuda_disponível": True}
    )
    colab.require_gpu(cfg)  # com GPU não levanta
    smoke = load_config(_nb02(tmp_path, smoke=True))
    monkeypatch.setattr(
        colab, "environment_report", lambda include_cuda=True: {"cuda_disponível": False}
    )
    colab.require_gpu(smoke)  # backend fake dispensa GPU
