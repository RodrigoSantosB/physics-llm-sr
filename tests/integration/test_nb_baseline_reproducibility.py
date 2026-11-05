"""Com as mesmas sementes, o baseline do notebook 02 coincide com o do notebook 01 (mesmo caso)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from physics_llm_sr import colab
from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV

COMPARED = ["expression", "mae", "mse", "one_minus_r2", "tree_distance", "data_hash", "seed"]


@pytest.mark.parametrize("braco", ["literal", "normalized"])
def test_nb02_baseline_equals_nb01_baseline_for_same_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, braco: str
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    nb01 = load_config(
        colab.build_baseline_config(
            scenarios=["free_fall"],
            algorithms=["deap", "gplearn"],
            repetitions=2,
            noise_levels=[0.01],
            noise_locations=["target"],
            braco=braco,
            out_dir=tmp_path,
            smoke=True,
        )
    )
    nb02 = load_config(
        colab.build_single_llm_config(
            scenario="free_fall",
            experiment="benchmark",
            llm="mistral",
            algorithms=["deap", "gplearn"],
            repetitions=2,
            prompt="E",
            prompts=["E"],
            braco=braco,
            noise_levels=[0.01],
            noise_locations=["target"],
            out_dir=tmp_path,
            smoke=True,
        )
    )
    dir01, _ = colab.run_notebook_stage(nb01, ["exp1"], colab.notebook_run_dir(nb01))
    dir02, _ = colab.run_notebook_stage(
        nb02, ["exp2"], colab.notebook_run_dir(nb02), arm="baseline"
    )
    assert dir01 != dir02
    r1 = colab.load_results(dir01).sort_values("cell_key").reset_index(drop=True)
    r2 = colab.load_results(dir02).sort_values("cell_key").reset_index(drop=True)
    assert list(r1["cell_key"]) == list(r2["cell_key"]) and len(r1) == 4
    pd.testing.assert_frame_equal(r1[COMPARED], r2[COMPARED])
