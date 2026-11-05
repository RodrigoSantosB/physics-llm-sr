"""Plots: apenas verifica criação dos arquivos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physics_llm_sr.evaluation import plots


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sr in ("deap", "pysr"):
        for llm in ("mistral_7b", "none"):
            for prompt in "ABCDEFGH":
                for loc in ("inputs", "target"):
                    for lvl in (0.01, 0.02, 0.03):
                        for rep in range(3):
                            rows.append(dict(sr=sr, llm=llm, prompt=prompt, noise_location=loc,
                                             noise_level=lvl, repetition=rep, mae=rng.random(),
                                             tree_distance=rng.random()))  # fmt: skip
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


def test_plots_create_files(tmp_path: Path) -> None:
    df = _df()
    hist = pd.DataFrame(
        {"config": ["a"] * 4 + ["b"] * 4, "generation": [0, 1, 2, 3] * 2, "best_loss": range(8)}
    )
    outs = [
        plots.plot_loss_per_generation(hist, tmp_path / "loss.png"),
        plots.plot_metric_bars(df, "mae", tmp_path / "bars.png"),
        plots.plot_prompt_heatmap(df, "tree_distance", tmp_path / "heat.png"),
        plots.plot_noise_curves(df, "tree_distance", tmp_path / "sub" / "noise.png"),
    ]
    for o in outs:
        assert o.exists() and o.stat().st_size > 0
