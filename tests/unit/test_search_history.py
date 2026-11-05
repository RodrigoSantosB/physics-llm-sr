"""Histórico de busca: DataFrame compatível com o plotter e serialização JSON."""

from __future__ import annotations

import math

import pandas as pd

from physics_llm_sr.evaluation.plots import plot_loss_per_generation
from physics_llm_sr.optimization.search_history import SearchHistory
from physics_llm_sr.sr.interfaces import GenerationRecord


def rec(g: int, loss: float) -> GenerationRecord:
    return GenerationRecord(g, loss, loss / 2, 3.0, 0.25, "x + 1", 4, 1, 0.5)


def test_dataframe_columns_and_rows() -> None:
    h = SearchHistory("deap|mistral|E")
    h.add(rec(0, 2.0))
    h.add(rec(1, 1.0))
    df = h.to_dataframe()
    assert list(df["generation"]) == [0, 1]
    assert set(df["config"]) == {"deap|mistral|E"}
    for col in ("best_loss", "best_e", "best_s", "best_c", "n_llm_calls", "n_cache_hits"):
        assert col in df.columns
    assert h.best_loss == 1.0 and len(h) == 2


def test_empty_dataframe_has_columns() -> None:
    df = SearchHistory("c").to_dataframe()
    assert df.empty and "best_loss" in df.columns


def test_json_roundtrip_with_nonfinite(tmp_path) -> None:  # type: ignore[no-untyped-def]
    h = SearchHistory("c", metadata={"seed": 1})
    h.add(rec(0, math.inf))
    h.add(rec(1, 1.5))
    path = h.save(tmp_path / "h.json")
    back = SearchHistory.load(path)
    assert back.config == "c" and back.metadata == {"seed": 1}
    assert back.records == h.records


def test_compatible_with_plotter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames = []
    for name in ("a", "b"):
        h = SearchHistory(name)
        h.add(rec(0, 2.0))
        h.add(rec(1, 1.0))
        frames.append(h.to_dataframe())
    out = plot_loss_per_generation(pd.concat(frames), tmp_path / "l.png")
    assert out.exists()
