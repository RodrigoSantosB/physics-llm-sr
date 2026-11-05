"""Backends no runner: no máximo um modelo LLM residente (matriz ordenada por LLM, backend
anterior descarregado ao trocar) e manifesto por célula com o backend efetivo.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from physics_llm_sr.config.loader import load_config, parse_config
from physics_llm_sr.config.paths import PROJECT_ROOT_ENV
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.experiments.runner import build_matrix, run_experiments
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.symbolic.expression import Expr

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
OK = EquationEvaluation(0.5, 0.5, 0.5, "ok")


@pytest.fixture(autouse=True)
def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))


class Residency:
    """Conta modelos vivos (carregados e não descarregados)."""

    def __init__(self) -> None:
        self.live = 0
        self.max_live = 0
        self.loads: list[str] = []


class StubBackend:
    """Backend que "carrega" o modelo no primeiro uso e o libera em ``close``."""

    def __init__(self, name: str, residency: Residency) -> None:
        self.name = name
        self.residency = residency
        self.loaded = False

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.loaded = True
            self.residency.live += 1
            self.residency.loads.append(self.name)
            self.residency.max_live = max(self.residency.max_live, self.residency.live)

    def close(self) -> None:
        if self.loaded:
            self.loaded = False
            self.residency.live -= 1

    def model_metadata(self) -> dict[str, Any]:
        return {"backend": "stub", "checkpoint": self.name, "prompt_format": "plain"}


class StubEvaluator:
    def __init__(self, backend: StubBackend) -> None:
        self.backend = backend

    def evaluate_many(
        self, expressions: Sequence[Expr], ctx: PhysicalContext, variant: str
    ) -> list[EquationEvaluation]:
        self.backend.ensure_loaded()
        return [OK for _ in expressions]


def _two_models() -> ExperimentConfig:
    data = load_config(SMOKE).model_dump(mode="json")
    first = data["llm"]["models"][0]
    data["llm"]["models"] = [first, {**first, "name": "fake2", "checkpoint": "fake-2"}]
    data["experiments"]["models"] = ["fake", "fake2"]
    return parse_config(data)


def test_matrix_is_grouped_by_llm() -> None:
    llms = [p.cell.llm for p in build_matrix(_two_models())]
    changes = sum(1 for a, b in zip(llms, llms[1:], strict=False) if a != b)
    assert changes == len(set(llms)) - 1  # cada LLM aparece num bloco contíguo
    assert llms[0] == "none"  # baselines pareados primeiro (sem LLM)


def test_at_most_one_model_resident(tmp_path: Path) -> None:
    residency = Residency()
    summary = run_experiments(
        _two_models(),
        run_dir=tmp_path / "run",
        evaluator_factory=lambda m: StubEvaluator(StubBackend(m, residency)),
    )
    assert summary.failed == 0
    assert residency.max_live <= 1
    assert sorted(residency.loads) == ["fake", "fake2"]  # cada modelo carregado uma vez
    assert residency.live == 0  # tudo descarregado ao final


def test_cell_manifest_records_effective_backend(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run_experiments(load_config(SMOKE), run_dir=run, algorithms=["deap"])
    manifests = {
        p.parent.name: json.loads(p.read_text(encoding="utf-8"))
        for p in (run / "cells").glob("*/manifest.json")
    }
    llm_cells = [m for k, m in manifests.items() if "__fake__" in k]
    assert llm_cells
    for manifest in llm_cells:
        (model,) = manifest["llm_models"]
        assert model["effective"]["backend"] == "fake"
        assert model["effective"]["checkpoint"] == "fake-evaluator"
