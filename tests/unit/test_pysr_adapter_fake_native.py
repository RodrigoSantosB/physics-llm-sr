"""Adaptador PySR com o PySR/Julia SIMULADOS (sem Julia): laço externo, sementes, auditoria.

Um módulo ``pysr`` falso substitui o pacote real e um ``jl`` falso substitui o
``juliacall.Main``. O ``fit`` nativo falso chama o callback registrado com o texto da
árvore (como a perda Julia faria) e devolve um hall of fame com a perda calculada como
no Julia: ``e`` e ``s`` da árvore crua, ``c`` da forma simplificada (callback).

Cobre: falha ao subir Julia em ``fit`` (AdapterRuntimeError), ``simplify`` + memo no
registro por iteração e na auditoria (``cos(0.5)*mass``: 0 chamadas LLM novas), semente
por iteração no ``warm_start`` e checagem da versão do PySR.
"""

from __future__ import annotations

import math
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset, PhysicalContext
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.optimization.objective import CombinedObjective
from physics_llm_sr.reproducibility.seeds import derive_seed
from physics_llm_sr.sr import pysr_adapter
from physics_llm_sr.sr.interfaces import AdapterRuntimeError
from physics_llm_sr.symbolic.expression import evaluate, size

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
FEATURES = ("mass", "length", "height", "damping", "t")
CTX = PhysicalContext("free_fall", {f: f for f in FEATURES}, "exp", "mass", {})
#: Árvore com constante dobrável: o callback avalia ``0.8775825618903728*mass``.
JULIA_TREE = "cos(0.5) * x1"
GOOD = EquationEvaluation(0.9, 0.8, 0.7, "ok")
W = (1 / 3, 1 / 3, 1 / 3)


def _train() -> Dataset:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 2.0, size=(40, len(FEATURES)))  # noqa: N806
    y = 0.9 * X[:, 0]
    return Dataset(X, y, y, FEATURES, "v", "free_fall", np.arange(40), X[:, 4], {})


def _cfg(niterations: int = 3) -> ExperimentConfig:
    cfg = load_config(SMOKE)
    return cfg.model_copy(
        update={
            "sr": cfg.sr.model_copy(
                update={
                    "generations": niterations,
                    "early_stopping": cfg.sr.early_stopping.model_copy(update={"enabled": False}),
                }
            )
        }
    )


class FakeJl:
    """``juliacall.Main`` simulado: guarda callback e parâmetros."""

    def __init__(self) -> None:
        self.callback: Any = None
        self.params: tuple[Any, ...] = ()

    def plsr_set_callback_b(self, cb: Any) -> None:
        self.callback = cb

    def plsr_clear_callback_b(self) -> None:
        self.callback = None

    def plsr_set_params_b(self, *params: Any) -> None:
        self.params = params

    def seval(self, code: str) -> int:
        return 0


def _fake_pysr(jl: FakeJl, seeds: list[int], version: str = "2.7.2") -> types.ModuleType:
    """Módulo ``pysr`` falso cujo regressor imita a perda Julia (``plsr_loss``)."""

    class NativeRegressor:
        def __init__(self, tempdir: str, **params: Any) -> None:
            self.params = params
            self.random_state = params["random_state"]
            self.equations_: pd.DataFrame | None = None

        def set_params(self, **kw: Any) -> NativeRegressor:
            for k, v in kw.items():
                setattr(self, k, v)
                self.params[k] = v
            return self

        def fit(self, X: Any, y: Any, variable_names: list[str]) -> None:  # noqa: N803
            seeds.append(int(self.random_state))
            w1, w2, w3, _huber, delta, _norm, _var, _mn = jl.params
            raw = pysr_adapter.julia_text_to_expr(JULIA_TREE, FEATURES)
            pred = evaluate(raw, X, FEATURES)
            r = np.abs(pred - y)
            e = float(np.mean(np.where(r <= delta, 0.5 * r * r, delta * (r - 0.5 * delta))))
            c = float(jl.callback(JULIA_TREE)) if jl.callback is not None else 0.0
            loss = w1 * e + w2 * size(raw) + (w3 * c if w3 > 0 else 0.0)
            self.equations_ = pd.DataFrame(
                {"equation": [JULIA_TREE], "loss": [loss], "sympy_format": [None]}
            )

    module = types.ModuleType("pysr")
    module.__version__ = version  # type: ignore[attr-defined]
    module.PySRRegressor = NativeRegressor  # type: ignore[attr-defined]
    return module


@pytest.fixture()
def fake_native(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeJl, list[int]]:
    jl = FakeJl()
    seeds: list[int] = []
    monkeypatch.setattr(pysr_adapter, "_julia", lambda: jl)
    monkeypatch.setitem(__import__("sys").modules, "pysr", _fake_pysr(jl, seeds))
    return jl, seeds


def _objective(fake: FakeEquationEvaluator) -> CombinedObjective:
    return CombinedObjective(W, fake, CTX, FEATURES, error_metric="huber")


# ------------------------------------------------------------------ item 1: Julia indisponível


def test_julia_start_failure_in_fit_is_adapter_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> Any:
        raise ImportError("juliacall: falha simulada ao iniciar o Julia")

    monkeypatch.setattr(pysr_adapter, "_julia", boom)
    fake = FakeEquationEvaluator.from_table({}, default=GOOD)
    with pytest.raises(AdapterRuntimeError, match="Julia"):
        pysr_adapter.PySRRegressor().fit(_train(), _objective(fake), _cfg())


# ------------------------------------------------------- item 3: simplify + memo no registro


def test_record_and_audit_use_simplified_form_and_callback_memo(
    fake_native: tuple[FakeJl, list[int]],
) -> None:
    fake = FakeEquationEvaluator.from_table({}, default=GOOD)
    reg = pysr_adapter.PySRRegressor()
    result = reg.fit(_train(), _objective(fake), _cfg(niterations=2))
    # Uma única forma (simplificada) avaliada: registro e auditoria não chamam o LLM.
    assert fake.calls == 1
    assert reg.callback is not None and len(reg.callback.memo) == 1
    expected_c = 1 - (0.9 + 0.8 + 0.7) / 3
    assert all(math.isclose(r.best_c, expected_c) for r in result.history)
    # s do registro = nós da árvore crua (como o count_nodes do Julia)
    assert all(r.best_s == 4 for r in result.history)
    audit = reg.audit
    assert audit["ok"], audit
    assert audit["rows"][0]["c"] == pytest.approx(expected_c)


def test_audit_failure_is_flagged_in_metadata(
    fake_native: tuple[FakeJl, list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pysr_adapter, "AUDIT_REL_TOL", -1.0)  # força divergência
    fake = FakeEquationEvaluator.from_table({}, default=GOOD)
    result = pysr_adapter.PySRRegressor().fit(_train(), _objective(fake), _cfg(niterations=1))
    extra = result.metadata.extra or {}
    assert extra["audit"]["ok"] is False
    assert extra["audit_failed"] is True
    assert "auditoria" in extra["audit_reason"]


# ------------------------------------------------------- item 4: semente por iteração


def test_warm_start_iterations_use_derived_seeds(fake_native: tuple[FakeJl, list[int]]) -> None:
    _, seeds = fake_native
    fake = FakeEquationEvaluator.from_table({}, default=GOOD)
    cfg = _cfg(niterations=3)
    pysr_adapter.PySRRegressor().fit(_train(), _objective(fake), cfg)
    base = derive_seed(cfg.seeds.master, pysr_adapter.SEED_NAME)
    assert seeds == [derive_seed(base, f"it{i}") for i in range(3)]
    assert len(set(seeds)) == 3
    again: list[int] = []
    jl = FakeJl()
    pysr_adapter._julia = lambda: jl  # type: ignore[assignment]  # restaurado pelo monkeypatch
    __import__("sys").modules["pysr"] = _fake_pysr(jl, again)
    pysr_adapter.PySRRegressor().fit(_train(), _objective(fake), cfg)
    assert again == seeds  # determinístico


# ------------------------------------------------------- item 7: versão do PySR


def test_unsupported_pysr_version_is_refused(
    fake_native: tuple[FakeJl, list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    jl, seeds = fake_native
    monkeypatch.setitem(__import__("sys").modules, "pysr", _fake_pysr(jl, seeds, "2.8.0"))
    fake = FakeEquationEvaluator.from_table({}, default=GOOD)
    with pytest.raises(pysr_adapter.UnsupportedPysrError, match=r"2\.8\.0"):
        pysr_adapter.PySRRegressor().fit(_train(), _objective(fake), _cfg())
