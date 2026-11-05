"""Execução de UMA célula: dados → objetivo → busca SR → métricas → linha/histórico/manifesto.

Falhas esperadas (LLM, adaptador indisponível, tempo excedido, predição não finita) viram
uma linha ``status='error'`` com o motivo — nunca nota substituta nem silêncio. Erros de
programação propagam.

Auditoria de paridade reprovada (PySR: ``metadata.extra["audit_failed"]``) nunca é
descartada: a linha recebe ``audit_failed=True`` e ``audit_reason``; em
``full`` + ``paper_replication`` a célula vira ``error`` (:class:`AuditFailedError`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.config.schemas import ExperimentConfig, SeedsConfig
from physics_llm_sr.core_types import Dataset, Mode, Track
from physics_llm_sr.evaluation.metrics import MetricInputError, regression_metrics
from physics_llm_sr.experiments.cells import NONE, Cell
from physics_llm_sr.experiments.data_provider import (
    DataProvider,
    data_seed_name,
    noise_seed_name,
    search_seed,
    split_seed_name,
)
from physics_llm_sr.experiments.sr_registry import AdapterUnavailableError, make_regressor
from physics_llm_sr.experiments.storage import STATUS_ERROR, STATUS_OK
from physics_llm_sr.experiments.structural import structural_row
from physics_llm_sr.llm.interfaces import (
    BackendError,
    EvaluationFailedError,
    PhysicsEquationEvaluator,
)
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.optimization.objective import DEFAULT_PROMPT_VARIANT, build_objective
from physics_llm_sr.optimization.search_history import SearchHistory
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.manifests import RunManifest
from physics_llm_sr.reproducibility.seeds import derive_seed
from physics_llm_sr.sr.interfaces import (
    AdapterRuntimeError,
    FitResult,
    FloatArray,
    LossBreakdown,
)
from physics_llm_sr.symbolic.expression import Expr, depth, parse_infix, size, to_infix

PARTITION_PREFIXES = {"train": "train", "validation": "validation", "test": "test"}
REGRESSION_KEYS = ("mae", "mse", "one_minus_r2")


class CellTimeLimitError(RuntimeError):
    """Tempo de parede da célula excedeu ``experiments.max_cell_seconds``."""


class AuditFailedError(RuntimeError):
    """Auditoria de paridade do adaptador reprovada em full + paper_replication."""


#: Falhas registradas como célula ``error`` (as demais exceções propagam).
CELL_FAILURES: tuple[type[BaseException], ...] = (
    EvaluationFailedError,
    BackendError,
    AdapterUnavailableError,
    AdapterRuntimeError,
    AuditFailedError,
    CellTimeLimitError,
    MetricInputError,
)


def is_strict(cfg: ExperimentConfig) -> bool:
    """``full`` + ``paper_replication``: falha de célula aborta; auditoria reprovada é erro."""
    return cfg.mode is Mode.FULL and cfg.track is Track.PAPER_REPLICATION


class TimeLimitedObjective:
    """Proxy do objetivo que interrompe a busca ao exceder o limite de tempo.

    Verifica o relógio antes de cada lote (geração); demais atributos (``counters``,
    ``weights``, ``metadata``...) são delegados ao objetivo original.
    """

    def __init__(
        self, inner: Any, limit_seconds: float, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        """Começa a contar agora."""
        self._inner = inner
        self._limit = limit_seconds
        self._clock = clock
        self._start = clock()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Delegado."""
        return tuple(self._inner.feature_names)

    def evaluate_population(
        self,
        exprs: Sequence[Expr],
        X: FloatArray,  # noqa: N803 - nome do contrato (PRD)
        y: FloatArray,
    ) -> Sequence[LossBreakdown]:
        """Delegado, após conferir o tempo."""
        elapsed = self._clock() - self._start
        if elapsed > self._limit:
            raise CellTimeLimitError(
                f"limite de {self._limit:g} s por célula excedido ({elapsed:.1f} s)"
            )
        result: Sequence[LossBreakdown] = self._inner.evaluate_population(exprs, X, y)
        return result

    def __getattr__(self, name: str) -> Any:
        """Demais atributos do objetivo original."""
        return getattr(self._inner, name)


@dataclass
class CellContext:
    """Recursos compartilhados pelas células de uma execução."""

    cfg: ExperimentConfig
    run_id: str
    data: DataProvider
    evaluator_for: Callable[[str], PhysicsEquationEvaluator]
    prompts: Callable[[], PromptLibrary]
    base_manifest: Callable[[], RunManifest]


@dataclass
class CellOutcome:
    """Linha de resultado, histórico e manifesto de uma célula."""

    row: dict[str, Any]
    history: pd.DataFrame | None
    manifest: dict[str, Any] | None
    exception: BaseException | None = None

    @property
    def ok(self) -> bool:
        """Célula concluída com sucesso."""
        return bool(self.row["status"] == STATUS_OK)


def identity_columns(cfg: ExperimentConfig, cell: Cell, run_id: str) -> dict[str, Any]:
    """Colunas que identificam a célula (exigidas por ``evaluation``)."""
    return {
        "run_id": run_id,
        "cell_key": cell.key(),
        "scenario": cell.scenario,
        "sr": cell.sr,
        "llm": cell.llm,
        "prompt": cell.prompt,
        "noise_type": cell.noise.noise_type,
        "noise_location": cell.noise.noise_location,
        "noise_level": cell.noise.noise_level,
        "repetition": cell.repetition,
        "seed": search_seed(cfg, cell),
        "mode": cfg.mode.value,
        "track": cfg.track.value,
        "normalization": cfg.loss.normalization,
    }


def _metrics(prefix: str, y: FloatArray, yhat: FloatArray) -> dict[str, float]:
    m = regression_metrics(y, yhat)
    return {f"{prefix}_{k}": m[k] for k in (*REGRESSION_KEYS, "r2")}


def _counter(counters: Mapping[str, float], name: str) -> float:
    return float(counters.get(name, 0) or 0)


def _evaluator_metrics(evaluator: PhysicsEquationEvaluator | None) -> dict[str, float]:
    metrics = getattr(evaluator, "metrics", None)
    as_dict = getattr(metrics, "as_dict", None)
    return dict(as_dict()) if callable(as_dict) else {}


def run_cell(ctx: CellContext, cell: Cell) -> CellOutcome:
    """Executa a célula e devolve a linha de resultado (``ok`` ou ``error``)."""
    cfg = ctx.cfg
    start = time.perf_counter()
    row = identity_columns(cfg, cell, ctx.run_id)
    full, parts = ctx.data.for_cell(cell)
    train, validation, test = parts["train"], parts["validation"], parts["test"]
    row.update(
        data_hash=full.metadata["data_hash"],
        train_hash=train.metadata["data_hash"],
        n_train=int(train.y_observed.size),
        n_test=int(test.y_observed.size),
    )
    scenario = get_scenario(cell.scenario, cfg.physics)
    context = scenario.physical_context()
    gt = parse_infix(context.ground_truth)
    equivalents = [parse_infix(s) for s in scenario.equivalent_ground_truths()[1:]]
    row["ground_truth"] = context.ground_truth
    gt_disclosed = False
    prompt_hash: str | None = None
    if not cell.is_baseline:
        library = ctx.prompts()
        gt_disclosed = library.variant(cell.prompt).gt_disclosed
        prompt_hash = library.prompt_hash(cell.prompt)
    row.update(gt_disclosed=gt_disclosed, prompt_hash=prompt_hash)
    seed = int(row["seed"])
    # A semente mestre da célula é a semente de busca: os adaptadores derivam a sua dela.
    cell_cfg = cfg.model_copy(update={"seeds": SeedsConfig(master=seed)})
    evaluator: PhysicsEquationEvaluator | None = None
    before: dict[str, float] = {}
    row.update(audit_failed=False, audit_reason=None)
    try:
        if not cell.is_baseline:
            evaluator = ctx.evaluator_for(cell.llm)
            before = _evaluator_metrics(evaluator)
        objective: Any = build_objective(
            cfg,
            evaluator,
            context,
            cell.sr,
            baseline=cell.is_baseline,
            feature_names=train.feature_names,
            prompt_variant=DEFAULT_PROMPT_VARIANT if cell.is_baseline else cell.prompt,
        )
        if cfg.experiments.max_cell_seconds is not None:
            objective = TimeLimitedObjective(objective, cfg.experiments.max_cell_seconds)
        regressor = make_regressor(cell.sr)
        fit = regressor.fit(train, objective, cell_cfg, validation=validation)
        extra = fit.metadata.extra or {}
        if extra.get("audit_failed"):
            reason = str(extra.get("audit_reason") or "auditoria de paridade reprovada")
            row.update(audit_failed=True, audit_reason=reason)
            if is_strict(cfg):
                raise AuditFailedError(reason)
        metrics = _all_metrics(regressor.predict, train, validation, test)
    except CELL_FAILURES as exc:
        row.update(
            status=STATUS_ERROR,
            error=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            stop_reason="time_limit" if isinstance(exc, CellTimeLimitError) else "error",
            cell_seconds=time.perf_counter() - start,
        )
        manifest = _manifest(ctx, cell, row, full, parts, None, evaluator)
        return CellOutcome(row, None, manifest, exc)
    after = _evaluator_metrics(evaluator)
    row.update(metrics)
    row.update(mae=row["test_mae"], mse=row["test_mse"], one_minus_r2=row["test_one_minus_r2"])
    row.update(structural_row(fit.best_expression, gt, equivalents, cfg.symbolic))
    row.update(_fit_columns(fit, before, after))
    row.update(status=STATUS_OK, error=None, error_type=None)
    row["cell_seconds"] = time.perf_counter() - start
    history = _history_frame(fit, cell, row)
    return CellOutcome(row, history, _manifest(ctx, cell, row, full, parts, fit, evaluator))


def _all_metrics(
    predict: Callable[[FloatArray], FloatArray],
    train: Dataset,
    validation: Dataset,
    test: Dataset,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        yhat = np.asarray(predict(part.X), dtype=float)
        out.update(_metrics(PARTITION_PREFIXES[name], part.y_observed, yhat))
        if name == "test":
            out.update(_metrics("test_true", part.y_true, yhat))
    return out


def _fit_columns(
    fit: FitResult, before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, Any]:
    best = fit.best_expression
    last = fit.history[-1] if fit.history else None
    c = fit.llm_counters
    return {
        "expression": to_infix(best),
        "complexity": size(best),
        "expression_depth": depth(best),
        "n_generations": fit.n_generations,
        "stop_reason": fit.stop_reason,
        "integration_status": fit.metadata.integration_status,
        "llm_in_loop": fit.metadata.llm_in_loop,
        "final_loss": None if last is None else last.best_loss,
        "final_e": None if last is None else last.best_e,
        "final_s": None if last is None else last.best_s,
        "final_c": None if last is None else last.best_c,
        "train_seconds": fit.elapsed_seconds,
        "llm_requested": int(_counter(c, "requested")),
        "llm_unique": int(_counter(c, "unique_llm")),
        "llm_calls": int(_counter(c, "llm_calls")),
        "llm_cache_hits": int(_counter(c, "cache_hits")),
        "llm_batches": int(_counter(c, "llm_batches")),
        "llm_seconds": _counter(c, "llm_seconds"),
        "invalid_candidates": int(_counter(c, "invalid")),
        "llm_invalid_responses": int(
            _counter(after, "invalid_responses") - _counter(before, "invalid_responses")
        ),
    }


def _history_frame(fit: FitResult, cell: Cell, row: Mapping[str, Any]) -> pd.DataFrame:
    hist = SearchHistory(f"{cell.sr}|{cell.llm}|{cell.prompt}")
    for record in fit.history:
        hist.add(record)
    frame = hist.to_dataframe()
    for col in (
        "cell_key",
        "run_id",
        "scenario",
        "sr",
        "llm",
        "prompt",
        "noise_type",
        "noise_location",
        "noise_level",
        "repetition",
        "seed",
        "mode",
        "track",
    ):
        frame[col] = row[col]
    return frame


def _manifest(
    ctx: CellContext,
    cell: Cell,
    row: Mapping[str, Any],
    full: Dataset,
    parts: Mapping[str, Dataset],
    fit: FitResult | None,
    evaluator: PhysicsEquationEvaluator | None = None,
) -> dict[str, Any]:
    cfg = ctx.cfg
    base = ctx.base_manifest()
    master = cfg.seeds.master
    seeds = {
        "master": master,
        "search": int(row["seed"]),
        "data": derive_seed(master, data_seed_name(cell.scenario, cell.repetition)),
        "noise": derive_seed(master, noise_seed_name(cell.scenario, cell.repetition, cell.noise)),
        "split": derive_seed(master, split_seed_name(cell.scenario, cell.repetition)),
    }
    models = [dict(m) for m in base.llm_models if m.get("name") == cell.llm]
    backend = getattr(evaluator, "backend", None)
    metadata = getattr(backend, "model_metadata", None)
    if callable(metadata):
        for entry in models:
            entry["effective"] = metadata()  # digest/revisão/prompt_format efetivos
    prompt = (
        None
        if cell.is_baseline
        else {
            "variant": cell.prompt,
            "prompt_hash": row.get("prompt_hash"),
            "gt_disclosed": row.get("gt_disclosed"),
        }
    )
    metric_keys = [k for k in row if k.startswith(("train_", "validation_", "test_"))]
    metrics = {k: row[k] for k in metric_keys if k != "train_hash"}
    for k in ("tree_distance", "tree_distance_equivalent_min", "exact_recovery"):
        if k in row:
            metrics[k] = row[k]
    manifest = base.model_copy(
        update={
            "run_id": f"{ctx.run_id}/{cell.key()}",
            "data_hashes": {
                "full": full.metadata["data_hash"],
                **{name: p.metadata["data_hash"] for name, p in parts.items()},
            },
            "seeds": seeds,
            "llm_models": models,
            "prompt": prompt,
            "timings": {
                k: float(row[k])
                for k in ("train_seconds", "llm_seconds", "cell_seconds")
                if row.get(k) is not None
            },
            "interruption": row.get("error"),
            "metrics": metrics,
            "discovered_expressions": []
            if fit is None
            else [
                {
                    "expression": row["expression"],
                    "complexity": row["complexity"],
                    "ground_truth": row["ground_truth"],
                }
            ],
            "history_file": None if fit is None else "history.parquet",
            "stop_conditions": {
                **base.stop_conditions,
                "stop_reason": row.get("stop_reason"),
                "max_cell_seconds": cfg.experiments.max_cell_seconds,
                "interruption": row.get("error"),
            },
        }
    )
    data: dict[str, Any] = manifest.model_dump(mode="json")
    data["cell"] = {
        "key": cell.key(),
        "llm": cell.llm or NONE,
        "integration": row.get("integration_status"),
    }
    if fit is not None and fit.metadata.extra is not None:
        data["regressor"] = {
            "algorithm": fit.metadata.algorithm,
            "integration_status": fit.metadata.integration_status,
            "extra": fit.metadata.extra,
        }
    return data
