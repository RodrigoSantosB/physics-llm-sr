"""Runner dos experimentos 1–7 (matriz, execução por célula, checkpoint, retomada).

Uso programático (o notebook chama estas funções)::

    summary = run_experiments(load_config("configs/smoke.yaml"))
    report(summary.run_dir)        # reconstrói tabelas e figuras dos arquivos persistidos
    replot(summary.run_dir)

Política de falhas: uma célula que falha (LLM, adaptador indisponível, tempo) é gravada
com ``status='error'`` e o motivo. Em ``mode=full`` + ``track=paper_replication`` a
execução ABORTA (:class:`ExperimentAbortedError`) depois de gravar a célula e consolidar;
em smoke/pilot (e extended) continua e o relatório lista as falhas. Na retomada, células
``ok`` são puladas e células ``error`` são tentadas de novo.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Mode, Track
from physics_llm_sr.experiments import (
    baseline,
    biased_noise,
    llm_integration,
    noise_robustness,
    prompt_sensitivity,
)
from physics_llm_sr.experiments.cells import Cell
from physics_llm_sr.experiments.data_provider import DataProvider
from physics_llm_sr.experiments.execution import CellContext, CellOutcome, run_cell
from physics_llm_sr.experiments.reporting import (
    methodological_view,
    replot,
    report,
)
from physics_llm_sr.experiments.storage import (
    RUN_FILE,
    STATUS_OK,
    completed_keys,
    write_cell,
)
from physics_llm_sr.llm.inference import build_backend
from physics_llm_sr.llm.interfaces import PhysicsEquationEvaluator
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import LLMEquationEvaluator
from physics_llm_sr.reproducibility.artifacts import (
    create_run_dir,
    load_json,
    save_json,
    save_manifest,
)
from physics_llm_sr.reproducibility.manifests import (
    RunManifest,
    build_manifest,
    hash_config,
    new_run_id,
)

__all__ = [
    "EXPERIMENTS",
    "ExperimentAbortedError",
    "PlannedCell",
    "RunSummary",
    "build_matrix",
    "find_resumable_run",
    "methodological_view",
    "replot",
    "report",
    "resolve_experiments",
    "run_experiments",
]

LOGGER = logging.getLogger(__name__)
#: Experimento -> gerador de células (ordem = Exp1..Exp5). Exp6 está nas métricas de cada
#: célula; Exp7 é a comparação feita em :func:`report`.
EXPERIMENTS: dict[str, Callable[[ExperimentConfig], list[Cell]]] = {
    baseline.NAME: baseline.cells,
    llm_integration.NAME: llm_integration.cells,
    prompt_sensitivity.NAME: prompt_sensitivity.cells,
    noise_robustness.NAME: noise_robustness.cells,
    biased_noise.NAME: biased_noise.cells,
}
ALIASES = {
    "exp1": baseline.NAME,
    "exp2": llm_integration.NAME,
    "exp3": prompt_sensitivity.NAME,
    "exp4": noise_robustness.NAME,
    "exp5": biased_noise.NAME,
}
DERIVED = {"exp6": "comparação estrutural (métricas de cada célula)", "exp7": "use `report`"}
EvaluatorFactory = Callable[[str], PhysicsEquationEvaluator]
CellCallback = Callable[[Cell, CellOutcome], None]


class ExperimentAbortedError(RuntimeError):
    """Falha de célula em full/paper_replication: a execução é interrompida."""


@dataclass(frozen=True)
class PlannedCell:
    """Célula da matriz e os experimentos que a pedem (células iguais são deduplicadas)."""

    cell: Cell
    experiments: tuple[str, ...]


@dataclass
class RunSummary:
    """Resumo de uma invocação do runner."""

    run_dir: Path
    run_id: str
    planned: int
    executed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)


def resolve_experiments(names: Iterable[str] | None, cfg: ExperimentConfig) -> list[str]:
    """Nomes canônicos a executar (``--only``); default = ``experiments.run`` da config."""
    if names is None:
        return list(cfg.experiments.run)
    out: list[str] = []
    for raw in names:
        name = raw.strip().lower()
        if name in DERIVED:
            raise ValueError(f"{name} não é executável isoladamente: {DERIVED[name]}")
        name = ALIASES.get(name, name)
        if name not in EXPERIMENTS:
            raise ValueError(f"experimento desconhecido: {raw!r}; válidos: exp1..exp5")
        out.append(name)
    return out


def build_matrix(
    cfg: ExperimentConfig,
    only: Iterable[str] | None = None,
    algorithms: Sequence[str] | None = None,
) -> list[PlannedCell]:
    """Matriz deduplicada e ordenada (cenário × repetição × ruído × SR × LLM × prompt)."""
    exps = resolve_experiments(only, cfg)
    if algorithms is not None:
        unknown = sorted(set(algorithms) - set(cfg.experiments.algorithms))
        if unknown:
            raise ValueError(f"--sr fora de experiments.algorithms: {unknown}")
    by_key: dict[str, tuple[Cell, list[str]]] = {}
    for exp in exps:
        for cell in EXPERIMENTS[exp](cfg):
            if algorithms is not None and cell.sr not in algorithms:
                continue
            entry = by_key.setdefault(cell.key(), (cell, []))
            if exp not in entry[1]:
                entry[1].append(exp)
    planned = [PlannedCell(c, tuple(e)) for c, e in by_key.values()]
    return sorted(planned, key=lambda p: p.cell.sort_key())


def aborts_on_failure(cfg: ExperimentConfig) -> bool:
    """Falha de célula aborta só em full + paper_replication."""
    return cfg.mode is Mode.FULL and cfg.track is Track.PAPER_REPLICATION


def find_resumable_run(cfg: ExperimentConfig) -> Path | None:
    """Execução mais recente em ``output_dir`` com o mesmo hash de config."""
    out = resolve_project_path(cfg.output_dir)
    if not out.is_dir():
        return None
    target = hash_config(cfg)
    candidates = [
        d
        for d in sorted(out.iterdir())
        if (d / RUN_FILE).is_file() and load_json(d / RUN_FILE).get("config_hash") == target
    ]
    return candidates[-1] if candidates else None


def _default_evaluator_factory(cfg: ExperimentConfig) -> EvaluatorFactory:
    def factory(model: str) -> PhysicsEquationEvaluator:
        backend = build_backend(cfg.llm, model, track=cfg.track)
        return LLMEquationEvaluator.from_config(cfg.llm, backend)

    return factory


def _open_run_dir(cfg: ExperimentConfig, run_dir: Path | None, resume: bool) -> tuple[Path, str]:
    target = hash_config(cfg)
    if run_dir is None and resume:
        run_dir = find_resumable_run(cfg)
        if run_dir is None:
            LOGGER.info("nenhuma execução retomável em %s; iniciando nova", cfg.output_dir)
    if run_dir is not None and (Path(run_dir) / RUN_FILE).is_file():
        info = load_json(Path(run_dir) / RUN_FILE)
        if info["config_hash"] != target:
            raise ValueError(
                f"config diverge da execução em {run_dir} (hash {info['config_hash'][:12]} "
                f"!= {target[:12]}): a retomada exige a mesma config"
            )
        return Path(run_dir), str(info["run_id"])
    run_id = new_run_id()
    if run_dir is None:
        path = create_run_dir(cfg.output_dir, run_id)
    else:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": cfg.mode.value,
            "track": cfg.track.value,
            "config_hash": target,
            "config": cfg.model_dump(mode="json"),
        },
        path / RUN_FILE,
    )
    return path, run_id


def run_experiments(
    cfg: ExperimentConfig,
    *,
    run_dir: Path | None = None,
    resume: bool = False,
    only: Iterable[str] | None = None,
    algorithms: Sequence[str] | None = None,
    evaluator_factory: EvaluatorFactory | None = None,
    on_cell_done: CellCallback | None = None,
    write_report: bool = True,
) -> RunSummary:
    """Executa a matriz da config célula a célula, com checkpoint e retomada.

    Args:
        cfg: configuração validada.
        run_dir: diretório da execução; ``None`` cria ``output_dir/<run_id>`` (ou, com
            ``resume``, reaproveita a execução mais recente com o mesmo hash de config).
        resume: pula células já concluídas (``status='ok'``) no ``run_dir``.
        only: subconjunto de experimentos (``exp1``..``exp5`` ou nomes canônicos).
        algorithms: subconjunto de algoritmos SR.
        evaluator_factory: ``modelo -> avaliador``; default = backend da config
            (``build_backend`` + ``LLMEquationEvaluator.from_config``).
        on_cell_done: chamado após cada célula executada (e gravada).
        write_report: consolida e gera relatório/figuras ao final.

    Raises:
        ExperimentAbortedError: falha de célula em full/paper_replication.
    """
    path, run_id = _open_run_dir(cfg, run_dir, resume)
    planned = build_matrix(cfg, only, algorithms)
    summary = RunSummary(path, run_id, planned=len(planned))
    done = completed_keys(path) if resume else set()
    factory = evaluator_factory or _default_evaluator_factory(cfg)
    evaluators: dict[str, PhysicsEquationEvaluator] = {}
    cache: dict[str, Any] = {}

    def evaluator_for(model: str) -> PhysicsEquationEvaluator:
        if model not in evaluators:
            evaluators[model] = factory(model)
        return evaluators[model]

    def prompts() -> PromptLibrary:
        if "prompts" not in cache:
            cache["prompts"] = PromptLibrary.from_config(cfg.llm)
        lib: PromptLibrary = cache["prompts"]
        return lib

    def base_manifest() -> RunManifest:
        if "manifest" not in cache:
            cache["manifest"] = build_manifest(cfg, run_id=run_id)
        man: RunManifest = cache["manifest"]
        return man

    ctx = CellContext(cfg, run_id, DataProvider(cfg), evaluator_for, prompts, base_manifest)
    abort = aborts_on_failure(cfg)
    for item in planned:
        key = item.cell.key()
        if key in done:
            summary.skipped += 1
            continue
        outcome = run_cell(ctx, item.cell)
        outcome.row["experiments"] = ",".join(item.experiments)
        write_cell(path, key, outcome.row, outcome.history, outcome.manifest)
        summary.executed += 1
        if outcome.row["status"] != STATUS_OK:
            summary.failed += 1
            summary.errors.append({"cell_key": key, "error": outcome.row["error"]})
            LOGGER.warning("célula %s falhou: %s", key, outcome.row["error"])
        if on_cell_done is not None:
            on_cell_done(item.cell, outcome)
        if outcome.row["status"] != STATUS_OK and abort:
            _finish(path, cfg, run_id, evaluators, write_report)
            raise ExperimentAbortedError(
                f"célula {key} falhou em full/paper_replication: {outcome.row['error']}"
            ) from outcome.exception
    _finish(path, cfg, run_id, evaluators, write_report)
    return summary


def _finish(
    path: Path,
    cfg: ExperimentConfig,
    run_id: str,
    evaluators: dict[str, PhysicsEquationEvaluator],
    write_report: bool,
) -> None:
    backends = {name: ev.backend for name, ev in evaluators.items() if hasattr(ev, "backend")}
    manifest = build_manifest(cfg, run_id=run_id, backends=backends or None)
    manifest = manifest.model_copy(update={"history_file": "history.parquet"})
    save_manifest(manifest, path)
    if write_report:
        report(path)
