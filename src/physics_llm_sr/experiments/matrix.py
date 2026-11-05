"""Matriz de células dos experimentos 1–5 (deduplicada e na ordem de execução).

Separado do runner para que o relatório (``reporting``) compare a matriz executada com a
matriz planejada completa da config sem importar o runner.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.experiments import (
    baseline,
    biased_noise,
    llm_integration,
    noise_robustness,
    prompt_sensitivity,
)
from physics_llm_sr.experiments.cells import Cell

#: Experimento -> gerador de células (ordem = Exp1..Exp5). Exp6 está nas métricas de cada
#: célula; Exp7 é a comparação feita em ``reporting.report``.
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


@dataclass(frozen=True)
class PlannedCell:
    """Célula da matriz e os experimentos que a pedem (células iguais são deduplicadas)."""

    cell: Cell
    experiments: tuple[str, ...]


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
    """Matriz deduplicada na ordem de execução (:meth:`Cell.sort_key`: por LLM primeiro)."""
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


def planned_keys(cfg: ExperimentConfig) -> set[str]:
    """Chaves da matriz planejada COMPLETA da config (sem ``--only``/``--sr``)."""
    return {p.cell.key() for p in build_matrix(cfg)}
