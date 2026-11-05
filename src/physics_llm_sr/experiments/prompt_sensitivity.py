"""Experimento 3 — sensibilidade aos prompts A–H (PRD §7, Tabela 2).

Os prompts que revelam a ground truth (``gt_disclosed``: D, F, G, H) são marcados em cada
linha de resultado e analisados SEPARADAMENTE dos que não revelam: a recuperação da
estrutura com a GT no prompt não é descoberta.
"""

from __future__ import annotations

import pandas as pd

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.evaluation.statistics import summarize_groups
from physics_llm_sr.experiments.cells import NONE, Cell, default_noise, llm_cell, selected_models
from physics_llm_sr.experiments.storage import empty_frame

NAME = "prompt_sensitivity"
METRICS = ("mae", "mse", "one_minus_r2", "tree_distance", "tree_distance_equivalent_min")
GROUPS = ("gt_disclosed", "llm", "prompt")


def cells(cfg: ExperimentConfig) -> list[Cell]:
    """Cenários × algoritmos × LLMs × ``llm.prompt_variants`` × repetições (+ baseline)."""
    noise = default_noise(cfg)
    ex = cfg.experiments
    out: list[Cell] = []
    for scenario in ex.scenarios:
        for sr in ex.algorithms:
            for rep in range(ex.repetitions):
                for model in selected_models(cfg):
                    for prompt in cfg.llm.prompt_variants:
                        cell = llm_cell(scenario, sr, model, prompt, noise, rep)
                        out.extend((cell, cell.baseline()))
    return out


def analyze(results: pd.DataFrame) -> pd.DataFrame:
    """Resumo por (gt_disclosed, llm, prompt) e por algoritmo, + taxa de recuperação exata.

    ``results``: linhas ``status == 'ok'`` com LLM. Formato longo de ``summarize_groups``
    acrescido da coluna ``analysis`` (``gt_disclosed`` / ``gt_not_disclosed``).
    """
    sub = results[results["llm"] != NONE]
    if sub.empty:
        return empty_frame()
    frames = []
    for disclosed, part in sub.groupby("gt_disclosed"):
        summary = summarize_groups(part, [*GROUPS, "sr"], list(METRICS))
        recovery = (
            part.groupby([*GROUPS, "sr"])["exact_recovery"].mean().rename("exact_recovery_rate")
        )
        summary = summary.merge(recovery.reset_index(), on=[*GROUPS, "sr"], how="left")
        summary.insert(0, "analysis", "gt_disclosed" if disclosed else "gt_not_disclosed")
        frames.append(summary)
    out: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return out
