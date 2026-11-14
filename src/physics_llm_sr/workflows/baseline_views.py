"""Tabelas e figuras padronizadas do estudo de baseline local (consumidas pelo notebook 05).

Todas as funções recebem o desenho, os dados ou o relatório de consistência
(:func:`physics_llm_sr.workflows.baseline_study.consistency_report`) e funcionam igualmente
para a Fase 1 e para qualquer conjunto da Fase 2 (mesmo esquema de colunas). As figuras são
devolvidas fechadas (``plt.close``) para exibição explícita com ``display(fig)``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from physics_llm_sr.analysis.legacy_export import export_legacy
from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.validation import validate_dataset
from physics_llm_sr.experiments.data_provider import DataProvider
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import parse_infix, variables
from physics_llm_sr.workflows.baseline_study import (
    CLASS_RECOVERS,
    CONSISTENCY_FILE,
    STUDY_LOG_FILE,
    StudyDesign,
    StudyResult,
    build_config,
)

FIGURE_DPI = 110
COMPARISON_METRICS = (
    ("um_menos_r2_teste", "1 − R² (teste limpo)"),
    ("distancia_estrutural", "distância estrutural"),
    ("erro_max_abs_grid", "erro máx. absoluto (grid denso)"),
)
LOG_FLOOR = 1e-16
"""Piso para eixos logarítmicos (``1 − R²`` e erros exatamente 0 aparecem neste piso)."""


def reference_table(design: StudyDesign) -> pd.DataFrame:
    """Equação de referência (ground truth), formas equivalentes e variáveis por cenário."""
    cfg = build_config(design)
    rows = []
    for name in design.scenarios:
        scenario = get_scenario(name, cfg.physics)
        context = scenario.physical_context()
        forms = scenario.equivalent_ground_truths()
        used = sorted(variables(parse_infix(forms[0])))
        rows.append(
            {
                "cenario": name,
                "alvo": f"{scenario.target_name} ({context.units[scenario.target_name]})",
                "equacao_referencia": forms[0],
                "formas_equivalentes": "; ".join(forms[1:]) or "(nenhuma)",
                "variaveis_na_equacao": ", ".join(used),
                "variaveis_distratoras": ", ".join(
                    v for v in scenario.feature_names if v not in used
                ),
            }
        )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


def generate_clean_data(design: StudyDesign) -> dict[str, Dataset]:
    """Dados sem ruído por cenário (repetição 0), gerados da equação de referência."""
    provider = DataProvider(build_config(design))
    return {name: provider.clean(name, 0) for name in design.scenarios}


def clean_data_report(design: StudyDesign, datasets: dict[str, Dataset]) -> pd.DataFrame:
    """Amostras, simulações e a verificação ``y_observado == y_verdadeiro`` por cenário."""
    cfg = build_config(design)
    rows = []
    for name, ds in datasets.items():
        rows.append(
            {
                "cenario": name,
                "amostras": ds.X.shape[0],
                "simulacoes": int(np.unique(ds.simulation_ids).size),
                "alvo": ds.target_name,
                "desvio_max_y_obs_vs_y_verdadeiro": float(
                    np.max(np.abs(ds.y_observed - ds.y_true))
                ),
                "y_observado_igual_verdadeiro": bool(np.array_equal(ds.y_observed, ds.y_true)),
                "problemas_de_validacao": "; ".join(validate_dataset(ds, cfg.physics)) or "nenhum",
            }
        )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


def domain_table(datasets: dict[str, Dataset]) -> pd.DataFrame:
    """Domínio amostrado (mínimo e máximo observados) de cada variável e do alvo, por cenário."""
    rows = []
    for name, ds in datasets.items():
        units = ds.metadata.get("units", {})
        for i, feature in enumerate((*ds.feature_names,)):
            rows.append(_domain_row(name, feature, units, ds.X[:, i]))
        rows.append(_domain_row(name, ds.target_name, units, ds.y_true))
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


def _domain_row(
    scenario: str, name: str, units: dict[str, str], values: np.ndarray
) -> dict[str, object]:
    return {
        "cenario": scenario,
        "variavel": name,
        "unidade": units.get(name, ""),
        "minimo": float(values.min()),
        "maximo": float(values.max()),
    }


def plot_trajectories(datasets: dict[str, Dataset], n_simulations: int = 3) -> Figure:
    """Trajetórias (alvo × tempo) de algumas simulações por cenário."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.6), squeeze=False)
    for ax, (name, ds) in zip(axes[0], datasets.items(), strict=True):
        for sim in np.unique(ds.simulation_ids)[:n_simulations]:
            mask = ds.simulation_ids == sim
            ax.plot(ds.t[mask], ds.y_true[mask], ".", ms=3, label=f"simulação {sim}")
        ax.set(title=name, xlabel="t (s)", ylabel=ds.target_name)
        ax.legend(fontsize="small")
    fig.tight_layout()
    plt.close(fig)
    return fig


def convergence_table(result: StudyResult) -> pd.DataFrame:
    """Melhor perda por geração (uma linha por cenário × repetição × geração × condição)."""
    frames = []
    for run in result.runs:
        if run.history.empty:
            continue
        keep = run.history[["scenario", "repetition", "generation", "best_loss"]]
        frames.append(keep.assign(condicao=run.condition))
    if not frames:
        empty: pd.DataFrame = pd.DataFrame(
            columns=["scenario", "repetition", "generation", "best_loss", "condicao"]
        )
        return empty
    table: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return table


def plot_convergence(result: StudyResult) -> Figure:
    """Melhor perda por geração, um painel por cenário (repetições finas, média em destaque)."""
    table = convergence_table(result)
    scenarios = list(dict.fromkeys(table["scenario"])) or list(result.design.scenarios)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 3.6), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios, strict=True):
        sub = table[table["scenario"] == scenario]
        for condition, cond in sub.groupby("condicao", sort=False):
            for _, rep in cond.groupby("repetition"):
                ax.plot(rep["generation"], rep["best_loss"], alpha=0.35, lw=1)
            mean = cond.groupby("generation")["best_loss"].mean()
            ax.plot(mean.index, mean.to_numpy(), lw=2.2, label=str(condition))
        ax.set(title=scenario, xlabel="geração", ylabel="melhor perda composta")
        if not sub.empty and (sub["best_loss"] > 0).all():
            ax.set_yscale("log")
        ax.legend(fontsize="small")
    fig.tight_layout()
    plt.close(fig)
    return fig


def scenario_comparison(report: pd.DataFrame) -> pd.DataFrame:
    """Tabela padronizada: médias entre repetições por fase × condição × algoritmo × cenário."""
    keys = ["fase", "condicao", "algoritmo", "cenario"]
    grouped = report.groupby(keys, sort=False)
    table = grouped.agg(
        n=("repeticao", "size"),
        mae_media=("mae_teste", "mean"),
        um_menos_r2_media=("um_menos_r2_teste", "mean"),
        um_menos_r2_mediana=("um_menos_r2_teste", "median"),
        erro_max_grid_maximo=("erro_max_abs_grid", "max"),
        distancia_media=("distancia_estrutural", "mean"),
        distancia_equivalente_media=("distancia_equivalente_min", "mean"),
        geracoes_media=("geracoes", "mean"),
        tempo_medio_s=("tempo_s", "mean"),
    )
    table["taxa_recupera_estrutura"] = grouped["classificacao"].apply(
        lambda s: float((s == CLASS_RECOVERS).mean())
    )
    out: pd.DataFrame = table.reset_index()
    return out


def plot_scenario_comparison(report: pd.DataFrame) -> Figure:
    """Comparação entre cenários: pontos por repetição e média, nas métricas padronizadas."""
    report = report.assign(grupo=report["condicao"].astype(str) + " | " + report["algoritmo"])
    scenarios = list(dict.fromkeys(report["cenario"]))
    groups = list(dict.fromkeys(report["grupo"]))
    width = 0.8 / max(1, len(groups))
    fig, axes = plt.subplots(
        1, len(COMPARISON_METRICS), figsize=(5.2 * len(COMPARISON_METRICS), 3.8)
    )
    for ax, (column, title) in zip(np.atleast_1d(axes), COMPARISON_METRICS, strict=True):
        for g, group in enumerate(groups):
            sub = report[report["grupo"] == group]
            for s, scenario in enumerate(scenarios):
                values = sub.loc[sub["cenario"] == scenario, column].to_numpy(dtype=float)
                if values.size == 0:
                    continue
                x = s + (g - (len(groups) - 1) / 2) * width
                if column != "distancia_estrutural":
                    values = np.maximum(values, LOG_FLOOR)
                ax.plot([x] * values.size, values, "o", ms=4, alpha=0.5, color=f"C{g}")
                ax.plot([x], [values.mean()], "_", ms=18, mew=2.5, color=f"C{g}",
                        label=group if s == 0 else None)  # fmt: skip
        ax.set_xticks(range(len(scenarios)), scenarios, rotation=15)
        ax.set_title(title)
        if column != "distancia_estrutural":
            ax.set_yscale("log")
        ax.legend(fontsize="x-small")
    fig.tight_layout()
    plt.close(fig)
    return fig


def export_study(result: StudyResult, out_dir: Path | str | None = None) -> dict[str, Path]:
    """Exporta CSV padronizado, log agregado e o formato legado de cada execução.

    Grava em ``out_dir`` (padrão: ``<output_root>/exportacao/<fase>``): ``consistencia.csv`` e
    ``study_log.jsonl`` (concatenação dos logs das execuções) e, por execução, a exportação
    legada de :mod:`physics_llm_sr.analysis.legacy_export` em ``legado/<hash do desenho>``.
    """
    from physics_llm_sr.workflows.baseline_study import consistency_report

    target = (
        Path(out_dir)
        if out_dir is not None
        else resolve_project_path(result.design.output_root) / "exportacao" / result.phase
    )
    target.mkdir(parents=True, exist_ok=True)
    consistency_path = target / "consistencia.csv"
    consistency_report(result).to_csv(consistency_path, index=False)
    log_path = target / STUDY_LOG_FILE
    with log_path.open("w", encoding="utf-8") as out:
        for run in result.runs:
            out.write((run.run_dir / STUDY_LOG_FILE).read_text(encoding="utf-8"))
    paths = {"consistencia": consistency_path, "log": log_path}
    for run in result.runs:
        legacy_dir = target / "legado" / run.run_dir.name
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)
        export_legacy([run.run_dir], legacy_dir)
        paths[f"legado:{run.condition}"] = legacy_dir
        shutil.copyfile(run.run_dir / CONSISTENCY_FILE, legacy_dir / "consistency.csv")
    return paths
