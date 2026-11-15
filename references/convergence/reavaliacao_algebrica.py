"""Reavaliação offline (sem retreinar) das buscas de convergência com a distância algébrica.

Fontes (somente leitura):

1. ``results.parquet`` de cada execução das rodadas O14/O15, nas worktrees indicadas por
   ``--results`` (``1 − R²`` do teste limpo registrado pela execução: ``fonte_r2 = teste``);
2. ``references/convergence/<alg>/<cenário>/search_table.csv`` para as configurações sem
   ``results.parquet`` disponível: cada expressão × semente vira uma linha e ``1 − R²`` é
   calculado no grid denso limpo (``fonte_r2 = grid``; aproximação declarada).

As distâncias primária, equivalente e algébrica são recalculadas com o código atual e as formas
equivalentes declaradas hoje. Classificação por repetição com o critério da busca
(``ConvergenceCriterion``: distância ≤ 0,05 e ``1 − R² ≤ 1e-3``): ``antiga`` =
``use_algebraic=False``; ``nova`` = ``use_algebraic=True``. Configuração converge com ≥ 2/3.

Uso: ``uv run python references/convergence/reavaliacao_algebrica.py --results <dir> [...]``.
"""

from __future__ import annotations

import argparse
import functools
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from physics_llm_sr.evaluation.metrics import one_minus_r2, structural_metrics
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import Expr, evaluate, parse_infix
from physics_llm_sr.workflows import baseline_study as bs
from physics_llm_sr.workflows import convergence_search as cs

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reavaliacao_algebrica.csv"
ALGORITHMS = ("gplearn", "deap", "pysr")
SCENARIOS = ("free_fall", "harmonic_motion", "damped_wave")
INTERRUPTED = " (interrompida)"
ROUND_BY_ROOT = {"convergence_search": "o14", "convergence_o15": "o15"}
OLD = cs.ConvergenceCriterion(use_algebraic=False)
NEW = cs.ConvergenceCriterion(use_algebraic=True)
COLUMNS = (
    "algoritmo",
    "cenario",
    "rodada",
    "etapa",
    "rotulo",
    "execucao",
    "semente",
    "expressao",
    "distancia_primaria",
    "distancia_equivalente",
    "distancia_algebrica",
    "um_menos_r2",
    "fonte_r2",
    "classificacao_antiga",
    "classificacao_nova",
    "converge_antigo",
    "converge_novo",
)


def _forms(scenario: str) -> tuple[Expr, list[Expr]]:
    cfg = bs.build_config(bs.StudyDesign(algorithm="gplearn", scenarios=(scenario,)))
    declared = [
        parse_infix(f) for f in get_scenario(scenario, cfg.physics).equivalent_ground_truths()
    ]
    return declared[0], declared[1:]


def _search_tables(roots: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in sorted(HERE.glob("*/*/search_table.csv")):
        df = pd.read_csv(path)
        df["_fonte"] = "references"
        frames.append(df)
    for root in roots:
        for path in sorted(root.glob("*/*/*/*/search_table.csv")):
            df = pd.read_csv(path)
            df["_rodada"] = ROUND_BY_ROOT.get(path.parents[3].name, path.parents[3].name)
            df["_fonte"] = "results"
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # etapas abortadas aparecem como "triagem (interrompida)" nas tabelas; o diretório é "triagem"
    out["etapa"] = out["etapa"].astype(str).str.replace(INTERRUPTED, "", regex=False)
    return out


def _label(
    tables: pd.DataFrame,
    used: set[tuple[str, str, str, str]],
    key: tuple[str, str, str],
    seeds: list[int],
    exprs: list[str],
) -> str:
    """Rótulo da configuração cuja linha contém todas as sementes e expressões da execução.

    Configurações com resultados idênticos (mesmas sementes e expressões) recebem as execuções
    na ordem, uma por rótulo (``used``).
    """
    alg, scenario, stage = key
    sub = tables[
        (tables["algoritmo"] == alg) & (tables["cenario"] == scenario) & (tables["etapa"] == stage)
    ]
    matches = []
    for row in sub.itertuples():
        table_seeds = str(getattr(row, "sementes", "") or "").split(",")
        table_exprs = str(getattr(row, "expressoes", "") or "").split(" | ")
        if all(str(s) in table_seeds for s in seeds) and all(e in table_exprs for e in exprs):
            matches.append(str(row.rotulo))
    for label in dict.fromkeys(matches):
        if (*key, label) not in used:
            used.add((*key, label))
            return label
    return matches[0] if matches else ""


def _from_parquet(roots: Iterable[Path], tables: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[tuple[str, str, str, str]] = set()
    for root in roots:
        rodada = ROUND_BY_ROOT.get(root.name, root.name)
        for path in sorted(root.glob("*/*/*/fase1_validacao_limpa/*/*/results.parquet")):
            stage = path.parents[3].name
            df = pd.read_parquet(path)
            if "expression" not in df.columns:  # execução interrompida sem expressão
                continue
            ok = df[(df["status"] == "ok") & df["expression"].map(lambda e: isinstance(e, str))]
            if ok.empty:
                continue
            alg, scenario = str(ok["sr"].iloc[0]), str(ok["scenario"].iloc[0])
            label = _label(
                tables,
                used,
                (alg, scenario, stage),
                [int(s) for s in ok["seed"]],
                list(ok["expression"]),
            )
            for rec in ok.to_dict("records"):
                rows.append(
                    {
                        "algoritmo": alg,
                        "cenario": scenario,
                        "rodada": rodada,
                        "etapa": stage,
                        "rotulo": label or f"<sem rótulo: {path.parent.name}>",
                        "execucao": path.parent.name,
                        "semente": int(rec["seed"]),
                        "expressao": rec["expression"],
                        "um_menos_r2": float(rec["test_true_one_minus_r2"]),
                        "fonte_r2": "teste",
                    }
                )
    return rows


@functools.cache
def _grid(scenario: str) -> tuple[object, object, tuple[str, ...]]:
    cfg = bs.build_config(bs.StudyDesign(algorithm="gplearn", scenarios=(scenario,)))
    x, y = bs.dense_grid(cfg, scenario)
    return x, y, get_scenario(scenario, cfg.physics).feature_names


def _grid_r2(expr: str, scenario: str) -> float:
    x, y, names = _grid(scenario)
    try:
        return one_minus_r2(y, evaluate(parse_infix(expr), x, names))  # type: ignore[arg-type]
    except Exception:
        return math.nan


def _from_tables(
    tables: pd.DataFrame, covered: set[tuple[str, str, str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    refs = tables[tables["_fonte"] == "references"]
    for rec in refs.to_dict("records"):
        key = (rec["algoritmo"], rec["cenario"], rec["etapa"], rec["rotulo"])
        exprs = [
            e for e in str(rec.get("expressoes") or "").split(" | ") if e and not e.startswith("<")
        ]
        if key in covered or not exprs or str(rec.get("expressoes")) == "nan":
            continue
        seeds = [s for s in str(rec.get("sementes") or "").split(",") if s and s != "nan"]
        rodada = rec.get("rodada")
        rodada = "o14" if not isinstance(rodada, str) or not rodada else rodada
        for i, expr in enumerate(exprs):
            rows.append(
                {
                    "algoritmo": rec["algoritmo"],
                    "cenario": rec["cenario"],
                    "rodada": rodada,
                    "etapa": rec["etapa"],
                    "rotulo": rec["rotulo"],
                    "execucao": "",
                    "semente": int(seeds[i]) if i < len(seeds) else -1,
                    "expressao": expr,
                    "um_menos_r2": _grid_r2(expr, rec["cenario"]),
                    "fonte_r2": "grid",
                }
            )
    return rows


def _classify(ok: bool) -> str:
    return "sucesso" if ok else "fracasso"


def reassess(roots: Iterable[Path]) -> pd.DataFrame:
    """Tabela de reavaliação (uma linha por repetição) com classificação antiga e nova."""
    roots = [Path(r) for r in roots if Path(r).exists()]
    tables = _search_tables(roots)
    rows = _from_parquet(roots, tables)
    covered = {(r["algoritmo"], r["cenario"], r["etapa"], r["rotulo"]) for r in rows}
    rows += _from_tables(tables, covered)
    forms = {s: _forms(s) for s in SCENARIOS}
    for row in rows:
        gt, eq = forms[str(row["cenario"])]
        try:
            m = structural_metrics(parse_infix(str(row["expressao"])), gt, eq)
        except Exception:
            m = dict.fromkeys(
                ("tree_distance", "tree_distance_equivalent_min", "tree_distance_algebraic"),
                math.nan,
            )
        row["distancia_primaria"] = m["tree_distance"]
        row["distancia_equivalente"] = m["tree_distance_equivalent_min"]
        row["distancia_algebrica"] = m["tree_distance_algebraic"]
        args = (
            float(row["um_menos_r2"]),
            m["tree_distance"],
            m["tree_distance_equivalent_min"],
            m["tree_distance_algebraic"],
        )
        row["classificacao_antiga"] = _classify(OLD.repetition_succeeds(*args))
        row["classificacao_nova"] = _classify(NEW.repetition_succeeds(*args))
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["algoritmo", "cenario", "rodada", "etapa", "rotulo", "semente"], keep="last"
    )
    keys = ["algoritmo", "cenario", "rodada", "etapa", "rotulo"]
    grouped = df.groupby(keys, dropna=False)
    for col, name in (
        ("classificacao_antiga", "converge_antigo"),
        ("classificacao_nova", "converge_novo"),
    ):
        conv = grouped[col].transform(lambda s: NEW.converges(int((s == "sucesso").sum()), len(s)))
        df[name] = conv.astype(bool)
    df = df.sort_values(keys + ["semente"], kind="mergesort").reset_index(drop=True)
    result: pd.DataFrame = df[list(COLUMNS)]
    return result


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por configuração: sucessos antigo/novo e convergência."""
    keys = ["algoritmo", "cenario", "rodada", "etapa", "rotulo"]
    out: pd.DataFrame = (
        df.groupby(keys, dropna=False)
        .agg(
            n=("semente", "size"),
            sucesso_antigo=("classificacao_antiga", lambda s: int((s == "sucesso").sum())),
            sucesso_novo=("classificacao_nova", lambda s: int((s == "sucesso").sum())),
            min_um_menos_r2=("um_menos_r2", "min"),
            converge_antigo=("converge_antigo", "first"),
            converge_novo=("converge_novo", "first"),
            min_dist_algebrica=("distancia_algebrica", "min"),
        )
        .reset_index()
    )
    return out


def main() -> None:
    """Gera ``reavaliacao_algebrica.csv`` e imprime as configurações que mudam de classificação."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", nargs="*", type=Path, default=[], help="raízes results/<rodada>"
    )
    args = parser.parse_args()
    df = reassess(args.results)
    df.to_csv(OUTPUT, index=False, float_format="%.6g")
    s = summary(df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(f"gravado {OUTPUT} ({len(df)} linhas, {len(s)} configurações)")
    print(s[s["sucesso_novo"] != s["sucesso_antigo"]].to_string(index=False))
    print("\nconvergem pelo critério novo:")
    print(s[s["converge_novo"]].to_string(index=False))


if __name__ == "__main__":
    main()
