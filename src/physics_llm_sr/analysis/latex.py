# ruff: noqa: E501, UP031
# mypy: disable-error-code="no-any-return,unused-ignore,no-untyped-call"
"""Tabelas LaTeX no estilo IEEE/relatório em português (versaletes, booktabs, vírgula decimal).

Funções puras: recebem DataFrames já calculados e devolvem texto LaTeX. Numeração romana
configurável; legenda em versaletes centralizada acima; ``n/e`` = não executado/indefinido.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

NA_TEX = "n/e"
ND_TEX = "n/d"
ROMAN = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
SCENARIO_PT = {
    "free_fall": "Queda livre",
    "harmonic_motion": "MHS",
    "damped_wave": "Onda amortecida",
}
LLM_PT = {"mistral": "Mistral", "llama2": "LLaMA", "falcon": "Falcon", "none": "Sem LLM"}
SR_PT = {"deap": "DEAP", "pysr": "PySR", "gplearn": "gplearn"}
LOC_PT = {"inputs": "Ruído nas entradas", "target": "Ruído no alvo", "both": "Ruído em ambos"}
SOURCE_PT = {"artigo": "Artigo", "com_llm": "Com LLM", "sem_llm": "Sem LLM"}
DECIMALS = {"mae": 3, "mse": 3, "one_minus_r2": 2, "tree_distance": 2}
METRIC_PT = {"mae": "MAE", "mse": "MSE", "one_minus_r2": "$1-R^2$", "tree_distance": "Dist."}
_ESC = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
}  # fmt: skip
Row = tuple[list[str], str]
"""(células, regra após a linha: '', 'cmidrule' ou 'midrule')."""


def roman(n: int) -> str:
    """Número romano (1-39)."""
    out = ""
    for value, sym in ROMAN:
        while n >= value:
            out += sym
            n -= value
    return out


def esc(text: str) -> str:
    """Escapa caracteres especiais do LaTeX."""
    return "".join(_ESC.get(c, c) for c in str(text))


def num(x: Any, decimals: int = 2, suffix: str = "") -> str:
    """Número com vírgula decimal; NaN/None vira ``n/e``."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA:
        return NA_TEX
    return f"{float(x):.{decimals}f}".replace(".", ",") + suffix.replace("%", r"\%")


def _cap(caption: str) -> str:
    return r"\textsc{" + esc(caption) + "}"


def wrap(number: int, caption: str, tabular: str, resize: bool = False) -> str:
    """Envolve o ``tabular`` com cabeçalho 'Tabela N' e legenda em versaletes acima."""
    body = tabular
    if resize:
        body = "\\resizebox{\\textwidth}{!}{%\n" + tabular + "}"
    return "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering\footnotesize",
            f"Tabela {roman(number)}\\\\[2pt]",
            r"\parbox{0.92\textwidth}{\centering " + _cap(caption) + "}\\\\[4pt]",
            body,
            r"\end{table}",
            "",
        ]
    )


def _tabular(colspec: str, header: Sequence[str], rows: Sequence[Row]) -> str:
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", *header, r"\midrule"]
    for i, (cells, rule) in enumerate(rows):
        lines.append(" & ".join(cells) + r" \\")
        last = i == len(rows) - 1
        if rule and not last:
            lines.append(r"\midrule" if rule == "midrule" else rule)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def generic(number: int, caption: str, df: pd.DataFrame, decimals: int = 3) -> str:
    """Tabela simples: todas as colunas, vírgula decimal."""
    cols = list(df.columns)
    rows: list[Row] = []
    for rec in df.itertuples(index=False):
        cells = []
        for v in rec:
            if isinstance(v, float):
                cells.append(num(v, decimals))
            elif v is None or v is pd.NA:
                cells.append(NA_TEX)
            else:
                cells.append(esc(str(v)))
        rows.append((cells, ""))
    header = [" & ".join(esc(str(c)) for c in cols) + r" \\"]
    return wrap(number, caption, _tabular("l" * len(cols), header, rows), resize=True)


# ----------------------------------------------------------------------------- ruído (Tabela VII)
def noise_table(long: pd.DataFrame, number: int, caption: str) -> str:
    """Formato 'Tabela VII': Cenário | RS | Fonte | 1%..5% por localização.

    ``long``: scenario, sr, noise_location, noise_level, source (artigo/com_llm/sem_llm), mean.
    """
    locs = [loc for loc in LOC_PT if loc in set(long["noise_location"])] or list(LOC_PT)
    levels = sorted(long["noise_level"].unique()) or [0.01, 0.02, 0.03, 0.04, 0.05]
    nl = len(levels)
    head1 = [r"\multicolumn{3}{c}{} & " + " & ".join(
        r"\multicolumn{%d}{c}{%s}" % (nl, LOC_PT[loc]) for loc in locs) + r" \\"]  # fmt: skip
    cm = " ".join(f"\\cmidrule(lr){{{4 + i * nl}-{3 + (i + 1) * nl}}}" for i in range(len(locs)))
    head2 = (
        "Cenário & RS & Fonte & "
        + " & ".join(" & ".join(f"{round(lv * 100)}\\%" for lv in levels) for _ in locs)
        + r" \\"
    )
    index = long.set_index(["scenario", "sr", "source", "noise_location", "noise_level"])["mean"]
    rows: list[Row] = []
    scen_list = [s for s in SCENARIO_PT if s in set(long["scenario"])]
    for si, scen in enumerate(scen_list):
        srs = [s for s in SR_PT if s in set(long[long["scenario"] == scen]["sr"])]
        for ri, sr in enumerate(srs):
            for k, src in enumerate(SOURCE_PT):
                cells = [
                    esc(SCENARIO_PT[scen]) if (ri == 0 and k == 0) else "",
                    esc(SR_PT[sr]) if k == 0 else "",
                    SOURCE_PT[src],
                ]
                for loc in locs:
                    for lv in levels:
                        v = index.get((scen, sr, src, loc, lv), math.nan)
                        cells.append(num(v, 2))
                last_sr = ri == len(srs) - 1
                rule = ""
                if k == 2:
                    rule = (
                        "midrule"
                        if last_sr and si < len(scen_list) - 1
                        else ("" if last_sr else r"\cmidrule(lr){2-%d}" % (3 + nl * len(locs)))
                    )
                rows.append((cells, rule))
    spec = "lll" + ("r" * nl) * len(locs)
    tab = _tabular(spec, [*head1, cm, head2], rows)
    return wrap(number, caption, tab, resize=True)


# ----------------------------------------------------------------------------- custo (Tabela V)
def cost_table(df: pd.DataFrame, number: int, caption: str) -> str:
    """Formato 'Tabela V': LLM | RS | Exec. | Chamadas | Falha | LLM/RS."""
    rows: list[Row] = []
    llms = list(dict.fromkeys(df["llm"]))
    for li, llm in enumerate(llms):
        sub = df[df["llm"] == llm].copy()
        sub["_o"] = sub["sr"].map({k: i for i, k in enumerate(SR_PT)})
        sub = sub.sort_values("_o").drop(columns="_o")
        for k, rec in enumerate(sub.itertuples(index=False)):
            fail = ND_TEX if pd.isna(rec.failure_rate_pct) else num(rec.failure_rate_pct, 1, "%")  # type: ignore[attr-defined]
            ratio = ND_TEX if pd.isna(rec.median_llm_sr_ratio) else num(rec.median_llm_sr_ratio, 1)  # type: ignore[attr-defined]
            calls = ND_TEX if pd.isna(rec.calls_mean) else num(rec.calls_mean, 1)  # type: ignore[attr-defined]
            cells = [
                esc(LLM_PT.get(llm, llm)) if k == 0 else "",
                esc(SR_PT.get(str(rec.sr), str(rec.sr))),  # type: ignore[attr-defined]
                str(rec.n_exec),  # type: ignore[attr-defined]
                calls,
                fail,
                ratio,
            ]
            rows.append((cells, "midrule" if (k == len(sub) - 1 and li < len(llms) - 1) else ""))
    header = [r"LLM & RS & Exec. & Chamadas & Falha & LLM/RS \\"]
    return wrap(number, caption, _tabular("llrrrr", header, rows))


# ----------------------------------------------------------------------------- T5
def benchmark_table(df: pd.DataFrame, number: int, caption: str) -> str:
    """Tabela 5: Cenário | RS | LLM | Fonte (Artigo, Com LLM, Sem LLM) x 4 métricas."""
    metrics = list(DECIMALS)
    head = ["Cenário & RS & LLM & Fonte & " + " & ".join(METRIC_PT[m] for m in metrics) + r" \\"]
    idx = df.set_index(["scenario", "sr", "llm", "metric"])
    rows: list[Row] = []
    scen_list = [s for s in SCENARIO_PT if s in set(df["scenario"])]
    for si, scen in enumerate(scen_list):
        srs = [s for s in SR_PT if s in set(df[df["scenario"] == scen]["sr"])]
        for ri, sr in enumerate(srs):
            llms = [m for m in LLM_PT if m != "none" and m in set(df["llm"])]
            for li, llm in enumerate(llms):
                lines = [
                    ("Artigo", "published"),
                    ("Com LLM", "local_mean"),
                    ("Artigo (sem LLM)", "baseline_published"),
                    ("Sem LLM", "baseline_local_mean"),
                ]
                for k, (name, col) in enumerate(lines):
                    first = ri == 0 and li == 0 and k == 0
                    cells = [
                        esc(SCENARIO_PT[scen]) if first else "",
                        esc(SR_PT[sr]) if (li == 0 and k == 0) else "",
                        esc(LLM_PT[llm]) if k == 0 else "",
                        name,
                    ]
                    for m in metrics:
                        try:
                            v = idx.loc[(scen, sr, llm, m), col]
                        except KeyError:
                            v = math.nan
                        cells.append(num(v, DECIMALS[m]))
                    end_llm = k == len(lines) - 1
                    rule = ""
                    if end_llm:
                        if li == len(llms) - 1:
                            last_sr = ri == len(srs) - 1
                            rule = (
                                ("midrule" if si < len(scen_list) - 1 else "")
                                if last_sr
                                else r"\cmidrule(lr){2-8}"
                            )
                        else:
                            rule = r"\cmidrule(lr){3-8}"
                    rows.append((cells, rule))
    return wrap(number, caption, _tabular("llllrrrr", head, rows), resize=True)


# ----------------------------------------------------------------------------- T2
def prompt_table(df: pd.DataFrame, number: int, caption: str) -> str:
    """Tabela 2: Prompt | LLM | Fonte (Artigo/Local) x SR x métricas."""
    metrics = list(DECIMALS)
    srs = [s for s in SR_PT if s in set(df["sr"])]
    nm = len(metrics)
    head1 = [
        r"\multicolumn{3}{c}{} & "
        + " & ".join(r"\multicolumn{%d}{c}{%s}" % (nm, SR_PT[s]) for s in srs)
        + r" \\",
        " ".join(f"\\cmidrule(lr){{{4 + i * nm}-{3 + (i + 1) * nm}}}" for i in range(len(srs))),
        "Prompt & LLM & Fonte & " + " & ".join(" & ".join(METRIC_PT[m] for m in metrics) for _ in srs)
        + r" \\",
    ]  # fmt: skip
    idx = df.set_index(["llm", "prompt", "sr", "metric"])
    rows: list[Row] = []
    llms = [m for m in LLM_PT if m != "none" and m in set(df["llm"])]
    for li, llm in enumerate(llms):
        prompts = sorted(set(df[df["llm"] == llm]["prompt"]))
        for pi, p in enumerate(prompts):
            for k, (name, col) in enumerate((("Artigo", "published"), ("Local", "local_mean"))):
                cells = [
                    esc(p) if k == 0 else "",
                    esc(LLM_PT[llm]) if (pi == 0 and k == 0) else "",
                    name,
                ]
                for s in srs:
                    for m in metrics:
                        try:
                            v = idx.loc[(llm, p, s, m), col]
                        except KeyError:
                            v = math.nan
                        cells.append(num(v, DECIMALS[m]))
                rule = ""
                if k == 1:
                    rule = (
                        "midrule"
                        if pi == len(prompts) - 1 and li < len(llms) - 1
                        else (
                            ""
                            if pi == len(prompts) - 1
                            else r"\cmidrule(lr){1-%d}" % (3 + nm * len(srs))
                        )
                    )
                rows.append((cells, rule))
    tab = _tabular("lll" + "r" * nm * len(srs), head1, rows)
    return wrap(number, caption, tab, resize=True)


# ----------------------------------------------------------------------------- efeito
def effect_table(df: pd.DataFrame, number: int, caption: str) -> str:
    """Efeito pareado: Cenário | RS | LLM | Métrica | Sem LLM | Com LLM | Δ | IC95 | p | n | alt."""
    head = [
        r"Cenário & RS & LLM & Métrica & Sem LLM & Com LLM & $\Delta$ & IC95\% & $p$ & $n$ & "
        r"expr. alt. \\"
    ]
    rows: list[Row] = []
    for rec in df.itertuples(index=False):
        d = DECIMALS[str(rec.metric)]  # type: ignore[attr-defined]
        ic = f"[{num(rec.ci_low, d)}; {num(rec.ci_high, d)}]"  # type: ignore[attr-defined]
        rows.append(
            (
                [
                    esc(SCENARIO_PT.get(str(rec.scenario), str(rec.scenario))),  # type: ignore[attr-defined]
                    esc(SR_PT.get(str(rec.sr), str(rec.sr))),  # type: ignore[attr-defined]
                    esc(LLM_PT.get(str(rec.llm), str(rec.llm))),  # type: ignore[attr-defined]
                    METRIC_PT[str(rec.metric)],  # type: ignore[attr-defined]
                    num(rec.mean_base, d),  # type: ignore[attr-defined]
                    num(rec.mean_llm, d),  # type: ignore[attr-defined]
                    num(rec.mean_diff, d),  # type: ignore[attr-defined]
                    ic,
                    num(rec.p_value, 3),  # type: ignore[attr-defined]
                    str(rec.n),  # type: ignore[attr-defined]
                    str(rec.n_expr_changed),  # type: ignore[attr-defined]
                ],
                "",
            )
        )
    return wrap(number, caption, _tabular("llllrrrrrrr", head, rows), resize=True)


def arms_table(df: pd.DataFrame, number: int, caption: str, arms: Sequence[str]) -> str:
    """Efeito do LLM (Δ e IC95) de cada braço lado a lado."""
    head = [
        r"Cenário & RS & LLM & Métrica & "
        + " & ".join(r"\multicolumn{2}{c}{%s}" % esc(a) for a in arms)
        + r" \\",
        " ".join(f"\\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}" for i in range(len(arms))),
        "& & & & " + " & ".join(r"$\Delta$ & IC95\%" for _ in arms) + r" \\",
    ]
    rows: list[Row] = []
    for rec in df.to_dict("records"):
        d = DECIMALS[str(rec["metric"])]
        cells = [
            esc(SCENARIO_PT.get(str(rec["scenario"]), str(rec["scenario"]))),
            esc(SR_PT.get(str(rec["sr"]), str(rec["sr"]))),
            esc(LLM_PT.get(str(rec["llm"]), str(rec["llm"]))),
            METRIC_PT[str(rec["metric"])],
        ]
        for a in arms:
            cells.append(num(rec.get(f"{a}__mean_diff"), d))
            cells.append(f"[{num(rec.get(f'{a}__ci_low'), d)}; {num(rec.get(f'{a}__ci_high'), d)}]")
        rows.append((cells, ""))
    return wrap(number, caption, _tabular("llll" + "rr" * len(arms), head, rows), resize=True)


def master(files: Sequence[str]) -> str:
    """``tabelas.tex`` mestre, compilável isoladamente."""
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{booktabs,graphicx}",
        r"\usepackage[landscape,margin=1.2cm]{geometry}",
        r"\begin{document}",
    ]
    for f in files:
        lines += [f"\\input{{{f}}}", r"\clearpage"]
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"
