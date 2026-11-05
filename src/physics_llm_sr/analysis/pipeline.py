# ruff: noqa: E501
"""Orquestra a análise: carrega as execuções, gera tabelas, figuras e o ``INDEX.md``."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from physics_llm_sr.analysis import figures, loading, tables
from physics_llm_sr.analysis.artifacts import Artifact

DEFAULT_ROOT = Path("results") / "analysis"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"


def write_index(out: Path, data: loading.AnalysisData, arts: list[Artifact]) -> Path:
    """``INDEX.md``: cada artefato, a tabela do artigo correspondente e o rótulo de execução."""
    lines = ["# Comparação artigo × experimentos locais", "", "## Execuções", ""]
    lines += ["| braço | diretório | rótulo |", "|---|---|---|"]
    for r in data.runs:
        lines.append(f"| {r.arm} | `{r.run_dir}` | {r.label} |")
    lines += ["", "## Artefatos", "", "| id | arquivo | corresponde a | rótulo de execução |",
              "|---|---|---|---|"]  # fmt: skip
    for a in arts:
        rel = a.path.relative_to(out)
        lines.append(f"| {a.artifact_id} | [{rel}]({rel}) | {a.paper_ref} | {a.label} |")
    lines += [
        "",
        "Valores publicados vêm de `references/paper_results/*.csv` (artigo e suplemento); "
        "valores locais vêm dos arquivos persistidos das execuções. Célula sem dado local = "
        "`n/e` (não executado), nunca zero. Figuras: publicado = vazado/tracejado, local = "
        "cheio/sólido.",
        "",
    ]
    path = out / "INDEX.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analysis(run_dirs: list[Path] | list[str], out_dir: Path | str | None = None) -> Path:
    """Gera todas as tabelas e figuras; devolve o diretório de saída."""
    data = loading.load_runs(run_dirs)
    out = (
        Path(out_dir)
        if out_dir is not None
        else DEFAULT_ROOT / dt.datetime.now().strftime(TIMESTAMP_FORMAT)
    )
    out.mkdir(parents=True, exist_ok=True)
    built = tables.build_tables(data)
    arts: list[Artifact] = []
    for spec in built.values():
        arts += tables.write_table(spec, out / "tables")
    arts.append(tables.write_master_tex(built, out / "tables"))
    arts += figures.build_figures(data, built, out / "figures")
    write_index(out, data, arts)
    return out
