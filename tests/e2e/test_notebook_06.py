"""T:notebook-06 — busca de convergência: estrutura, terminologia e execução mínima."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NB06 = NOTEBOOKS_DIR / "06_busca_de_convergencia.ipynb"
BUILDER = NOTEBOOKS_DIR / "build_06_busca_convergencia.py"
FORBIDDEN_TERMS = re.compile(r"\b(bra[cç]o|arms?|preset|hoje)\b", re.IGNORECASE)
REQUIRED_TERMS = (
    "equação de referência",
    "forma equivalente",
    "critério de convergência",
    "configuração candidata",
    "desvio declarado",
    "formulação da função de perda",
    "diagnóstico de expressividade",
    "sementes novas",
    "notebook 05",
)
TOP_PARAMETERS = (
    "ALGORITMO",
    "CENARIO",
    "CRITERIO",
    "ORCAMENTO_MAXIMO",
    "REPETICOES",
    "REPETICOES_CONFIRMACAO",
    "SEMENTE_MESTRA",
    "DESLOCAMENTO_SEMENTE",
    "PARALELO",
)
STAGES = (
    "Ambiente",
    "Parâmetros",
    "Diagnóstico de expressividade",
    "Triagem",
    "Refinamento",
    "Confirmação em sementes novas",
    "Exportação",
)
EXEC_TIMEOUT_S = 900


def _nb() -> nbformat.NotebookNode:
    return nbformat.read(NB06, as_version=4)  # type: ignore[no-untyped-call,unused-ignore,no-any-return]


def _code_cells() -> list[str]:
    return [c.source for c in _nb().cells if c.cell_type == "code"]


def test_notebook_matches_its_generator() -> None:
    spec = importlib.util.spec_from_file_location(BUILDER.stem, BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = module.build()
    assert [(c.cell_type, c.source) for c in built.cells] == [
        (c.cell_type, c.source) for c in _nb().cells
    ]


def test_uses_scientific_terminology_only() -> None:
    text = "\n".join(c.source for c in _nb().cells)
    assert not FORBIDDEN_TERMS.findall(text), FORBIDDEN_TERMS.findall(text)
    for term in REQUIRED_TERMS:
        assert term in text, term


def test_cells_define_no_logic_and_import_no_llm() -> None:
    for source in _code_cells():
        for node in ast.walk(ast.parse(source)):
            assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            assert not isinstance(node, ast.Lambda)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"transformers", "torch"}


def test_structure_stages_and_top_parameters() -> None:
    nb = _nb()
    headers = [
        m.group(1, 2)
        for c in nb.cells
        if c.cell_type == "markdown"
        for m in re.finditer(r"^## Etapa (\d+) — (.+)$", c.source, re.MULTILINE)
    ]
    assert [int(n) for n, _ in headers] == list(range(len(STAGES)))
    for (_, title), stage in zip(headers, STAGES, strict=True):
        assert title.startswith(stage), (title, stage)
    params = next(c.source for c in nb.cells if c.cell_type == "code" and "ALGORITMO =" in c.source)
    for name in TOP_PARAMETERS:
        assert re.search(rf"^{name}\s*=", params, re.MULTILINE), name
    sections = re.split(
        r"^## Etapa \d+",
        "\n".join(c.source for c in nb.cells if c.cell_type == "markdown"),
        flags=re.MULTILINE,
    )[1:]
    for section in sections:
        for word in ("**Objetivo:**", "**Chamada:**", "**Saída a conferir", "**Critério"):
            assert word in section, word


@pytest.mark.slow
def test_notebook_executes_with_minimal_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nbclient import NotebookClient

    monkeypatch.setenv("PHYSICS_LLM_SR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("PHYSICS_LLM_SR_REPO_DIR", raising=False)
    monkeypatch.setenv("PHYSICS_LLM_SR_NB06_MINIMO", "1")
    nb = _nb()
    NotebookClient(
        nb,
        timeout=EXEC_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB06.parent)}},
    ).execute()
    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors, errors
    root = tmp_path / "results/convergence_search/gplearn/free_fall"
    for name in ("best_config.yaml", "best_config.json", "search_table.csv"):
        assert (root / name).exists(), name
    assert (root / "triagem" / "study_log.jsonl").exists()
    assert (root / "confirmacao" / "study_log.jsonl").exists()
