"""T:notebook-05 — notebook de baseline local: estrutura, terminologia e execução mínima."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NB05 = NOTEBOOKS_DIR / "05_baseline_local.ipynb"
BUILDER = NOTEBOOKS_DIR / "build_05_baseline_local.py"
FORBIDDEN_TERMS = re.compile(r"\b(bra[cç]o|arms?|preset|hoje)\b", re.IGNORECASE)
REQUIRED_TERMS = (
    "formulação da função de perda",
    "condição experimental",
    "regime sem ruído",
    "equação de referência",
    "desenho experimental",
    "orçamento de busca",
    "controle positivo",
    "repetição (semente derivada)",
)
TOP_PARAMETERS = (
    "ALGORITMO",
    "CENARIOS",
    "REPETICOES",
    "SEMENTE_MESTRA",
    "FORMULACAO_PERDA",
    "ORCAMENTO",
    "LIMIAR_AJUSTE",
    "LIMIAR_ESTRUTURA",
    "EXECUTAR_FASE2_RUIDO",
    "EXECUTAR_FASE2_HIPERPARAMETROS",
    "EXECUTAR_FASE2_ALGORITMOS",
)
EXEC_TIMEOUT_S = 900


def _nb() -> nbformat.NotebookNode:
    return nbformat.read(NB05, as_version=4)  # type: ignore[no-untyped-call,unused-ignore,no-any-return]


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
    assert "Colab" not in text and "Drive" not in text and "git clone" not in text


def test_cells_define_no_scientific_logic_and_import_no_llm() -> None:
    for source in _code_cells():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            assert not isinstance(node, ast.Lambda)
            if isinstance(node, ast.Import):
                assert not any(
                    a.name.split(".")[0] in {"transformers", "torch"} for a in node.names
                )
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"transformers", "torch"}


def test_structure_and_top_parameters() -> None:
    nb = _nb()
    headers = [
        m.group(1)
        for c in nb.cells
        if c.cell_type == "markdown"
        for m in re.finditer(r"^## Etapa (\d+)", c.source, re.MULTILINE)
    ]
    assert headers == [str(n) for n in range(12)]
    params = next(c.source for c in nb.cells if c.cell_type == "code" and "ALGORITMO =" in c.source)
    for name in TOP_PARAMETERS:
        assert re.search(rf"^{name}\s*=", params, re.MULTILINE), name
    for flag in (
        "EXECUTAR_FASE2_RUIDO",
        "EXECUTAR_FASE2_HIPERPARAMETROS",
        "EXECUTAR_FASE2_ALGORITMOS",
    ):
        assert re.search(rf"^{flag}\s*=\s*False\b", params, re.MULTILINE), flag
    assert re.search(r'^ALGORITMO\s*=\s*"gplearn"', params, re.MULTILINE)
    first = _code_cells()[0]
    assert "sys.version_info" in first and "pip install -e ." in first
    assert "modulos_de_llm_importados" in first


def test_each_stage_states_objective_call_output_and_criterion() -> None:
    sections = re.split(
        r"^## Etapa \d+",
        "\n".join(c.source for c in _nb().cells if c.cell_type == "markdown"),
        flags=re.MULTILINE,
    )[1:]
    assert len(sections) == 12
    for index, section in enumerate(sections):
        if index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10):
            for word in ("**Objetivo:**", "**Chamada:**", "**Saída a conferir", "**Critério"):
                assert word in section, (index, word)


@pytest.mark.slow
def test_notebook_executes_with_minimal_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nbclient import NotebookClient

    monkeypatch.setenv("PHYSICS_LLM_SR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("PHYSICS_LLM_SR_REPO_DIR", raising=False)
    monkeypatch.setenv("PHYSICS_LLM_SR_NB05_MINIMO", "1")
    nb = _nb()
    NotebookClient(
        nb,
        timeout=EXEC_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB05.parent)}},
    ).execute()
    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors
    logs = sorted(
        tmp_path.glob("results/nb05_baseline_local/fase1_validacao_limpa/gplearn/*/study_log.jsonl")
    )
    assert len(logs) == 3  # um desenho por cenário
    assert sum(len(p.read_text(encoding="utf-8").splitlines()) for p in logs) == 3 * 2
    assert not list(tmp_path.glob("results/nb05_baseline_local/fase2_*"))  # Fase 2 desligada
