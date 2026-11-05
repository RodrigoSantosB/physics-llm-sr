"""C:notebook (T:notebook): o notebook principal só chama ``physics_llm_sr`` e executa em smoke.

(a) verificação estática via AST de cada célula de código (sem ``def``/``class``/``lambda`` de
lógica científica) e das 19 etapas (00-18) com os 4 componentes do PRD §10;
(b) execução de ponta a ponta em ``RUN_PROFILE="smoke"`` (nbclient), sem LLM real.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "01_full_paper_reproduction.ipynb"
#: Auxiliares de exibição triviais permitidos nas células (nenhum é lógica científica).
ALLOWED_HELPERS: frozenset[str] = frozenset({"show_table"})
STAGES = tuple(f"{n:02d}" for n in range(19))
COMPONENTS = (
    "1. Explicação teórica",
    "2. Referência ao artigo",
    "3. Chamada ao módulo",
    "4. Visualização e interpretação",
)
HEADER = re.compile(r"^##\s+Etapa\s+(\d{2})\b", re.MULTILINE)
EXEC_TIMEOUT_S = 1800


def _load() -> Any:
    return nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call,unused-ignore]


def _sections(nb: Any) -> dict[str, list[Any]]:
    """Células de cada etapa: da célula-título ``## Etapa NN`` até a seguinte."""
    sections: dict[str, list[Any]] = {}
    current: str | None = None
    for cell in nb.cells:
        match = HEADER.search(cell.source) if cell.cell_type == "markdown" else None
        if match:
            current = match.group(1)
            sections[current] = []
        if current is not None:
            sections[current].append(cell)
    return sections


def test_notebook_exists_with_run_parameters() -> None:
    nb = _load()
    first = next(c.source for c in nb.cells if c.cell_type == "code")
    assert re.search(r'^RUN_PROFILE\s*=\s*"smoke"', first, re.MULTILINE)
    assert re.search(r'^ARM\s*=\s*"replication"', first, re.MULTILINE)
    assert "LOAD_EXISTING_RUN" in first


def test_all_19_stages_with_the_four_components() -> None:
    sections = _sections(_load())
    assert tuple(sections) == STAGES
    for stage, cells in sections.items():
        text = "\n".join(c.source for c in cells)
        for component in COMPONENTS:
            assert component in text, f"etapa {stage}: falta '{component}'"
        assert any(c.cell_type == "code" for c in cells), f"etapa {stage} sem célula de código"
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        assert "physics_llm_sr" in code, f"etapa {stage} não chama o módulo Python"


def test_cells_define_no_scientific_logic() -> None:
    offenders: list[str] = []
    for idx, cell in enumerate(_load().cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.Lambda)):
                offenders.append(f"célula {idx}: {type(node).__name__}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name not in ALLOWED_HELPERS:
                    offenders.append(f"célula {idx}: def {node.name}")
    assert not offenders, offenders


def test_cells_import_only_package_and_standard_stack() -> None:
    allowed = {
        "physics_llm_sr",
        "pathlib",
        "json",
        "os",
        "sys",
        "pandas",
        "numpy",
        "matplotlib",
        "IPython",
        "importlib",
        "textwrap",
        "subprocess",
        "google",
    }
    bad: list[str] = []
    for idx, cell in enumerate(_load().cells):
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            bad += [f"célula {idx}: {n}" for n in names if n.split(".")[0] not in allowed]
    assert not bad, bad


def _first_code() -> str:
    return str(next(c.source for c in _load().cells if c.cell_type == "code"))


def test_first_setup_cell_is_self_contained_and_ordered() -> None:
    """Item 1: a 1ª célula não importa o pacote; ordem drive, clone, verificação, pip, path."""
    src = _first_code()
    assert "physics_llm_sr" not in src.replace("PHYSICS_LLM_SR_", "").replace("physics-llm-sr", "")
    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("physics_llm_sr") for name in imported)
    tokens = ("drive.mount(", '"git", "clone"', "is_file()", '"pip", "install"', "sys.path.insert")
    order = [src.index(token) for token in tokens]
    assert order == sorted(order), order
    assert "raise FileNotFoundError" in src and "REPO_URL" in src
    assert re.search(r'^HF_HOME_DIR\s*=\s*""', src, re.MULTILINE)


def test_notebook_has_numbered_colab_roadmap_and_stage18_uses_module() -> None:
    nb = _load()
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "## Roteiro Colab A100" in text
    road = text[text.index("## Roteiro Colab A100") :]
    for n in range(1, 7):
        assert f"\n{n}. " in "\n" + road
    code18 = "\n".join(c.source for c in _sections(nb)["18"] if c.cell_type == "code")
    assert "loss_term_contributions" in code18
    assert 'llm["w1·e"]' not in code18


def test_notebook_matches_its_generator() -> None:
    """O .ipynb é gerado por notebooks/build_notebook.py (conteúdo das células idêntico)."""
    import importlib.util

    path = NOTEBOOK.parent / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("build_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = module.build()
    current = _load()
    assert [(c.cell_type, c.source) for c in built.cells] == [
        (c.cell_type, c.source) for c in current.cells
    ]


@pytest.mark.slow
def test_notebook_executes_in_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbclient import NotebookClient

    monkeypatch.setenv("PHYSICS_LLM_SR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("PHYSICS_LLM_SR_REPO_DIR", raising=False)
    nb = _load()
    client = NotebookClient(
        nb,
        timeout=EXEC_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors
    runs = list(tmp_path.glob("results/notebook_smoke/replication/notebook_run/results.parquet"))
    assert runs, "o notebook smoke deveria ter persistido results.parquet"
    assert os.environ.get("PHYSICS_LLM_SR_PROJECT_ROOT") == str(tmp_path)
