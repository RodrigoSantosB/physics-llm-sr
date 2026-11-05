# ruff: noqa: E501
"""C:notebook (T:notebook): os três notebooks só chamam ``physics_llm_sr`` e executam em smoke.

(a) verificação estática via AST de cada célula de código (sem ``def``/``class``/``lambda`` de
lógica científica); notebook 03: 19 etapas (00-18) com os 4 componentes do PRD §10;
notebooks 01 (baseline sem LLM), 02 (um LLM, um caso) e 03 (pipeline completo): preparo
autossuficiente antes de importar o pacote, parâmetros no topo, independência entre si;
(b) execução de ponta a ponta em smoke (nbclient), sem LLM real.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOK = NOTEBOOKS_DIR / "03_pipeline_completo.ipynb"  # notebook 03 (etapas 00-18)
NB01 = NOTEBOOKS_DIR / "01_baseline_sem_llm.ipynb"
NB02 = NOTEBOOKS_DIR / "02_um_llm_um_caso.ipynb"
NB04 = NOTEBOOKS_DIR / "04_comparacao_com_artigo.ipynb"
BUILDERS = {
    NB01: "build_01_baseline.py",
    NB02: "build_02_um_llm.py",
    NOTEBOOK: "build_03_pipeline_completo.py",
    NB04: "build_04_comparacao_com_artigo.py",
}
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
    assert re.search(r'^PRESET\s*=.*"hoje"\)', first, re.MULTILINE)
    for name in ("MODELOS", "CENARIOS", "EXPERIMENTOS", "ALGORITMOS", "REPETICOES", "PROMPTS"):
        assert re.search(rf"^{name}\s*=\s*None", first, re.MULTILINE), name
    assert re.search(
        r'^REPO_URL\s*=\s*"https://github.com/RodrigoSantosB/physics-llm-sr.git"',
        first,
        re.MULTILINE,
    )
    assert re.search(r'^REPO_BRANCH\s*=\s*"main"', first, re.MULTILINE)
    assert re.search(r'^REPO_DIR\s*=\s*"/content/physics-llm-sr"', first, re.MULTILINE)
    assert re.search(r"^MOUNT_DRIVE\s*=\s*True", first, re.MULTILINE)
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
    tokens = (
        "drive.mount(",
        '"git", "clone"',
        '"pull"',
        "is_file()",
        '"pip", "install"',
        "sys.path.insert",
    )
    order = [src.index(token) for token in tokens]
    assert order == sorted(order), order
    assert "raise FileNotFoundError" in src and "REPO_URL" in src
    assert re.search(r'^HF_HOME_DIR\s*=\s*"/content/hf_home"', src, re.MULTILINE)


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
    """Cada .ipynb é gerado por seu notebooks/build_*.py (conteúdo das células idêntico)."""
    import importlib.util

    for notebook, builder in BUILDERS.items():
        path = NOTEBOOKS_DIR / builder
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        built = module.build()
        current = nbformat.read(notebook, as_version=4)  # type: ignore[no-untyped-call,unused-ignore]
        assert [(c.cell_type, c.source) for c in built.cells] == [
            (c.cell_type, c.source) for c in current.cells
        ], notebook.name


# ------------------------------------------------------------------ notebooks 01, 02 e 03
ALL_NOTEBOOKS = (NB01, NB02, NOTEBOOK, NB04)
OWN_TAG = {NB01: "nb01", NB02: "nb02", NOTEBOOK: "nb03", NB04: "nb04"}
TOP_PARAMETERS = {
    NB01: ("CENARIOS", "ALGORITMOS", "REPETICOES", "NIVEIS_RUIDO", "LOCALIZACOES", "BRACO"),
    NB02: (
        "CENARIO",
        "EXPERIMENTO",
        "LLM",
        "ALGORITMOS",
        "REPETICOES",
        "PROMPT",
        "PROMPTS",
        "BRACO",
        "NIVEIS_RUIDO",
        "LOCALIZACOES",
    ),
    NOTEBOOK: ("PRESET", "MODELOS", "CENARIOS", "EXPERIMENTOS", "ALGORITMOS", "REPETICOES"),
    NB04: ("RUN_DIRS", "OUT_DIR"),
}
SCIENCE_LITERALS = re.compile(
    r'["\'](free_fall|harmonic_motion|damped_wave|mistral|llama2|falcon)["\']'
)


def _read(path: Path) -> Any:
    return nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call,unused-ignore]


def _code_cells(path: Path) -> list[str]:
    return [str(c.source) for c in _read(path).cells if c.cell_type == "code"]


def _imports(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_each_notebook_has_self_contained_setup_before_importing_the_package(path: Path) -> None:
    """A 1ª célula de código (parâmetros + preparo) vem antes de qualquer import do pacote."""
    cells = _code_cells(path)
    first = cells[0]
    assert not any(m.startswith("physics_llm_sr") for m in _imports(first))
    tokens = (
        "drive.mount(",
        '"git", "clone"',
        '"pull"',
        "is_file()",
        '"pip", "install"',
        "sys.path.insert",
    )
    order = [first.index(token) for token in tokens]
    assert order == sorted(order), path.name
    assert "https://github.com/RodrigoSantosB/physics-llm-sr.git" in first
    first_package_use = next(
        i
        for i, src in enumerate(cells)
        if any(m.startswith("physics_llm_sr") for m in _imports(src))
    )
    assert first_package_use >= 1


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_each_notebook_has_parameters_at_the_top(path: Path) -> None:
    first = _code_cells(path)[0]
    for name in TOP_PARAMETERS[path]:
        assert re.search(rf"^{name}\s*=", first, re.MULTILINE), f"{path.name}: falta {name}"
    for cell in _read(path).cells[:1]:
        assert cell.cell_type == "markdown"  # a explicação vem antes dos parâmetros


@pytest.mark.parametrize("path", (NB01, NB02), ids=lambda p: p.name)
def test_no_science_literals_after_the_parameter_cell(path: Path) -> None:
    """Nada fixo no meio: cenários e LLMs só aparecem na célula de parâmetros."""
    for idx, src in enumerate(_code_cells(path)[1:], start=1):
        assert not SCIENCE_LITERALS.search(src), f"{path.name}: célula de código {idx}"


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_define_no_scientific_logic_and_import_only_allowed_modules(path: Path) -> None:
    allowed = {
        "physics_llm_sr", "pathlib", "json", "os", "sys", "pandas", "numpy", "matplotlib",
        "IPython", "importlib", "textwrap", "subprocess", "google",
    }  # fmt: skip
    for idx, src in enumerate(_code_cells(path)):
        for node in ast.walk(ast.parse(src)):
            assert not isinstance(node, (ast.ClassDef, ast.Lambda)), f"{path.name}[{idx}]"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name in ALLOWED_HELPERS, f"{path.name}[{idx}] def {node.name}"
        assert {m.split(".")[0] for m in _imports(src)} <= allowed, f"{path.name}[{idx}]"


def test_notebook_01_never_imports_transformers_or_torch() -> None:
    text = "\n".join(_code_cells(NB01))
    for src in _code_cells(NB01):
        modules = {m.split(".")[0] for m in _imports(src)}
        assert not modules & {"transformers", "torch", "huggingface_hub", "accelerate"}
    assert "[hf]" not in text and "HF_HOME" not in text and "HF_TOKEN" not in text
    assert "environment_report(include_cuda=False)" in text  # não importa o torch do ambiente
    cfg_text = (NOTEBOOKS_DIR.parent / "configs" / "nb01_baseline.yaml").read_text(encoding="utf-8")
    assert "baseline_only: true" in cfg_text and "backend: fake" in cfg_text


def test_notebook_01_has_one_resumable_training_cell_per_algorithm() -> None:
    for algorithm in ("deap", "gplearn", "pysr"):
        cells = [
            c
            for c in _code_cells(NB01)
            if f'algorithms=["{algorithm}"]' in c and "run_notebook_stage" in c
        ]
        assert len(cells) == 1, algorithm
    first = _code_cells(NB01)[0]
    assert re.search(r"^REPETICOES\s*=\s*3\b.*25", first, re.MULTILINE)
    assert re.search(r'^BRACO\s*=\s*"literal"', first, re.MULTILINE)


def test_notebook_02_defaults_and_steps() -> None:
    first = _code_cells(NB02)[0]
    for pattern in (
        r'^CENARIO\s*=\s*"free_fall"',
        r'^EXPERIMENTO\s*=\s*"benchmark"',
        r'^LLM\s*=\s*"mistral"',
        r"^REPETICOES\s*=\s*3\b",
        r'^PROMPT\s*=\s*"E"',
        r'^BRACO\s*=\s*"normalized"',
    ):
        assert re.search(pattern, first, re.MULTILINE), pattern
    md = "\n".join(c.source for c in _read(NB02).cells if c.cell_type == "markdown")
    assert "GAP-02" in md and "HF_TOKEN" in md and "A100" in md and "Wilcoxon" in md
    code = "\n".join(_code_cells(NB02))
    for needle in (
        'arm="baseline"',
        'arm="llm"',
        "measure_llm_latency",
        "probe_expressions",
        "read_cache_records",
        "paired_effect",
        "loss_decomposition",
        "figure8",
        "cost_summary",
        "legacy_export",
        "ensure_hf_access",
    ):
        assert needle in code, needle


@pytest.mark.parametrize(
    "path", (NB01, NB02, NOTEBOOK), ids=lambda p: p.name
)  # o 04 lê runs do 01/02
def test_notebooks_are_independent_and_use_only_their_own_directories(path: Path) -> None:
    """Cada notebook grava só em ``results/nbXX*`` próprio e não referencia pastas dos outros."""
    own = OWN_TAG[path]
    text = "\n".join(c.source for c in _read(path).cells)
    for other in set(OWN_TAG.values()) - {own}:
        assert not re.search(
            rf"{other}[_/]|results/{other}|configs_derivadas/{other}|llm_{other}", text
        ), f"{path.name} referencia {other}"
    for other_nb in (NB01, NB02, NOTEBOOK, NB04):
        if other_nb != path:
            assert other_nb.name not in text, f"{path.name} referencia {other_nb.name}"


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_have_open_in_colab_badge(path: Path) -> None:
    text = "\n".join(c.source for c in _read(path).cells if c.cell_type == "markdown")
    assert (
        "https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/"
        f"notebooks/{path.name}"
    ) in text


@pytest.mark.slow
@pytest.mark.parametrize("path", (NB01, NB02), ids=lambda p: p.name)
def test_notebooks_01_and_02_execute_in_smoke(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nbclient import NotebookClient

    monkeypatch.setenv("PHYSICS_LLM_SR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("PHYSICS_LLM_SR_REPO_DIR", raising=False)
    monkeypatch.setenv("PHYSICS_LLM_SR_NOTEBOOK_SMOKE", "1")
    nb = _read(path)
    NotebookClient(
        nb,
        timeout=EXEC_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors
    runs = list(tmp_path.glob(f"results/{OWN_TAG[path]}_*/**/notebook_run/results.parquet"))
    assert len(runs) == 1, runs
    assert not list(tmp_path.glob("results/nb0[!" + OWN_TAG[path][-1] + "]*"))


@pytest.mark.slow
def test_notebook_executes_in_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbclient import NotebookClient

    monkeypatch.setenv("PHYSICS_LLM_SR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("PHYSICS_LLM_SR_REPO_DIR", raising=False)
    monkeypatch.setenv("PHYSICS_LLM_SR_NOTEBOOK_PRESET", "smoke")
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
    runs = list(
        tmp_path.glob("results/nb03_notebook_smoke/replication/notebook_run/results.parquet")
    )
    assert runs, "o notebook smoke deveria ter persistido results.parquet"
    assert os.environ.get("PHYSICS_LLM_SR_PROJECT_ROOT") == str(tmp_path)
