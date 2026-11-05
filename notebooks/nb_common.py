"""Utilidades compartilhadas pelos geradores dos notebooks 01 e 02 (não faz parte do pacote).

``NotebookBuilder`` acumula células e grava o ``.ipynb`` com IDs determinísticos (hash do
índice e do conteúdo), de modo que o diff é estável. ``setup_cell`` devolve o texto da célula
de preparo autossuficiente do Colab (clone/pull do GitHub, pip, Drive, caminhos): cada
notebook recebe a SUA cópia (a duplicação é intencional: cada notebook roda sozinho).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import nbformat

NBFORMAT_MINOR = 5
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
REPO_URL = "https://github.com/RodrigoSantosB/physics-llm-sr.git"
COLAB_BADGE = (
    "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/"
    "notebooks/{name}.ipynb)"
)


RUFF_CONFIG = Path(__file__).resolve().parent / "ruff.toml"


def format_code(text: str) -> str:
    """Formata o código da célula com ``ruff format`` (o repositório formata os ``.ipynb``)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--config",
            str(RUFF_CONFIG),
            "--stdin-filename",
            "cell.py",
            "-",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip("\n")


class NotebookBuilder:
    """Acumula células e grava o notebook."""

    def __init__(self, notebook: Path) -> None:
        """``notebook``: caminho do ``.ipynb`` de saída."""
        self.notebook = notebook
        self.cells: list[tuple[str, str]] = []

    def md(self, text: str) -> None:
        """Acrescenta uma célula markdown."""
        self.cells.append(("markdown", text.strip("\n")))

    def code(self, text: str) -> None:
        """Acrescenta uma célula de código (formatada com ruff)."""
        self.cells.append(("code", format_code(text.strip("\n"))))

    def build(self) -> nbformat.NotebookNode:
        """Monta o notebook a partir das células acumuladas."""
        nb = nbformat.v4.new_notebook()
        nb.nbformat_minor = NBFORMAT_MINOR
        nb.metadata = {"kernelspec": KERNELSPEC, "language_info": {"name": "python"}}
        for index, (kind, source) in enumerate(self.cells):
            make = (
                nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
            )
            cell = make(source)
            cell["id"] = hashlib.sha1(f"{index}:{source}".encode()).hexdigest()[:8]
            nb.cells.append(cell)
        nbformat.validate(nb)
        return nb

    def write(self) -> None:
        """Grava o notebook em disco."""
        nbformat.write(self.build(), self.notebook)
        print(f"gravado {self.notebook} ({len(self.cells)} células)")


def setup_cell(*, parameters: str, pip_extras: str, hf_home: bool) -> str:
    """Célula 1 de código: parâmetros do usuário + preparo autossuficiente (sem importar o pacote).

    ``parameters``: bloco de parâmetros (texto Python) colocado no topo; ``pip_extras``:
    ``""`` (CPU, sem LLM) ou ``"[hf]"``; ``hf_home``: define ``HF_HOME`` no disco local do Colab.
    """
    hf_params = (
        "# Pesos (~15 GB por modelo) no disco LOCAL do Colab. Drive só com >= 20 GB livres por modelo\n"
        "# (não recomendado: cota de 15 GB e escrita via FUSE truncam o arquivo -> 'header too large').\n"
        'HF_HOME_DIR = "/content/hf_home"\n'
        if hf_home
        else ""
    )
    hf_env = (
        "    if HF_HOME_DIR:  # se transformers/huggingface_hub já foi importado, reinicie o ambiente\n"
        '        os.environ["HF_HOME"] = HF_HOME_DIR\n'
        if hf_home
        else ""
    )
    return f'''{parameters.strip()}

# ---- Somente Colab: de onde vem o código e onde persistem os resultados -------------------
# O código vem do GitHub (disco local do Colab, rápido); o Drive só PERSISTE os resultados
# (PHYSICS_LLM_SR_PROJECT_ROOT = DRIVE_PROJECT_DIR).
REPO_URL = "{REPO_URL}"  # alternativa: pasta no Drive com .git (REPO_URL = "")
REPO_BRANCH = "main"
REPO_DIR = "/content/physics-llm-sr"
MOUNT_DRIVE = True  # monta o Drive para persistir os resultados (não para o código)
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/physics-llm-sr-runs"
{hf_params}
# Preparo autossuficiente (não importa o pacote do projeto): Drive -> clone/pull -> verificação -> pip -> path.
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
_repo = Path(REPO_DIR or "/content/physics-llm-sr")
if IN_COLAB:
    if MOUNT_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
    if REPO_URL and not _repo.is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(_repo)], check=True
        )
    elif REPO_URL and (_repo / ".git").exists():
        subprocess.run(["git", "-C", str(_repo), "pull", "--ff-only"], check=True)
    if not (_repo / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"repositório não encontrado em {{_repo}} (falta pyproject.toml). Preencha REPO_URL "
            "(GitHub) para clonar, ou aponte REPO_DIR para uma pasta com o código (Drive, com .git)."
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{{_repo}}{pip_extras}"], check=True)
    os.environ.setdefault("PHYSICS_LLM_SR_REPO_DIR", str(_repo))
    os.environ.setdefault("PHYSICS_LLM_SR_PROJECT_ROOT", DRIVE_PROJECT_DIR if MOUNT_DRIVE else "/content/physics-llm-sr-work")
{hf_env}# O .pth do pip -e não vale no kernel já aberto: garante o src no sys.path.
if (_repo / "src").is_dir() and str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")  # figuras dentro do notebook
'''
