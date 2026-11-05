"""C:fisica-desacoplada: physics/ e data/ não dependem de llm, sr nem optimization."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "physics_llm_sr"
FORBIDDEN = ("llm", "sr", "optimization")
DECOUPLED = ("physics", "data")


def _forbidden(module: str) -> bool:
    return any(
        module == f"physics_llm_sr.{p}" or module.startswith(f"physics_llm_sr.{p}.")
        for p in FORBIDDEN
    )


@pytest.mark.parametrize("package", DECOUPLED)
def test_runtime_import_does_not_load_forbidden_modules(package: str) -> None:
    code = (
        "import sys\n"
        f"import physics_llm_sr.{package}\n"
        f"bad = sorted(m for m in sys.modules if any(m == 'physics_llm_sr.' + p or "
        f"m.startswith('physics_llm_sr.' + p + '.') for p in {FORBIDDEN!r}))\n"
        "print(','.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"módulos proibidos carregados: {result.stdout.strip()} {result.stderr}"
    )


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            found.append(base)
            found.extend(f"{base}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("package", DECOUPLED)
def test_static_imports_do_not_reference_forbidden_packages(package: str) -> None:
    files = sorted((PACKAGE_ROOT / package).rglob("*.py"))
    assert files, f"nenhum arquivo em {package}/"
    for path in files:
        for module in _imported_modules(path):
            assert not _forbidden(module), f"{path.name} importa {module}"


def test_static_scan_detects_violation(tmp_path: Path) -> None:
    sample = tmp_path / "bad.py"
    sample.write_text("from physics_llm_sr.llm import x\nimport physics_llm_sr.sr.deap_runner\n")
    modules = _imported_modules(sample)
    assert [m for m in modules if _forbidden(m)]
