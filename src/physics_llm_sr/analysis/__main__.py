# ruff: noqa: E501
"""``python -m physics_llm_sr.analysis --runs <dir1> <dir2> ... --out <dir>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from physics_llm_sr.analysis.pipeline import run_analysis


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da linha de comando."""
    parser = argparse.ArgumentParser(prog="physics_llm_sr.analysis", description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="diretórios de runs")
    parser.add_argument(
        "--out", type=Path, default=None, help="saída (padrão results/analysis/<ts>)"
    )
    args = parser.parse_args(argv)
    out = run_analysis(args.runs, args.out)
    print(f"análise gerada em {out} (veja {out / 'INDEX.md'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
