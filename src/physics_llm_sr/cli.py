"""Linha de comando ``physics-llm-sr``.

- ``physics-llm-sr run --config configs/X.yaml [--resume [RUN_DIR]] [--only exp1,exp2]
  [--sr deap,gplearn] [--allow-code-change]``: executa a matriz da config (checkpoint por
  célula). ``RUN_DIR`` relativo é resolvido contra ``PHYSICS_LLM_SR_PROJECT_ROOT`` (como
  ``output_dir``); retomar com código git diferente exige ``--allow-code-change``.
- ``physics-llm-sr report <run_dir>``: reconstrói tabelas, comparação com o artigo e
  figuras a partir dos arquivos persistidos (sem treinar).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from physics_llm_sr.config.loader import ConfigError, load_config
from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.experiments.runner import ExperimentAbortedError, report, run_experiments
from physics_llm_sr.logging_setup import setup_logging

#: Valor de ``--resume`` sem argumento: retomar a execução mais recente com a mesma config.
RESUME_LATEST = "latest"
EXIT_OK = 0
EXIT_CELL_ERRORS = 1
EXIT_USAGE = 2
EXIT_ABORTED = 3


def _csv(text: str | None) -> list[str] | None:
    if text is None:
        return None
    items = [t.strip() for t in text.split(",") if t.strip()]
    return items or None


def build_parser() -> argparse.ArgumentParser:
    """Parser dos subcomandos ``run`` e ``report``."""
    parser = argparse.ArgumentParser(prog="physics-llm-sr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="executa os experimentos da config")
    run.add_argument("--config", required=True, type=Path, help="arquivo YAML (configs/)")
    run.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_LATEST,
        default=None,
        metavar="RUN_DIR",
        help="retoma RUN_DIR (ou a execução mais recente com a mesma config)",
    )
    run.add_argument(
        "--allow-code-change",
        action="store_true",
        help="permite retomar com código (git) diferente; fica registrado e avisado no relatório",
    )
    run.add_argument("--only", help="experimentos: exp1..exp5 (ou nomes), separados por vírgula")
    run.add_argument("--sr", help="algoritmos SR, separados por vírgula (ex.: deap,gplearn)")
    run.add_argument("--verbose", action="store_true", help="log DEBUG")
    rep = sub.add_parser("report", help="relatório e figuras de uma execução")
    rep.add_argument("run_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada (``[project.scripts]``)."""
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
    if args.command == "report":
        info = report(args.run_dir)
        print(
            json.dumps(
                {k: info[k] for k in ("label", "n_cells", "n_ok", "n_error")},
                indent=2,
                ensure_ascii=False,
            )
        )
        return EXIT_OK
    try:
        cfg = load_config(args.config)
        run_dir = (
            None if args.resume in (None, RESUME_LATEST) else resolve_project_path(args.resume)
        )
        summary = run_experiments(
            cfg,
            run_dir=run_dir,
            resume=args.resume is not None,
            only=_csv(args.only),
            algorithms=_csv(args.sr),
            allow_code_change=args.allow_code_change,
        )
    except (ConfigError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ExperimentAbortedError as exc:
        print(f"execução abortada: {exc}", file=sys.stderr)
        return EXIT_ABORTED
    print(
        f"run_dir={summary.run_dir} mode={cfg.mode.value} track={cfg.track.value} "
        f"planejadas={summary.planned} executadas={summary.executed} "
        f"puladas={summary.skipped} falhas={summary.failed}"
    )
    for err in summary.errors:
        print(f"  ERRO {err['cell_key']}: {err['error']}", file=sys.stderr)
    return EXIT_CELL_ERRORS if summary.failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
