"""Logging simples e idempotente."""

from __future__ import annotations

import logging
from pathlib import Path

from physics_llm_sr.logging_setup import setup_logging


def test_setup_logging_is_idempotent_and_writes_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    setup_logging(log_file=log_file)
    logger = setup_logging(log_file=log_file)
    assert len(logger.handlers) == 2
    logging.getLogger("physics_llm_sr.test").info("olá")
    for h in logger.handlers:
        h.flush()
    assert "olá" in log_file.read_text(encoding="utf-8")
