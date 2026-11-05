"""Sementes: uma semente mestre gera sementes derivadas estáveis por nome."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random

import numpy as np

#: Sementes ficam em [0, 2**32) para serem aceitas por numpy legado, sklearn e gplearn.
SEED_BITS = 32
SEED_UPPER_BOUND = 2**SEED_BITS


def derive_seed(master: int, name: str) -> int:
    """Deriva uma semente estável (independente de processo/plataforma) a partir do nome."""
    digest = hashlib.sha256(f"{master}:{name}".encode()).digest()
    return int(int.from_bytes(digest[:8], "big") % SEED_UPPER_BOUND)


def derive_rng(master: int, name: str) -> np.random.Generator:
    """Gerador numpy independente, semeado com a semente derivada de ``name``."""
    return np.random.default_rng(np.random.SeedSequence(derive_seed(master, name)))


def seed_everything(seed: int) -> None:
    """Fixa ``random``, numpy (API legada) e, se instalado, torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
