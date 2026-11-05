"""Parada antecipada (PRD §4.1): melhoria relativa < ``rel_tol`` por ``patience`` gerações.

Melhoria relativa entre gerações consecutivas::

    improvement = (prev_best - new_best) / |prev_best|

Casos de borda: ``prev_best == 0`` -> melhoria 0 se ``new_best >= 0`` (nada a melhorar) e
``+inf`` se ``new_best < 0``; ``prev_best = +inf`` e ``new_best`` finito -> ``+inf``;
ambos ``+inf`` -> 0. Piora (melhoria negativa) conta como estagnação. Melhoria
``>= rel_tol`` zera o contador. A primeira chamada só registra a linha de base.
"""

from __future__ import annotations

import math

from physics_llm_sr.config.schemas import EarlyStoppingConfig

#: Defaults do PRD §4.1 (0,1% e 3 gerações).
DEFAULT_REL_TOL = 0.001
DEFAULT_PATIENCE = 3


def relative_improvement(prev_best: float, new_best: float) -> float:
    """Melhoria relativa ``(prev - new) / |prev|`` com os casos de borda documentados."""
    if math.isnan(prev_best) or math.isnan(new_best):
        raise ValueError("best_loss não pode ser NaN")
    if math.isinf(prev_best):
        return 0.0 if math.isinf(new_best) else math.inf
    if prev_best == 0.0:
        return 0.0 if new_best >= 0.0 else math.inf
    return (prev_best - new_best) / abs(prev_best)


class EarlyStopping:
    """Detector de estagnação da melhor perda composta."""

    def __init__(
        self,
        rel_tol: float = DEFAULT_REL_TOL,
        patience: int = DEFAULT_PATIENCE,
        enabled: bool = True,
    ) -> None:
        """Configura o detector (``rel_tol > 0``, ``patience >= 1``)."""
        if not rel_tol > 0:
            raise ValueError("rel_tol deve ser > 0")
        if patience < 1:
            raise ValueError("patience deve ser >= 1")
        self.rel_tol = rel_tol
        self.patience = patience
        self.enabled = enabled
        self.stagnant = 0
        self._prev: float | None = None

    @classmethod
    def from_config(cls, cfg: EarlyStoppingConfig) -> EarlyStopping:
        """Constrói a partir de ``sr.early_stopping``."""
        return cls(cfg.min_relative_improvement, cfg.patience, cfg.enabled)

    def update(self, best_loss: float) -> bool:
        """Registra a melhor perda da geração; ``True`` se a busca deve parar."""
        prev, self._prev = self._prev, best_loss
        if prev is None:
            if math.isnan(best_loss):
                raise ValueError("best_loss não pode ser NaN")
            return False
        if relative_improvement(prev, best_loss) >= self.rel_tol:
            self.stagnant = 0
        else:
            self.stagnant += 1
        return self.enabled and self.stagnant >= self.patience
