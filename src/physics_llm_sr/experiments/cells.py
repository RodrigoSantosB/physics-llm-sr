"""Célula experimental: a unidade de execução, checkpoint e retomada do runner.

Uma célula é (cenário × algoritmo × LLM × prompt × condição de ruído × repetição). A chave
:meth:`Cell.key` é determinística (sem tempo nem aleatoriedade): a mesma config gera as
mesmas chaves, e a retomada pula as células cuja chave já está concluída.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from physics_llm_sr.config.schemas import ExperimentConfig

#: Rótulo de LLM e prompt das células sem LLM (baseline, ``w3 = 0``).
NONE = "none"
ZERO_MEAN = "zero_mean"
BIASED = "biased"
#: Casas usadas para formatar o nível de ruído na chave.
LEVEL_KEY_DECIMALS = 4
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class NoiseCondition:
    """Ruído aplicado ao dataset limpo da repetição (GAP-12)."""

    noise_type: str
    noise_location: str
    noise_level: float

    def label(self) -> str:
        """Rótulo estável ``tipo-localização-nível``."""
        return f"{self.noise_type}-{self.noise_location}-{self.noise_level:.{LEVEL_KEY_DECIMALS}f}"


def default_noise(cfg: ExperimentConfig) -> NoiseCondition:
    """Ruído padrão do artigo (p.6): média zero, ``sigma_fraction`` do desvio, ``location``."""
    return NoiseCondition(ZERO_MEAN, cfg.data.noise.location, cfg.data.noise.sigma_fraction)


def _safe(text: str) -> str:
    return _UNSAFE.sub("_", text)


@dataclass(frozen=True)
class Cell:
    """Uma configuração × repetição. Baseline: ``llm == prompt == 'none'``."""

    scenario: str
    sr: str
    llm: str
    prompt: str
    noise: NoiseCondition
    repetition: int

    @property
    def is_baseline(self) -> bool:
        """Sem LLM na perda (``w3 = 0``)."""
        return self.llm == NONE

    def key(self) -> str:
        """Chave determinística (também nome de diretório)."""
        parts = (
            self.scenario,
            self.sr,
            self.llm,
            self.prompt,
            self.noise.label(),
            f"r{self.repetition:03d}",
        )
        return "__".join(_safe(p) for p in parts)

    def baseline(self) -> Cell:
        """Célula baseline pareada (mesmos dados, split e semente de busca)."""
        return dataclasses.replace(self, llm=NONE, prompt=NONE)

    def sort_key(self) -> tuple[object, ...]:
        """Ordem de execução: baselines primeiro, depois um bloco contíguo por LLM.

        Agrupar por LLM antes de SR/cenário mantém no máximo um modelo residente (o runner
        descarrega o backend anterior ao trocar de LLM). Dentro de cada bloco a ordem é por
        dados (cenário, repetição, ruído) e SR. O pareamento baseline × LLM não depende da
        ordem: dados, split e semente de busca são derivados da chave da célula.
        """
        n = self.noise
        return (
            not self.is_baseline,
            self.llm,
            self.scenario,
            self.repetition,
            n.noise_type,
            n.noise_location,
            n.noise_level,
            self.sr,
            self.prompt,
        )


def llm_cell(
    scenario: str, sr: str, llm: str, prompt: str, noise: NoiseCondition, repetition: int
) -> Cell:
    """Atalho para uma célula com LLM."""
    return Cell(scenario, sr, llm, prompt, noise, repetition)


def selected_models(cfg: ExperimentConfig) -> list[str]:
    """Modelos da matriz: ``experiments.models`` ou, se vazio, todos de ``llm.models``."""
    return list(cfg.experiments.models) or [m.name for m in cfg.llm.models]
