"""Dados por célula: limpo por repetição → ruído da condição → split (GAP-09, GAP-12).

Sementes derivadas (todas de ``seeds.master``, rastreáveis pelo nome):

- dados limpos: ``data/<cenário>/rep<r>`` (cada repetição regenera os dados, GAP-09);
- ruído: ``noise/<cenário>/rep<r>/<tipo>/<localização>/<nível>`` aplicado ao MESMO
  dataset limpo da repetição (Exp4/Exp5 partem do mesmo conjunto limpo);
- split: ``split/<cenário>/rep<r>`` (por ``simulation_id``; igual em todas as condições
  de ruído da repetição, pois o ruído não altera os ids);
- busca: ``search/<cenário>/<algoritmo>/rep<r>`` — a mesma para baseline e LLM
  (pareamento) e para todas as condições de ruído.
"""

from __future__ import annotations

from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.noise import NoiseSpec, apply_noise
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.experiments.cells import BIASED, Cell, NoiseCondition
from physics_llm_sr.reproducibility.seeds import derive_rng, derive_seed


def data_seed_name(scenario: str, repetition: int) -> str:
    """Nome da semente dos dados limpos."""
    return f"data/{scenario}/rep{repetition}"


def noise_seed_name(scenario: str, repetition: int, noise: NoiseCondition) -> str:
    """Nome da semente do ruído da condição."""
    return (
        f"noise/{scenario}/rep{repetition}/{noise.noise_type}/{noise.noise_location}/"
        f"{noise.noise_level:g}"
    )


def split_seed_name(scenario: str, repetition: int) -> str:
    """Nome da semente do split."""
    return f"split/{scenario}/rep{repetition}"


def search_seed_name(scenario: str, sr: str, repetition: int) -> str:
    """Nome da semente da busca SR (compartilhada por baseline e LLM)."""
    return f"search/{scenario}/{sr}/rep{repetition}"


def search_seed(cfg: ExperimentConfig, cell: Cell) -> int:
    """Semente da busca da célula."""
    return derive_seed(cfg.seeds.master, search_seed_name(cell.scenario, cell.sr, cell.repetition))


class DataProvider:
    """Gera e memoriza (última chave) os dados das células, na ordem de execução."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        """Guarda a config; nada é gerado até o primeiro pedido."""
        self.cfg = cfg
        self._clean_key: tuple[str, int] | None = None
        self._clean: Dataset | None = None
        self._parts_key: tuple[str, int, NoiseCondition] | None = None
        self._parts: tuple[Dataset, dict[str, Dataset]] | None = None

    def clean(self, scenario: str, repetition: int) -> Dataset:
        """Dataset limpo (``y_observed == y_true``) da repetição."""
        key = (scenario, repetition)
        if self._clean_key != key or self._clean is None:
            self._clean = generate_dataset(
                self.cfg, scenario, seed_name=data_seed_name(scenario, repetition), noisy=False
            )
            self._clean_key = key
        return self._clean

    def noisy(self, scenario: str, repetition: int, noise: NoiseCondition) -> Dataset:
        """Dataset da condição de ruído (ruído aplicado ao conjunto inteiro antes do split)."""
        clean = self.clean(scenario, repetition)
        if noise.noise_level == 0.0:
            return clean
        spec = NoiseSpec(
            "biased" if noise.noise_type == BIASED else "zero_mean",
            noise.noise_level,
            noise.noise_location,  # type: ignore[arg-type]
            bias_fraction=self.cfg.data.exp5.mean_fraction,
        )
        rng = derive_rng(self.cfg.seeds.master, noise_seed_name(scenario, repetition, noise))
        return apply_noise(clean, spec, rng)

    def for_cell(self, cell: Cell) -> tuple[Dataset, dict[str, Dataset]]:
        """``(dataset completo, {train, validation, test})`` da célula."""
        key = (cell.scenario, cell.repetition, cell.noise)
        if self._parts_key != key or self._parts is None:
            full = self.noisy(cell.scenario, cell.repetition, cell.noise)
            rng = derive_rng(self.cfg.seeds.master, split_seed_name(cell.scenario, cell.repetition))
            self._parts = (full, split_by_simulation(full, self.cfg.data.split, rng))
            self._parts_key = key
        return self._parts
