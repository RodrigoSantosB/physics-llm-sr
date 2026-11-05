"""Ruído sobre um dataset limpo (função pura): gaussiano de média zero ou enviesado."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np

from physics_llm_sr.config.schemas import Exp5Config, NoiseLocation
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.validation import compute_data_hash

NoiseKind = Literal["zero_mean", "biased"]
#: Fração da média do atributo usada como viés no ruído ``biased``: fonte única, a config
#: (``Exp5Config.mean_fraction``).
DEFAULT_BIAS_FRACTION = Exp5Config().mean_fraction


@dataclass(frozen=True)
class NoiseSpec:
    """Especificação do ruído.

    ``zero_mean``: ``N(0, (level*std)^2)``. ``biased``: ``N(bias_fraction*mean, (level*std)^2)``
    com média e desvio do atributo limpo perturbado.
    """

    kind: NoiseKind
    level: float
    location: NoiseLocation
    bias_fraction: float = DEFAULT_BIAS_FRACTION

    def __post_init__(self) -> None:
        """Valida kind, level e location."""
        if self.kind not in get_args(NoiseKind):
            raise ValueError(f"kind inválido: {self.kind!r}")
        if not self.level >= 0.0:
            raise ValueError(f"level deve ser >= 0; recebido {self.level}")
        if self.location not in get_args(NoiseLocation):
            raise ValueError(f"location inválida: {self.location!r}")


def _perturb(values: np.ndarray[Any, Any], spec: NoiseSpec, rng: np.random.Generator) -> Any:
    """Soma ruído a ``values`` (1-D); o original não é modificado."""
    bias = spec.bias_fraction * float(np.mean(values)) if spec.kind == "biased" else 0.0
    sigma = spec.level * float(np.std(values))
    return values + bias + sigma * rng.standard_normal(values.shape)


def apply_noise(dataset: Dataset, spec: NoiseSpec, rng: np.random.Generator) -> Dataset:
    """Devolve novo ``Dataset`` com ruído; ``y_observed`` parte sempre de ``y_true``.

    ``simulation_ids``, ``t`` (identificador) e ``y_true`` nunca são perturbados. Em
    ``inputs``/``both`` todas as colunas de ``X`` (inclusive o tempo como feature) recebem
    ruído; ordem do consumo do ``rng``: colunas de X, depois o alvo.
    """
    x = dataset.X.copy()
    y_obs = dataset.y_true.copy()
    if spec.location in ("inputs", "both"):
        for j in range(x.shape[1]):
            x[:, j] = _perturb(dataset.X[:, j], spec, rng)
    if spec.location in ("target", "both"):
        y_obs = _perturb(dataset.y_true, spec, rng)
    meta = dict(dataset.metadata)
    meta["noise"] = {
        "kind": spec.kind,
        "level": spec.level,
        "location": spec.location,
        "bias_fraction": spec.bias_fraction,
    }
    out = dataclasses.replace(dataset, X=x, y_observed=y_obs, metadata=meta)
    out.metadata["data_hash"] = compute_data_hash(out)
    return out
