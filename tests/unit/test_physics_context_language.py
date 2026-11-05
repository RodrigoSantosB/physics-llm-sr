"""T:prompts - contexto físico em inglês (os prompts são em inglês)."""

from __future__ import annotations

import re

import pytest

from physics_llm_sr.physics.registry import SCENARIO_NAMES, get_scenario

PORTUGUESE_MARKERS = re.compile(
    r"[áàâãéêíóôõúç]|\b(massa|velocidade|queda|tempo|mola|onda|amplitude da|campo|altura|"
    r"irrelevante|sem|com|desde|ao longo|prevê|objeto|solta|solto|decorrido|coeficiente de)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_physical_context_is_english(scenario: str) -> None:
    ctx = get_scenario(scenario).physical_context()
    texts = [ctx.experiment_description, *ctx.variable_descriptions.values()]
    offenders = [t for t in texts if PORTUGUESE_MARKERS.search(t)]
    assert not offenders, offenders
