"""Amostragem de simulações (etapas 2-8 do PRD 3.2): parâmetros, instantes e GT."""

from __future__ import annotations

from typing import Any

import numpy as np

from physics_llm_sr.config.schemas import DataConfig, PhysicsConfig
from physics_llm_sr.physics.base import PhysicalScenario


def sample_simulations(
    scenario: PhysicalScenario,
    physics: PhysicsConfig,
    data: DataConfig,
    rng: np.random.Generator,
) -> tuple[Any, Any, Any, Any]:
    """Simula ``data.n_simulations`` trajetórias e as concatena.

    Devolve ``(X, y_true, t, simulation_ids)``; cada simulação tem ``n_time_points`` instantes
    ordenados (teto ``max_steps``). Os ids são ``0..n_simulations-1``.
    """
    xs, ys, ts, ids = [], [], [], []
    for sim_id in range(data.n_simulations):
        traj = scenario.simulate(rng, physics.sampling, data.n_time_points, data.max_steps)
        xs.append(traj.X)
        ys.append(traj.y_true)
        ts.append(traj.t)
        ids.append(np.full(traj.t.shape[0], sim_id, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(ids)
