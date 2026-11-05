"""C:fisica-gt: valores de referência calculados à mão e domínio de validade."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physics_llm_sr.config.schemas import PhysicsConfig
from physics_llm_sr.physics import get_scenario
from physics_llm_sr.physics.damped_wave import damped_wave_field
from physics_llm_sr.physics.free_fall import free_fall_velocity
from physics_llm_sr.physics.harmonic_motion import harmonic_displacement

TOL = 1e-9
COLUMNS = ("mass", "length", "height", "damping", "t")


def test_free_fall_hand_value() -> None:
    assert free_fall_velocity(5.0) == pytest.approx(9.904544411531507, abs=TOL)
    np.testing.assert_allclose(free_fall_velocity(np.array([0.0, 5.0])), [0.0, 9.904544411531507])


def test_free_fall_energy_conservation() -> None:
    h = np.linspace(0.0, 100.0, 11)
    v = free_fall_velocity(h)
    np.testing.assert_allclose(v**2, 2 * 9.81 * h, atol=TOL)


def test_harmonic_hand_values() -> None:
    assert harmonic_displacement(0.2, 1.0, 10.0, 0.0) == pytest.approx(0.2, abs=TOL)
    half_period = math.pi / math.sqrt(10.0)
    assert harmonic_displacement(0.2, 1.0, 10.0, half_period) == pytest.approx(-0.2, abs=TOL)
    # quarto de período: cruza zero
    assert harmonic_displacement(0.2, 1.0, 10.0, half_period / 2) == pytest.approx(0.0, abs=TOL)


def test_harmonic_period() -> None:
    m, k = 2.5, 10.0
    period = 2 * math.pi * math.sqrt(m / k)
    t = np.linspace(0.0, period, 7)
    x = harmonic_displacement(0.3, m, k, t)
    assert x[-1] == pytest.approx(x[0], abs=TOL)


def test_wave_hand_values() -> None:
    assert damped_wave_field(0.0, 0.0, 0.0) == pytest.approx(1.0, abs=TOL)
    # alpha=0, x=0, t=T/2 com omega=2pi -> cos(-pi) = -1
    assert damped_wave_field(0.0, 0.0, 0.5) == pytest.approx(-1.0, abs=TOL)
    # alpha=1, x=0, t=1: exp(-0.5)*cos(-2pi)
    assert damped_wave_field(0.0, 1.0, 1.0) == pytest.approx(math.exp(-0.5), abs=TOL)


def test_wave_envelope_decreasing_and_bounded() -> None:
    t = np.linspace(0.0, 1.0, 200)
    for x in (0.0, 0.13, 0.4):
        e = damped_wave_field(x, 0.8, t)
        assert np.all(np.abs(e) <= 1.0 * np.exp(-0.8 * t / 2) + TOL)
        assert np.all(np.abs(e) <= 1.0 + TOL)


@pytest.mark.parametrize(
    "call, parameter",
    [
        (lambda: free_fall_velocity(-1.0), "height"),
        (lambda: free_fall_velocity(1.0, gravity=0.0), "gravity"),
        (lambda: free_fall_velocity(float("nan")), "height"),
        (lambda: harmonic_displacement(-0.1, 1.0, 10.0, 0.0), "amplitude"),
        (lambda: harmonic_displacement(0.1, 0.0, 10.0, 0.0), "mass"),
        (lambda: harmonic_displacement(0.1, 1.0, -10.0, 0.0), "spring_constant"),
        (lambda: harmonic_displacement(0.1, 1.0, 10.0, -1.0), "t"),
        (lambda: damped_wave_field(0.0, -0.1, 0.0), "damping"),
        (lambda: damped_wave_field(0.0, 0.1, -1.0), "t"),
        (lambda: damped_wave_field(0.0, 0.1, 0.0, amplitude=0.0), "amplitude"),
    ],
)
def test_out_of_domain_raises_naming_parameter(call, parameter: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=parameter):
        call()


@pytest.mark.parametrize("name", ["free_fall", "harmonic_motion", "damped_wave"])
def test_scenario_trajectory_and_context(name: str) -> None:
    cfg = PhysicsConfig()
    scenario = get_scenario(name, cfg)
    assert scenario.feature_names == COLUMNS
    rng = np.random.default_rng(0)
    traj = scenario.simulate(rng, cfg.sampling, n_points=100, max_steps=1000)
    assert traj.X.shape == (100, 5)
    assert traj.y_true.shape == (100,)
    assert np.all(np.diff(traj.t) >= 0)
    assert np.all(np.isfinite(traj.y_true))
    np.testing.assert_allclose(scenario.reference(traj.X), traj.y_true, atol=TOL)
    # reprodutível dado o gerador
    again = scenario.simulate(np.random.default_rng(0), cfg.sampling, 100, 1000)
    np.testing.assert_array_equal(again.X, traj.X)
    ctx = scenario.physical_context()
    assert ctx.scenario == name
    assert set(ctx.variable_descriptions) >= set(COLUMNS)
    assert set(ctx.units) >= set(COLUMNS)
    assert ctx.ground_truth


def test_ground_truth_strings_evaluate_to_reference() -> None:
    cfg = PhysicsConfig()
    env = {"sqrt": np.sqrt, "exp": np.exp, "cos": np.cos, "sin": np.sin, "log": np.log}
    for name in ("free_fall", "harmonic_motion", "damped_wave"):
        scenario = get_scenario(name, cfg)
        traj = scenario.simulate(np.random.default_rng(1), cfg.sampling, 50, 1000)
        ns = {c: traj.X[:, i] for i, c in enumerate(COLUMNS)}
        # eval confiável: string produzida pelo próprio projeto, ambiente restrito
        got = eval(scenario.physical_context().ground_truth, {"__builtins__": {}}, {**env, **ns})  # noqa: S307
        np.testing.assert_allclose(got, traj.y_true, atol=TOL)
    assert get_scenario("free_fall", cfg).physical_context().ground_truth == "sqrt(2*9.81*height)"


def test_free_fall_sampling_domain() -> None:
    cfg = PhysicsConfig()
    scenario = get_scenario("free_fall", cfg)
    rng = np.random.default_rng(3)
    for _ in range(20):
        traj = scenario.simulate(rng, cfg.sampling, 100, 1000)
        h = traj.X[:, 2]
        assert h.min() >= 0.0 and h.max() <= cfg.sampling.height.high + TOL


def test_unknown_scenario_and_bad_points() -> None:
    with pytest.raises(ValueError, match="scenario"):
        get_scenario("nope", PhysicsConfig())
    cfg = PhysicsConfig()
    sc = get_scenario("harmonic_motion", cfg)
    with pytest.raises(ValueError, match="n_points"):
        sc.simulate(np.random.default_rng(0), cfg.sampling, 0, 1000)
