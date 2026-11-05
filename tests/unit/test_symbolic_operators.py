"""T:operadores - semântica protegida explícita dos operadores do registro."""

from __future__ import annotations

import numpy as np
import pytest

from physics_llm_sr.config.schemas import SymbolicConfig
from physics_llm_sr.symbolic.operators import (
    DEFAULT_PROTECTION,
    NAN_REPLACEMENT,
    OPERATORS,
    ProtectionParams,
    UnknownOperatorError,
    apply_operator,
    get_operator,
)

P = DEFAULT_PROTECTION


def _arr(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=float)


def test_default_protection_comes_from_config_defaults() -> None:
    cfg = SymbolicConfig()
    assert ProtectionParams.from_config(cfg) == DEFAULT_PROTECTION
    assert P.division_epsilon == cfg.division_epsilon
    assert P.division_fallback == cfg.division_fallback
    assert P.log_epsilon == cfg.log_epsilon
    assert P.exp_clip == cfg.exp_clip
    assert P.power_clip == cfg.power_clip


def test_registry_has_required_operators_with_arity_and_commutativity() -> None:
    expected = {
        "add": (2, True),
        "sub": (2, False),
        "mul": (2, True),
        "div": (2, False),
        "neg": (1, False),
        "pow": (2, False),
        "exp": (1, False),
        "log": (1, False),
        "sqrt": (1, False),
        "sin": (1, False),
        "cos": (1, False),
    }
    assert set(OPERATORS) == set(expected)
    for name, (arity, commutative) in expected.items():
        spec = get_operator(name)
        assert spec.name == name
        assert spec.arity == arity
        assert spec.commutative is commutative


def test_unknown_operator_rejected() -> None:
    with pytest.raises(UnknownOperatorError):
        get_operator("tan")


def test_division_by_zero_returns_fallback() -> None:
    out = apply_operator("div", _arr(3.0, -2.0, 6.0), _arr(0.0, 1e-12, 2.0), params=P)
    np.testing.assert_array_equal(out, _arr(P.division_fallback, P.division_fallback, 3.0))
    assert P.division_fallback == 1.0


def test_log_of_zero_is_zero_and_log_of_negative_uses_abs() -> None:
    out = apply_operator("log", _arr(0.0, -1.0, -np.e, 1e-12), params=P)
    np.testing.assert_allclose(out, _arr(0.0, 0.0, 1.0, 0.0))


def test_sqrt_uses_abs() -> None:
    out = apply_operator("sqrt", _arr(-4.0, 9.0, 0.0), params=P)
    np.testing.assert_array_equal(out, _arr(2.0, 3.0, 0.0))


def test_exp_argument_is_clipped() -> None:
    out = apply_operator("exp", _arr(1000.0, -1000.0, 1.0), params=P)
    np.testing.assert_allclose(out, _arr(np.exp(P.exp_clip), np.exp(-P.exp_clip), np.e))
    assert np.all(np.isfinite(out))


def test_pow_negative_base_fractional_exponent_uses_abs() -> None:
    out = apply_operator("pow", _arr(-8.0, -2.0, 4.0), _arr(1.0 / 3.0, 3.0, 0.5), params=P)
    # expoente não inteiro -> |a|**b ; expoente inteiro -> potência real com sinal
    np.testing.assert_allclose(out, _arr(2.0, -8.0, 2.0))


def test_pow_overflow_and_zero_negative_exponent_are_clipped() -> None:
    out = apply_operator("pow", _arr(10.0, 0.0, -10.0), _arr(400.0, -1.0, 401.0), params=P)
    np.testing.assert_array_equal(out, _arr(P.power_clip, P.power_clip, -P.power_clip))


def test_nonfinite_results_are_replaced() -> None:
    big = np.finfo(float).max
    mul = apply_operator("mul", _arr(big, -big), _arr(10.0, 10.0), params=P)
    np.testing.assert_array_equal(mul, _arr(P.power_clip, -P.power_clip))
    add = apply_operator("add", _arr(np.nan), _arr(1.0), params=P)
    np.testing.assert_array_equal(add, _arr(NAN_REPLACEMENT))
    sin = apply_operator("sin", _arr(np.inf), params=P)
    np.testing.assert_array_equal(sin, _arr(NAN_REPLACEMENT))


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_every_operator_is_finite_on_hostile_inputs(name: str) -> None:
    hostile = _arr(0.0, -0.0, -1.0, 1e-300, -1e300, 1e300, 1000.0, -1000.0, np.nan, np.inf)
    arity = get_operator(name).arity
    args = [hostile] * arity if arity == 1 else [hostile, hostile[::-1]]
    out = apply_operator(name, *args, params=P)
    assert out.shape == hostile.shape
    assert np.all(np.isfinite(out))


def test_arity_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        apply_operator("add", _arr(1.0), params=P)
