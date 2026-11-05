"""T:parser-neg - parse_infix rejeita construções fora da whitelist sem executar nada."""

from __future__ import annotations

import builtins
import re
from pathlib import Path

import pytest

import physics_llm_sr.symbolic as symbolic_pkg
from physics_llm_sr.symbolic.expression import (
    MAX_INFIX_LENGTH,
    ExpressionError,
    UnsafeExpressionError,
    from_sympy,
    parse_infix,
)

FEATURES = ("mass", "length", "height", "damping", "t")

SENTINEL_CALLS: list[str] = []


def _sentinel(*args: object) -> int:
    SENTINEL_CALLS.append(repr(args))
    return 0


HOSTILE = [
    "__import__('os').system('x')",
    "().__class__",
    "().__class__.__bases__[0].__subclasses__()",
    "lambda: 0",
    "(lambda x: x)(1)",
    "t.real",
    "mass[0]",
    "[t][0]",
    "velocity",  # nome não permitido
    "'abc'",
    "b'abc'",
    "f'{t}'",
    "sentinel_fn(1)",
    "exp(t, base=2)",
    "exp(*[t])",
    "t if mass else length",
    "t == mass",
    "t and mass",
    "t ^ 2",
    "t // 2",
    "t % 2",
    "~t",
    "not t",
    "{t: 1}",
    "{t}",
    "(t, mass)",
    "[t for t in (1,)]",
    "(x := 1)",
    "1j",
    "True",
    "None",
    "...",
    "exp",  # nome de operador sem chamada
    "_hidden",
    "__builtins__",
    "eval('1')",
    "exec('1')",
    "getattr(t, 'real')",
    "open('/etc/passwd')",
]


@pytest.fixture(autouse=True)
def _install_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    SENTINEL_CALLS.clear()
    monkeypatch.setattr(builtins, "sentinel_fn", _sentinel, raising=False)
    original_import = builtins.__import__

    def _spy_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "os" and args and args[0] is None:
            SENTINEL_CALLS.append("import os")
        return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _spy_import)


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_input_rejected_without_execution(text: str) -> None:
    with pytest.raises(ExpressionError):
        parse_infix(text, variables=FEATURES)
    assert SENTINEL_CALLS == []


@pytest.mark.parametrize(
    "text",
    ["__import__('os').system('x')", "().__class__", "lambda: 0", "t.real", "mass[0]", "'s'"],
)
def test_structural_violations_are_unsafe_errors(text: str) -> None:
    with pytest.raises(UnsafeExpressionError):
        parse_infix(text, variables=FEATURES)


def test_name_not_in_allowed_variables_rejected() -> None:
    with pytest.raises(UnsafeExpressionError):
        parse_infix("mass + velocity", variables=FEATURES)


def test_syntax_error_and_oversized_input_rejected() -> None:
    with pytest.raises(ExpressionError):
        parse_infix("t +", variables=FEATURES)
    with pytest.raises(ExpressionError):
        parse_infix("t" + "+t" * MAX_INFIX_LENGTH, variables=FEATURES)


def test_deeply_nested_input_rejected_cleanly() -> None:
    text = "exp(" * 2000 + "t" + ")" * 2000
    with pytest.raises(ExpressionError):
        parse_infix(text, variables=FEATURES)


def test_non_string_inputs_rejected() -> None:
    with pytest.raises(ExpressionError):
        parse_infix(b"t", variables=FEATURES)  # type: ignore[arg-type]
    with pytest.raises(ExpressionError):
        from_sympy("t + 1")  # string NUNCA vai para sympify


def test_symbolic_package_has_no_eval_exec_or_sympify() -> None:
    pkg_dir = Path(symbolic_pkg.__file__).resolve().parent
    pattern = re.compile(r"(?<![\w.])(eval|exec|sympify|compile)\s*\(|parse_expr|__import__")
    offenders = []
    for path in sorted(pkg_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert offenders == []
