"""Registro dos adaptadores SR com import preguiçoso.

O PySR (``sr/pysr_adapter.py``, classe ``PySRRegressor``) depende de Julia; importar o
runner nunca importa o PySR. Ausência do módulo, do pacote ou do Julia vira
:class:`AdapterUnavailableError` com mensagem clara, tratada pelo runner como falha da
célula (smoke/pilot continuam; full paper_replication aborta).
"""

from __future__ import annotations

import importlib

from physics_llm_sr.sr.interfaces import SymbolicRegressor

#: algoritmo -> (módulo, classe).
ADAPTERS: dict[str, tuple[str, str]] = {
    "deap": ("physics_llm_sr.sr.deap_adapter", "DeapRegressor"),
    "gplearn": ("physics_llm_sr.sr.gplearn_adapter", "GplearnRegressor"),
    "pysr": ("physics_llm_sr.sr.pysr_adapter", "PySRRegressor"),
}
_HINTS = {
    "pysr": "o PySR exige o pacote 'pysr' e um Julia funcional (veja `make test-julia`)",
}


class AdapterUnavailableError(RuntimeError):
    """Adaptador não pôde ser carregado (módulo ausente, dependência ou Julia indisponível)."""


def make_regressor(name: str) -> SymbolicRegressor:
    """Instancia o adaptador ``name`` (import preguiçoso)."""
    if name not in ADAPTERS:
        raise ValueError(f"algoritmo desconhecido: {name!r}; válidos: {sorted(ADAPTERS)}")
    module_name, class_name = ADAPTERS[name]
    hint = _HINTS.get(name, "")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        regressor: SymbolicRegressor = cls()
    except Exception as exc:  # import de Julia/juliacall pode levantar qualquer erro
        raise AdapterUnavailableError(
            f"adaptador {name!r} indisponível ({module_name}.{class_name}): "
            f"{type(exc).__name__}: {exc}. {hint}".strip()
        ) from exc
    return regressor
