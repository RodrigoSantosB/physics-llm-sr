# ruff: noqa: D102, D103, D107, D403, E501, N806, N813, B905
# (protótipo descartável do spike GAP-08, mantido como documentação)
"""Spike GAP-08: termo LLM (w3*c) DENTRO da seleção do PySR via callback Julia -> Python.

Mecanismo: ``loss_function`` (objetivo completo em Julia, ``(tree, dataset, options) -> L``)
calcula ``w1*Huber + w2*nós`` e chama um callable Python guardado em ``Main.INLOOP_CB[]``
(PythonCall.Py) com ``string_tree(tree, options)``. No Python: texto -> ``parse_infix`` ->
``canonical_key`` -> cache -> ``FakeEquationEvaluator.evaluate`` -> ``c = 1 - mean(scores)``.

Uso (de dentro do worktree)::

    uv run python spikes/pysr_inloop/inloop_spike.py influence   # prova de influência + determinismo
    uv run python spikes/pysr_inloop/inloop_spike.py cost        # chamadas por iteração / overhead
    uv run python spikes/pysr_inloop/inloop_spike.py threads     # tenta parallelism="multithreading"

Saída: JSON em ``spikes/pysr_inloop/out/<modo>.json`` e resumo no stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.scoring import FakeEquationEvaluator
from physics_llm_sr.symbolic.canonicalization import canonical_key, simplify
from physics_llm_sr.symbolic.expression import Expr, Op, parse_infix

_NUM = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|[-+]?(inf|nan)")
OUT = Path(__file__).resolve().parent / "out"
SEED = 7
VARS = ["x1", "x2", "x3"]  # h (m), massa (kg, distratora), ruído (distratora)

# --------------------------------------------------------------------------------------
# Julia: objetivo completo. Pesos e callback ficam em globais de Main (Refs), assim a
# mesma definição serve para todas as configurações sem redefinir a função.
# --------------------------------------------------------------------------------------
JULIA_PRELUDE = """
using PythonCall
const INLOOP_CB = Ref{Any}(nothing)        # PythonCall.Py (callable Python)
const INLOOP_W = Ref((1/3, 1/3, 1/3))      # (w1, w2, w3)
const INLOOP_CALL_PY = Ref(true)           # false => não chama o Python (mede overhead)
const INLOOP_NCALLS = Ref(0)
set_inloop_cb!(f) = (INLOOP_CB[] = f; nothing)
set_inloop_w!(w1, w2, w3, callpy) = (INLOOP_W[] = (Float64(w1), Float64(w2), Float64(w3)); INLOOP_CALL_PY[] = Bool(callpy); INLOOP_NCALLS[] = 0; nothing)
"""

LOSS_FUNCTION = """
function inloop_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    pred, ok = eval_tree_array(tree, dataset.X, options)
    !ok && return L(Inf)
    δ = T(1.0)                                   # GAP-05: Huber, δ = 1
    e = zero(T)
    @inbounds for i in eachindex(pred)
        r = abs(pred[i] - dataset.y[i])
        e += r <= δ ? r^2 / 2 : δ * (r - δ / 2)
    end
    e /= length(pred)
    s = count_nodes(tree)                        # s = nº de nós
    w1, w2, w3 = INLOOP_W[]
    c = 0.0
    if INLOOP_CALL_PY[]
        INLOOP_NCALLS[] += 1
        txt = string_tree(tree, options)
        # PySR chama equation_search com _jl_call_nogil: o GIL está LIBERADO aqui.
        # Sem reaquisição (PyGILState_Ensure) o pycall dá segfault em PyTuple_New.
        c = PythonCall.GIL.@lock pyconvert(Float64, INLOOP_CB[](txt))
    end
    return L(w1 * e + w2 * s + w3 * c)
end
"""


# --------------------------------------------------------------------------------------
# Python: texto PySR -> Expr -> canonical_key -> cache -> avaliador -> c
# --------------------------------------------------------------------------------------
def pysr_text_to_expr(text: str) -> Expr:
    """``string_tree`` do SymbolicRegression.jl -> AST comum (``^`` -> ``**``)."""
    return parse_infix(text.replace("^", "**"), variables=VARS)


def contains_op(expr: Expr, name: str) -> bool:
    if isinstance(expr, Op):
        return expr.name == name or any(contains_op(c, name) for c in expr.children)
    return False


def make_evaluator(favored: str | None) -> FakeEquationEvaluator:
    """c baixo (escores 0.9) se a árvore contém ``favored``; c alto (escores 0.1) caso contrário."""

    def fn(expr: Expr, _ctx: Any, _variant: str) -> EquationEvaluation:
        good = favored is not None and contains_op(expr, favored)
        v = 0.9 if good else 0.1
        return EquationEvaluation(v, v, v, "fake")

    return FakeEquationEvaluator.from_function(fn)


class Callback:
    """Callable registrado em Julia. Cache por ``canonical_key``; conta chamadas."""

    def __init__(self, evaluator: FakeEquationEvaluator, score_simplified: bool = False) -> None:
        self.evaluator = evaluator
        self.score_simplified = score_simplified
        self.cache: dict[str, float] = {}
        self.calls = 0
        self.unique_text: set[str] = set()
        self.parse_errors: Counter[str] = Counter()
        self.keys_3sig: set[str] = set()  # canonical_key com 3 algarismos significativos
        self.keys_struct: set[str] = set()  # só estrutura (constantes mascaradas)
        self.seconds = 0.0

    def __call__(self, text: Any) -> float:
        t0 = time.perf_counter()
        text = str(text)
        self.calls += 1
        self.unique_text.add(text)
        try:
            expr = pysr_text_to_expr(text)
            if self.score_simplified:  # dobra constantes: cos(0.3) deixa de "ter cos"
                expr = simplify(expr)
        except Exception as exc:  # noqa: BLE001 - spike: contabiliza e penaliza
            self.parse_errors[type(exc).__name__] += 1
            self.seconds += time.perf_counter() - t0
            return 1.0
        key = canonical_key(expr)
        self.keys_3sig.add(canonical_key(expr, 3))
        self.keys_struct.add(" ".join("c" if _NUM.fullmatch(t) else t for t in key.split()))
        c = self.cache.get(key)
        if c is None:
            ev = self.evaluator.evaluate(expr, None, "E")  # type: ignore[arg-type]
            c = 1.0 - sum(ev.scores) / 3.0
            self.cache[key] = c
        self.seconds += time.perf_counter() - t0
        return c

    def stats(self) -> dict[str, Any]:
        return {
            "callback_calls": self.calls,
            "unique_strings": len(self.unique_text),
            "unique_canonical_keys": len(self.cache),
            "unique_keys_3sig": len(self.keys_3sig),
            "unique_structures": len(self.keys_struct),
            "evaluator_calls": self.evaluator.calls,
            "cache_hit_rate": 1 - self.evaluator.calls / self.calls if self.calls else None,
            "parse_errors": dict(self.parse_errors),
            "python_seconds": round(self.seconds, 3),
        }


# --------------------------------------------------------------------------------------
def free_fall(n: int = 200, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.uniform(1.0, 100.0, n)
    m = rng.uniform(0.1, 10.0, n)
    z = rng.normal(0.0, 1.0, n)
    v = np.sqrt(2 * 9.81 * h)
    return np.column_stack([h, m, z]), v


_JL_READY = False


def jl_main() -> Any:
    global _JL_READY
    import pysr  # noqa: F401 - importar ANTES de juliacall (PySR configura threads/sinais)
    from juliacall import Main as jl  # o mesmo Main usado pelo PySR

    if not _JL_READY:
        jl.seval(JULIA_PRELUDE)
        _JL_READY = True
    return jl


def run(
    *,
    weights: tuple[float, float, float],
    favored: str | None,
    call_py: bool = True,
    niterations: int = 8,
    populations: int = 2,
    population_size: int = 30,
    parallelism: str = "serial",
    optimize: bool = True,
    should_simplify: bool = True,
    score_simplified: bool = True,
    seed: int = SEED,
    **extra: Any,
) -> dict[str, Any]:
    from pysr import PySRRegressor

    jl = jl_main()
    cb = Callback(make_evaluator(favored), score_simplified)
    jl.seval("set_inloop_cb!")(cb)
    jl.seval("set_inloop_w!")(*weights, call_py)
    X, y = free_fall()
    kw: dict[str, Any] = dict(extra)
    if not optimize:
        kw["optimize_probability"] = 0.0
        kw["should_optimize_constants"] = False
    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["neg", "exp", "log", "sin", "cos"],
        loss_function=LOSS_FUNCTION,
        parsimony=0.0,
        niterations=niterations,
        populations=populations,
        population_size=population_size,
        maxsize=20,
        batching=False,
        should_simplify=should_simplify,
        parallelism=parallelism,
        deterministic=parallelism == "serial",
        random_state=seed,
        model_selection="accuracy",
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        tempdir=tempfile.mkdtemp(prefix="pysr_inloop_"),
        **kw,
    )
    t0 = time.perf_counter()
    model.fit(X, y, variable_names=VARS)
    wall = time.perf_counter() - t0
    eqs = model.equations_
    hof = [str(e) for e in eqs["equation"]]
    best = str(model.get_best()["equation"])
    frac = (
        sum(1 for e in hof if favored and re.search(rf"\b{favored}\(", e)) / len(hof)
        if hof
        else 0.0
    )
    # Auditoria: recalcula L de cada equação do HoF no Python e compara com a perda do PySR.
    calls_before = cb.calls
    hof_rows = []
    for i, (cpx, lo, eq) in enumerate(zip(eqs["complexity"], eqs["loss"], hof)):
        r = np.abs(model.predict(X, index=i) - y)
        e = float(np.mean(np.where(r <= 1.0, r**2 / 2, r - 0.5)))
        c_py = cb(eq)
        w1, w2, w3 = weights
        hof_rows.append(
            {
                "complexity": int(cpx),
                "loss": float(lo),
                "equation": eq,
                "e": round(e, 4),
                "c": round(c_py, 3),
                "L_recomputed": round(w1 * e + w2 * int(cpx) + w3 * (c_py if call_py else 0.0), 4),
            }
        )
    cb.calls = calls_before
    return {
        "weights": weights,
        "favored": favored,
        "call_py": call_py,
        "niterations": niterations,
        "populations": populations,
        "population_size": population_size,
        "parallelism": parallelism,
        "extra": extra,
        "optimize_constants": optimize,
        "should_simplify": should_simplify,
        "score_simplified": score_simplified,
        "wall_seconds": round(wall, 2),
        "julia_loss_calls": int(jl.seval("INLOOP_NCALLS[]")),
        "best": best,
        "best_has_favored": bool(favored and re.search(rf"\b{favored}\(", best)),
        "hof_frac_favored": round(frac, 3),
        "max_abs_loss_minus_recomputed": round(
            max(abs(h["loss"] - h["L_recomputed"]) for h in hof_rows), 4
        ),
        "hof": hof_rows,
        **cb.stats(),
    }


def summarize(r: dict[str, Any]) -> str:
    return (
        f"w={tuple(round(w, 3) for w in r['weights'])} fav={r['favored']} callpy={r['call_py']} "
        f"it={r['niterations']} | best={r['best']!r} has_fav={r['best_has_favored']} "
        f"hof_frac={r['hof_frac_favored']} | calls={r['callback_calls']} uniq_str={r['unique_strings']} "
        f"uniq_key={r['unique_canonical_keys']} k3={r['unique_keys_3sig']} struct={r['unique_structures']} hit={r['cache_hit_rate'] and round(r['cache_hit_rate'], 3)} "
        f"parse_err={r['parse_errors']} audit={r['max_abs_loss_minus_recomputed']} | wall={r['wall_seconds']}s py={r['python_seconds']}s"
    )


def mode_influence() -> list[dict[str, Any]]:
    """Mesma semente/dados. A: favorece cos; B: não favorece nada. w3 = 0 vs w3 > 0.

    Prova GAP-08: com w3 = 0, trocar o avaliador (A <-> B) NÃO muda nada; com w3 > 0 muda.
    Repetição idêntica (A, w3>0) testa determinismo com o callback ativo.
    """
    w0 = (0.5, 0.5, 0.0)  # baseline GAP-01
    wp = (1 / 3, 1 / 3, 1 / 3)  # pilot.yaml
    wbig = (1 / 3, 1 / 3, 5.0)  # w3 grande, para isolar o efeito
    cfgs = [
        (w0, "cos"),
        (w0, None),
        (wp, "cos"),
        (wp, None),
        (wp, "cos"),
        (wbig, "cos"),
        (wbig, None),
    ]
    out = []
    for w, fav in cfgs:
        r = run(weights=w, favored=fav)
        print(summarize(r), flush=True)
        out.append(r)
    # Armadilha documentada: escore sobre a árvore crua (sem dobrar constantes).
    for kw in ({"score_simplified": False}, {"score_simplified": False, "should_simplify": False}):
        r = run(weights=wbig, favored="cos", **kw)
        print(f"[{kw}] " + summarize(r), flush=True)
        out.append(r)
    a, b = out[0], out[1]
    print(
        "w3=0: HoF idêntico entre avaliadores A/B:",
        a["hof"] == b["hof"]
        or [h["equation"] for h in a["hof"]] == [h["equation"] for h in b["hof"]],
    )
    print(
        "determinismo (A, w3=1/3, 2x):",
        [h["equation"] for h in out[2]["hof"]] == [h["equation"] for h in out[4]["hof"]]
        and out[2]["callback_calls"] == out[4]["callback_calls"],
    )
    return out


def mode_cost() -> list[dict[str, Any]]:
    out = []
    for it in (1, 2, 4):
        for call_py in (False, True):
            r = run(weights=(1 / 3, 1 / 3, 1 / 3), favored="cos", call_py=call_py, niterations=it)
            print(summarize(r) + f" jl_calls={r['julia_loss_calls']}", flush=True)
            out.append(r)
    for it in (1, 4):
        r = run(weights=(1 / 3, 1 / 3, 1 / 3), favored="cos", niterations=it, optimize=False)
        print("[sem otimização de constantes] " + summarize(r), flush=True)
        out.append(r)
    # Escala da hipótese GAP-08: populations=1, population_size=100
    r = run(
        weights=(1 / 3, 1 / 3, 1 / 3),
        favored="cos",
        niterations=1,
        populations=1,
        population_size=100,
    )
    print("[pop 1x100] " + summarize(r), flush=True)
    out.append(r)
    for ncyc in (50, 100):
        r = run(
            weights=(1 / 3, 1 / 3, 1 / 3),
            favored="cos",
            niterations=1,
            populations=1,
            population_size=100,
            ncycles_per_iteration=ncyc,
        )
        print(f"[pop 1x100, ncycles_per_iteration={ncyc}] " + summarize(r), flush=True)
        out.append(r)
    return out


def mode_threads() -> list[dict[str, Any]]:
    r = run(
        weights=(1 / 3, 1 / 3, 1 / 3), favored="cos", parallelism="multithreading", niterations=2
    )
    print(summarize(r), flush=True)
    return [r]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["influence", "cost", "threads"])
    args = ap.parse_args()
    res = {"influence": mode_influence, "cost": mode_cost, "threads": mode_threads}[args.mode]()
    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.mode}.json").write_text(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
