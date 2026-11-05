# mypy: disable-error-code="no-any-return"
"""Inspeção do LLM para o notebook 02: sonda de expressões e leitura do cache de avaliações.

Só leitura e exibição. Nenhuma nota substituta: expressão sem avaliação no cache aparece com
``found = False`` e escores ``NaN``; resposta inválida na sonda aparece com ``valida = False``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physics_llm_sr.config.paths import resolve_project_path
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.cache import (
    CACHE_FILENAME,
    DEFAULT_CACHE_KEY_MODE,
    DEFAULT_CACHE_SIGNIFICANT_DIGITS,
    STATUS_OK,
    expression_key,
)
from physics_llm_sr.llm.interfaces import EquationEvaluation
from physics_llm_sr.llm.response_parser import InvalidResponseError, parse_evaluation
from physics_llm_sr.llm.scoring import LLMEquationEvaluator, penalty
from physics_llm_sr.physics.base import PhysicalScenario
from physics_llm_sr.symbolic.expression import parse_infix

CACHE_COLUMNS = (
    "cache_key",
    "status",
    "expression",
    "canonical_key",
    "scenario",
    "prompt_variant",
    "model_id",
    "model_revision",
    "c1",
    "c2",
    "c3",
    "penalty",
    "feedback",
    "raw_response",
    "created_at",
)
SCORE_COLUMNS = ("c1", "c2", "c3", "penalty")
#: Expressão absurda padrão da sonda (sem sentido físico para qualquer cenário).
ABSURD_EXPRESSION = "exp(height*height) + log(mass)"
#: Rótulos da sonda de três expressões (GT, forma equivalente, absurda).
PROBE_LABELS = ("GT", "equivalente", "absurda")


def cache_file(cfg: ExperimentConfig) -> Path:
    """Arquivo JSONL do cache de avaliações da config (``llm.cache_dir`` na raiz do projeto)."""
    return resolve_project_path(cfg.llm.cache_dir) / CACHE_FILENAME


def _record_row(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = EquationEvaluation.from_dict(record["evaluation"])
    meta = record.get("metadata") or {}
    c1, c2, c3 = evaluation.scores
    return {
        "cache_key": record["key"],
        "status": STATUS_OK,
        "expression": meta.get("expression"),
        "canonical_key": meta.get("canonical_key"),
        "scenario": meta.get("scenario"),
        "prompt_variant": meta.get("prompt_variant"),
        "model_id": meta.get("model_id"),
        "model_revision": meta.get("model_revision"),
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "penalty": penalty(evaluation),
        "feedback": evaluation.feedback,
        "raw_response": record.get("raw_response"),
        "created_at": record.get("created_at"),
    }


def read_cache_records(path: str | Path) -> pd.DataFrame:
    """Avaliações válidas do cache JSONL (último registro ``ok`` de cada chave).

    Erros de avaliação e linhas corrompidas ou fora do contrato são ignorados (como no
    :class:`~physics_llm_sr.llm.cache.EvaluationCache`). Arquivo ausente: tabela vazia com
    as colunas de :data:`CACHE_COLUMNS`.
    """
    file = Path(path)
    by_key: dict[str, dict[str, Any]] = {}
    if file.is_file():
        with file.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record["status"] == STATUS_OK:
                        by_key[str(record["key"])] = _record_row(record)
                except (ValueError, KeyError, TypeError):
                    continue
    return pd.DataFrame(list(by_key.values()), columns=list(CACHE_COLUMNS))


def lookup_evaluations(
    records: pd.DataFrame,
    expressions: Sequence[str],
    *,
    scenario: str,
    prompt: str,
    features: Sequence[str] | None = None,
    model_id: str | None = None,
    significant_digits: int = DEFAULT_CACHE_SIGNIFICANT_DIGITS,
    key_mode: str = DEFAULT_CACHE_KEY_MODE,
) -> pd.DataFrame:
    """Avaliação em cache de cada expressão (casada pela forma canônica da chave do cache).

    ``features`` default: colunas de :data:`~physics_llm_sr.physics.base.FEATURE_NAMES`.
    Expressão sem registro (ou que não parseia) devolve ``found = False`` e escores ``NaN``.
    """
    from physics_llm_sr.physics.base import FEATURE_NAMES

    names = tuple(features) if features is not None else FEATURE_NAMES
    part = records[(records["scenario"] == scenario) & (records["prompt_variant"] == prompt)]
    if model_id is not None:
        part = part[part["model_id"] == model_id]
    index = part.drop_duplicates("canonical_key", keep="last").set_index("canonical_key")
    rows: list[dict[str, Any]] = []
    for text in expressions:
        row: dict[str, Any] = {"expression": text, "found": False}
        row.update({c: np.nan for c in SCORE_COLUMNS})
        row.update(feedback=None, raw_response=None)
        try:
            key = expression_key(parse_infix(text, names), significant_digits, key_mode)
        except ValueError:
            key = None
        if key is not None and key in index.index:
            hit = index.loc[key]
            row.update({c: hit[c] for c in (*SCORE_COLUMNS, "feedback", "raw_response")})
            row["found"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def probe_set(
    scenario: PhysicalScenario, equivalent: str | None = None, absurd: str = ABSURD_EXPRESSION
) -> tuple[str, str, str]:
    """As três expressões da sonda: GT do artigo, forma equivalente e uma expressão absurda.

    ``equivalent``: forma equivalente escolhida; ``None`` usa a primeira forma equivalente
    declarada pelo cenário e diferente da GT; se não houver, a GT multiplicada por 1.0
    (reescrita trivial).
    """
    gt = scenario.physical_context().ground_truth
    if equivalent is None:
        declared = [e for e in scenario.equivalent_ground_truths() if e != gt]
        equivalent = declared[0] if declared else f"1.0*({gt})"
    return (gt, equivalent, absurd)


def close_backend(evaluator: LLMEquationEvaluator) -> None:
    """Libera o backend do avaliador (descarrega o modelo), se ele tiver ``close()``."""
    close = getattr(evaluator.backend, "close", None)
    if callable(close):
        close()


def probe_expressions(
    evaluator: LLMEquationEvaluator,
    expressions: Sequence[str],
    context: PhysicalContext,
    prompt_variant: str,
    features: Sequence[str],
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Pergunta ao LLM sobre cada expressão e devolve resposta crua, escores e penalidade.

    Não usa o cache nem as tentativas extras: mostra exatamente o que o modelo respondeu
    numa chamada. Resposta fora do contrato: ``valida = False``, ``erro`` com o motivo e
    escores ``NaN`` (nenhuma nota substituta).
    """
    names = list(labels) if labels is not None else list(PROBE_LABELS)
    if len(names) != len(expressions):
        names = [f"expressão {i + 1}" for i in range(len(expressions))]
    rendered = [
        evaluator.prompt_library.render(parse_infix(t, tuple(features)), context, prompt_variant)
        for t in expressions
    ]
    responses = evaluator.backend.generate([(r.system, r.user) for r in rendered])
    rows: list[dict[str, Any]] = []
    for label, text, prompt, raw in zip(names, expressions, rendered, responses, strict=True):
        row: dict[str, Any] = {
            "rotulo": label,
            "expression": text,
            "resposta_crua": raw,
            "valida": False,
            "erro": None,
            "feedback": None,
            "prompt_user": prompt.user,
        }
        row.update({c: np.nan for c in SCORE_COLUMNS})
        try:
            evaluation = parse_evaluation(raw)
        except InvalidResponseError as exc:
            row["erro"] = f"{type(exc).__name__}: {exc}"
        else:
            row.update(valida=True, feedback=evaluation.feedback, penalty=penalty(evaluation))
            row.update(zip(("c1", "c2", "c3"), evaluation.scores, strict=True))
        rows.append(row)
    return pd.DataFrame(rows)
