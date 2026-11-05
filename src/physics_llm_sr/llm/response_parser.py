"""Validação da resposta do LLM (PRD §6.3). Nunca executa conteúdo da resposta.

Política:

1. A resposta é texto não confiável. Só ``json`` da stdlib a lê; nada de ``eval``,
   ``ast.literal_eval``, ``sympify`` ou similares.
2. Extrai-se o **primeiro** objeto JSON decodificável: tenta-se ``raw_decode`` a partir de
   cada ``{`` em ordem (texto/markdown ao redor é tolerado; ``{`` em prosa é pulado).
3. O objeto deve ter **exatamente** as chaves ``dimensional_consistency``, ``simplicity``,
   ``physical_realism`` e ``feedback``. Chave extra é rejeitada (:class:`UnexpectedKeysError`):
   o prompt exige "exactly these keys and no others", e aceitar extras esconderia respostas
   fora do contrato.
4. Escores: ``int``/``float`` JSON (``bool`` recusado), finitos (``NaN``/``Infinity`` e
   números que estouram para ``inf`` recusados) e em ``[0, 1]`` fechado; ``feedback``: ``str``.
5. Qualquer violação levanta subclasse de :class:`InvalidResponseError`; o chamador decide
   retry/abort. Não existe nota substituta.

A lista Python ``[dim corr, simp, sim, feedback]`` do artigo não é aceita (desvio GAP-13:
o prompt reconstruído pede JSON).
"""

from __future__ import annotations

import json
import math
from typing import Any

from physics_llm_sr.llm.interfaces import (
    FEEDBACK_KEY,
    RESPONSE_KEYS,
    SCORE_KEYS,
    SCORE_MAX,
    SCORE_MIN,
    EquationEvaluation,
    LLMError,
)

#: Respostas maiores são recusadas sem análise (limita custo de varredura).
MAX_RESPONSE_CHARS = 32_768
#: Máximo de posições ``{`` testadas na busca do primeiro objeto.
MAX_OBJECT_CANDIDATES = 64
#: Tamanho do trecho da resposta citado nas mensagens de erro.
ERROR_EXCERPT_CHARS = 200


class InvalidResponseError(LLMError, ValueError):
    """Resposta fora do contrato; ``raw`` guarda um trecho para auditoria."""

    def __init__(self, message: str, raw: str = "") -> None:
        """Mensagem + trecho da resposta original."""
        super().__init__(message)
        self.raw = raw[:ERROR_EXCERPT_CHARS]


class ResponseTooLongError(InvalidResponseError):
    """Resposta acima de :data:`MAX_RESPONSE_CHARS`."""


class NoJSONObjectError(InvalidResponseError):
    """Nenhum objeto JSON decodificável na resposta."""


class MissingKeysError(InvalidResponseError):
    """Faltam chaves obrigatórias."""


class UnexpectedKeysError(InvalidResponseError):
    """Chaves além das quatro do contrato."""


class ScoreTypeError(InvalidResponseError):
    """Escore não numérico (ou ``bool``) ou feedback não textual."""


class ScoreRangeError(InvalidResponseError):
    """Escore fora de ``[0, 1]`` ou não finito."""


class _NonFiniteConstantError(Exception):
    def __init__(self, token: str) -> None:
        super().__init__(token)
        self.token = token


def _reject_constant(token: str) -> Any:
    raise _NonFiniteConstantError(token)


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Devolve o primeiro objeto JSON decodificável de ``text``.

    Raises:
        NoJSONObjectError: nenhum objeto encontrado.
        ScoreRangeError: o objeto contém ``NaN``/``Infinity``.
    """
    start = text.find("{")
    attempts = 0
    while start != -1 and attempts < MAX_OBJECT_CANDIDATES:
        attempts += 1
        try:
            obj, _ = _DECODER.raw_decode(text, start)
        except _NonFiniteConstantError as exc:
            raise ScoreRangeError(f"constante não finita na resposta: {exc.token}", text) from None
        except (json.JSONDecodeError, RecursionError):
            obj = None
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise NoJSONObjectError("nenhum objeto JSON na resposta", text)


def _score(name: str, value: object, raw: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreTypeError(f"{name} deve ser número, recebeu {type(value).__name__}", raw)
    number = float(value)
    if not math.isfinite(number):
        raise ScoreRangeError(f"{name} não finito", raw)
    if not SCORE_MIN <= number <= SCORE_MAX:
        raise ScoreRangeError(f"{name}={number!r} fora de [{SCORE_MIN}, {SCORE_MAX}]", raw)
    return number


def parse_evaluation(text: str) -> EquationEvaluation:
    """Valida a resposta textual do LLM e devolve :class:`EquationEvaluation`.

    Raises:
        InvalidResponseError: (subclasses) qualquer violação do contrato.
    """
    if not isinstance(text, str):
        raise InvalidResponseError(f"resposta deve ser str, recebeu {type(text).__name__}")
    if len(text) > MAX_RESPONSE_CHARS:
        raise ResponseTooLongError(f"resposta com {len(text)} caracteres", text)
    obj = extract_first_json_object(text)
    keys = set(obj)
    missing = sorted(RESPONSE_KEYS - keys)
    if missing:
        raise MissingKeysError(f"chaves ausentes: {missing}", text)
    extra = sorted(keys - RESPONSE_KEYS)
    if extra:
        raise UnexpectedKeysError(f"chaves inesperadas: {extra}", text)
    scores = {name: _score(name, obj[name], text) for name in SCORE_KEYS}
    feedback = obj[FEEDBACK_KEY]
    if not isinstance(feedback, str):
        raise ScoreTypeError("feedback deve ser str", text)
    return EquationEvaluation(
        dimensional_consistency=scores["dimensional_consistency"],
        simplicity=scores["simplicity"],
        physical_realism=scores["physical_realism"],
        feedback=feedback,
    )
