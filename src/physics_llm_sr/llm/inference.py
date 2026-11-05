"""Backends de geração de texto (PRD §6.1, §6.2, GAP-07).

- :class:`FakeBackend`: simulado e determinístico (testes e modo smoke).
- :class:`OllamaBackend`: HTTP local (``/api/chat`` ou ``/api/generate``), temperatura 0,
  semente fixa. Usado na validação local (Q4 = desvio de quantização).
- :class:`TransformersBackend`: Hugging Face ``transformers`` em fp16 (Colab A100),
  ``do_sample=False``, lote com padding à esquerda, revisão fixada por SHA.
  ``torch``/``transformers`` são importados só no primeiro uso.

:func:`build_backend` instancia o backend a partir de ``LLMConfig``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from physics_llm_sr.llm.interfaces import BackendError

#: Identificador do backend simulado.
FAKE_CHECKPOINT = "fake"
FAKE_REVISION = "fake-v1"
#: Casas decimais dos escores do responder simulado padrão.
FAKE_SCORE_DECIMALS = 4
_HEX_CHUNK = 4
_HEX_CHUNK_MAX = 16**_HEX_CHUNK - 1

Responder = Callable[[str, str], str]


def deterministic_fake_response(system: str, user: str) -> str:
    """Resposta JSON válida derivada do hash do prompt (rótulo explícito: simulada)."""
    digest = hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()
    values = [
        round(int(digest[i * _HEX_CHUNK : (i + 1) * _HEX_CHUNK], 16) / _HEX_CHUNK_MAX, 4)
        for i in range(3)
    ]
    return json.dumps(
        {
            "dimensional_consistency": values[0],
            "simplicity": values[1],
            "physical_realism": values[2],
            "feedback": "simulated (FakeBackend)",
        }
    )


class FakeBackend:
    """Backend simulado: resposta fixa, roteiro sequencial ou função ``(system, user)``.

    Contadores: ``calls`` (pares recebidos), ``prompt_count`` e ``generate_calls``.
    """

    def __init__(
        self,
        responses: str | Sequence[str] | Responder | None = None,
        *,
        checkpoint: str = FAKE_CHECKPOINT,
        revision: str | None = FAKE_REVISION,
    ) -> None:
        """``responses=None`` usa :func:`deterministic_fake_response`."""
        self._fixed: str | None = None
        self._script: list[str] | None = None
        self._fn: Responder | None = None
        if responses is None:
            self._fn = deterministic_fake_response
        elif isinstance(responses, str):
            self._fixed = responses
        elif callable(responses):
            self._fn = responses
        else:
            self._script = list(responses)
        self.checkpoint = checkpoint
        self.revision = revision
        self.calls: list[tuple[str, str]] = []
        self.generate_calls = 0

    @property
    def prompt_count(self) -> int:
        """Número total de prompts recebidos."""
        return len(self.calls)

    def _respond(self, system: str, user: str) -> str:
        if self._fixed is not None:
            return self._fixed
        if self._fn is not None:
            return self._fn(system, user)
        assert self._script is not None
        if not self._script:
            raise BackendError("FakeBackend: roteiro de respostas esgotado")
        return self._script.pop(0)

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Uma resposta por prompt, na ordem."""
        self.generate_calls += 1
        out = []
        for system, user in prompts:
            self.calls.append((system, user))
            out.append(self._respond(system, user))
        return out

    def model_metadata(self) -> dict[str, Any]:
        """Metadados do backend simulado."""
        return {
            "backend": "fake",
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "quantization": None,
            "infra": "simulated",
        }
