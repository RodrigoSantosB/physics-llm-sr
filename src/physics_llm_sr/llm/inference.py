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
import importlib
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import requests

from physics_llm_sr.config.schemas import LLMConfig
from physics_llm_sr.core_types import Track
from physics_llm_sr.llm.interfaces import (
    BackendConfigurationError,
    BackendError,
    TextGenerationBackend,
)

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
        round(
            int(digest[i * _HEX_CHUNK : (i + 1) * _HEX_CHUNK], 16) / _HEX_CHUNK_MAX,
            FAKE_SCORE_DECIMALS,
        )
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


# ----------------------------------------------------------------------------- Ollama

#: Endereço padrão do servidor Ollama (sobrescrito por ``OLLAMA_HOST``).
DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_ENDPOINTS: tuple[str, ...] = ("chat", "generate")
#: Semente de inferência quando o backend é criado sem config (``llm.seed``).
DEFAULT_INFERENCE_SEED = 0
#: Timeout HTTP por requisição sem config, em segundos (``llm.request_timeout_s``).
DEFAULT_REQUEST_TIMEOUT_S = 300.0
#: Tokens máximos gerados quando não vem da config (``llm.max_new_tokens``).
DEFAULT_MAX_NEW_TOKENS = 256
#: Temperatura exigida pelo artigo (p.6).
REQUIRED_TEMPERATURE = 0.0


def _ollama_base_url(base_url: str | None) -> str:
    url = base_url or os.environ.get(OLLAMA_HOST_ENV) or DEFAULT_OLLAMA_URL
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


class OllamaBackend:
    """Backend HTTP do Ollama local (``/api/chat`` por padrão), uma requisição por prompt.

    Opções enviadas: ``temperature=0``, ``seed`` fixa e ``num_predict``; ``stream=False``.
    Não usa ``format=json`` (decodificação restrita seria desvio do protocolo do artigo).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        endpoint: str = "chat",
        temperature: float = REQUIRED_TEMPERATURE,
        seed: int = DEFAULT_INFERENCE_SEED,
        num_predict: int = DEFAULT_MAX_NEW_TOKENS,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_S,
        revision: str | None = None,
        quantization: str | None = None,
        infra: str = "local-ollama",
        session: Any = None,
    ) -> None:
        """Configura o cliente; ``session`` (``requests.Session``-like) permite simular HTTP."""
        if endpoint not in OLLAMA_ENDPOINTS:
            raise BackendConfigurationError(f"endpoint deve ser um de {OLLAMA_ENDPOINTS}")
        if temperature != REQUIRED_TEMPERATURE:
            raise BackendConfigurationError("temperatura deve ser 0 (artigo, p.6)")
        self.model = model
        self.base_url = _ollama_base_url(base_url)
        self.endpoint = endpoint
        self.temperature = temperature
        self.seed = seed
        self.num_predict = num_predict
        self.timeout = timeout
        self.revision = revision
        self.quantization = quantization
        self.infra = infra
        self._session = session if session is not None else requests.Session()
        self._digest: str | None = None

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        options = {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
        }
        if self.endpoint == "chat":
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return {"model": self.model, "messages": messages, "stream": False, "options": options}
        return {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": options,
        }

    def _extract(self, data: Any) -> str:
        try:
            text = data["message"]["content"] if self.endpoint == "chat" else data["response"]
        except (KeyError, TypeError):
            raise BackendError(f"resposta Ollama sem texto: {str(data)[:200]}") from None
        if not isinstance(text, str):
            raise BackendError("resposta Ollama com texto não textual")
        return text

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Uma requisição por prompt; erros HTTP/rede viram :class:`BackendError`."""
        url = f"{self.base_url}/api/{self.endpoint}"
        out = []
        for system, user in prompts:
            try:
                response = self._session.post(
                    url, json=self._payload(system, user), timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise BackendError(f"falha Ollama em {url}: {exc}") from exc
            out.append(self._extract(data))
        return out

    def fetch_digest(self) -> str | None:
        """Digest do modelo em ``/api/tags`` (revisão local reprodutível), se disponível."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get("models", [])
        except (requests.RequestException, ValueError, AttributeError) as exc:
            raise BackendError(f"falha ao consultar modelos Ollama: {exc}") from exc
        for entry in models:
            if entry.get("name") == self.model or entry.get("model") == self.model:
                digest = entry.get("digest")
                return str(digest) if digest else None
        return None

    def resolve_identity(self) -> None:
        """Consulta o digest efetivo em ``/api/tags`` (uma vez); :class:`BackendError` se falhar.

        Separado de :meth:`model_metadata` para que esta não faça I/O de rede.
        """
        if self._digest is None:
            self._digest = self.fetch_digest()

    def model_metadata(self) -> dict[str, Any]:
        """Checkpoint, digest (após :meth:`resolve_identity`), revisão, quantização e opções."""
        return {
            "backend": "ollama",
            "checkpoint": self.model,
            "digest": self._digest,
            "revision": self.revision,
            "quantization": self.quantization,
            "infra": self.infra,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
        }


# ----------------------------------------------------------------------- transformers

#: Tamanho de lote sem config (``llm.batch_size``).
DEFAULT_HF_BATCH_SIZE = 8
#: Rótulo de precisão registrado no manifesto (GAP-07: fp16).
HF_QUANTIZATION = "fp16"
HF_DEVICE_MAP = "auto"
PROMPT_FORMAT_CHAT = "chat_template"
PROMPT_FORMAT_CHAT_MERGED = "chat_template_merged_system"
PROMPT_FORMAT_PLAIN = "plain"
_HF_INSTALL_HINT = "pip install -e '.[hf]' (torch, transformers, accelerate)"


class TransformersBackend:
    """Hugging Face ``transformers`` em fp16, decodificação gulosa e lote com padding à esquerda.

    - ``trust_remote_code`` é sempre ``False`` (recusado se pedido).
    - Em ``Track.PAPER_REPLICATION`` a revisão (SHA do commit no Hub) é obrigatória.
    - Usa o chat template do tokenizer quando existe; se o template recusar o papel
      ``system`` (ex.: Mistral v0.1), o system é prefixado à mensagem do usuário; sem
      template, ``system``, linha em branco e ``user`` em texto simples.
    - ``torch``/``transformers`` são importados só no primeiro :meth:`generate`.
    """

    def __init__(
        self,
        checkpoint: str,
        revision: str | None,
        *,
        track: Track = Track.PAPER_REPLICATION,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        batch_size: int = DEFAULT_HF_BATCH_SIZE,
        seed: int = DEFAULT_INFERENCE_SEED,
        infra: str = "unspecified",
        device_map: str | None = HF_DEVICE_MAP,
        trust_remote_code: bool = False,
    ) -> None:
        """Valida a configuração sem carregar o modelo."""
        if trust_remote_code:
            raise BackendConfigurationError(
                "trust_remote_code=True é proibido (execução de código remoto do Hub)"
            )
        if revision is not None and not revision.strip():
            revision = None
        if revision is None and track is Track.PAPER_REPLICATION:
            raise BackendConfigurationError(
                f"revisão (SHA) obrigatória para {checkpoint} em paper_replication"
            )
        if batch_size < 1 or max_new_tokens < 1:
            raise BackendConfigurationError("batch_size e max_new_tokens devem ser >= 1")
        self.checkpoint = checkpoint
        self.revision = revision
        self.track = track
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.seed = seed
        self.infra = infra
        self.device_map = device_map
        self._torch: Any = None
        self._transformers: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._prompt_format: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise BackendError(f"backend transformers indisponível: {_HF_INSTALL_HINT}") from exc
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.checkpoint, revision=self.revision, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            revision=self.revision,
            torch_dtype=torch.float16,
            trust_remote_code=False,
            device_map=self.device_map,
        )
        model.eval()
        self._torch, self._transformers = torch, transformers
        self._tokenizer, self._model = tokenizer, model

    def _format(self, system: str, user: str) -> tuple[str, str]:
        tokenizer = self._tokenizer
        merged = f"{system}\n\n{user}"
        if getattr(tokenizer, "chat_template", None):
            attempts = (
                (
                    PROMPT_FORMAT_CHAT,
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                ),
                (PROMPT_FORMAT_CHAT_MERGED, [{"role": "user", "content": merged}]),
            )
            for label, messages in attempts:
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:  # templates Jinja levantam tipos variados
                    continue
                return str(text), label
        return merged + "\n", PROMPT_FORMAT_PLAIN

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Geração gulosa em lotes de ``batch_size``, na ordem de entrada."""
        self._load()
        torch, tokenizer, model = self._torch, self._tokenizer, self._model
        out: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            formatted = [self._format(s, u) for s, u in chunk]
            self._prompt_format = formatted[-1][1]
            templated = self._prompt_format != PROMPT_FORMAT_PLAIN
            encoding = tokenizer(
                [text for text, _ in formatted],
                return_tensors="pt",
                padding=True,
                add_special_tokens=not templated,
            ).to(model.device)
            input_len = encoding["input_ids"].shape[1]
            torch.manual_seed(self.seed)
            with torch.no_grad():
                generated = model.generate(
                    **encoding,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            out.extend(
                str(t)
                for t in tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)
            )
        return out

    def model_metadata(self) -> dict[str, Any]:
        """Checkpoint, revisão (SHA), fp16, infra e parâmetros de decodificação."""
        return {
            "backend": "transformers",
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "quantization": HF_QUANTIZATION,
            "infra": self.infra,
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "prompt_format": self._prompt_format,
            "transformers_version": getattr(self._transformers, "__version__", None),
        }


# ---------------------------------------------------------------------------- fábrica


def build_backend(
    llm_config: LLMConfig,
    model_name: str,
    *,
    track: Track,
) -> TextGenerationBackend:
    """Instancia o backend de ``llm_config.backend`` para o modelo ``model_name``.

    Lê da config: ``batch_size``, ``seed``, ``request_timeout_s``, ``ollama_endpoint``,
    ``max_new_tokens`` e ``temperature``. ``track`` (``ExperimentConfig.track``) decide se a
    revisão (SHA) do modelo é obrigatória (``paper_replication``).
    """
    by_name = {m.name: m for m in llm_config.models}
    if model_name not in by_name:
        raise BackendConfigurationError(
            f"modelo {model_name!r} não declarado em llm.models: {sorted(by_name)}"
        )
    model = by_name[model_name]
    if llm_config.backend == "fake":
        return FakeBackend(checkpoint=model.checkpoint, revision=model.revision or FAKE_REVISION)
    if llm_config.backend == "ollama":
        return OllamaBackend(
            model.checkpoint,
            base_url=llm_config.ollama_endpoint,
            temperature=llm_config.temperature,
            seed=llm_config.seed,
            num_predict=llm_config.max_new_tokens,
            timeout=llm_config.request_timeout_s,
            revision=model.revision,
            quantization=model.quantization,
            infra=model.infra,
        )
    if llm_config.backend == "transformers":
        return TransformersBackend(
            model.checkpoint,
            model.revision,
            track=track,
            max_new_tokens=llm_config.max_new_tokens,
            batch_size=llm_config.batch_size,
            seed=llm_config.seed,
            infra=model.infra,
        )
    raise BackendConfigurationError(f"backend desconhecido: {llm_config.backend!r}")
