"""T:llm-hf / T:llm-hf-neg (transformers com stub) e backend Ollama com HTTP simulado."""

from __future__ import annotations

import contextlib
import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import requests

from physics_llm_sr.config.schemas import LLMConfig, LLMModelConfig
from physics_llm_sr.core_types import Track
from physics_llm_sr.llm.inference import (
    FakeBackend,
    OllamaBackend,
    TransformersBackend,
    build_backend,
)
from physics_llm_sr.llm.interfaces import (
    BackendConfigurationError,
    BackendError,
    TextGenerationBackend,
)

VALID = json.dumps(
    {"dimensional_consistency": 1.0, "simplicity": 1.0, "physical_realism": 1.0, "feedback": ""}
)
CKPT = "mistralai/Mistral-7B-Instruct-v0.1"
SHA = "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca"
PROMPTS = [("SYS", "user one"), ("SYS", "user two"), ("SYS", "user three")]


# ------------------------------------------------------------------ stubs transformers/torch


class _Encoding(dict[str, Any]):
    device: Any = None

    def to(self, device: Any) -> _Encoding:
        self.device = device
        return self


class StubTokenizer:
    def __init__(self, chat_template: str | None, reject_system: bool) -> None:
        self.chat_template = chat_template
        self.reject_system = reject_system
        self.padding_side = "right"
        self.pad_token: str | None = None
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.template_calls: list[list[dict[str, str]]] = []
        self.encoded: list[tuple[list[str], dict[str, Any]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False and add_generation_prompt is True
        self.template_calls.append(messages)
        if self.reject_system and any(m["role"] == "system" for m in messages):
            raise ValueError("Conversation roles must alternate user/assistant")
        return "<s>[INST] " + " || ".join(m["content"] for m in messages) + " [/INST]"

    def __call__(self, texts: list[str], **kwargs: Any) -> _Encoding:
        self.encoded.append((list(texts), kwargs))
        return _Encoding(input_ids=np.zeros((len(texts), 5), dtype=int))

    def batch_decode(self, tokens: np.ndarray, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens
        assert tokens.shape[1] == 1  # só os tokens novos
        return [VALID for _ in range(tokens.shape[0])]


class StubModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.eval_called = False

    def eval(self) -> StubModel:
        self.eval_called = True
        return self

    def generate(self, input_ids: np.ndarray, **kwargs: Any) -> np.ndarray:
        self.generate_calls.append(kwargs)
        n = input_ids.shape[0]
        return np.concatenate([input_ids, np.ones((n, 1), dtype=int)], axis=1)


class HFStub:
    def __init__(self, chat_template: str | None = "tmpl", reject_system: bool = False) -> None:
        self.tokenizer = StubTokenizer(chat_template, reject_system)
        self.model = StubModel()
        self.tokenizer_kwargs: dict[str, Any] = {}
        self.model_kwargs: dict[str, Any] = {}
        self.seeds: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self
        torch = types.ModuleType("torch")
        torch.float16 = "float16-sentinel"  # type: ignore[attr-defined]
        torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
        torch.manual_seed = stub.seeds.append  # type: ignore[attr-defined]

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> StubTokenizer:
                stub.tokenizer_kwargs = {"name": name, **kwargs}
                return stub.tokenizer

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> StubModel:
                stub.model_kwargs = {"name": name, **kwargs}
                return stub.model

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
        transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
        transformers.__version__ = "stub-4.0"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "transformers", transformers)


@pytest.fixture
def hf(monkeypatch: pytest.MonkeyPatch) -> HFStub:
    stub = HFStub()
    stub.install(monkeypatch)
    return stub


def _hf(**kwargs: Any) -> TransformersBackend:
    params: dict[str, Any] = {
        "checkpoint": CKPT,
        "revision": SHA,
        "track": Track.PAPER_REPLICATION,
        "max_new_tokens": 77,
        "batch_size": 2,
    }
    params.update(kwargs)
    return TransformersBackend(**params)


# ------------------------------------------------------------------------ T:llm-hf


def test_hf_generates_in_left_padded_greedy_batches(hf: HFStub) -> None:
    backend = _hf()
    assert backend.generate(PROMPTS) == [VALID] * 3
    assert hf.model_kwargs["revision"] == SHA
    assert hf.model_kwargs["trust_remote_code"] is False
    assert hf.model_kwargs["torch_dtype"] == "float16-sentinel"
    assert hf.tokenizer_kwargs["revision"] == SHA
    assert hf.tokenizer_kwargs["trust_remote_code"] is False
    assert hf.tokenizer.padding_side == "left"
    assert hf.tokenizer.pad_token == hf.tokenizer.eos_token
    assert hf.model.eval_called
    assert len(hf.model.generate_calls) == 2  # lotes de 2 + 1
    for call in hf.model.generate_calls:
        assert call["do_sample"] is False
        assert call["max_new_tokens"] == 77
        assert "temperature" not in call
    assert [len(t) for t, _ in hf.tokenizer.encoded] == [2, 1]
    assert all(kw["padding"] is True for _, kw in hf.tokenizer.encoded)


def test_hf_uses_chat_template_with_system(hf: HFStub) -> None:
    _hf().generate(PROMPTS[:1])
    messages = hf.tokenizer.template_calls[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    texts, kwargs = hf.tokenizer.encoded[0]
    assert texts[0].startswith("<s>[INST]") and kwargs["add_special_tokens"] is False


def test_hf_chat_template_without_system_role_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = HFStub(reject_system=True)
    stub.install(monkeypatch)
    backend = _hf()
    backend.generate(PROMPTS[:1])
    merged = stub.tokenizer.template_calls[-1]
    assert [m["role"] for m in merged] == ["user"]
    assert merged[0]["content"].startswith("SYS") and "user one" in merged[0]["content"]
    assert backend.model_metadata()["prompt_format"] == "chat_template_merged_system"


def test_hf_without_chat_template_uses_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = HFStub(chat_template=None)
    stub.install(monkeypatch)
    backend = _hf()
    backend.generate(PROMPTS[:1])
    texts, kwargs = stub.tokenizer.encoded[0]
    assert texts[0].startswith("SYS") and "user one" in texts[0]
    assert kwargs["add_special_tokens"] is True
    assert backend.model_metadata()["prompt_format"] == "plain"


def test_hf_imports_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    _hf()
    assert "transformers" not in sys.modules and "torch" not in sys.modules


def test_hf_metadata(hf: HFStub) -> None:
    backend = _hf(infra="colab-a100")
    backend.generate(PROMPTS[:1])
    meta = backend.model_metadata()
    assert meta["backend"] == "transformers" and meta["checkpoint"] == CKPT
    assert meta["revision"] == SHA and meta["quantization"] == "fp16"
    assert meta["infra"] == "colab-a100" and meta["do_sample"] is False
    assert meta["transformers_version"] == "stub-4.0"
    assert isinstance(backend, TextGenerationBackend)


# ---------------------------------------------------------------------- T:llm-hf-neg


def test_hf_refuses_trust_remote_code() -> None:
    with pytest.raises(BackendConfigurationError, match="trust_remote_code"):
        _hf(trust_remote_code=True)
    with pytest.raises(BackendConfigurationError, match="trust_remote_code"):
        _hf(trust_remote_code=True, track=Track.EXTENDED_ANALYSIS)


def test_hf_requires_revision_in_paper_replication() -> None:
    with pytest.raises(BackendConfigurationError, match="revis"):
        _hf(revision=None)
    with pytest.raises(BackendConfigurationError, match="revis"):
        _hf(revision="  ")


def test_hf_revision_optional_in_extended(hf: HFStub) -> None:
    backend = _hf(revision=None, track=Track.EXTENDED_ANALYSIS)
    backend.generate(PROMPTS[:1])
    assert hf.model_kwargs["revision"] is None


def test_hf_missing_dependency_is_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(BackendError, match="pip install"):
        _hf().generate(PROMPTS[:1])


# ------------------------------------------------------------------------ Ollama


class _Resp:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, responses: Iterator[_Resp]) -> None:
        self.responses = responses
        self.posts: list[tuple[str, dict[str, Any], float]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> _Resp:
        self.posts.append((url, json, timeout))
        return next(self.responses)


def test_ollama_chat_payload_and_parsing() -> None:
    session = _Session(iter([_Resp({"message": {"content": VALID}})] * 3))
    backend = OllamaBackend(
        "mistral:7b-instruct",
        base_url="http://host:1234/",
        seed=7,
        num_predict=99,
        timeout=12.5,
        session=session,
    )
    assert backend.generate(PROMPTS) == [VALID] * 3
    url, payload, timeout = session.posts[0]
    assert url == "http://host:1234/api/chat" and timeout == 12.5
    assert payload["model"] == "mistral:7b-instruct" and payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "user one"},
    ]
    assert payload["options"] == {"temperature": 0.0, "seed": 7, "num_predict": 99}
    assert "format" not in payload


def test_ollama_generate_endpoint() -> None:
    session = _Session(iter([_Resp({"response": VALID})]))
    backend = OllamaBackend("m", endpoint="generate", session=session)
    assert backend.generate(PROMPTS[:1]) == [VALID]
    url, payload, _ = session.posts[0]
    assert url.endswith("/api/generate")
    assert payload["system"] == "SYS" and payload["prompt"] == "user one"


@pytest.mark.parametrize(
    "resp",
    [_Resp({}, status=500), _Resp(ValueError("not json")), _Resp({"message": {}})],
    ids=["http500", "bad-json", "missing-content"],
)
def test_ollama_failures_are_backend_errors(resp: _Resp) -> None:
    backend = OllamaBackend("m", session=_Session(iter([resp])))
    with pytest.raises(BackendError):
        backend.generate(PROMPTS[:1])


def test_ollama_connection_error_is_backend_error() -> None:
    class Down:
        def post(self, url: str, json: dict[str, Any], timeout: float) -> _Resp:
            raise requests.ConnectionError("refused")

    with pytest.raises(BackendError, match="refused"):
        OllamaBackend("m", session=Down()).generate(PROMPTS[:1])


def test_ollama_rejects_nonzero_temperature_and_bad_endpoint() -> None:
    with pytest.raises(BackendConfigurationError):
        OllamaBackend("m", temperature=0.7)
    with pytest.raises(BackendConfigurationError):
        OllamaBackend("m", endpoint="completions")


def test_ollama_host_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:5555")
    assert OllamaBackend("m").base_url == "http://127.0.0.1:5555"


# ------------------------------------------------------------------- fábrica


def _cfg(backend: str, revision: str | None = SHA, checkpoint: str = CKPT) -> LLMConfig:
    model = LLMModelConfig(
        name="mistral", checkpoint=checkpoint, revision=revision, quantization="fp16"
    )
    return LLMConfig(backend=backend, models=[model], max_new_tokens=33)


def test_build_backend_variants(hf: HFStub) -> None:
    fake = build_backend(_cfg("fake"), "mistral", track=Track.PAPER_REPLICATION)
    assert isinstance(fake, FakeBackend)
    ollama = build_backend(
        _cfg("ollama", checkpoint="mistral:7b-instruct"), "mistral", track=Track.PAPER_REPLICATION
    )
    assert isinstance(ollama, OllamaBackend) and ollama.num_predict == 33
    assert ollama.model_metadata()["checkpoint"] == "mistral:7b-instruct"
    hf_backend = build_backend(_cfg("transformers"), "mistral", track=Track.PAPER_REPLICATION)
    assert isinstance(hf_backend, TransformersBackend)
    assert hf_backend.max_new_tokens == 33 and hf_backend.revision == SHA


def test_build_backend_rejects_unknown_model_and_missing_revision() -> None:
    with pytest.raises(BackendConfigurationError, match="llama2"):
        build_backend(_cfg("fake"), "llama2", track=Track.PAPER_REPLICATION)
    with pytest.raises(BackendConfigurationError, match="revis"):
        build_backend(_cfg("transformers", revision=None), "mistral", track=Track.PAPER_REPLICATION)


def test_fake_backend_default_is_deterministic_and_valid() -> None:
    a = FakeBackend().generate(PROMPTS)
    b = FakeBackend().generate(PROMPTS)
    assert a == b and len(set(a)) == 3
