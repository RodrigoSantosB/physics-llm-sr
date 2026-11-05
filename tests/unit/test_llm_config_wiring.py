"""T:config-llm - config YAML -> backend/cache/prompts sem rede (stubs)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import requests

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.config.schemas import ExperimentConfig, LLMConfig, LLMModelConfig
from physics_llm_sr.core_types import Track
from physics_llm_sr.llm.cache import EvaluationCache
from physics_llm_sr.llm.inference import (
    FakeBackend,
    OllamaBackend,
    TransformersBackend,
    build_backend,
)
from physics_llm_sr.llm.interfaces import BackendConfigurationError
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.llm.scoring import LLMEquationEvaluator

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
YAMLS = ("smoke", "pilot", "colab_pilot", "replication", "extended")
REVISIONS = {
    "mistralai/Mistral-7B-Instruct-v0.1": "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca",
    "meta-llama/Llama-2-7b-chat-hf": "f5db02db724555f92da89c216ac04704f23d4590",
    "tiiuae/falcon-7b-instruct": "8782b5c5d8c9290412416618f36a133653e85285",
}
OLLAMA_TAGS = {"mistral:7b-instruct", "llama2:7b-chat", "falcon:7b-instruct"}


def _cfg(name: str) -> ExperimentConfig:
    return load_config(CONFIGS_DIR / f"{name}.yaml")


@pytest.fixture(autouse=True)
def _no_hf_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garante que nada importa torch/transformers nem usa a rede."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

    def _boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("rede proibida em teste")

    monkeypatch.setattr(requests.Session, "request", _boom)


@pytest.mark.parametrize("name", YAMLS)
def test_build_backend_from_every_yaml(name: str) -> None:
    cfg = _cfg(name)
    for model in cfg.llm.models:
        backend = build_backend(cfg.llm, model.name, track=cfg.track)
        meta = backend.model_metadata()
        assert meta["backend"] == cfg.llm.backend
        if cfg.llm.backend == "transformers":
            assert isinstance(backend, TransformersBackend)
            assert backend.batch_size == cfg.llm.batch_size
            assert backend.seed == cfg.llm.seed
            assert backend.revision == REVISIONS[model.checkpoint]
        elif cfg.llm.backend == "ollama":
            assert isinstance(backend, OllamaBackend)
            assert backend.timeout == cfg.llm.request_timeout_s
            assert backend.seed == cfg.llm.seed
            assert backend.base_url == cfg.llm.ollama_endpoint.rstrip("/")
            assert meta["quantization"] == "Q4"
        else:
            assert isinstance(backend, FakeBackend)


@pytest.mark.parametrize("name", ["replication", "colab_pilot", "extended"])
def test_transformers_yamls_pin_revisions(name: str) -> None:
    cfg = _cfg(name)
    assert cfg.llm.backend == "transformers"
    for model in cfg.llm.models:
        assert model.revision == REVISIONS[model.checkpoint]


def test_pilot_local_declares_three_ollama_models_q4() -> None:
    cfg = _cfg("pilot")
    assert {m.checkpoint for m in cfg.llm.models} == OLLAMA_TAGS
    assert all(m.quantization == "Q4" for m in cfg.llm.models)
    assert cfg.experiments.models == ["mistral"]


def test_ollama_endpoint_default() -> None:
    assert LLMConfig().ollama_endpoint == "http://localhost:11434"


def test_ollama_endpoint_read_from_config() -> None:
    llm = LLMConfig(
        backend="ollama",
        ollama_endpoint="http://example.invalid:9999",
        models=[LLMModelConfig(name="m", checkpoint="mistral:7b-instruct")],
    )
    backend = build_backend(llm, "m", track=Track.EXTENDED_ANALYSIS)
    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == "http://example.invalid:9999"


def test_build_backend_requires_track() -> None:
    llm = LLMConfig(models=[LLMModelConfig(name="m", checkpoint="c")])
    with pytest.raises(TypeError):
        build_backend(llm, "m")  # type: ignore[call-arg]


def test_replication_requires_revision_at_build_and_in_schema() -> None:
    llm = LLMConfig(
        backend="transformers",
        models=[LLMModelConfig(name="m", checkpoint="org/model", revision=None)],
    )
    with pytest.raises(BackendConfigurationError, match="revis"):
        build_backend(llm, "m", track=Track.PAPER_REPLICATION)
    # em extended a revisão é opcional
    assert (
        build_backend(llm, "m", track=Track.EXTENDED_ANALYSIS).model_metadata()["revision"] is None
    )
    data = _cfg("replication").model_dump(mode="json")
    data["llm"]["models"][0]["revision"] = None
    with pytest.raises(ValueError, match="revision"):
        ExperimentConfig.model_validate(data)


def test_cache_and_prompts_dir_come_from_config(tmp_path: Path) -> None:
    prompts_src = Path(__file__).resolve().parents[2] / "prompts"
    llm = LLMConfig(
        cache_dir=str(tmp_path / "cache"),
        prompts_dir=str(prompts_src),
        max_retries=1,
        failure_policy="abort",
        models=[LLMModelConfig(name="m", checkpoint="c")],
    )
    cache = EvaluationCache.from_config(llm)
    assert cache is not None and cache.path is not None
    assert cache.path.parent == tmp_path / "cache"
    assert PromptLibrary.from_config(llm).source_dir == prompts_src
    evaluator = LLMEquationEvaluator.from_config(llm, FakeBackend())
    assert evaluator.max_retries == 1 and evaluator.failure_policy == "abort"
    assert evaluator.cache is not None and evaluator.cache.path == cache.path


def test_cache_disabled_returns_none() -> None:
    llm = LLMConfig(cache_enabled=False)
    assert EvaluationCache.from_config(llm) is None
    assert LLMEquationEvaluator.from_config(llm, FakeBackend()).cache is None
