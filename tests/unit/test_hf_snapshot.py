"""Pré-download verificado dos pesos HF: corrupção ('header too large'), espaço e padrões."""

from __future__ import annotations

import json
import shutil
import struct
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from physics_llm_sr.llm import hf_snapshot
from physics_llm_sr.llm.interfaces import BackendError

REPO = "mistralai/Mistral-7B-Instruct-v0.1"
SHA = "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca"
NAME = "model-00001-of-00002.safetensors"


def good_file(path: Path, payload: int = 16) -> int:
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload)
    return path.stat().st_size


def bad_file(path: Path) -> None:
    path.write_bytes(struct.pack("<Q", 10**15) + b"x" * 64)  # cabeçalho gigante


class Hub:
    """Stub de huggingface_hub: o 1º download grava `first`, os seguintes gravam o bom."""

    def __init__(
        self, tmp: Path, first_bad: bool, always_bad: bool = False, size: int | None = None
    ):
        self.snap = tmp / "snap"
        self.snap.mkdir()
        self.first_bad, self.always_bad, self.size = first_bad, always_bad, size
        self.calls: list[dict[str, Any]] = []

    def snapshot_download(self, repo: str, **kw: Any) -> str:
        self.calls.append({"repo": repo, **kw})
        target = self.snap / NAME
        bad = self.always_bad or (self.first_bad and len(self.calls) == 1)
        if bad:
            bad_file(target)
        else:
            good_file(target)
        return str(self.snap)

    def try_to_load_from_cache(self, *a: Any, **k: Any) -> None:
        return None

    def HfApi(self) -> Any:  # noqa: N802
        hub = self

        class Api:
            def model_info(self, *a: Any, **k: Any) -> Any:
                if hub.size is None:
                    raise OSError("offline")
                sib = types.SimpleNamespace(rfilename=NAME, size=hub.size)
                junk = types.SimpleNamespace(rfilename="pytorch_model-00001.bin", size=10**9)
                return types.SimpleNamespace(siblings=[sib, junk])

        return Api()


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))

    def _install(**kw: Any) -> Hub:
        hub = Hub(tmp_path, **kw)
        mod = types.ModuleType("huggingface_hub")
        mod.snapshot_download = hub.snapshot_download  # type: ignore[attr-defined]
        mod.try_to_load_from_cache = hub.try_to_load_from_cache  # type: ignore[attr-defined]
        mod.HfApi = hub.HfApi  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
        return hub

    return _install


def test_valid_header_passes_without_redownload(install: Any) -> None:
    hub = install(first_bad=False)
    assert hf_snapshot.download_verified_snapshot(REPO, SHA) == hub.snap
    assert len(hub.calls) == 1 and "force_download" not in hub.calls[0]
    assert hub.calls[0]["revision"] == SHA


def test_bin_files_not_in_allow_patterns(install: Any) -> None:
    hub = install(first_bad=False)
    hf_snapshot.download_verified_snapshot(REPO, SHA)
    patterns = hub.calls[0]["allow_patterns"]
    assert "*.safetensors" in patterns
    assert not any(".bin" in p for p in patterns)


def test_huge_header_detected_removed_and_redownloaded_once(install: Any) -> None:
    hub = install(first_bad=True)
    path = hf_snapshot.download_verified_snapshot(REPO, SHA)
    assert [c.get("force_download") for c in hub.calls] == [None, True]
    assert hf_snapshot.header_problem(path / NAME) is None


def test_size_mismatch_detected_even_with_plausible_header(install: Any) -> None:
    hub = install(first_bad=False, size=10**6)  # tamanho esperado != do arquivo gerado
    with pytest.raises(BackendError, match="esperado 1000000"):
        hf_snapshot.download_verified_snapshot(REPO, SHA)
    assert len(hub.calls) == 2


def test_second_failure_raises_clear_backend_error(install: Any, tmp_path: Path) -> None:
    hub = install(first_bad=True, always_bad=True)
    with pytest.raises(BackendError) as err:
        hf_snapshot.download_verified_snapshot(REPO, SHA)
    msg = str(err.value)
    assert NAME in msg and str(tmp_path / "hf_home") in msg and "/content/hf_home" in msg
    assert "header too large" in msg
    assert len(hub.calls) == 2


def test_truncated_file_has_problem(tmp_path: Path) -> None:
    f = tmp_path / "x.safetensors"
    f.write_bytes(b"\x01\x02")
    assert hf_snapshot.header_problem(f) is not None
    f.write_bytes(struct.pack("<Q", 8) + b"notjson!")
    assert "JSON" in str(hf_snapshot.header_problem(f))


def test_insufficient_space_fails_early(install: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install(first_bad=False, size=10**9)
    usage = types.SimpleNamespace(total=0, used=0, free=10**9)  # < 1 GB + 1 GB
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage)
    with pytest.raises(BackendError, match="espaço insuficiente"):
        hf_snapshot.download_verified_snapshot(REPO, SHA)
    assert hub.calls == []


def test_colab_plan_defaults_hf_home_to_local_disk() -> None:
    from physics_llm_sr import colab

    plan = colab.setup_environment(
        repo_dir=Path("/r"), mount_drive=True, in_colab=True, dry_run=True, environ={}
    )
    hf = next(a for a in plan.actions if a.name == "HF_HOME")
    assert hf.value == "/content/hf_home"
