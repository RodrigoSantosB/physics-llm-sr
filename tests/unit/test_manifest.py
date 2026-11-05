"""T:manifest - campos obrigatórios e hashes estáveis."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from physics_llm_sr.config.loader import load_config
from physics_llm_sr.core_types import Dataset
from physics_llm_sr.data.validation import compute_data_hash
from physics_llm_sr.llm.inference import FakeBackend, OllamaBackend
from physics_llm_sr.reproducibility.artifacts import (
    create_run_dir,
    load_json,
    load_manifest,
    load_parquet,
    save_json,
    save_manifest,
    save_parquet,
)
from physics_llm_sr.reproducibility.manifests import (
    UNKNOWN,
    build_manifest,
    git_sha,
    hash_config,
    hash_dataset,
    new_run_id,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
SMOKE = CONFIGS_DIR / "smoke.yaml"

REQUIRED_FIELDS = {
    "run_id",
    "created_at",
    "mode",
    "track",
    "config_hash",
    "data_hashes",
    "metrics",
    "discovered_expressions",
    "history_file",
    "stop_conditions",
    "git_sha",
    "python_version",
    "julia_version",
    "library_versions",
    "seeds",
    "hardware",
    "llm_models",
    "inference",
    "prompt",
    "timings",
    "interruption",
}


def _dataset(scale: float = 1.0) -> Dataset:
    x = np.arange(12, dtype=float).reshape(4, 3) * scale
    return Dataset(
        X=x,
        y_true=x[:, 0],
        y_observed=x[:, 0],
        feature_names=("a", "b", "c"),
        target_name="y",
        scenario="free_fall",
        simulation_ids=np.array([0, 0, 1, 1]),
        t=np.arange(4, dtype=float),
        metadata={},
    )


def test_run_ids_are_unique() -> None:
    assert len({new_run_id() for _ in range(200)}) == 200


def test_manifest_has_all_required_fields() -> None:
    cfg = load_config(SMOKE)
    m = build_manifest(
        cfg,
        datasets={"free_fall/clean": _dataset(), "free_fall/target_1pct": _dataset(2.0)},
        seeds={"data": 1},
        prompt={"variant": "E", "text": "hello"},
        timings={"fit": 1.5},
    )
    dumped = m.model_dump(mode="json")
    assert REQUIRED_FIELDS <= set(dumped)
    assert dumped["config_hash"] == hash_config(cfg)
    assert dumped["data_hashes"] == {
        "free_fall/clean": compute_data_hash(_dataset()),
        "free_fall/target_1pct": compute_data_hash(_dataset(2.0)),
    }
    assert dumped["seeds"] == {"data": 1}
    assert dumped["timings"] == {"fit": 1.5}
    assert dumped["prompt"]["variant"] == "E"
    assert dumped["llm_models"][0]["checkpoint"] == "fake-evaluator"
    assert dumped["inference"]["temperature"] == 0.0
    assert "numpy" in dumped["library_versions"]
    assert dumped["python_version"].count(".") >= 2
    assert dumped["interruption"] is None


def test_two_manifests_get_distinct_run_ids() -> None:
    cfg = load_config(SMOKE)
    assert build_manifest(cfg).run_id != build_manifest(cfg).run_id


def test_interruption_condition_is_recorded() -> None:
    cfg = load_config(SMOKE)
    m = build_manifest(cfg, interruption="early_stopping: 3 gerações sem melhoria")
    assert m.interruption is not None
    assert "early_stopping" in m.interruption


def test_config_hash_is_stable_and_sensitive() -> None:
    cfg = load_config(SMOKE)
    assert hash_config(cfg) == hash_config(load_config(SMOKE))
    assert re.fullmatch(r"[0-9a-f]{64}", hash_config(cfg))
    changed = cfg.model_copy(update={"output_dir": "elsewhere"})
    assert hash_config(changed) != hash_config(cfg)


def test_dataset_hash_is_stable_and_sensitive() -> None:
    assert hash_dataset(_dataset()) == hash_dataset(_dataset())
    assert hash_dataset(_dataset()) != hash_dataset(_dataset(scale=2.0))


def test_git_sha_unknown_outside_repo(tmp_path: Path) -> None:
    assert git_sha(tmp_path) == UNKNOWN


def test_git_sha_in_repo_is_hex() -> None:
    sha = git_sha(Path(__file__).parent)
    assert sha == UNKNOWN or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_manifest_roundtrip_on_disk(tmp_path: Path) -> None:
    cfg = load_config(SMOKE)
    m = build_manifest(cfg, datasets={"free_fall/clean": _dataset()})
    run_dir = create_run_dir(tmp_path, m.run_id)
    assert run_dir.is_dir()
    path = save_manifest(m, run_dir)
    assert path.name == "manifest.json"
    assert load_manifest(run_dir) == m


def test_json_and_parquet_roundtrip(tmp_path: Path) -> None:
    payload = {"a": np.float64(1.5), "b": np.array([1, 2]), "c": "x"}
    save_json(payload, tmp_path / "p.json")
    assert load_json(tmp_path / "p.json") == {"a": 1.5, "b": [1, 2], "c": "x"}
    df = pd.DataFrame({"x": [1.0, 2.0], "s": ["u", "v"]})
    save_parquet(df, tmp_path / "d.parquet")
    pd.testing.assert_frame_equal(load_parquet(tmp_path / "d.parquet"), df)


def test_create_run_dir_refuses_existing(tmp_path: Path) -> None:
    create_run_dir(tmp_path, "run-x")
    try:
        create_run_dir(tmp_path, "run-x")
    except FileExistsError:
        return
    raise AssertionError("esperava FileExistsError")


def test_single_dataset_hash_algorithm() -> None:
    assert hash_dataset(_dataset()) == compute_data_hash(_dataset())


def test_section_11_fields_are_recorded(tmp_path: Path) -> None:
    cfg = load_config(SMOKE)
    m = build_manifest(
        cfg,
        metrics={"free_fall/deap/mistral": {"mae": 0.1, "tree_distance": 0.0}},
        discovered_expressions=[{"scenario": "free_fall", "expression": "sqrt(2*9.81*height)"}],
        history_file="runs/x/history.parquet",
        interruption="early_stopping: 3 gerações sem melhoria",
        timings={"total": 9.0, "llm": 3.0},
    )
    d = m.model_dump(mode="json")
    assert d["metrics"]["free_fall/deap/mistral"]["mae"] == 0.1
    assert d["discovered_expressions"][0]["expression"].startswith("sqrt")
    assert d["history_file"] == "runs/x/history.parquet"
    ee = cfg.sr.early_stopping
    assert d["stop_conditions"]["early_stopping"]["patience"] == ee.patience
    assert d["stop_conditions"]["interruption"].startswith("early_stopping")
    assert d["timings"] == {"total": 9.0, "llm": 3.0}
    save_manifest(m, tmp_path)
    assert load_manifest(tmp_path).metrics == m.metrics


def test_llm_models_carry_effective_model_metadata() -> None:
    cfg = load_config(SMOKE)
    name = cfg.llm.models[0].name

    class _Session:
        def get(self, url: str, timeout: float) -> object:
            class R:
                def raise_for_status(self) -> None: ...
                def json(self) -> dict[str, object]:
                    return {"models": [{"name": "m", "digest": "sha256:abc"}]}

            return R()

    ollama = OllamaBackend("m", session=_Session())
    m = build_manifest(cfg, backends={name: ollama})
    entry = next(e for e in m.llm_models if e["name"] == name)
    assert entry["effective"]["digest"] == "sha256:abc"
    assert entry["effective"]["backend"] == "ollama"
    fake = build_manifest(cfg, backends={name: FakeBackend()})
    eff = next(e for e in fake.llm_models if e["name"] == name)["effective"]
    assert eff["backend"] == "fake"
