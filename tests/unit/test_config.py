"""T:config - carga válida e rejeições da configuração tipada."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from physics_llm_sr.config.loader import ConfigError, load_config, parse_config
from physics_llm_sr.config.schemas import ExperimentConfig
from physics_llm_sr.core_types import Mode, Track

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
ALL_CONFIGS = ["smoke", "pilot", "colab_pilot", "replication", "extended"]


def _smoke_dict() -> dict[str, Any]:
    with (CONFIGS_DIR / "smoke.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def test_smoke_yaml_loads_as_typed_object() -> None:
    cfg = load_config(CONFIGS_DIR / "smoke.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.mode is Mode.SMOKE
    assert cfg.data.n_simulations == 5
    assert cfg.sr.population == 10
    assert cfg.sr.generations == 3
    assert cfg.llm.backend == "fake"


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_every_shipped_config_is_valid(name: str) -> None:
    cfg = load_config(CONFIGS_DIR / f"{name}.yaml")
    assert cfg.loss.w1 + cfg.loss.w2 + cfg.loss.w3 == pytest.approx(1.0)


def test_replication_uses_paper_values() -> None:
    cfg = load_config(CONFIGS_DIR / "replication.yaml")
    assert cfg.track is Track.PAPER_REPLICATION
    assert cfg.mode is Mode.FULL
    assert cfg.experiments.repetitions == 25
    assert cfg.sr.population == 100
    assert cfg.sr.generations == 50
    assert cfg.sr.deap.crossover_prob == 0.05
    assert cfg.sr.deap.mutation_prob == 0.01
    assert cfg.sr.deap.tournament_size == 3
    assert cfg.sr.deap.max_depth == 8
    assert cfg.loss.normalization == "literal"
    assert cfg.sr.pysr.loss == "huber"
    assert cfg.data.n_simulations == 500
    assert cfg.data.max_steps == 1000


def test_extended_uses_normalized_loss() -> None:
    cfg = load_config(CONFIGS_DIR / "extended.yaml")
    assert cfg.track is Track.EXTENDED_ANALYSIS
    assert cfg.loss.normalization == "normalized"


def test_pilot_definitions() -> None:
    local = load_config(CONFIGS_DIR / "pilot.yaml")
    colab = load_config(CONFIGS_DIR / "colab_pilot.yaml")
    for cfg in (local, colab):
        assert cfg.mode is Mode.PILOT
        assert cfg.experiments.scenarios == ["free_fall"]
        assert sorted(cfg.experiments.algorithms) == ["deap", "gplearn", "pysr"]
        assert cfg.experiments.repetitions == 3
        assert cfg.llm.prompt_variants == ["E"]
        assert [m.name for m in cfg.llm.models] == ["mistral"]
    assert (local.llm.backend, local.sr.population, local.sr.generations) == ("ollama", 50, 10)
    assert (colab.llm.backend, colab.sr.population, colab.sr.generations) == (
        "transformers",
        100,
        50,
    )


def test_weights_not_summing_to_one_name_the_field() -> None:
    data = _smoke_dict()
    data["loss"].update(w1=0.5, w2=0.5, w3=0.5)
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "loss" in str(exc.value)
    assert "w1+w2+w3" in str(exc.value)


def test_negative_weight_names_the_field() -> None:
    data = _smoke_dict()
    data["loss"].update(w1=-0.2, w2=0.6, w3=0.6)
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "loss.w1" in str(exc.value)


def test_weight_sum_within_tolerance_is_accepted() -> None:
    data = _smoke_dict()
    data["loss"].update(w1=0.3333333, w2=0.3333333, w3=0.3333334)
    assert parse_config(data).loss.w3 == pytest.approx(1 / 3, abs=1e-6)


def test_unknown_top_level_field_is_rejected() -> None:
    data = _smoke_dict()
    data["populaton"] = 3
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "populaton" in str(exc.value)


def test_unknown_nested_field_is_rejected_with_path() -> None:
    data = _smoke_dict()
    data["sr"].setdefault("deap", {})["mutation"] = 0.1
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "sr.deap.mutation" in str(exc.value)


def test_invalid_backend_names_the_field() -> None:
    data = _smoke_dict()
    data["llm"]["backend"] = "openai"
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "llm.backend" in str(exc.value)


def test_nonzero_temperature_is_rejected() -> None:
    data = _smoke_dict()
    data["llm"]["temperature"] = 0.7
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "llm.temperature" in str(exc.value)


def test_split_must_sum_to_one() -> None:
    data = _smoke_dict()
    data["data"]["split"] = {"train": 0.7, "validation": 0.2, "test": 0.2}
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "data.split" in str(exc.value)


def test_normalized_loss_is_rejected_in_paper_replication() -> None:
    data = _smoke_dict()
    data["track"] = "paper_replication"
    data["loss"]["normalization"] = "normalized"
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "loss.normalization" in str(exc.value)


def test_experiment_must_reference_declared_model() -> None:
    data = _smoke_dict()
    data["experiments"]["models"] = ["gpt-unknown"]
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert "experiments.models" in str(exc.value)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="nao encontrado|não encontrado"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


# --- alinhamento com a auditoria (GAP-01, GAP-06, GAP-10, decisões pós-auditoria) ---

GAP06_OPERATORS = {
    "deap": ["add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos"],
    "gplearn": ["add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos"],
    "pysr": ["add", "sub", "mul", "div", "pow", "neg", "exp", "log", "sin", "cos"],
}


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_shipped_configs_follow_audit(name: str) -> None:
    cfg = load_config(CONFIGS_DIR / f"{name}.yaml")
    assert cfg.symbolic.tree_distance_constant_penalty == 1.0
    assert cfg.symbolic.tree_distance.descend_on_operator_mismatch is True
    assert cfg.symbolic.tree_distance.canonicalize is True
    assert cfg.symbolic.division_epsilon == 0.001
    assert cfg.symbolic.log_epsilon == 0.001
    assert cfg.symbolic.exp_clip == 100.0
    assert {k: sorted(v) for k, v in cfg.sr.operators.items()} == {
        k: sorted(v) for k, v in GAP06_OPERATORS.items()
    }
    assert cfg.loss.baseline_weights == pytest.approx((0.5, 0.5, 0.0))
    assert cfg.llm.evaluation_scope == "all"


def test_defaults_match_gap06_and_gap10() -> None:
    cfg = ExperimentConfig(mode=Mode.SMOKE, track=Track.PAPER_REPLICATION)
    assert cfg.symbolic.tree_distance_constant_penalty == 1.0
    assert "sqrt" not in cfg.sr.operators["deap"]
    assert {k: sorted(v) for k, v in cfg.sr.operators.items()} == {
        k: sorted(v) for k, v in GAP06_OPERATORS.items()
    }


def test_sqrt_is_a_valid_operator_name_for_extended() -> None:
    data = _smoke_dict()
    data["track"] = "extended_analysis"
    data["sr"] = {"operators": {"deap": ["add", "sqrt", "neg"]}}
    assert parse_config(data).sr.operators["deap"] == ["add", "sqrt", "neg"]


def test_baseline_weights_must_sum_to_one_and_have_w3_zero() -> None:
    data = _smoke_dict()
    data["loss"]["baseline_weights"] = [0.5, 0.4, 0.0]
    with pytest.raises(ConfigError, match="baseline_weights"):
        parse_config(data)
    data["loss"]["baseline_weights"] = [0.4, 0.3, 0.3]
    with pytest.raises(ConfigError, match="baseline_weights"):
        parse_config(data)


def test_llm_new_fields_defaults() -> None:
    llm = ExperimentConfig(mode=Mode.SMOKE, track=Track.PAPER_REPLICATION).llm
    assert llm.evaluation_scope == "all"
    assert llm.top_k is None
    assert llm.prompts_dir is None
    assert llm.batch_size >= 1
    assert llm.request_timeout_s > 0
    assert llm.seed >= 0


def test_top_k_forbidden_in_paper_replication() -> None:
    data = _smoke_dict()
    data["llm"] |= {"evaluation_scope": "top_k", "top_k": 10}
    with pytest.raises(ConfigError, match="top_k"):
        parse_config(data)
    data["llm"] |= {"top_k": 10}
    with pytest.raises(ConfigError, match="top_k"):
        parse_config(data)


def test_top_k_allowed_in_extended_and_requires_value() -> None:
    data = _smoke_dict()
    data["track"] = "extended_analysis"
    data["llm"] |= {"evaluation_scope": "top_k", "top_k": 10}
    assert parse_config(data).llm.top_k == 10
    data["llm"] |= {"evaluation_scope": "top_k", "top_k": None}
    with pytest.raises(ConfigError, match="top_k"):
        parse_config(data)
    data["llm"] |= {"evaluation_scope": "all", "top_k": 5}
    with pytest.raises(ConfigError, match="top_k"):
        parse_config(data)


def test_symbolic_consumers_read_new_config_fields() -> None:
    from physics_llm_sr.config.schemas import SymbolicConfig
    from physics_llm_sr.symbolic.tree_distance import TreeDistanceParams

    cfg = SymbolicConfig.model_validate(
        {"tree_distance": {"descend_on_operator_mismatch": False, "canonicalize": False}}
    )
    params = TreeDistanceParams.from_config(cfg)
    assert params.descend_on_operator_mismatch is False
    assert params.canonicalize is False
    assert params.alpha == 1.0
