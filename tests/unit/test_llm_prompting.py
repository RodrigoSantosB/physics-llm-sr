"""T:prompts - 8 variantes A-H renderizadas; só D, F, G, H revelam a GT."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.prompting import (
    PROMPTS_DIR_ENV,
    VARIANT_IDS,
    PromptLibrary,
    PromptTemplateError,
    resolve_prompts_dir,
)
from physics_llm_sr.physics.registry import SCENARIO_NAMES, get_scenario
from physics_llm_sr.symbolic.expression import parse_infix, to_infix

REPO_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
GT_VARIANTS = {"D", "F", "G", "H"}
CANDIDATE = parse_infix("mass * t + cos(length)")


@pytest.fixture(scope="module")
def library() -> PromptLibrary:
    return PromptLibrary.load(REPO_PROMPTS)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
@pytest.mark.parametrize("variant", VARIANT_IDS)
def test_gt_disclosed_only_in_d_f_g_h(library: PromptLibrary, scenario: str, variant: str) -> None:
    ctx = get_scenario(scenario).physical_context()
    rendered = library.render(CANDIDATE, ctx, variant)
    full = rendered.system + rendered.user
    assert rendered.variant == variant
    assert to_infix(CANDIDATE) in rendered.user
    assert "{{" not in full and "}}" not in full
    assert rendered.gt_disclosed is (variant in GT_VARIANTS)
    assert (ctx.ground_truth in full) is (variant in GT_VARIANTS)


def test_all_eight_variants_loaded(library: PromptLibrary) -> None:
    assert tuple(sorted(library.variants)) == ("A", "B", "C", "D", "E", "F", "G", "H")
    assert all(library.variant(v).reconstructed for v in VARIANT_IDS)


def test_sections_follow_includes(library: PromptLibrary) -> None:
    ctx = get_scenario("free_fall").physical_context()
    a = library.render(CANDIDATE, ctx, "A").user
    e = library.render(CANDIDATE, ctx, "E").user
    assert "VARIABLE DESCRIPTIONS" not in a and "EXPERIMENT DESCRIPTION" not in a
    assert "VARIABLE DESCRIPTIONS" in e and ctx.experiment_description in e
    assert e.index("VARIABLE DESCRIPTIONS") < e.index("EXPERIMENT DESCRIPTION")
    for name, desc in ctx.variable_descriptions.items():
        assert name in e and desc in e


def test_literal_substitution_does_not_reexpand(library: PromptLibrary) -> None:
    ctx = PhysicalContext(
        scenario="x",
        variable_descriptions={"t": "{{ground_truth}} {0} {__class__}"},
        experiment_description="{{expression}}",
        ground_truth="SECRET_GT",
        units={},
    )
    user = library.render(CANDIDATE, ctx, "E").user
    assert "{{ground_truth}} {0} {__class__}" in user
    assert "{{expression}}" in user
    assert "SECRET_GT" not in user


def test_prompt_hash_stable_and_variant_specific(library: PromptLibrary) -> None:
    again = PromptLibrary.load(REPO_PROMPTS)
    assert library.prompt_hash("E") == again.prompt_hash("E")
    assert len({library.prompt_hash(v) for v in VARIANT_IDS}) == len(VARIANT_IDS)


def test_prompt_hash_changes_with_template(tmp_path: Path) -> None:
    copy = tmp_path / "prompts"
    shutil.copytree(REPO_PROMPTS, copy)
    before = PromptLibrary.load(copy).prompt_hash("E")
    system = copy / "base" / "system.txt"
    system.write_text(system.read_text(encoding="utf-8") + "\nextra\n", encoding="utf-8")
    assert PromptLibrary.load(copy).prompt_hash("E") != before


def test_unknown_placeholder_rejected(tmp_path: Path) -> None:
    copy = tmp_path / "prompts"
    shutil.copytree(REPO_PROMPTS, copy)
    evaluation = copy / "base" / "evaluation.txt"
    evaluation.write_text("{{expression}} {{unknown_name}}", encoding="utf-8")
    lib = PromptLibrary.load(copy)
    with pytest.raises(PromptTemplateError, match="unknown_name"):
        lib.render(CANDIDATE, get_scenario("free_fall").physical_context(), "A")


def test_inconsistent_variant_rejected(tmp_path: Path) -> None:
    copy = tmp_path / "prompts"
    shutil.copytree(REPO_PROMPTS, copy)
    d = copy / "variants" / "D.yaml"
    d.write_text(d.read_text(encoding="utf-8").replace("gt_disclosed: true", "gt_disclosed: false"))
    with pytest.raises(PromptTemplateError, match="gt_disclosed"):
        PromptLibrary.load(copy)


def test_unknown_variant_rejected(library: PromptLibrary) -> None:
    with pytest.raises(PromptTemplateError):
        library.render(CANDIDATE, get_scenario("free_fall").physical_context(), "Z")


def test_resolve_prompts_dir_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit"
    shutil.copytree(REPO_PROMPTS, explicit)
    env_dir = tmp_path / "env"
    shutil.copytree(REPO_PROMPTS, env_dir)
    monkeypatch.setenv(PROMPTS_DIR_ENV, str(env_dir))
    assert resolve_prompts_dir(explicit) == explicit.resolve()
    assert resolve_prompts_dir() == env_dir.resolve()
    monkeypatch.delenv(PROMPTS_DIR_ENV)
    monkeypatch.chdir(tmp_path)
    assert resolve_prompts_dir() == REPO_PROMPTS.resolve()


def test_resolve_prompts_dir_invalid_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_prompts_dir(tmp_path / "missing")
