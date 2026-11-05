"""T:prompts (vazamento) - o prompt não revela a GT nem exemplos equivalentes a ela."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from physics_llm_sr.llm.prompting import VARIANT_IDS, PromptLibrary
from physics_llm_sr.physics.registry import SCENARIO_NAMES, get_scenario
from physics_llm_sr.symbolic.expression import Const, Expr, Op, Var, parse_infix
from physics_llm_sr.symbolic.tree_distance import tree_distance

REPO_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
GT_VARIANTS = {"D", "F", "G", "H"}
EXAMPLE_LINE = re.compile(r"^Example\s+\d+\s*-\s*expression:\s*(?P<expr>.+?),\s*target:", re.M)
MIN_EXAMPLES = 3
CANDIDATE = parse_infix("mass * t + cos(length)")


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _anonymize(expr: Expr) -> Expr:
    """Unifica folhas: nomes de variáveis e valores de constantes viram a mesma folha."""
    if isinstance(expr, (Var, Const)):
        return Var("v")
    assert isinstance(expr, Op)
    return Op(expr.name, tuple(_anonymize(c) for c in expr.children))


def _example_expressions(system_text: str) -> list[str]:
    return [m.group("expr").strip() for m in EXAMPLE_LINE.finditer(system_text)]


@pytest.fixture(scope="module")
def library() -> PromptLibrary:
    return PromptLibrary.load(REPO_PROMPTS)


def _ground_truths() -> list[tuple[str, str]]:
    return [
        (name, gt)
        for name in SCENARIO_NAMES
        for gt in get_scenario(name).equivalent_ground_truths()
    ]


@pytest.mark.parametrize("variant", sorted(set(VARIANT_IDS) - GT_VARIANTS))
@pytest.mark.parametrize(("scenario", "gt"), _ground_truths())
def test_non_disclosing_variants_do_not_contain_gt(
    library: PromptLibrary, scenario: str, gt: str, variant: str
) -> None:
    ctx = get_scenario(scenario).physical_context()
    rendered = library.render(CANDIDATE, ctx, variant)
    assert rendered.gt_disclosed is False
    assert _squash(gt) not in _squash(rendered.system + rendered.user)


@pytest.mark.parametrize("variant", sorted(GT_VARIANTS))
@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_disclosing_variants_do_contain_gt(
    library: PromptLibrary, scenario: str, variant: str
) -> None:
    ctx = get_scenario(scenario).physical_context()
    rendered = library.render(CANDIDATE, ctx, variant)
    assert _squash(ctx.ground_truth) in _squash(rendered.user)


def test_examples_are_extracted(library: PromptLibrary) -> None:
    assert len(_example_expressions(library.system_template)) >= MIN_EXAMPLES


@pytest.mark.parametrize(("scenario", "gt"), _ground_truths())
def test_prompt_examples_not_structurally_equivalent_to_gt(
    library: PromptLibrary, scenario: str, gt: str
) -> None:
    gt_tree = _anonymize(parse_infix(gt))
    for text in _example_expressions(library.system_template):
        example = _anonymize(parse_infix(text))
        assert tree_distance(example, gt_tree) > 0.0, (
            f"exemplo {text!r} tem a mesma estrutura de {gt!r} ({scenario})"
        )


def test_prompt_examples_do_not_contain_gt_text(library: PromptLibrary) -> None:
    squashed = _squash(library.system_template)
    for _, gt in _ground_truths():
        assert _squash(gt) not in squashed


def test_readme_declares_reconstructed_v11() -> None:
    readme = (REPO_PROMPTS / "README.md").read_text(encoding="utf-8")
    assert "v1.1-reconstructed" in readme
    assert "reconstructed" in readme
