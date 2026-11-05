"""Templates de prompt versionados e variantes A-H (PRD §6.4, GAP-13).

Arquivos em ``prompts/`` (ver ``prompts/README.md``):

- ``base/system.txt``: mensagem de sistema, sem placeholders.
- ``base/evaluation.txt``: mensagem do usuário, com ``{{expression}}`` e ``{{context_block}}``.
- ``variants/<ID>.yaml``: ``includes`` e ``sections`` (com ``{{variable_descriptions}}``,
  ``{{experiment_description}}``, ``{{ground_truth}}``).

Renderização por substituição literal, em passagem única: cada ``{{nome}}`` do template é
trocado pelo valor correspondente; valores inseridos nunca são reprocessados (sem
``str.format``/``string.Template``), então texto do contexto não injeta placeholders.

Resolução do diretório (:func:`resolve_prompts_dir`): argumento explícito → variável de
ambiente :data:`PROMPTS_DIR_ENV` → ``prompts/`` em algum ancestral do pacote instalado
(``pip install -e .`` no Colab) → ``prompts/`` em algum ancestral do diretório corrente.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from physics_llm_sr.config.schemas import LLMConfig
from physics_llm_sr.core_types import PhysicalContext
from physics_llm_sr.llm.interfaces import LLMError
from physics_llm_sr.symbolic.expression import Expr, to_infix

#: Variável de ambiente que aponta para o diretório ``prompts/``.
PROMPTS_DIR_ENV = "PHYSICS_LLM_SR_PROMPTS_DIR"
#: Nome do diretório procurado nos ancestrais do pacote e do cwd.
PROMPTS_DIRNAME = "prompts"
SYSTEM_FILE = Path("base") / "system.txt"
EVALUATION_FILE = Path("base") / "evaluation.txt"
VARIANTS_DIR = "variants"
#: Identificadores das variantes experimentais (PRD §6.4).
VARIANT_IDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
#: Ordem fixa das seções no ``context_block`` (prompts/README.md).
SECTION_ORDER: tuple[str, ...] = ("variables", "experiment", "ground_truth")
GT_SECTION = "ground_truth"
#: Versão da lógica de renderização (entra no hash do prompt).
RENDERER_VERSION = "1"
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


class PromptTemplateError(LLMError, ValueError):
    """Template/variante inválido ou placeholder sem valor."""


def _is_prompts_dir(path: Path) -> bool:
    return (path / SYSTEM_FILE).is_file() and (path / EVALUATION_FILE).is_file()


def resolve_prompts_dir(explicit: str | Path | None = None) -> Path:
    """Localiza o diretório ``prompts/`` (ver docstring do módulo para a ordem).

    Raises:
        FileNotFoundError: nenhum candidato válido; a mensagem lista os tentados.
    """
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not _is_prompts_dir(path):
            raise FileNotFoundError(f"diretório de prompts inválido: {path}")
        return path
    env = os.environ.get(PROMPTS_DIR_ENV)
    if env:
        path = Path(env).expanduser().resolve()
        if not _is_prompts_dir(path):
            raise FileNotFoundError(f"{PROMPTS_DIR_ENV}={env!r} não contém prompts válidos")
        return path
    tried: list[Path] = []
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for ancestor in (start, *start.parents):
            candidate = ancestor / PROMPTS_DIRNAME
            tried.append(candidate)
            if _is_prompts_dir(candidate):
                return candidate
    raise FileNotFoundError(
        f"diretório 'prompts/' não encontrado; defina {PROMPTS_DIR_ENV}. Tentados: {tried}"
    )


@dataclass(frozen=True)
class PromptVariantSpec:
    """Definição de uma variante (``variants/<ID>.yaml``)."""

    id: str
    description: str
    includes: tuple[str, ...]
    gt_disclosed: bool
    reconstructed: bool
    sections: Mapping[str, str]


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt pronto para o backend, com metadados de rastreabilidade."""

    variant: str
    system: str
    user: str
    gt_disclosed: bool
    prompt_hash: str


def substitute(template: str, values: Mapping[str, str]) -> str:
    """Troca cada ``{{nome}}`` por ``values[nome]`` em passagem única.

    Raises:
        PromptTemplateError: placeholder sem valor em ``values``.
    """
    missing = sorted({m for m in _PLACEHOLDER.findall(template) if m not in values})
    if missing:
        raise PromptTemplateError(f"placeholders sem valor: {missing}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def format_variable_descriptions(ctx: PhysicalContext) -> str:
    """Uma variável por linha: ``- nome (unidade): descrição``."""
    lines = []
    for name, description in ctx.variable_descriptions.items():
        unit = ctx.units.get(name)
        label = f"{name} ({unit})" if unit else name
        lines.append(f"- {label}: {description}")
    return "\n".join(lines)


def _parse_variant(path: Path) -> PromptVariantSpec:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PromptTemplateError(f"{path.name}: YAML deve ser um mapeamento")
    vid = data.get("id")
    if vid != path.stem:
        raise PromptTemplateError(f"{path.name}: id {vid!r} difere do nome do arquivo")
    includes = data.get("includes") or []
    sections = data.get("sections") or {}
    if not isinstance(includes, list) or not all(i in SECTION_ORDER for i in includes):
        raise PromptTemplateError(f"{path.name}: includes inválido {includes!r}")
    if not isinstance(sections, dict) or set(sections) != set(includes):
        raise PromptTemplateError(f"{path.name}: sections deve ter exatamente {includes}")
    if not all(isinstance(v, str) for v in sections.values()):
        raise PromptTemplateError(f"{path.name}: seções devem ser texto")
    gt_disclosed = data.get("gt_disclosed")
    if gt_disclosed is not (GT_SECTION in includes):
        raise PromptTemplateError(
            f"{path.name}: gt_disclosed={gt_disclosed!r} inconsistente com includes"
        )
    reconstructed = data.get("reconstructed")
    if not isinstance(reconstructed, bool):
        raise PromptTemplateError(f"{path.name}: reconstructed deve ser bool")
    return PromptVariantSpec(
        id=vid,
        description=str(data.get("description", "")),
        includes=tuple(i for i in SECTION_ORDER if i in includes),
        gt_disclosed=gt_disclosed,
        reconstructed=reconstructed,
        sections=dict(sections),
    )


class PromptLibrary:
    """Templates base + variantes A-H carregados de um diretório ``prompts/``."""

    def __init__(
        self,
        system_template: str,
        evaluation_template: str,
        variants: Mapping[str, PromptVariantSpec],
        source_dir: Path | None = None,
    ) -> None:
        """Monta a biblioteca; use :meth:`load` para ler do disco."""
        missing = [v for v in VARIANT_IDS if v not in variants]
        if missing:
            raise PromptTemplateError(f"variantes ausentes: {missing}")
        self.system_template = system_template
        self.evaluation_template = evaluation_template
        self._variants = dict(variants)
        self.source_dir = source_dir
        self._hashes = {vid: self._compute_hash(spec) for vid, spec in self._variants.items()}

    @classmethod
    def load(cls, prompts_dir: str | Path | None = None) -> PromptLibrary:
        """Lê templates e variantes de :func:`resolve_prompts_dir` (``prompts_dir``)."""
        root = resolve_prompts_dir(prompts_dir)
        variants = {p.stem: _parse_variant(p) for p in sorted((root / VARIANTS_DIR).glob("*.yaml"))}
        return cls(
            system_template=(root / SYSTEM_FILE).read_text(encoding="utf-8"),
            evaluation_template=(root / EVALUATION_FILE).read_text(encoding="utf-8"),
            variants=variants,
            source_dir=root,
        )

    @classmethod
    def from_config(cls, llm_config: LLMConfig) -> PromptLibrary:
        """Carrega de ``llm.prompts_dir`` (``None`` = ``prompts/`` do repositório)."""
        return cls.load(llm_config.prompts_dir)

    @property
    def variants(self) -> tuple[str, ...]:
        """Identificadores carregados."""
        return tuple(self._variants)

    def variant(self, variant_id: str) -> PromptVariantSpec:
        """Especificação da variante ``variant_id``."""
        try:
            return self._variants[variant_id]
        except KeyError:
            raise PromptTemplateError(f"variante desconhecida: {variant_id!r}") from None

    def _compute_hash(self, spec: PromptVariantSpec) -> str:
        payload = {
            "renderer": RENDERER_VERSION,
            "system": self.system_template,
            "evaluation": self.evaluation_template,
            "variant": {
                "id": spec.id,
                "includes": list(spec.includes),
                "sections": dict(spec.sections),
            },
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def prompt_hash(self, variant_id: str) -> str:
        """SHA-256 da versão do prompt (templates + variante + versão do renderizador)."""
        self.variant(variant_id)
        return self._hashes[variant_id]

    def render(
        self, expression: Expr, physical_context: PhysicalContext, variant_id: str
    ) -> RenderedPrompt:
        """Renderiza ``(system, user)`` com a expressão em infixo."""
        spec = self.variant(variant_id)
        section_values = {
            "variable_descriptions": format_variable_descriptions(physical_context),
            "experiment_description": physical_context.experiment_description,
            "ground_truth": physical_context.ground_truth,
        }
        context_block = "".join(
            substitute(spec.sections[name], section_values) for name in spec.includes
        )
        user = substitute(
            self.evaluation_template,
            {"expression": to_infix(expression), "context_block": context_block},
        )
        return RenderedPrompt(
            variant=spec.id,
            system=substitute(self.system_template, {}),
            user=user,
            gt_disclosed=spec.gt_disclosed,
            prompt_hash=self._hashes[spec.id],
        )
