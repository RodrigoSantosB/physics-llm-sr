"""T:docs — auditoria metodológica em ``references/``.

Verifica que ``scientific_assumptions.md`` cobre GAP-01..GAP-14 com os campos
obrigatórios, que os exemplos numéricos da distância entre árvores são
internamente consistentes, que a rastreabilidade cobre R1..R13 e que os CSVs
de ``references/paper_results`` seguem o esquema documentado.

Usa apenas a biblioteca-padrão e pytest.
"""

from __future__ import annotations

import csv
import re
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCES = REPO_ROOT / "references"
ASSUMPTIONS = REFERENCES / "scientific_assumptions.md"
TRACEABILITY = REFERENCES / "methodology_traceability.md"
PROMPTS_SOURCE = REFERENCES / "prompts_source.md"
PAPER_RESULTS = REFERENCES / "paper_results"

GAP_IDS = tuple(f"GAP-{i:02d}" for i in range(1, 15))
REQUIRED_GAP_FIELDS = (
    "Questão",
    "Evidência",
    "Hipótese adotada",
    "Justificativa",
    "Chave de configuração",
    "Impacto na comparabilidade",
    "Status",
)
ALLOWED_STATUS = {"hipótese", "desvio", "reconstruído", "limitação"}
COMPARISON_CLASSES = (
    "numericamente próxima",
    "qualitativa",
    "parcial",
    "divergente",
    "inconclusiva",
)
REQUIREMENT_IDS = tuple(f"R{i}" for i in range(1, 14))

RESULT_COLUMNS = [
    "table",
    "row_idx",
    "scenario",
    "sr",
    "llm",
    "prompt",
    "noise_type",
    "noise_location",
    "noise_level",
    "metric",
    "value",
    "source_page",
    "notes",
]
RESULT_FILES = {
    "table2_prompt_sensitivity.csv": ("2", 72 * 4),
    "table3_noise_zero_mean.csv": ("3", 9 * 15),
    "table4_noise_biased.csv": ("4", 9 * 15),
    "table5_benchmark_supplement.csv": ("5", 36 * 4),
}
SCENARIOS = {"free_fall", "harmonic_motion", "damped_wave", "average_3_scenarios"}
SR_NAMES = {"deap", "gplearn", "pysr"}
LLM_NAMES = {"mistral_7b", "llama2_7b", "falcon_7b", "none", "unspecified"}
PROMPTS = set("ABCDEFGH") | {"none", "unspecified"}
NOISE_TYPES = {"zero_mean", "biased"}
NOISE_LOCATIONS = {"inputs", "target", "both"}
NOISE_LEVELS = {"0.01", "0.02", "0.03", "0.04", "0.05"}
# Casas decimais publicadas por métrica (precisão usada nos critérios de comparação).
METRIC_DECIMALS = {"mae": 3, "mse": 3, "one_minus_r2": 2, "tree_distance": 2}
SOURCE_PAGE_PATTERN = re.compile(r"^(?:\d{1,2}|S\d+)$")
DECIMAL_PATTERN = re.compile(r"^\d+\.(\d+)$")

RUNTIME_COLUMNS = [
    "table",
    "row_idx",
    "scenario",
    "sr",
    "llm",
    "total_sr_s",
    "total_llm_s",
    "llm_sr_ratio",
    "avg_sr_per_call_s",
    "avg_llm_per_call_s",
    "avg_sr_per_iter_s",
    "avg_llm_per_iter_s",
    "source_page",
    "notes",
]


def _read(path: Path) -> str:
    assert path.is_file(), f"arquivo ausente: {path.relative_to(REPO_ROOT)}"
    return path.read_text(encoding="utf-8")


def _gap_sections(text: str) -> dict[str, str]:
    """Divide o documento em seções ``### GAP-NN`` (até o próximo ## ou ###)."""
    sections: dict[str, str] = {}
    pattern = re.compile(r"^### (GAP-\d{2})\b.*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        next_h2 = re.search(r"^## ", text[match.end() : end], re.MULTILINE)
        if next_h2:
            end = match.end() + next_h2.start()
        sections[match.group(1)] = text[match.start() : end]
    return sections


def _field_value(section: str, field: str) -> str | None:
    match = re.search(rf"^- \*\*{re.escape(field)}:\*\*(.*)$", section, re.MULTILINE)
    return match.group(1).strip() if match else None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    assert path.is_file(), f"arquivo ausente: {path.relative_to(REPO_ROOT)}"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


# --------------------------------------------------------------------------- GAPs


def test_all_gaps_present() -> None:
    sections = _gap_sections(_read(ASSUMPTIONS))
    missing = [gap for gap in GAP_IDS if gap not in sections]
    assert not missing, f"GAPs ausentes: {missing}"


@pytest.mark.parametrize("gap_id", GAP_IDS)
def test_gap_has_required_fields(gap_id: str) -> None:
    section = _gap_sections(_read(ASSUMPTIONS)).get(gap_id)
    assert section is not None, f"{gap_id} ausente"
    for field in REQUIRED_GAP_FIELDS:
        value = _field_value(section, field)
        assert value, f"{gap_id}: campo '{field}' ausente ou vazio"
    config_keys = re.findall(
        r"`([a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+)`",
        _field_value(section, "Chave de configuração") or "",
    )
    assert config_keys, f"{gap_id}: chave de configuração deve ser pontilhada e entre crases"
    evidence = _field_value(section, "Evidência") or ""
    assert re.search(r"p\.\s?\d+|suplemento|Tabela \d", evidence), (
        f"{gap_id}: evidência sem página/fonte"
    )


@pytest.mark.parametrize("gap_id", GAP_IDS)
def test_gap_status_is_allowed(gap_id: str) -> None:
    section = _gap_sections(_read(ASSUMPTIONS))[gap_id]
    status = (_field_value(section, "Status") or "").lower()
    tokens = {token.strip(" .") for token in re.split(r"[+,/]", status) if token.strip(" .")}
    assert tokens, f"{gap_id}: status vazio"
    assert tokens <= ALLOWED_STATUS, f"{gap_id}: status inválido {tokens - ALLOWED_STATUS}"


def test_gap06_defines_operator_set_per_algorithm() -> None:
    section = _gap_sections(_read(ASSUMPTIONS))["GAP-06"]
    for key in ("operators.deap", "operators.gplearn", "operators.pysr"):
        assert f"`{key}" in section, f"GAP-06 sem conjunto para {key}"


def test_gap08_covers_each_library() -> None:
    section = _gap_sections(_read(ASSUMPTIONS))["GAP-08"]
    for library in ("DEAP", "gplearn", "PySR"):
        assert library in section
    assert "make_fitness" in section
    assert "juliacall" in section or "PythonCall" in section


def test_gap14_has_cost_for_both_gpus() -> None:
    section = _gap_sections(_read(ASSUMPTIONS))["GAP-14"]
    assert "A100" in section and "GTX 1060" in section


def test_tree_distance_examples_are_consistent() -> None:
    """Cada exemplo ``| TD-xx | ... | S | N | d |`` satisfaz d = S/N ∈ [0, 1]."""
    section = _gap_sections(_read(ASSUMPTIONS))["GAP-10"]
    rows = re.findall(
        r"^\|\s*(TD-\d+)\s*\|[^\n]*?\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*$",
        section,
        re.MULTILINE,
    )
    assert len(rows) >= 8, "GAP-10 deve trazer ao menos 8 exemplos numéricos TD-xx"
    for example_id, total, count, distance in rows:
        expected = Decimal(total) / Decimal(count)
        published = Decimal(distance)
        tolerance = Decimal(1).scaleb(-(len(distance.split(".")[1]) if "." in distance else 0)) / 2
        assert abs(expected - published) <= tolerance, (
            f"{example_id}: {total}/{count} != {distance}"
        )
        assert Decimal(0) <= published <= Decimal(1), f"{example_id}: fora de [0,1]"


def test_inconsistencies_are_registered() -> None:
    text = _read(ASSUMPTIONS)
    for inc in ("INC-01", "INC-02", "INC-03", "INC-04", "INC-05", "INC-06"):
        assert re.search(rf"^\|\s*{inc}\s*\|", text, re.MULTILINE), f"{inc} não registrada"


def test_preregistered_comparison_criteria() -> None:
    text = _read(ASSUMPTIONS)
    match = re.search(
        r"^## Critérios de comparação pré-registrados(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "seção 'Critérios de comparação pré-registrados' ausente"
    body = match.group(1).lower()
    for label in COMPARISON_CLASSES:
        assert label in body, f"classe '{label}' não definida"
    for concept in ("ic95", "precisão publicada", "baseline"):
        assert concept in body


# ---------------------------------------------------------------- rastreabilidade


def test_traceability_covers_all_requirements() -> None:
    text = _read(TRACEABILITY)
    for requirement in REQUIREMENT_IDS:
        assert re.search(rf"\b{requirement}\b", text), f"{requirement} ausente na matriz"
    matrix = re.search(r"^## Matriz\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert matrix, "seção '## Matriz' ausente"
    rows = [line for line in matrix.group(1).splitlines() if line.startswith("|") and "T:" in line]
    assert len(rows) >= 13, "matriz deve ter linhas ligando seção → requisito → módulo → teste"
    for row in rows:
        assert re.search(r"\bR\d{1,2}\b", row), f"linha sem requisito: {row}"
        assert re.search(
            r"`(?:src/|prompts/|references/|configs/|notebooks/|tests/|Makefile|pyproject)",
            row,
        ), f"linha sem módulo: {row}"
    assert "T:docs" in text


def test_prompts_source_covers_variants() -> None:
    text = _read(PROMPTS_SOURCE)
    for variant in "ABCDEFGH":
        assert re.search(rf"^\|\s*{variant}\s*\|", text, re.MULTILINE), (
            f"variante {variant} ausente"
        )
    for criterion in ("c1", "c2", "c3"):
        assert criterion in text


# ---------------------------------------------------------------- paper_results


def test_paper_results_readme_mentions_supplement() -> None:
    readme = _read(PAPER_RESULTS / "README.md")
    assert "Tabela 5" in readme and "suplemento" in readme.lower()


@pytest.mark.parametrize("filename", sorted(RESULT_FILES))
def test_result_csv_schema_and_types(filename: str) -> None:
    table_id, expected_rows = RESULT_FILES[filename]
    header, rows = _read_csv(PAPER_RESULTS / filename)
    assert header == RESULT_COLUMNS
    assert len(rows) == expected_rows
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        assert row["table"] == table_id
        assert int(row["row_idx"]) >= 1
        assert row["scenario"] in SCENARIOS
        assert row["sr"] in SR_NAMES
        assert row["llm"] in LLM_NAMES
        assert row["prompt"] in PROMPTS
        assert row["noise_type"] in NOISE_TYPES
        assert row["noise_location"] in NOISE_LOCATIONS
        assert row["noise_level"] in NOISE_LEVELS
        assert row["metric"] in METRIC_DECIMALS
        match = DECIMAL_PATTERN.match(row["value"])
        assert match, f"valor não decimal: {row['value']!r}"
        assert len(match.group(1)) == METRIC_DECIMALS[row["metric"]], f"precisão inesperada: {row}"
        value = float(row["value"])
        assert value >= 0.0
        if row["metric"] in {"tree_distance", "one_minus_r2"}:
            assert value <= 1.0
        assert SOURCE_PAGE_PATTERN.match(row["source_page"]), row["source_page"]
        key = (row["row_idx"], row["metric"], row["noise_location"], row["noise_level"])
        assert key not in seen, f"célula duplicada: {key}"
        seen.add(key)


def test_table2_prompt_blocks_and_inc01() -> None:
    _, rows = _read_csv(PAPER_RESULTS / "table2_prompt_sensitivity.csv")
    by_prompt: dict[str, list[str]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt"], []).append(row["value"])
    assert sorted(by_prompt) == list("ABCDEFGH")
    assert all(len(values) == 36 for values in by_prompt.values())
    # INC-01: o artigo publica F, G e H idênticos a E (transcrição fiel).
    assert by_prompt["F"] == by_prompt["E"] == by_prompt["G"] == by_prompt["H"]
    assert by_prompt["D"] != by_prompt["E"]


def test_mse_not_below_squared_mae() -> None:
    """Desigualdade de Jensen (MSE ≥ MAE²), tolerando o arredondamento publicado."""
    half_unit = 0.0005
    for filename in (
        "table2_prompt_sensitivity.csv",
        "table5_benchmark_supplement.csv",
    ):
        _, rows = _read_csv(PAPER_RESULTS / filename)
        cells: dict[str, dict[str, float]] = {}
        for row in rows:
            cells.setdefault(row["row_idx"], {})[row["metric"]] = float(row["value"])
        for row_idx, metrics in cells.items():
            assert metrics["mse"] + half_unit >= (metrics["mae"] - half_unit) ** 2, (
                filename,
                row_idx,
            )


def test_runtime_csv_schema() -> None:
    header, rows = _read_csv(PAPER_RESULTS / "table6_runtime_supplement.csv")
    assert header == RUNTIME_COLUMNS
    assert len(rows) == 27
    for row in rows:
        assert row["table"] == "6"
        assert row["sr"] in SR_NAMES
        assert row["llm"] in {"mistral_7b", "llama2_7b", "falcon_7b"}
        for column in RUNTIME_COLUMNS[5:12]:
            assert float(row[column]) >= 0.0
        assert SOURCE_PAGE_PATTERN.match(row["source_page"])
