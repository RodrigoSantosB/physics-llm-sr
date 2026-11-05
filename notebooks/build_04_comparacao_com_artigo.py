"""Gera ``04_comparacao_com_artigo.ipynb`` (``uv run python notebooks/build_04_comparacao_com_artigo.py``).

O .ipynb versionado é a saída deste script. Células só chamam ``physics_llm_sr.analysis``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common  # noqa: E402

NAME = "04_comparacao_com_artigo"
NOTEBOOK = Path(__file__).resolve().parent / "04_comparacao_com_artigo.ipynb"
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Acrescenta uma célula markdown."""
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    """Acrescenta uma célula de código."""
    CELLS.append(("code", nb_common.format_code(text.strip("\n"))))


def build() -> nbformat.NotebookNode:
    """Monta o notebook."""
    nb = nbformat.v4.new_notebook()
    nb.nbformat_minor = 5
    nb.metadata = {"kernelspec": KERNELSPEC, "language_info": {"name": "python"}}
    for i, (kind, src) in enumerate(CELLS):
        make = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        cell = make(src)
        cell["id"] = hashlib.sha1(f"{i}:{src}".encode()).hexdigest()[:8]
        nb.cells.append(cell)
    nbformat.validate(nb)
    return nb


def main() -> None:
    """Grava o notebook."""
    nbformat.write(build(), NOTEBOOK)
    print(f"gravado {NOTEBOOK} ({len(CELLS)} células)")


md(r"""
# Comparação artigo × experimentos locais

Taskin, Xie & Lazebnik, *Sci. Rep.* 16:1614 (2026), Tabelas 2–4 e suplemento (Tabelas 5–6), lado a lado com os
resultados locais persistidos. **Os valores publicados vêm de `references/paper_results/*.csv`; os locais, de
`results.parquet`/`history.parquet`.** O artigo só tem figuras esquemáticas (Fig. 1–2): os gráficos aqui são
visualizações próprias das tabelas publicadas. Publicado = vazado/tracejado; local = cheio/sólido.
Célula sem dado local aparece como `n/e` (não executado), nunca como zero. Cada artefato leva o rótulo de
modo/trilha: **piloto e smoke são execuções reduzidas e não são reprodução.**

As células só chamam `physics_llm_sr.analysis`. Defina `RUN_DIRS` (ou a variável `PHYSICS_LLM_SR_RUN_DIRS`); caminhos relativos valem a partir da raiz do projeto (no Colab, o Drive). A primeira célula de código prepara o ambiente (Drive, clone/pull do GitHub, `pip install -e` sem LLM).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/notebooks/04_comparacao_com_artigo.ipynb)
""")
PARAMETERS = r"""
# =====================================================================================
# PARÂMETROS DO NOTEBOOK 04 (edite SÓ esta célula)
# =====================================================================================
# Caminhos relativos são resolvidos na RAIZ DO PROJETO (no Colab, o Drive:
# /content/drive/MyDrive/physics-llm-sr-runs). Uso principal: baseline (01) x 1 LLM (02).
RUN_DIRS = [
    # "results/nb01_baseline/<hash>/notebook_run",
    # "results/nb02_free_fall_benchmark_mistral/<hash>/notebook_run",
]
OUT_DIR = None  # None => <raiz>/results/nb04_analise/<timestamp>/
"""
code(nb_common.setup_cell(parameters=PARAMETERS, pip_extras="", hf_home=False))
code(r"""
from physics_llm_sr.analysis import notebook as nb

OUT = nb.generate(RUN_DIRS, OUT_DIR)
FACTS = nb.facts(OUT)
print("saída:", OUT)
for label in FACTS["labels"]:
    print("rótulo:", label)
""")

SECTIONS = [
    (
        "Tabela 2: sensibilidade a prompts A–H (F1)",
        "t2_prompt_sensitivity",
        "F1",
        "Artigo p.9–10, Tabela 2: MAE, MSE, 1−R² e distância por prompt (A–H), LLM e SR, média dos três cenários (ruído "
        "padrão 1%, hipótese). Prompts D, F, G, H revelam a GT. F, G, H são idênticos a E (INC-01).",
        [
            "Em quais células houve execução local? Os prompts que revelam a GT (D, F, G, H) ficam abaixo dos que não revelam?",
            "A Tabela 2 é média de três cenários: o piloto cobre só um cenário. Que conclusão ela permite?",
        ],
        [
            "arm_short",
            "llm",
            "sr",
            "prompt",
            "metric",
            "published",
            "local_mean",
            "n",
            "status_local",
        ],
    ),
    (
        "Tabelas 3 e 4: robustez a ruído (F2)",
        "t3_noise_zero_mean",
        "F2",
        "Artigo p.10, Tabelas 3 (média zero) e 4 (enviesado): distância vs nível 1–5%, por cenário, SR e localização. "
        "LLM e prompt não são informados (GAP-12).",
        [
            "A distância local cresce com o nível de ruído como a publicada?",
            "Quais localizações e cenários não foram executados localmente?",
        ],
        [
            "arm",
            "scenario",
            "sr",
            "noise_location",
            "noise_level",
            "published",
            "local_mean",
            "n",
            "status_local",
        ],
    ),
    (
        "Tabela 5 (suplemento): benchmark baseline + 3 LLM (F3)",
        "t5_benchmark",
        "F3",
        'Suplemento, Tabela 5: 3 cenários × (baseline + 3 LLM) × 3 SR. INC-08: as linhas de "Dropping ball" com LLM '
        "coincidem com o prompt A da Tabela 2.",
        [
            "O erro local do baseline tem a mesma ordem de grandeza do publicado? Por que (escala literal, GAP-02)?",
            "O IC95 local inclui o valor publicado?",
        ],
        [
            "arm_short",
            "scenario",
            "sr",
            "llm",
            "metric",
            "published",
            "local_mean",
            "abs_diff",
            "classe",
        ],
    ),
    (
        "Tabela 6 (suplemento): tempo e chamadas (F4)",
        "t6_runtime",
        "F4",
        "Suplemento, Tabela 6: tempo de SR e de LLM (s) e razão LLM/SR. INC-09: as linhas Falcon são incompatíveis "
        "com os totais.",
        [
            "O tempo do LLM domina o tempo local, como no artigo?",
            "Quantas chamadas novas por execução? O custo é comparável ao publicado (derivado, impreciso)?",
        ],
        [
            "arm_short",
            "scenario",
            "sr",
            "llm",
            "published_total_llm_s",
            "local_total_llm_s",
            "published_ratio",
            "local_ratio",
            "local_calls_mean",
            "status_local",
        ],
    ),
    (
        "Custo do LLM (formato Tabela V de estilo)",
        "t6_llm_cost",
        "F4",
        "Sem equivalente publicado exato: chamadas novas, taxa de falha de interpretação e mediana da razão LLM/RS.",
        ["Há campos `n/d`? Quais deixam de ser persistidos?"],
        None,
    ),
    (
        "Efeito pareado do LLM (F5)",
        "effect_paired",
        "F5",
        "O artigo não reporta pareamento por repetição; aqui, baseline − LLM na mesma repetição (positivo = LLM melhor).",
        [
            "O IC95 da diferença exclui zero? O Wilcoxon tem poder com este n?",
            "Em quantas repetições a expressão final mudou com o LLM?",
        ],
        [
            "arm_short",
            "sr",
            "llm",
            "metric",
            "n",
            "mean_diff",
            "ci_low",
            "ci_high",
            "p_value",
            "n_expr_changed",
        ],
    ),
    (
        "Braços literal × normalized (F6)",
        "effect_arms",
        "F6",
        "Sem equivalente publicado. `normalized` é desvio declarado (extended_analysis) e nunca vira reprodução.",
        ["O efeito do LLM muda de sinal ou de magnitude entre os braços?"],
        None,
    ),
    (
        "Decomposição da perda (F7)",
        "loss_decomposition",
        "F7",
        "Sem equivalente publicado. Mostra o peso de w1·e, w2·s, w3·c na perda final (GAP-01/GAP-02).",
        ["Se a fração de w3·c é pequena na escala literal, o LLM pode influenciar a busca?"],
        None,
    ),
    (
        "Convergência (F8)",
        None,
        "F8",
        "Sem equivalente publicado. best_loss por geração (history.parquet). O baseline usa w3 = 0: valores absolutos de L "
        "não são comparáveis.",
        ["O LLM acelera a convergência (em gerações) ou só muda o nível de L?"],
        None,
    ),
]

for title, tname, fig, paper, questions, cols in SECTIONS:
    md(f"## {title}\n\n**O que o artigo reporta.** {paper}")
    if tname:
        code(f'nb.table(OUT, "{tname}", {cols!r})' if cols else f'nb.table(OUT, "{tname}")')
    code(f'nb.show_figures(OUT, "{fig}")')
    qs = "\n".join(f"- {q}" for q in questions)
    md(f"**Leitura guiada.** Perguntas para o pesquisador (sem conclusões prontas):\n\n{qs}")

md("## Resumo das classificações pré-registradas")
code(r"""
print(f"células comparáveis: {FACTS['cells_total']}; com execução local: {FACTS['cells_executed']}")
print("classes (somente células executadas):", FACTS["classes"])
print(f"células da Tabela 5 com dado local: {FACTS['t5_executed']}")
print(f"repetições com expressão alterada pelo LLM (soma): {FACTS['expr_changed']}")
FACTS["effect_mae"]
""")
md(
    "**Leitura guiada.** Qual fração das células publicadas foi realmente executada? Em execução reduzida, "
    "toda classe é `inconclusiva` por regra (n < 25 ou modo ≠ full): o que falta para uma reprodução completa?"
)
code(r"""
FACTS["loss_frac_w3c"]
""")

if __name__ == "__main__":
    main()
