# ruff: noqa: E501
"""Gera ``06_busca_de_convergencia.ipynb`` (``uv run python notebooks/build_06_busca_convergencia.py``).

O .ipynb versionado é a saída deste script; edite as células aqui e regenere. O teste
``tests/e2e/test_notebook_06.py`` confere que o arquivo corresponde ao gerador.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common  # noqa: E402

NAME = "06_busca_de_convergencia"
NOTEBOOK = Path(__file__).resolve().parent / f"{NAME}.ipynb"
B = nb_common.NotebookBuilder(NOTEBOOK)
md, code = B.md, B.code


def build():  # type: ignore[no-untyped-def]
    """Monta o notebook (usado pelo teste de correspondência com o .ipynb)."""
    return B.build()


def main() -> None:
    """Grava o notebook."""
    B.write()


md(r"""
# Notebook 06 — Busca de configuração para a convergência do baseline (sem LLM)

**Pergunta:** que configuração de busca faz o baseline de regressão simbólica **recuperar a equação de referência** (ou uma forma
fisicamente equivalente declarada pelo cenário) no **regime sem ruído**, de forma reproduzível, antes da inserção do LLM?

O notebook 05 mostrou que, com o orçamento padrão (população 100 × 50 gerações) e parada antecipada agressiva (0,1% por 3 gerações),
as repetições na queda livre "não convergem". Aqui a busca de configuração é feita de forma **sistemática**:

1. **diagnóstico de expressividade** — o conjunto de operadores consegue escrever a referência? qual a perda de cada forma e de candidatas triviais?
2. **triagem** — configurações candidatas, em ordem de hipótese (H1 parada antecipada, H2 faixa de constantes, H3 pressão seletiva, H4 orçamento, H5 operadores);
3. **refinamento** — a configuração **mais barata** que ainda converge;
4. **confirmação em sementes novas** — a configuração escolhida é reexecutada com sementes que não participaram da busca;
5. **exportação** — `best_config.yaml`/`best_config.json` e o bloco para colar no notebook 05.

**Glossário**

| termo | significado |
|---|---|
| **equação de referência** | a lei física analítica que gera os dados (ex.: `sqrt(2*9.81*height)`) |
| **forma equivalente** | outra expressão fisicamente idêntica declarada pelo cenário (queda livre: `9.81*t`, pois `height = g t²/2`) |
| **critério de convergência** | repetição bem sucedida: `min(distância estrutural, distância equivalente, distância algébrica) ≤ limiar` **e** `1 − R²` (teste limpo) `≤ limiar`; a configuração converge se a fração de repetições bem sucedidas `≥ fração mínima` |
| **distância algébrica** | métrica secundária: distância estrutural após uma forma normal algébrica (identidades exatas como `u + u = 2u`, `u/c = (1/c)u`, `cos` par); a primária continua sendo a da comparação com o artigo |
| **configuração candidata** | dicionário de caminhos de configuração (`sr.*`, `loss.normalization`) → valor, com os **desvios** em relação ao baseline literal marcados |
| **desvio declarado** | valor diferente do artigo / das hipóteses H-02 e H-03 do baseline literal; a reprodução literal continua usando os valores do artigo |
| **repetição (semente derivada)** | execução independente; dados, divisão treino/validação/teste e busca usam sementes derivadas da semente mestra |

**Como usar:** edite só a célula de parâmetros (Etapa 1) e execute de cima para baixo. Nenhum módulo de LLM é importado; basta CPU
(PySR exige o Julia, instalado na primeira importação). Toda a lógica está em `physics_llm_sr.workflows.convergence_search`.
""")

md(r"""
## Etapa 0 — Ambiente local

**Objetivo:** confirmar o ambiente antes de qualquer cálculo.
**Chamada:** verifica o Python, importa `physics_llm_sr` e mostra o relatório do ambiente.
**Saída a conferir:** `python_ok = True`; `modulos_de_llm_importados = []`.
**Critério:** a célula não pode levantar erro (se o pacote não for importável, rode `pip install -e .` na raiz do repositório).
""")

code(r"""
# Etapa 0 — ambiente local
import sys

import pandas as pd
from IPython.display import display

if sys.version_info < (3, 10):  # noqa: UP036
    raise RuntimeError(f"Python >= 3.10 é necessário; encontrado {sys.version.split()[0]}")
try:
    from physics_llm_sr.workflows import baseline_study
    from physics_llm_sr.workflows import convergence_search as busca
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "O pacote physics_llm_sr não está instalado neste ambiente. Na raiz do repositório rode:  pip install -e ."
    ) from exc

pd.set_option("display.max_colwidth", 80)
AMBIENTE = baseline_study.environment_report()
display(pd.Series(AMBIENTE, name="valor").to_frame())
assert AMBIENTE["python_ok"], "versão de Python insuficiente"
assert not AMBIENTE["modulos_de_llm_importados"], "um módulo de LLM foi importado: este notebook não usa LLM"
""")

md(r"""
## Etapa 1 — Parâmetros da busca

**Objetivo:** fixar, num só lugar, o algoritmo, o cenário, o critério de convergência e o orçamento máximo da busca.
**Chamada:** `ConvergenceCriterion(...)` e `SearchSpace.default(ALGORITMO).within_cost(ORCAMENTO_MAXIMO)`.
**Saída a conferir:** a lista ordenada de configurações candidatas, a hipótese que cada uma testa e os desvios declarados.
**Critério:** sem erro de validação. `ORCAMENTO_MAXIMO` é o custo nominal máximo (população × gerações no DEAP/gplearn;
populações × tamanho × ciclos × iterações no PySR); candidatas mais caras são descartadas antes de executar.
""")

code(r"""
# Etapa 1 — parâmetros (edite SÓ esta célula)
ALGORITMO = "gplearn"  # "gplearn" | "deap" | "pysr"
CENARIO = "free_fall"  # "free_fall" | "harmonic_motion" | "damped_wave"
CRITERIO = busca.ConvergenceCriterion(
    max_tree_distance=0.05,  # min(distância estrutural, equivalente, algébrica) <= 0,05
    max_one_minus_r2=1e-3,  # 1 - R² no teste limpo <= 1e-3
    min_success_fraction=2 / 3,  # fração mínima de repetições bem sucedidas
    use_algebraic=True,  # False = critério das rodadas O14/O15 (sem a distância algébrica)
)
ORCAMENTO_MAXIMO = 400_000  # custo nominal máximo de uma candidata (ver texto acima)
REPETICOES = 3  # repetições por candidata na triagem e no refinamento
REPETICOES_CONFIRMACAO = 5  # repetições na confirmação (sementes novas)
SEMENTE_MESTRA = 42
DESLOCAMENTO_SEMENTE = 1000  # confirmação usa SEMENTE_MESTRA + DESLOCAMENTO_SEMENTE
PARALELO = 4  # processos simultâneos (DEAP/gplearn; PySR roda em série)
PARAR_NO_PRIMEIRO_SUCESSO = True  # a triagem para no primeiro lote com uma configuração que converge
N_SIMULACOES, N_INSTANTES = 500, 100  # dados por cenário (artigo: 500 x 100)
RAIZ_SAIDA = "results/convergence_search"  # relativa a PHYSICS_LLM_SR_PROJECT_ROOT (ou ao diretório atual)

MODO_MINIMO = busca.minimal_mode_requested()  # testes: dados e orçamento mínimos (sem valor científico)
ESPACO = busca.SearchSpace.default(ALGORITMO).within_cost(ORCAMENTO_MAXIMO)
if MODO_MINIMO:
    ESPACO = busca.SearchSpace.minimal(ALGORITMO)
    N_SIMULACOES, N_INSTANTES = baseline_study.MINIMAL_SIMULATIONS, baseline_study.MINIMAL_TIME_POINTS
    REPETICOES, REPETICOES_CONFIRMACAO, PARALELO = 2, 2, 1
DADOS = {"n_simulations": N_SIMULACOES, "n_time_points": N_INSTANTES}
baseline_study.require_algorithm(ALGORITMO)
display(
    pd.DataFrame(
        [
            {
                "ordem": i,
                "configuração candidata": c.label,
                "hipótese": c.hypothesis,
                "custo estimado": busca.estimated_cost(ALGORITMO, c.overrides),
                "desvios declarados": ", ".join(c.deviations) or "(nenhum)",
            }
            for i, c in enumerate(ESPACO.candidates)
        ]
    )
)
""")

md(r"""
## Etapa 2 — Diagnóstico de expressividade

**Objetivo:** antes de culpar o algoritmo, verificar (a) se o conjunto de operadores consegue escrever a equação de referência ou uma
forma equivalente e (b) se a **formulação da função de perda** premia a referência frente a candidatas triviais.
**Chamada:** `expressiveness_diagnosis(ALGORITMO, CENARIO)`: número de nós, operadores ausentes e `L = w1·e + w2·s`
(pesos do baseline 0,5/0,5/0) para a referência, as formas equivalentes, a constante média, cada variável e `c·variável`.
**Saída a conferir:** `expressavel = True` em ao menos uma forma de referência; `vence_referencia = False` em todas as triviais.
**Critério:** se alguma trivial tem `L` menor que todas as formas de referência (`loss_blocks_convergence = True`), **nenhuma** busca
converge sob esta formulação da perda: na perda literal (erro em unidades SI, tamanho em nós) isso ocorre nos cenários oscilatórios,
cuja variância do alvo é menor que o custo de um nó. Nesse caso a única alavanca é a formulação `normalized` (desvio declarado).
""")

code(r"""
# Etapa 2 — diagnóstico de expressividade
DIAGNOSTICO = busca.expressiveness_diagnosis(ALGORITMO, CENARIO, **DADOS)
display(DIAGNOSTICO)
print("a perda impede a convergência (trivial vence a referência):", busca.loss_blocks_convergence(DIAGNOSTICO))
""")

md(r"""
## Etapa 3 — Triagem

**Objetivo:** executar as configurações candidatas na ordem das hipóteses, com as **mesmas sementes** para todas (comparação pareada).
**Chamada:** `run_stage(ALGORITMO, CENARIO, ESPACO, REPETICOES, SEMENTE_MESTRA, PARALELO, criterion=CRITERIO, stop_on_success=...)`.
**Saída a conferir:** uma linha por configuração: taxa de sucesso, mediana de `1 − R²`, mediana/mínimo das distâncias, gerações,
tempo médio e expressões típicas; `study_log.jsonl` e `search_table.csv` em `<raiz>/<algoritmo>/<cenário>/triagem/`.
**Critério:** a primeira linha (tabela ordenada) é a melhor configuração; `converge = True` se atinge o critério.
""")

code(r"""
# Etapa 3 — triagem
TRIAGEM = busca.run_stage(
    ALGORITMO,
    CENARIO,
    ESPACO,
    REPETICOES,
    SEMENTE_MESTRA,
    PARALELO,
    criterion=CRITERIO,
    stop_on_success=PARAR_NO_PRIMEIRO_SUCESSO,
    output_root=RAIZ_SAIDA,
    progress=print,
    **DADOS,
)
COLUNAS = [
    "rotulo",
    "converge",
    "n_sucesso",
    "n_execucoes",
    "mediana_um_menos_r2",
    "mediana_distancia_min",
    "geracoes_media",
    "tempo_medio_s",
    "expressao_tipica",
    "desvios",
]
display(TRIAGEM.table[COLUNAS])
print(f"tempo da etapa: {TRIAGEM.elapsed_seconds:.0f} s; parou no primeiro sucesso: {TRIAGEM.stopped_early}")
MELHOR = TRIAGEM.best
""")

md(r"""
## Etapa 4 — Refinamento

**Objetivo:** a partir da melhor configuração, encontrar a **mais barata** (população e gerações menores) que ainda converge.
**Chamada:** `refine(ALGORITMO, CENARIO, MELHOR, ...)` (só se a triagem encontrou uma configuração que converge).
**Saída a conferir:** a tabela com a configuração e suas reduções; a escolhida é a de menor custo estimado entre as que convergem.
**Critério:** se nenhuma redução converge, a escolhida é a própria melhor da triagem.
""")

code(r"""
# Etapa 4 — refinamento
ETAPAS = [TRIAGEM]
ESCOLHIDA = MELHOR
if bool(TRIAGEM.table.iloc[0]["converge"]):
    ESCOLHIDA, REFINAMENTO = busca.refine(
        ALGORITMO,
        CENARIO,
        MELHOR,
        REPETICOES,
        SEMENTE_MESTRA,
        PARALELO,
        criterion=CRITERIO,
        output_root=RAIZ_SAIDA,
        progress=print,
        **DADOS,
    )
    ETAPAS.append(REFINAMENTO)
    display(REFINAMENTO.table[COLUNAS])
else:
    print("a triagem não encontrou configuração que converge: refinamento não executado")
print("configuração escolhida:", ESCOLHIDA.label)
display(pd.Series(ESCOLHIDA.overrides, name="valor").to_frame())
""")

md(r"""
## Etapa 5 — Confirmação em sementes novas

**Objetivo:** evitar escolher a semente sortuda: a configuração escolhida é reexecutada com a semente mestra deslocada
(`SEMENTE_MESTRA + DESLOCAMENTO_SEMENTE`): dados, divisão e busca novos.
**Chamada:** `confirm_cheapest(...)` percorre as configurações que convergiram na triagem e no refinamento, da mais barata à mais cara,
e chama `confirm(...)` até uma convergir também em sementes novas (com 3 repetições, "converge" ainda é ruidoso). Em paralelo, cada
repetição usa a semente mestra `SEMENTE_MESTRA + DESLOCAMENTO_SEMENTE + k`. A confirmação é recusada se alguma semente já usada na
busca reaparecer.
**Saída a conferir:** taxa de sucesso nas sementes novas e as sementes usadas (disjuntas das da busca).
**Critério:** a configuração é aceita se `converge = True` também na confirmação.
""")

code(r"""
# Etapa 5 — confirmação em sementes novas
SEMENTES_DA_BUSCA = set().union(*(etapa.seeds for etapa in ETAPAS))
if bool(TRIAGEM.table.iloc[0]["converge"]):
    # da mais barata à mais cara entre as que convergiram, até uma confirmar (máx. 3 tentativas)
    ESCOLHIDA, CONFIRMACAO, TENTATIVAS = busca.confirm_cheapest(
        ALGORITMO,
        CENARIO,
        ETAPAS,
        REPETICOES_CONFIRMACAO,
        DESLOCAMENTO_SEMENTE,
        search_master_seed=SEMENTE_MESTRA,
        criterion=CRITERIO,
        output_root=RAIZ_SAIDA,
        parallel=PARALELO,
        progress=print,
        **DADOS,
    )
else:
    CONFIRMACAO = busca.confirm(
        ALGORITMO,
        CENARIO,
        ESCOLHIDA,
        REPETICOES_CONFIRMACAO,
        DESLOCAMENTO_SEMENTE,
        search_master_seed=SEMENTE_MESTRA,
        used_seeds=SEMENTES_DA_BUSCA,
        criterion=CRITERIO,
        output_root=RAIZ_SAIDA,
        parallel=PARALELO,
        progress=print,
        **DADOS,
    )
    TENTATIVAS = [CONFIRMACAO]
display(busca.load_stage_table(ALGORITMO, CENARIO, busca.STAGE_CONFIRMATION, RAIZ_SAIDA)[COLUNAS + ["sementes"]])
display(CONFIRMACAO.reports[ESCOLHIDA.label][["repeticao", "semente_busca", "equacao_descoberta", "um_menos_r2_teste", "distancia_estrutural", "distancia_equivalente_min", "distancia_algebrica"]])
print("configuração confirmada:", ESCOLHIDA.label, "| converge:", bool(CONFIRMACAO.table.iloc[0]["converge"]))
print("sementes disjuntas das da busca:", not (CONFIRMACAO.seeds & SEMENTES_DA_BUSCA))
""")

md(r"""
## Etapa 6 — Exportação do conjunto vencedor e como usar no notebook 05

**Objetivo:** gravar a configuração vencedora e as tabelas de todas as etapas.
**Chamada:** `export_best(...)` grava `<raiz>/<algoritmo>/<cenário>/best_config.yaml`, `best_config.json` e `search_table.csv`.
**Saída a conferir:** o bloco abaixo, para colar na **Etapa 1 do notebook 05** (substitui `ORCAMENTO` e acrescenta
`hyperparameters=tuple(HIPERPARAMETROS.items())` ao `StudyDesign(...)`). Alternativa: `load_best_config(caminho)` devolve
`(SearchBudget, hyperparameters, formulação da perda)`.
**Critério:** os valores exportados que diferem do baseline literal aparecem em `desvios_do_baseline_literal` (desvio declarado):
a reprodução literal do artigo continua usando os valores do artigo; esta é a configuração **calibrada** usada antes da inserção do LLM.
""")

code(r"""
# Etapa 6 — exportação
ARQUIVOS = busca.export_best(
    ALGORITMO,
    CENARIO,
    ESCOLHIDA,
    [*ETAPAS, *TENTATIVAS],
    criterion=CRITERIO,
    confirmation=CONFIRMACAO,
    output_root=RAIZ_SAIDA,
)
display(pd.Series({k: str(v) for k, v in ARQUIVOS.items()}, name="arquivo").to_frame())
VENCEDORA = busca.best_config_dict(ALGORITMO, CENARIO, ESCOLHIDA, criterion=CRITERIO, confirmation=CONFIRMACAO)
print("desvios do baseline literal:", VENCEDORA["desvios_do_baseline_literal"])
print("\n# ---- colar na Etapa 1 do notebook 05 ----")
print(VENCEDORA["notebook_05"])
""")


if __name__ == "__main__":
    main()
