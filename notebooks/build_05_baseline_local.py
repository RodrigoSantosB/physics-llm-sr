# ruff: noqa: E501
"""Gera ``05_baseline_local.ipynb`` (``uv run python notebooks/build_05_baseline_local.py``).

O .ipynb versionado é a saída deste script; edite as células aqui e regenere. O teste
``tests/e2e/test_notebook_05.py`` confere que o arquivo corresponde ao gerador.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common  # noqa: E402

NAME = "05_baseline_local"
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
# Notebook 05 — Baseline de regressão simbólica, execução local (sem LLM)

**Pergunta:** dado um algoritmo de regressão simbólica e uma equação analítica de referência, o pipeline
*equação → dados → busca → métrica* recupera o comportamento esperado? O estudo é **incremental**:

- **Fase 1 — validação limpa:** tudo em **regime sem ruído** (σ = 0), para validar a consistência matemática do pipeline e a convergência da busca.
- **Fase 2 — refinamento progressivo** (desligada por padrão): ruído controlado, variação de hiperparâmetros e outros algoritmos, reaproveitando as **mesmas** funções de execução, relatório e figura.

Os três sistemas físicos (queda livre, movimento harmônico simples e onda amortecida) passam pelas mesmas funções, parametrizadas por cenário, com logs e métricas no mesmo formato.

**Glossário (termos usados no notebook)**

| termo | significado |
|---|---|
| **equação de referência (ground truth)** | a lei física analítica que gera os dados e que a busca deve redescobrir |
| **desenho experimental** | o conjunto de decisões que define o estudo: algoritmo, cenários, repetições, semente mestra, formulação da perda, orçamento de busca e condição de ruído |
| **condição experimental** | uma combinação de fatores executada (por exemplo, um nível de σ ou um conjunto de hiperparâmetros) |
| **regime sem ruído / com ruído** | σ = 0 (observação = valor verdadeiro) / σ > 0 (σ é uma fração do desvio-padrão do sinal) |
| **formulação da função de perda** | `literal` = perda do artigo, com o erro quadrático médio em unidades SI (trilha `paper_replication`); `normalized` = perda normalizada, um desvio declarado (trilha `extended_analysis`) |
| **orçamento de busca** | tamanho da população, número de gerações e critério de parada antecipada |
| **repetição (semente derivada)** | execução independente; cada repetição usa uma semente derivada da semente mestra (dados, divisão treino/validação/teste e busca) |
| **controle positivo** | avaliar a *própria* equação de referência pelo mesmo caminho de métricas: deve dar distância estrutural 0 e R² = 1; se falhar, o erro está nos dados ou nas métricas, não no algoritmo |

**Como usar:** edite só a célula de parâmetros (Etapa 1) e execute de cima para baixo. Cada etapa traz *objetivo*, *chamada*, *saída a conferir* e *critério*.
Nenhum módulo de LLM (`transformers`, `torch`) é importado; basta CPU. Toda a lógica está em `physics_llm_sr.workflows`; as células só chamam funções.
""")

md(r"""
## Etapa 0 — Ambiente local

**Objetivo:** confirmar que o ambiente Conda/venv está pronto antes de qualquer cálculo.
**Chamada:** verifica a versão do Python, a importação de `physics_llm_sr` e mostra o relatório do ambiente.
**Saída a conferir:** `python_ok = True`; `modulos_de_llm_importados = []`; `raiz_de_resultados` é onde os resultados serão gravados
(variável `PHYSICS_LLM_SR_PROJECT_ROOT`, ou o diretório atual se ela não existir). `pysr_instalado`/`julia_no_path` só importam se você escolher `ALGORITMO = "pysr"`.
**Critério:** a célula não pode levantar erro. Se o pacote não for importável, rode `pip install -e .` a partir da raiz do repositório (veja o README).
""")

code(r"""
# Etapa 0 — ambiente local
import sys

import pandas as pd
from IPython.display import display

if sys.version_info < (3, 10):  # noqa: UP036  (a verificação é para quem abre o notebook com Python antigo)
    raise RuntimeError(f"Python >= 3.10 é necessário; encontrado {sys.version.split()[0]}")
try:
    from physics_llm_sr.workflows import baseline_study
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "O pacote physics_llm_sr não está instalado neste ambiente. A partir da raiz do repositório "
        "(a pasta com pyproject.toml), rode:  pip install -e .   e reinicie o kernel."
    ) from exc

AMBIENTE = baseline_study.environment_report()
display(pd.Series(AMBIENTE, name="valor").to_frame())
assert AMBIENTE["python_ok"], "versão de Python insuficiente"
assert not AMBIENTE["modulos_de_llm_importados"], "um módulo de LLM foi importado: este notebook não usa LLM"
""")

md(r"""
## Etapa 1 — Desenho experimental

**Objetivo:** fixar, num só lugar, todas as decisões do estudo (nada fica escondido no restante do notebook).
**Chamada:** `StudyDesign(...)` valida os parâmetros e `build_config(...)` mostra a configuração resolvida.
**Saída a conferir:** `llm.backend = fake` e `baseline_only = True` (nenhum LLM participa); `track`/`normalization` coerentes com `FORMULACAO_PERDA`;
`sigma_fraction = 0.0` (regime sem ruído); a pasta de saída `<raiz>/<fase>/<algoritmo>/<hash do desenho>/`.
**Critério:** sem erro de validação. Os **limiares de consistência** (`LIMIAR_AJUSTE`, `LIMIAR_ESTRUTURA`) são os únicos critérios usados para classificar os resultados;
não há tolerância escondida.

Alternativas de `ALGORITMO`: `"gplearn"` (padrão: programação genética, rápido em CPU, sem Julia), `"deap"` (algoritmo genético, ainda mais rápido) e
`"pysr"` (busca evolutiva em Julia; **orçamento reduzido e declarado**, não comparável em esforço com os demais; a primeira importação instala o Julia).
""")

code(r"""
# Etapa 1 — desenho experimental (edite SÓ esta célula)
from physics_llm_sr.workflows.baseline_study import (
    ConsistencyCriteria,
    NoiseCondition,
    SearchBudget,
    StudyDesign,
)

ALGORITMO = "gplearn"  # "gplearn" | "deap" | "pysr"
CENARIOS = ["free_fall", "harmonic_motion", "damped_wave"]  # queda livre, MHS, onda amortecida
REPETICOES = 3  # execuções independentes por cenário (semente derivada de SEMENTE_MESTRA)
SEMENTE_MESTRA = 42
FORMULACAO_PERDA = "literal"  # "literal" (perda do artigo, SI) | "normalized" (desvio declarado)
ORCAMENTO = SearchBudget(population=100, generations=50, early_stopping=True, patience=3, min_relative_improvement=0.001)
N_SIMULACOES, N_INSTANTES = 500, 100  # tamanho do conjunto de dados por cenário (artigo: 500 x 100)
RAIZ_SAIDA = "results/nb05_baseline_local"  # relativa a PHYSICS_LLM_SR_PROJECT_ROOT (ou ao diretório atual)

# Critério de consistência da Fase 1 (limiares visíveis)
LIMIAR_AJUSTE = 1e-4  # ajuste numérico aceito quando 1 - R² (teste limpo) <= limiar
LIMIAR_ESTRUTURA = 0.0  # estrutura recuperada quando a distância estrutural à referência <= limiar (0 = idêntica)
CLASSES_ACEITAS = ["recupera estrutura", "ajusta numericamente sem recuperar estrutura"]  # o que conta como "recuperou o comportamento"
FRACAO_MINIMA = 1.0  # fração mínima de repetições aceitas, por cenário, para aprovar a Fase 1

# Fase 2 (DESLIGADA por padrão: ligue só depois da Fase 1 aprovada)
EXECUTAR_FASE2_RUIDO = False  # 9a: ruído controlado
EXECUTAR_FASE2_HIPERPARAMETROS = False  # 9b: variação de hiperparâmetros
EXECUTAR_FASE2_ALGORITMOS = False  # 9c: outros algoritmos
TIPO_RUIDO, NIVEIS_SIGMA, LOCALIZACOES = "zero_mean", [0.01, 0.03, 0.05], ["target"]  # 9a: tipo, σ (fração do desvio) e onde
GRADE_HIPERPARAMETROS = {"sr.population": [50, 100]}  # 9b: caminho em sr.* -> valores
ALGORITMOS_COMPARADOS = ["deap", "gplearn"]  # 9c

MODO_MINIMO = baseline_study.minimal_mode_requested()  # make notebook-05 / testes: orçamento e dados mínimos
if MODO_MINIMO:
    ORCAMENTO = SearchBudget.minimal()
    N_SIMULACOES, N_INSTANTES = baseline_study.MINIMAL_SIMULATIONS, baseline_study.MINIMAL_TIME_POINTS
    REPETICOES = min(REPETICOES, 2)

baseline_study.require_algorithm(ALGORITMO)
DESENHO = StudyDesign(
    algorithm=ALGORITMO,
    scenarios=tuple(CENARIOS),
    repetitions=REPETICOES,
    master_seed=SEMENTE_MESTRA,
    loss_formulation=FORMULACAO_PERDA,
    search_budget=ORCAMENTO,
    noise=NoiseCondition(),  # regime sem ruído (σ = 0): obrigatório na Fase 1
    output_root=RAIZ_SAIDA,
    n_simulations=N_SIMULACOES,
    n_time_points=N_INSTANTES,
    criteria=ConsistencyCriteria(max_one_minus_r2=LIMIAR_AJUSTE, max_tree_distance=LIMIAR_ESTRUTURA),
)
CONFIG = baseline_study.build_config(DESENHO)
display(
    pd.Series(
        {
            "algoritmo": DESENHO.algorithm,
            "cenários": ", ".join(DESENHO.scenarios),
            "repetições": DESENHO.repetitions,
            "semente mestra": DESENHO.master_seed,
            "formulação da perda": f"{DESENHO.loss_formulation} ({CONFIG.track.value})",
            "orçamento (população x gerações)": f"{ORCAMENTO.population} x {ORCAMENTO.generations}",
            "parada antecipada": ORCAMENTO.early_stopping,
            "σ (sigma_fraction)": CONFIG.data.noise.sigma_fraction,
            "llm.backend / baseline_only": f"{CONFIG.llm.backend} / {CONFIG.experiments.baseline_only}",
            "pesos do baseline (w1, w2, w3)": str(CONFIG.loss.baseline_weights),
            "hash do desenho": DESENHO.design_hash(),
            "modo mínimo (teste de encanamento)": MODO_MINIMO,
        },
        name="valor",
    ).to_frame()
)
""")

md(r"""
## Etapa 2 — Equações de referência e domínios

**Objetivo:** mostrar, por cenário, a equação de referência (ground truth), as formas fisicamente equivalentes, as variáveis que entram na equação e as **distratoras** (colunas de entrada irrelevantes de propósito).
**Chamada:** `reference_table(DESENHO)`.
**Saída a conferir:** queda livre `v = sqrt(2·g·height)` (e a forma equivalente `g·t`); MHS e onda amortecida com suas variáveis; `variaveis_distratoras` lista o que a busca deve ignorar.
**Critério:** as equações conferem com a física esperada. As formas equivalentes alimentam só a métrica **secundária**; a distância primária é sempre à equação de referência do artigo.
""")

code(r"""
# Etapa 2 — equações de referência
from physics_llm_sr.workflows import baseline_views

pd.set_option("display.max_colwidth", 80)
display(baseline_views.reference_table(DESENHO))
""")

md(r"""
## Etapa 3 — Geração dos dados sem ruído

**Objetivo:** gerar, a partir da equação, o conjunto de dados de cada cenário no regime sem ruído e verificar que `y_observado == y_verdadeiro`.
**Chamada:** `generate_clean_data`, `clean_data_report`, `domain_table` e `plot_trajectories`.
**Saída a conferir:** `y_observado_igual_verdadeiro = True` e `desvio_max_y_obs_vs_y_verdadeiro = 0.0` em todos os cenários; `problemas_de_validacao = nenhum`;
a tabela de domínio com mínimo e máximo de cada variável; o gráfico com o comportamento esperado (queda livre: velocidade crescente até o solo; MHS e onda: oscilação dentro de um período).
**Critério:** o `assert` da célula passa (sem ruído, a observação é idêntica ao valor verdadeiro).
""")

code(r"""
# Etapa 3 — dados limpos por cenário
DADOS = baseline_views.generate_clean_data(DESENHO)
RELATORIO_DADOS = baseline_views.clean_data_report(DESENHO, DADOS)
display(RELATORIO_DADOS)
display(baseline_views.domain_table(DADOS))
display(baseline_views.plot_trajectories(DADOS))
assert RELATORIO_DADOS["y_observado_igual_verdadeiro"].all(), "no regime sem ruído y_observado deve ser igual a y_verdadeiro"
""")

md(r"""
## Etapa 4 — Controle positivo

**Objetivo:** validar dados e métricas **antes** de culpar o algoritmo: a equação de referência (e suas formas equivalentes) é avaliada nos dados limpos pelo mesmo código de métricas.
**Chamada:** `positive_control(cenario, DESENHO)` para cada cenário.
**Saída a conferir:** `distancia_estrutural = 0` na forma primária, `um_menos_r2` próximo de 0 (no máximo `POSITIVE_CONTROL_TOLERANCE = 1e-9`, erro de ponto flutuante),
`distancia_equivalente_min = 0` em todas as formas, `erro_max_abs_grid` ~ 0 e `aprovado = True`.
**Critério:** todas as linhas aprovadas. Se alguma falhar, **pare**: o problema está nos dados ou nas métricas e qualquer resultado da busca seria inconclusivo.
""")

code(r"""
# Etapa 4 — controle positivo
CONTROLE = pd.concat([baseline_study.positive_control(c, DESENHO) for c in DESENHO.scenarios], ignore_index=True)
display(CONTROLE)
assert CONTROLE["aprovado"].all(), "o controle positivo falhou: revise dados e métricas antes de rodar a busca"
""")

md(r"""
## Etapa 5 — Fase 1: execução do baseline por cenário

**Objetivo:** executar a busca sem ruído em cada cenário, com a **mesma função** (`phase1_clean_validation`), uma chamada por cenário.
**Chamada:** laço sobre `CENARIOS`; cada chamada grava manifesto, `results.parquet`, `consistency.csv` e `study_log.jsonl` em `<raiz>/fase1_validacao_limpa/<algoritmo>/<hash>/`.
É **retomável**: repetir a célula reaproveita as repetições já concluídas (o runner pula o que já está `ok`).
A função recusa qualquer desenho com ruído (`CleanPhaseError`).
**Saída a conferir:** por cenário, execuções feitas/puladas/com falha (`failed = 0`) e o diretório de saída.
**Critério:** nenhuma falha de execução. Uma busca que não converge **não é falha de execução**: aparece na Etapa 6 como "não converge".
""")

code(r"""
# Etapa 5 — Fase 1, um cenário por vez (mesma função para os três)
import time
from dataclasses import replace

RESULTADOS_POR_CENARIO = []
for cenario in DESENHO.scenarios:
    inicio = time.perf_counter()
    resultado = baseline_study.phase1_clean_validation(replace(DESENHO, scenarios=(cenario,)))
    resumo = resultado.runs[0].summary
    print(
        f"{cenario:16s} planejadas={resumo.planned} executadas={resumo.executed} "
        f"puladas={resumo.skipped} falhas={resumo.failed}  ({time.perf_counter() - inicio:.1f} s)  {resultado.runs[0].run_dir}"
    )
    RESULTADOS_POR_CENARIO.append(resultado)
FASE1 = baseline_study.combine_results(RESULTADOS_POR_CENARIO)
""")

md(r"""
## Etapa 6 — Relatório de consistência e critério de aprovação

**Objetivo:** comparar, por cenário e repetição, o modelo descoberto com a equação de referência e classificar o resultado com os limiares da Etapa 1.
**Chamada:** `consistency_report(FASE1)` e `phase1_approval(...)`.
**Saída a conferir (por linha):** equação descoberta × de referência; MAE, MSE e 1 − R² no **teste limpo**; `erro_max_abs_grid` (erro máximo absoluto num grid denso do domínio físico, fora da amostra de treino);
`distancia_estrutural` (primária) e `distancia_equivalente_min` (secundária); `recuperacao_exata`; gerações; `motivo_parada`; tempo; e a **classificação**:

- `recupera estrutura`: 1 − R² ≤ `LIMIAR_AJUSTE` **e** distância ≤ `LIMIAR_ESTRUTURA`;
- `ajusta numericamente sem recuperar estrutura`: ajuste dentro do limiar, estrutura diferente (na queda livre, `9.81*t` é equivalente fisicamente, mas só a distância *secundária* o reconhece);
- `não converge`: 1 − R² acima do limiar;
- `erro de execução`: a execução falhou.

**Critério de aprovação da Fase 1:** por cenário, a fração de repetições em `CLASSES_ACEITAS` ≥ `FRACAO_MINIMA`. Um resultado "não aprovado" é informação científica (orçamento insuficiente, formulação da perda, algoritmo), não um defeito do pipeline: o controle positivo da Etapa 4 já descartou dados e métricas.
""")

code(r"""
# Etapa 6 — relatório de consistência e aprovação
from IPython.display import Markdown

RELATORIO = baseline_study.consistency_report(FASE1)
display(RELATORIO)
APROVACAO = baseline_study.phase1_approval(RELATORIO, min_fraction=FRACAO_MINIMA, accepted=CLASSES_ACEITAS)
display(APROVACAO)
FASE1_APROVADA = bool(APROVACAO["aprovado"].all())
display(
    Markdown(
        f"**Fase 1 {'APROVADA' if FASE1_APROVADA else 'NÃO aprovada'}** "
        f"(critério: fração >= {FRACAO_MINIMA} de repetições em {CLASSES_ACEITAS}; "
        f"1 - R² <= {LIMIAR_AJUSTE}, distância <= {LIMIAR_ESTRUTURA})."
    )
)
""")

md(r"""
## Etapa 7 — Convergência

**Objetivo:** ver se a busca converge: melhor perda composta por geração, um painel por cenário (repetições em linhas finas, média em destaque).
**Chamada:** `plot_convergence(FASE1)`.
**Saída a conferir:** curvas **não crescentes** que achatam; curvas curtas indicam parada antecipada (melhoria relativa < 0,1% por 3 gerações).
**Critério:** a melhor perda não aumenta; se a curva é plana desde o início, a busca estagnou (compare com `motivo_parada` na Etapa 6).
""")

code(r"""
# Etapa 7 — convergência
display(baseline_views.plot_convergence(FASE1))
""")

md(r"""
## Etapa 8 — Comparação entre cenários

**Objetivo:** comparar os três sistemas físicos nas mesmas métricas, com tabela e figura padronizadas (as mesmas funções servem à Fase 2).
**Chamada:** `scenario_comparison(RELATORIO)` e `plot_scenario_comparison(RELATORIO)`.
**Saída a conferir:** média entre repetições de MAE, 1 − R², distância estrutural, erro máximo no grid, gerações e tempo; `taxa_recupera_estrutura`. Na figura, pontos = repetições e traço = média (eixos logarítmicos onde indicado).
**Critério:** a leitura é por cenário; cenários mais difíceis (mais variáveis e funções transcendentes) tendem a ter erro e distância maiores.
""")

code(r"""
# Etapa 8 — comparação entre cenários
display(baseline_views.scenario_comparison(RELATORIO))
display(baseline_views.plot_scenario_comparison(RELATORIO))
""")

md(r"""
## Etapa 9 — Fase 2 (desligada por padrão)

**Quando ligar:** só **depois** de a Fase 1 ser aprovada (Etapa 6) e de o controle positivo passar (Etapa 4). Cada bloco abaixo é independente, tem uma flag `EXECUTAR_FASE2_*` na Etapa 1
e produz um conjunto de execuções **separado**, com o mesmo esquema de saída (`CONSISTENCY_COLUMNS`) e a mesma tabela/figura de comparação, **incluindo a Fase 1 como referência**.
Em cada bloco o resultado da Fase 1 aparece com a condição "sem ruído (σ = 0)" para comparação direta.

### 9a — Ruído controlado
**Objetivo:** medir a degradação do baseline quando σ > 0 (fração do desvio-padrão), por tipo (`zero_mean` ou `biased`) e localização (`inputs`, `target`, `both`).
**Chamada:** `phase2_noise_sweep`. **Saída a conferir:** uma condição por (localização × σ); `sigma` na tabela. **Critério:** a distância tende a crescer com σ.
""")

code(r"""
# Etapa 9a — ruído controlado (Fase 2)
if EXECUTAR_FASE2_RUIDO:
    if not FASE1_APROVADA:
        print("AVISO: a Fase 1 não foi aprovada; interprete a Fase 2 com cautela.")
    FASE2_RUIDO = baseline_study.phase2_noise_sweep(DESENHO, NIVEIS_SIGMA, LOCALIZACOES, TIPO_RUIDO)
    RELATORIO_RUIDO = baseline_study.consistency_report(FASE2_RUIDO)
    COMPARACAO_RUIDO = pd.concat([RELATORIO, RELATORIO_RUIDO], ignore_index=True)
    display(baseline_views.scenario_comparison(COMPARACAO_RUIDO))
    display(baseline_views.plot_scenario_comparison(COMPARACAO_RUIDO))
else:
    print("Fase 2a desligada (EXECUTAR_FASE2_RUIDO = False).")
""")

md(r"""
### 9b — Variação de hiperparâmetros
**Objetivo:** ver o efeito de hiperparâmetros da busca (qualquer caminho `sr.*`, por exemplo `sr.population` ou `sr.gplearn.p_crossover`) mantendo dados e sementes.
**Chamada:** `phase2_hyperparameter_sweep(DESENHO, GRADE_HIPERPARAMETROS)` (produto cartesiano da grade). **Saída a conferir:** uma condição por combinação. **Critério:** compare `taxa_recupera_estrutura` e 1 − R² entre combinações.
""")

code(r"""
# Etapa 9b — hiperparâmetros (Fase 2)
if EXECUTAR_FASE2_HIPERPARAMETROS:
    FASE2_HIPER = baseline_study.phase2_hyperparameter_sweep(DESENHO, GRADE_HIPERPARAMETROS)
    RELATORIO_HIPER = baseline_study.consistency_report(FASE2_HIPER)
    COMPARACAO_HIPER = pd.concat([RELATORIO, RELATORIO_HIPER], ignore_index=True)
    display(baseline_views.scenario_comparison(COMPARACAO_HIPER))
    display(baseline_views.plot_scenario_comparison(COMPARACAO_HIPER))
else:
    print("Fase 2b desligada (EXECUTAR_FASE2_HIPERPARAMETROS = False).")
""")

md(r"""
### 9c — Outros algoritmos
**Objetivo:** executar o mesmo desenho com outros algoritmos (`deap`, `gplearn`, `pysr`). O PySR roda com orçamento **reduzido e declarado** (campos `pysr_*` do orçamento): não é comparável em esforço.
**Chamada:** `phase2_algorithm_comparison(DESENHO, ALGORITMOS_COMPARADOS)`. **Saída a conferir:** uma condição por algoritmo. **Critério:** compare as classificações por cenário.
""")

code(r"""
# Etapa 9c — algoritmos (Fase 2)
if EXECUTAR_FASE2_ALGORITMOS:
    FASE2_ALGORITMOS = baseline_study.phase2_algorithm_comparison(DESENHO, ALGORITMOS_COMPARADOS)
    RELATORIO_ALGORITMOS = baseline_study.consistency_report(FASE2_ALGORITMOS)
    COMPARACAO_ALGORITMOS = pd.concat([RELATORIO, RELATORIO_ALGORITMOS], ignore_index=True)
    display(baseline_views.scenario_comparison(COMPARACAO_ALGORITMOS))
    display(baseline_views.plot_scenario_comparison(COMPARACAO_ALGORITMOS))
else:
    print("Fase 2c desligada (EXECUTAR_FASE2_ALGORITMOS = False).")
""")

md(r"""
## Etapa 10 — Exportação

**Objetivo:** gravar o resultado num CSV padronizado (esquema fixo, o mesmo em todas as fases) e no **formato legado** da reprodução anterior (`analysis.legacy_export`), para comparação direta.
**Chamada:** `export_study(FASE1)`.
**Saída a conferir:** os caminhos de `consistencia.csv`, `study_log.jsonl` (uma linha por execução) e da pasta `legado/<hash>/`.
**Critério:** os arquivos existem; o log tem uma linha por cenário × repetição.
""")

code(r"""
# Etapa 10 — exportação padronizada + formato legado
EXPORTADOS = baseline_views.export_study(FASE1)
display(pd.Series({k: str(v) for k, v in EXPORTADOS.items()}, name="caminho").to_frame())
print("linhas no log:", sum(1 for _ in open(EXPORTADOS["log"], encoding="utf-8")), "| execuções:", len(RELATORIO))
""")

md(r"""
## Etapa 11 — Resumo e próximos passos

A célula abaixo resume a execução **a partir dos arquivos** (nenhum número é digitado no texto) e reconfirma que nenhum módulo de LLM foi carregado.

**Próximos passos (progressão sugerida):**

1. Com a Fase 1 aprovada, ligue `EXECUTAR_FASE2_RUIDO` e acompanhe a degradação em função de σ.
2. Varie hiperparâmetros (`EXECUTAR_FASE2_HIPERPARAMETROS`) e, depois, outros algoritmos (`EXECUTAR_FASE2_ALGORITMOS`).
3. Se a Fase 1 não for aprovada, investigue na ordem: orçamento de busca (população, gerações, parada antecipada), formulação da perda (`normalized`), outro algoritmo.
4. Aumente `REPETICOES` (o artigo usa 25) antes de tirar conclusões estatísticas; com 3 repetições o desvio é pouco informativo.
5. O LLM entra nos notebooks 02 e 03; este baseline é a referência de comparação.
""")

code(r"""
# Etapa 11 — resumo final
import sys

from IPython.display import Markdown, display

display(
    Markdown(
        f"**Algoritmo:** {DESENHO.algorithm} | **formulação da perda:** {DESENHO.loss_formulation} | "
        f"**execuções:** {len(RELATORIO)} | **Fase 1:** {'aprovada' if FASE1_APROVADA else 'não aprovada'}  \n"
        f"**Pasta da Fase 1:** `{FASE1.runs[0].run_dir.parent}`"
    )
)
display(RELATORIO.groupby(["cenario", "classificacao"]).size().rename("repetições").to_frame())
carregados = [m for m in ("transformers", "torch") if m in sys.modules]
assert not carregados, f"um módulo de LLM foi carregado: {carregados}"
print("Verificação: nenhum módulo de LLM (transformers/torch) foi importado nesta sessão.")
""")

if __name__ == "__main__":
    main()
