# ruff: noqa: E501
"""Gera ``01_baseline_sem_llm.ipynb`` (``uv run python notebooks/build_01_baseline.py``).

O .ipynb versionado é a saída deste script; edite as células aqui e regenere. O teste
``tests/e2e/test_notebook.py`` confere que o arquivo corresponde ao gerador.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common  # noqa: E402

NAME = "01_baseline_sem_llm"
NOTEBOOK = Path(__file__).resolve().parent / f"{NAME}.ipynb"
B = nb_common.NotebookBuilder(NOTEBOOK)
md, code = B.md, B.code


def build():  # type: ignore[no-untyped-def]
    """Monta o notebook (usado pelo teste de correspondência com o .ipynb)."""
    return B.build()


def main() -> None:
    """Grava o notebook."""
    B.write()


PARAMETERS = r"""
# =====================================================================================
# PARÂMETROS DO NOTEBOOK 01 (edite SÓ esta célula; não há valores fixos no resto)
# =====================================================================================
CENARIOS = ["free_fall", "harmonic_motion", "damped_wave"]  # 1 a 3 dos três sistemas físicos
# PySR é o mais lento (Julia); retire-o para um teste rápido
ALGORITMOS = ["deap", "gplearn", "pysr"]
REPETICOES = 3  # 25 = número do artigo (horas a dias de CPU)
NIVEIS_RUIDO = [0.01, 0.03, 0.05]  # fração do desvio-padrão (Exp. 4 e 5); o artigo usa 0.01 a 0.05
LOCALIZACOES = ["target"]  # onde entra o ruído: "inputs", "target", "both" (o artigo usa as três)
# "literal" = perda do artigo (GAP-02); "normalized" = perda normalizada (desvio declarado)
BRACO = "literal"
"""

md(rf"""
# Notebook 01 — Baseline sem LLM (DEAP, gplearn e PySR)

{nb_common.COLAB_BADGE.format(name=NAME)}

**O que este notebook responde:** quão bem a regressão simbólica funciona **sem nenhum LLM**? Ele roda o pipeline completo
(dados, treino, métricas, equações descobertas, convergência, robustez a ruído, comparação com o que o artigo publicou para
"sem LLM") e serve de **linha de base** para avaliar, nos notebooks 02 e 03, o impacto de inserir o LLM.

**Ambiente:** o Colab pode estar em **CPU** (padrão, sem GPU). Nada aqui importa `transformers`, `torch` nem baixa modelo; nenhum LLM é carregado.
Os experimentos executados são o Exp. 1 (baseline) e as **células baseline** do Exp. 4 (ruído de média zero) e do Exp. 5 (ruído enviesado), todos com `w3 = 0` (pesos 0,5/0,5, GAP-01).

**Como usar (passo a passo):**

1. Edite os **parâmetros** (primeira célula de código): cenários, algoritmos, repetições, níveis e localizações do ruído, braço.
2. Execute as células **uma a uma, de cima para baixo**. Cada passo termina com uma lista **"O que verificar"** para conferir a saída antes de seguir.
3. O treino é feito **uma célula por algoritmo** e é **retomável**: se a sessão cair, repita o setup e os passos 1 e 2, e as células já concluídas são puladas (o diretório de resultados é determinístico).

**Independência:** este notebook é fechado em si. Tem a própria célula de preparo, grava só em `<PROJECT_ROOT>/results/nb01_baseline/<hash>/`
(configuração derivada em `<PROJECT_ROOT>/configs_derivadas/nb01/`) e **não lê saídas de outros notebooks**. Com as mesmas sementes, o baseline do notebook 02 para o mesmo
caso e a mesma repetição **coincide** com o daqui (as chaves das células e as sementes derivadas são idênticas; use o mesmo `BRACO`).
Para adaptar a outro cenário, só os parâmetros mudam.

**Modo smoke (testes):** com a variável de ambiente `PHYSICS_LLM_SR_NOTEBOOK_SMOKE=1` o notebook usa tamanhos mínimos e dispensa o PySR (valida o encanamento em minutos; os números não valem como resultado).
""")

code(nb_common.setup_cell(parameters=PARAMETERS, pip_extras="", hf_home=False))

md(r"""
## Passo 1 — Setup e configuração resolvida

Valida os parâmetros (erro claro **antes** de rodar qualquer coisa), deriva a configuração a partir de `configs/nb01_baseline.yaml`
(validada pelo schema e gravada em `configs_derivadas/nb01/`) e mostra o que será executado.

**O que verificar:**

- `llm.backend = fake` e `baseline_only = True`: **nenhum LLM** participa (a matriz só tem células `w3 = 0`).
- `track`/`normalization` correspondem ao `BRACO` escolhido (`literal` = `paper_replication`).
- `output_dir` aponta para `results/nb01_baseline/<hash>` (pasta própria deste notebook).
- Na tabela de bibliotecas, `transformers` e `torch` podem existir no Colab, mas **não são importados** aqui; `cuda_disponível` fica `None`.
""")

code(r"""
# Passo 1 — configuração resolvida
import pandas as pd
from IPython.display import Markdown, display
from physics_llm_sr import colab
from physics_llm_sr.config.loader import load_config

SMOKE = colab.smoke_requested()  # só make notebook-smoke / testes (tamanhos mínimos, sem PySR)
ROOT = colab.ensure_project_root()
CONFIG_PATH = colab.build_baseline_config(
    scenarios=CENARIOS,
    algorithms=ALGORITMOS,
    repetitions=REPETICOES,
    noise_levels=NIVEIS_RUIDO,
    noise_locations=LOCALIZACOES,
    braco=BRACO,
    out_dir=ROOT,
    smoke=SMOKE,
)
cfg = load_config(CONFIG_PATH)
RUN_DIR = colab.notebook_run_dir(cfg)
ANALYSIS_DIR = RUN_DIR.parent / "analise"
display(Markdown(f"Configuração derivada: `{CONFIG_PATH}`  \nDiretório da execução: `{RUN_DIR}`  \nModo smoke: **{SMOKE}**"))
display(pd.Series(colab.profile_summary(cfg), name="valor").to_frame())
display(pd.Series({"baseline_only": cfg.experiments.baseline_only, "pesos do baseline (w1, w2, w3)": str(cfg.loss.baseline_weights)}, name="valor").to_frame())
info = colab.environment_report(include_cuda=False)
display(pd.Series(info["libraries"], name="versão").to_frame())
""")

md(r"""
## Passo 2 — Plano: quantas células e quanto tempo (CPU)

A matriz tem uma célula por (cenário × algoritmo × repetição × condição de ruído). O Exp. 4 em 1% no alvo coincide com o Exp. 1 (ruído padrão do artigo) e entra **uma vez só**.
O tempo é uma **estimativa por hipótese** (segundos por célula por algoritmo, a medir: PySR inclui a partida do Julia); cada célula é limitada por `max_cell_seconds`.

**O que verificar:**

- O total de células confere com a sua conta (cenários × algoritmos × repetições × condições de ruído).
- O PySR domina o tempo estimado. Se for demais, reduza `ALGORITMOS`/`REPETICOES` e rode de novo a célula do Passo 1.
- Depois do treino, compare o tempo real (`train_seconds_mean`, Passo 4) com esta estimativa.
""")

code(r"""
# Passo 2 — plano e estimativa de CPU
from IPython.display import Markdown, display
from physics_llm_sr import colab

display(colab.plan_table(cfg))
estimate = colab.estimate_cpu_runtime(cfg)
display(estimate)
display(
    Markdown(
        f"Total: **{int(estimate['cells'].sum())} células**, cerca de **{estimate['hours'].sum():.2f} h** de CPU "
        f"(hipóteses: {colab.DEFAULT_CPU_SECONDS_PER_CELL} s por célula; limite por célula: {cfg.experiments.max_cell_seconds} s)."
    )
)
""")

md(r"""
## Passo 3 — Geração e exploração dos dados

Para cada cenário o gerador simula as trajetórias com a solução analítica (a GT) e cria o conjunto limpo (`y_true`); o ruído é aplicado depois, por célula, sobre este mesmo conjunto limpo.
Cada simulação tem parâmetros próprios (massa, comprimento, altura, amortecimento); as colunas que não entram na GT são distratoras.

**O que verificar:**

- `problemas de validação` vazio para todos os cenários (sem NaN, sem valores fora do domínio físico).
- A tabela de amostras mostra as 5 colunas de entrada (`mass`, `length`, `height`, `damping`, `t`) e o alvo.
- O gráfico mostra o comportamento esperado: queda livre com velocidade crescente até o solo; harmônico e onda amortecida oscilando dentro de um período.
""")

code(r"""
# Passo 3 — dados por cenário
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.validation import validate_dataset

datasets = {name: generate_dataset(cfg, name, noisy=False) for name in cfg.experiments.scenarios}
display(
    pd.DataFrame(
        [
            {
                "cenário": name,
                "amostras": ds.X.shape[0],
                "simulações": np.unique(ds.simulation_ids).size,
                "alvo": ds.target_name,
                "GT": ds.metadata.get("ground_truth", ""),
                "problemas de validação": validate_dataset(ds, cfg.physics),
            }
            for name, ds in datasets.items()
        ]
    )
)
for name, ds in datasets.items():
    display(Markdown(f"**{name}** (5 primeiras linhas)"))
    display(pd.DataFrame(ds.X, columns=list(ds.feature_names)).assign(**{ds.target_name: ds.y_true}).head(5))
fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.8), squeeze=False)
for ax, (name, ds) in zip(axes[0], datasets.items()):
    for sim in np.unique(ds.simulation_ids)[:3]:
        mask = ds.simulation_ids == sim
        ax.plot(ds.t[mask], ds.y_true[mask], ".", ms=3, label=f"simulação {sim}")
    ax.set(title=name, xlabel="t (s)", ylabel=ds.target_name)
    ax.legend()
plt.tight_layout()
plt.show()
""")

md(r"""
## Passo 4 — Treino por algoritmo (uma célula por algoritmo, retomável)

Cada célula executa **todas** as células da matriz daquele algoritmo (Exp. 1 e células baseline de Exp. 4 e 5) e grava cada uma atomicamente. Se a sessão cair, basta rodar de novo:
as concluídas são recarregadas, não recalculadas. Uma célula que estoura `max_cell_seconds` ou falha vira `status = error` registrado e **não interrompe** as demais.
A tabela de progresso mostra, por algoritmo, células `ok`, com `error` e o tempo médio de treino.

**O que verificar (após cada algoritmo):**

- `executed + skipped = planned` no resumo e `failed = 0` (ou os erros listados abaixo são esperados, por exemplo PySR sem Julia).
- Na tabela de progresso, `ok` cresce até o número de células do algoritmo no Passo 2.
""")

md(r"""
### 4a — DEAP (algoritmo genético; cruzamento 0,05, mutação 0,01, torneio 3, profundidade 8)
""")

code(r"""
# Passo 4a — DEAP (retomável)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

if "deap" in cfg.experiments.algorithms:
    RUN_DIR, summary = colab.run_notebook_stage(cfg, colab.NB01_EXPERIMENTS, RUN_DIR, algorithms=["deap"])
    print(summary)
else:
    print("DEAP fora de ALGORITMOS: nada a executar")
results = colab.load_results(RUN_DIR)
display(nv.status_table(results))
display(nv.error_rows(results))
""")

md(r"""
### 4b — gplearn (programação genética; hiperparâmetros padrão do gplearn 0.4.2, hipótese)
""")

code(r"""
# Passo 4b — gplearn (retomável)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

if "gplearn" in cfg.experiments.algorithms:
    RUN_DIR, summary = colab.run_notebook_stage(cfg, colab.NB01_EXPERIMENTS, RUN_DIR, algorithms=["gplearn"])
    print(summary)
else:
    print("gplearn fora de ALGORITMOS: nada a executar")
results = colab.load_results(RUN_DIR)
display(nv.status_table(results))
display(nv.error_rows(results))
""")

md(r"""
### 4c — PySR (busca evolutiva em Julia; **reduzido**, desvio declarado em `configs/nb01_baseline.yaml`)

A primeira importação instala o Julia e as dependências (vários minutos). O PySR roda aqui com uma população de 50, 50 ciclos e 10 iterações (o artigo e os outros algoritmos usam 100 × 50): os resultados do PySR **não são comparáveis em esforço** com DEAP/gplearn.
Sem Julia, as células PySR ficam como `error` e o restante do notebook segue.
""")

code(r"""
# Passo 4c — PySR (retomável; a primeira vez instala o Julia)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

if "pysr" in cfg.experiments.algorithms:
    RUN_DIR, summary = colab.run_notebook_stage(cfg, colab.NB01_EXPERIMENTS, RUN_DIR, algorithms=["pysr"])
    print(summary)
else:
    print("PySR fora de ALGORITMOS: nada a executar")
results = colab.load_results(RUN_DIR)
display(nv.status_table(results))
display(nv.error_rows(results))
""")

md(r"""
## Passo 5 — Métricas por cenário × algoritmo

Métricas no **conjunto de teste** (partição por simulação, GAP-09), média e desvio-padrão amostral entre as repetições, no **ruído padrão** do artigo (1% no alvo, o Exp. 1):
MAE, MSE, 1−R², **distância estrutural** à GT do artigo (0 = mesma estrutura) e **distância a formas equivalentes** (`tree_distance_equivalent_min`, métrica secundária, nunca substitui a primária).

**O que verificar:**

- `n` = número de repetições `ok` por linha; com poucas repetições o desvio é pouco informativo.
- Distância 0 significa que a estrutura da GT foi recuperada; compare `tree_distance` com a `tree_distance_equivalent_min` (na queda livre, `9.81*t` é equivalente fisicamente, mas só a secundária o reconhece).
- Métricas altas em MAE/MSE com distância baixa indicam estrutura certa com constantes erradas, e vice-versa.
""")

code(r"""
# Passo 5 — métricas por cenário x algoritmo (Exp. 1)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

results = colab.load_results(RUN_DIR)
BASE = nv.default_noise_rows(results, cfg)
metrics = nv.metrics_summary(BASE)
display(metrics)
""")

md(r"""
## Passo 6 — Equações descobertas × equação verdadeira (GT)

Uma linha por repetição: a melhor expressão encontrada, a GT e as distâncias. Constantes aparecem com várias casas: o que importa é a **estrutura** (operadores e variáveis).

**O que verificar:**

- Em `damped_wave` e `harmonic_motion` as expressões costumam ser bem maiores que a GT; veja se aparecem as variáveis certas (`t`, `length`, `damping`, `mass`).
- `exact_recovery = True` só quando a distância à GT do artigo é 0.
""")

code(r"""
# Passo 6 — equações descobertas vs GT
import pandas as pd
from IPython.display import display
from physics_llm_sr.analysis import notebook_views as nv

pd.set_option("display.max_colwidth", 120)
equations = nv.equations_table(BASE)
display(equations)
""")

md(r"""
## Passo 7 — Convergência (melhor perda por geração)

Melhor perda composta (`w1·e + w2·s`, com `w3 = 0`) por geração, média das repetições de cada cenário/algoritmo, no ruído padrão. A parada antecipada (melhoria relativa < 0,1% por 3 gerações) encurta as curvas.

**O que verificar:**

- As curvas **não crescem** (a melhor perda é não crescente) e achatam: a busca convergiu ou parou antes do limite de gerações.
- Curvas muito curtas indicam parada antecipada logo no início (busca estagnada).
""")

code(r"""
# Passo 7 — convergência
import pandas as pd
from IPython.display import Image, display
from physics_llm_sr.evaluation.plots import plot_loss_per_generation

history = pd.read_parquet(RUN_DIR / "history.parquet")
conv = nv.convergence_history(BASE, history)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
if conv.empty:
    print("sem histórico para mostrar")
else:
    display(Image(filename=str(plot_loss_per_generation(conv, ANALYSIS_DIR / "convergencia.png"))))
""")

md(r"""
## Passo 8 — Robustez a ruído (curvas no estilo do projeto)

Distância estrutural em função do nível de ruído, uma grade cenário × localização, para o ruído de **média zero** (Tabela 3 do artigo) e o **enviesado** (Tabela 4). As linhas pontilhadas com `x` (`sem LLM`) são as suas, com barras de IC95 quando `n ≥ 2`;
as tracejadas são as publicadas pelo artigo (o artigo não informa o LLM dessas tabelas, GAP-12). Painéis sem dado local indicam "não executado" (por exemplo, localizações não pedidas em `LOCALIZACOES`).

**O que verificar:**

- A distância tende a **crescer** com o nível de ruído; o ruído enviesado tende a degradar mais que o de média zero.
- Só aparecem níveis e localizações pedidos em `NIVEIS_RUIDO`/`LOCALIZACOES`.
""")

code(r"""
# Passo 8 — curvas de ruído (figures.figure2)
from IPython.display import Image, display
from physics_llm_sr.analysis import figures, loading

data = loading.load_runs([RUN_DIR])
for noise_type in ("zero_mean", "biased"):
    artifacts = figures.figure2(data, noise_type, data.arms[0], ANALYSIS_DIR / "figuras")
    display(Image(filename=str(next(a.path for a in artifacts if a.path.suffix == ".png"))))
""")

md(r"""
## Passo 9 — Comparação com as linhas "sem LLM" publicadas (Tabelas 3, 4 e 5)

Cada valor local é confrontado com o publicado por critérios **pré-registrados** (`references/scientific_assumptions.md`). A Tabela 5 (suplemento) traz o baseline "sem LLM" do artigo
(MAE, MSE, 1−R², distância; ruído padrão). As Tabelas 3 e 4 trazem a distância por nível de ruído sem informar o LLM (GAP-12); aqui a curva publicada é comparada com o **seu baseline** só como referência de direção.

**O que verificar:**

- A coluna `classe`: com poucas repetições e execução reduzida o resultado é, no máximo, `parcial`/`qualitativa`/`inconclusiva`. **Uma execução com 3 repetições e PySR reduzido não é reprodução do artigo.**
- `published` é o valor do artigo; `mean` é a sua média; `n` o número de repetições locais usadas.
""")

code(r"""
# Passo 9 — baseline local vs publicado
from IPython.display import display

published = nv.baseline_vs_published(results)
display(published.groupby(["table", "classe"]).size().rename("células").to_frame())
display(published.head(40))
""")

md(r"""
## Passo 10 — Exportação no formato legado

Gera os CSV (`resultados_benchmark.csv`, `resultados_prompts.csv`, `resultados_ruido.csv`) e as tabelas no formato da reprodução anterior, em `results/nb01_baseline/<hash>/legado/`, e as tabelas/figuras de comparação artigo × local (`INDEX.md`) em `analise/`.
No notebook 01 não há células com LLM: as tabelas de benchmark e de prompts saem vazias de LLM, e a tabela "baseline sem LLM" é a que tem conteúdo.

**O que verificar:** `exportação legada em ...` sem erro e os arquivos listados; `FORMATO_LEGADO.md` descreve o conteúdo.
""")

code(r"""
# Passo 10 — export legado + análise
import subprocess
import sys
from IPython.display import Markdown, display

LEGACY_DIR = RUN_DIR.parent / "legado"
subprocess.run([sys.executable, "-m", "physics_llm_sr.analysis", "--runs", str(RUN_DIR), "--out", str(ANALYSIS_DIR / "comparacao")], check=True)
subprocess.run([sys.executable, "-m", "physics_llm_sr.analysis.legacy_export", "--runs", str(RUN_DIR), "--out", str(LEGACY_DIR)], check=True)
print(sorted(p.name for p in LEGACY_DIR.iterdir()))
display(Markdown((ANALYSIS_DIR / "comparacao" / "INDEX.md").read_text(encoding="utf-8")[:1500]))
""")

md(r"""
## Passo 11 — Resumo final e o que observar

A célula abaixo resume a execução **a partir dos arquivos** (nenhum número é digitado aqui) e confirma que nenhum LLM foi carregado.

**O que observar** (para interpretar e, depois, comparar com os notebooks 02 e 03):

1. **Qualidade do baseline**: `tree_distance` perto de 0 e MAE baixo = o algoritmo recupera a lei física sem ajuda; valores altos mostram onde um LLM teria espaço para ajudar.
2. **Estrutura × constantes**: MAE baixo com distância alta (ou o contrário) separa "forma errada" de "constantes imprecisas".
3. **Dependência do algoritmo e do cenário**: compare DEAP, gplearn e PySR (lembrando que o PySR é reduzido aqui).
4. **Ruído**: a inclinação das curvas do Passo 8 é a degradação do baseline; é contra ela que se mede se o LLM torna a busca mais robusta.
5. **Limites**: poucas repetições (3) não dão poder estatístico; 25 repetições é o número do artigo. O baseline usa `w3 = 0` (GAP-01) e a escala da perda segue o `BRACO` (GAP-02).
6. **Próximo passo**: rode o notebook 02 para o mesmo cenário/algoritmos e compare **par a par** (mesmos dados e mesma semente de busca) o efeito de inserir o LLM.
""")

code(r"""
# Passo 11 — resumo final
import sys
from IPython.display import Markdown, display

ok = nv.ok_rows(results)
display(
    Markdown(
        f"**Execução:** `{RUN_DIR}`  \n"
        f"Braço: **{BRACO}** ({cfg.track.value}); células ok: **{len(ok)}** de {len(results)}; "
        f"com erro: **{len(results) - len(ok)}**; repetições: {cfg.experiments.repetitions}; modo: {cfg.mode.value}."
    )
)
display(metrics[["scenario", "sr", "n", "mae_mean", "tree_distance_mean", "tree_distance_equivalent_min_mean"]])
loaded = [m for m in ("transformers", "torch") if m in sys.modules]
assert not loaded, f"um LLM/framework de LLM foi carregado: {loaded}"
print("Verificação: nenhum módulo de LLM (transformers/torch) foi importado nesta sessão.")
""")

if __name__ == "__main__":
    main()
