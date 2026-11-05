# ruff: noqa: E501
"""Gera ``02_um_llm_um_caso.ipynb`` (``uv run python notebooks/build_02_um_llm.py``).

O .ipynb versionado é a saída deste script; edite as células aqui e regenere. O teste
``tests/e2e/test_notebook.py`` confere que o arquivo corresponde ao gerador.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common  # noqa: E402

NAME = "02_um_llm_um_caso"
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
# PARÂMETROS DO NOTEBOOK 02 (edite SÓ esta célula; não há valores fixos no resto)
# =====================================================================================
CENARIO = "free_fall"  # UM de: "free_fall" | "harmonic_motion" | "damped_wave"
# "benchmark" (Exp. 2) | "prompts" (Exp. 3) | "ruido" (Exp. 4) | "ruido_enviesado" (Exp. 5)
EXPERIMENTO = "benchmark"
LLM = "mistral"  # "mistral" | "llama2" (restrito: exige o segredo HF_TOKEN) | "falcon"
ALGORITMOS = ["deap", "gplearn", "pysr"]  # PySR é o mais lento (Julia + LLM no laço)
REPETICOES = 3  # 25 = número do artigo (dias de GPU)
# prompt A-H usado em "benchmark", "ruido" e "ruido_enviesado" (E = variáveis + experimento)
PROMPT = "E"
PROMPTS = ["A", "C", "E", "H"]  # só para EXPERIMENTO = "prompts" (D, F, G, H revelam a GT)
NIVEIS_RUIDO = [0.01, 0.05]  # só para "ruido"/"ruido_enviesado": fração do desvio-padrão
LOCALIZACOES = ["target"]  # só para "ruido"/"ruido_enviesado": "inputs", "target" ou "both"
# "normalized" (padrão) ou "literal" (ver o aviso GAP-02 no início do notebook)
BRACO = "normalized"
# Teste rápido do LLM (Passo 3): as três expressões enviadas ao modelo.
EXPRESSAO_EQUIVALENTE = None  # None = forma equivalente declarada pelo cenário (ou 1.0*(GT))
EXPRESSAO_ABSURDA = "exp(height*height) + log(mass)"  # sem sentido físico: o LLM deveria penalizar
"""

md(rf"""
# Notebook 02 — Um LLM, um caso: qual o impacto de inserir o LLM?

{nb_common.COLAB_BADGE.format(name=NAME)}

**O que este notebook responde:** para **um** cenário, **um** experimento e **um** LLM, a regressão simbólica fica melhor **com** o LLM na função de perda do que **sem** ele?
Ele roda o **seu próprio baseline pareado** (mesmo conjunto de dados, mesma partição e mesma semente de busca, com `w3 = 0`) e a versão com o LLM, e compara par a par, repetição por repetição, sem depender do notebook 01.

**Ambiente:** Colab com **GPU A100 (High-RAM)** (Ambiente de execução → Alterar tipo). O modelo de 7B roda em fp16 via `transformers`; o Llama 2 é restrito e exige o segredo `HF_TOKEN` (aceite a licença no Hugging Face e crie o segredo no Colab com "Acesso ao notebook"; a validação acontece **antes** de carregar qualquer modelo).

**Como usar (passo a passo):**

1. Edite os **parâmetros** (primeira célula de código): cenário, experimento, LLM, algoritmos, repetições, prompt e braço.
2. Execute as células **uma a uma, de cima para baixo**. Cada passo termina com **"O que verificar"**. O Passo 3 mostra **o que o LLM responde de verdade** para 3 expressões antes de gastar horas de busca.
3. As execuções são **retomáveis** (o diretório é determinístico): se a sessão cair, repita o setup e os Passos 1 e 2 e rode as células de treino de novo; as células concluídas são puladas.

**Perda e braço (`BRACO`):** a perda é `L = w1·e + w2·s + w3·c`, com `e` o erro de ajuste, `s` o tamanho da árvore e `c = 1 − (c1 + c2 + c3)/3` a penalidade dos escores do LLM (consistência dimensional, simplicidade e realismo físico).
No braço **`literal`** (perda do artigo: `e` = MSE em unidades SI, `s` = número de nós) o termo do LLM `w3·c ≤ w3` é **dominado pelo erro** `e` e o LLM praticamente não altera a busca (**GAP-02**; é um achado científico, não um bug).
Por isso o padrão aqui é **`normalized`** (`e = MSE/Var(y)`, `s = nós/máx.`), um **desvio declarado** em que o LLM tem influência mensurável; ele é rotulado `extended_analysis` e **não é reprodução do artigo**. Use `literal` para ver o efeito (pequeno) da escala do artigo.

**Independência:** este notebook é fechado em si: tem a própria célula de preparo, grava só em `<PROJECT_ROOT>/results/nb02_<cenario>_<experimento>_<llm>/<hash>/` (configuração em `configs_derivadas/nb02/`, cache do LLM em `.cache/llm_nb02`) e **não lê saídas de outros notebooks**.
**Checagem opcional de reprodutibilidade:** com as mesmas sementes, o baseline daqui para um caso coincide com o do notebook 01 para o mesmo caso e a mesma repetição (as chaves das células e as sementes derivadas são idênticas). Para conferir, rode o 01 com `CENARIOS=[CENARIO]`, os mesmos `ALGORITMOS`, `REPETICOES` e `BRACO`,
e compare a coluna `expression` e as métricas do Passo 4 daqui com as do Passo 5 do 01 (o `data_hash` e a `seed` também coincidem). Isso é uma conferência manual; nenhum notebook lê o outro.

**Modo smoke (testes):** com `PHYSICS_LLM_SR_NOTEBOOK_SMOKE=1` o notebook usa o avaliador simulado, tamanhos mínimos e dispensa GPU e PySR (valida o encanamento em minutos; os números não valem como resultado).
""")

code(nb_common.setup_cell(parameters=PARAMETERS, pip_extras="[hf]", hf_home=True))

md(r"""
## Passo 1 — Setup e configuração resolvida

Valida os parâmetros (**erro claro antes de rodar**: combinação inválida, `PROMPTS` desconhecidos, Llama 2 sem `HF_TOKEN`), deriva a configuração de `configs/nb02_um_llm.yaml`
(validada pelo schema; revisões dos modelos fixadas por SHA em `llm.models`; transformers fp16, lote 16) e a grava em `configs_derivadas/nb02/`.

**O que verificar:**

- `experiments`, `scenarios`, `models` e `algorithms` correspondem aos parâmetros; só **um** cenário e **um** modelo.
- `track`/`normalization`: `extended_analysis`/`normalized` (padrão) ou `paper_replication`/`literal` (`BRACO = "literal"`).
- `output_dir` = `results/nb02_<cenario>_<experimento>_<llm>/<hash>`.
- Com `LLM = "llama2"` sem `HF_TOKEN` a célula **para aqui** com as instruções; com Mistral/Falcon aparece "não necessário".
""")

code(r"""
# Passo 1 — configuração resolvida
import pandas as pd
from IPython.display import Markdown, display
from physics_llm_sr import colab
from physics_llm_sr.config.loader import load_config

SMOKE = colab.smoke_requested()  # só make notebook-smoke / testes (avaliador simulado, tamanhos mínimos)
ROOT = colab.ensure_project_root()
CONFIG_PATH = colab.build_single_llm_config(
    scenario=CENARIO,
    experiment=EXPERIMENTO,
    llm=LLM,
    algorithms=ALGORITMOS,
    repetitions=REPETICOES,
    prompt=PROMPT,
    prompts=PROMPTS,
    braco=BRACO,
    noise_levels=NIVEIS_RUIDO,
    noise_locations=LOCALIZACOES,
    out_dir=ROOT,
    smoke=SMOKE,
)
cfg = load_config(CONFIG_PATH)
RUN_DIR = colab.notebook_run_dir(cfg)
ANALYSIS_DIR = RUN_DIR.parent / "analise"
EXP = colab.NB02_EXPERIMENTS[EXPERIMENTO]  # alias do runner (exp2..exp5)
gated = colab.ensure_hf_access(cfg)  # Llama 2 exige HF_TOKEN; erro claro antes de carregar o modelo
display(Markdown(f"Configuração derivada: `{CONFIG_PATH}`  \nDiretório da execução: `{RUN_DIR}`  \nModo smoke: **{SMOKE}**  \nLogin no Hugging Face: {'ok (modelo restrito)' if gated else 'não necessário'}"))
display(pd.Series(colab.profile_summary(cfg), name="valor").to_frame())
display(pd.Series({"experimento": EXPERIMENTO, "alias": EXP, "LLM": LLM, "modelo (checkpoint)": cfg.llm.models[0].checkpoint, "revisão fixada": cfg.llm.models[0].revision, "prompt": PROMPT, "pesos (w1, w2, w3)": str((cfg.loss.w1, cfg.loss.w2, cfg.loss.w3))}, name="valor").to_frame())
""")

md(r"""
## Passo 2 — Plano, latência do LLM e estimativa de tempo

Antes de gastar horas: confere a GPU, **mede a latência real** do modelo (10 avaliações em lote e 5 individuais; a primeira chamada carrega o modelo, o que leva alguns minutos e baixa ~15 GB na primeira vez)
e projeta o tempo **só de LLM** a partir da matriz de células (o tempo de busca do SR não está incluído; a estimativa de chamadas por célula é uma hipótese a medir).

**O que verificar:**

- `cuda_disponível = True` (sem GPU a célula para com instruções; no smoke o backend é simulado).
- A tabela de plano mostra as células **com LLM** (`modelo:<LLM>`) e as **baseline** (`modelo:none`), em pares.
- A latência em lote é bem menor que a individual (DEAP e gplearn usam lote; o PySR chama em série). O total em horas cabe no seu orçamento? Senão reduza `REPETICOES`/`ALGORITMOS` e rode o Passo 1 de novo.
""")

code(r"""
# Passo 2 — GPU, plano, latência e estimativa
from IPython.display import Markdown, display
from physics_llm_sr import colab

colab.require_gpu(cfg)
display(pd.Series({k: v for k, v in colab.environment_report().items() if k != "libraries"}, name="valor").to_frame())
display(colab.plan_table(cfg))
latency = colab.measure_llm_latency(cfg)
print(f"latência medida: {latency['batched_s_per_call']:.3f} s/chamada em lote, {latency['serial_s_per_call']:.3f} s/chamada individual")
estimate = colab.estimate_runtime(cfg, latency["batched_s_per_call"], latency["serial_s_per_call"])
display(estimate)
display(Markdown(f"Tempo projetado só de LLM: **{estimate['llm_hours'].sum():.2f} h** (hipóteses de chamadas por célula: {colab.DEFAULT_CALLS_PER_CELL})."))
""")

md(r"""
## Passo 3 — Teste rápido do LLM em 3 expressões: qual vai ser o output?

Pergunta ao modelo, **uma vez, sem cache**, sobre três expressões do cenário escolhido e mostra a **resposta crua**, os escores `c1` (consistência dimensional), `c2` (simplicidade) e `c3` (realismo físico), o **feedback** textual e a **penalidade** `c = 1 − (c1 + c2 + c3)/3` que entraria na perda
(`c` perto de 0 = o LLM aprova; perto de 1 = o LLM reprova):

1. a **GT** do artigo (deveria receber escores altos e `c` baixo);
2. uma **forma equivalente** (`EXPRESSAO_EQUIVALENTE`, ou a declarada pelo cenário): a mesma física escrita de outro jeito;
3. uma expressão **absurda** (`EXPRESSAO_ABSURDA`): deveria ser reprovada (`c` alto).

**O que verificar:**

- `valida = True` nas três e a resposta crua é um JSON com `dimensional_consistency`, `simplicity`, `physical_realism` e `feedback`. `valida = False` mostra o motivo em `erro` (nenhuma nota é inventada; na busca o runner tenta de novo e, se falhar, registra a falha).
- Os escores **discriminam**: `c(GT) < c(absurda)`. Se o modelo dá escores parecidos às três, o LLM pouco informa a busca e não espere efeito grande no Passo 6.
- Um 7B fp16 é ruidoso: a GT pode não receber a nota máxima. É isso que o experimento mede.
""")

code(r"""
# Passo 3 — sonda do LLM (3 expressões)
from IPython.display import display
from physics_llm_sr.analysis import llm_inspection
from physics_llm_sr.physics.base import FEATURE_NAMES
from physics_llm_sr.physics.registry import get_scenario

scenario = get_scenario(CENARIO, cfg.physics)
context = scenario.physical_context()
expressions = llm_inspection.probe_set(scenario, EXPRESSAO_EQUIVALENTE, EXPRESSAO_ABSURDA)
evaluator = colab.build_evaluator(cfg)
probe = llm_inspection.probe_expressions(evaluator, expressions, context, PROMPT, FEATURE_NAMES)
display(probe[["rotulo", "expression", "valida", "c1", "c2", "c3", "penalty", "feedback", "erro"]])
for row in probe.itertuples():
    print(f"--- {row.rotulo}: {row.expression}\nresposta crua do LLM: {row.resposta_crua}\n")
print("prompt enviado (usuário), 1ª expressão:\n", probe.loc[0, "prompt_user"])
llm_inspection.close_backend(evaluator)  # descarrega o modelo: o runner carrega o seu próprio
""")

md(r"""
## Passo 4 — Execução do baseline (sem LLM, `w3 = 0`)

Roda as células **baseline** do experimento: mesmos dados (semente por cenário e repetição), mesma partição e mesma semente de busca das células com LLM, mas com a perda `w1·e + w2·s` (pesos renormalizados 0,5/0,5, GAP-01). É o **seu próprio baseline pareado**:
o notebook não precisa de nenhum resultado externo. Retomável.

**O que verificar:** `failed = 0` (ou erros esperados, como PySR sem Julia), `ok` igual ao número de células `modelo:none` do plano e tempos de treino coerentes.
""")

code(r"""
# Passo 4 — baseline pareado (retomável)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

RUN_DIR, summary = colab.run_notebook_stage(cfg, [EXP], RUN_DIR, arm="baseline")
print(summary)
results = colab.load_results(RUN_DIR)
display(nv.status_table(results[results["llm"] == "none"]))
display(nv.error_rows(results))
""")

md(r"""
## Passo 5 — Execução com o LLM (uma célula por algoritmo, retomável)

Roda as células **com LLM**: a cada geração o LLM avalia cada equação candidata única (com cache de avaliações em `.cache/llm_nb02`) e a penalidade `c` entra na perda `w1·e + w2·s + w3·c`.
DEAP e gplearn avaliam a geração **em lote**; o PySR chama o LLM **em série** dentro do callback e é muito mais lento. Uma falha de avaliação do LLM **nunca** vira nota substituta: a célula fica `error` e é listada.

**O que verificar (após cada algoritmo):** `failed = 0`; `ok` cresce; `llm_calls > 0` na tabela de custo do Passo 10 (prova que o LLM participou da busca).
""")

md(r"""
### 5a — DEAP + LLM
""")

code(r"""
# Passo 5a — DEAP com LLM (retomável)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

if "deap" in cfg.experiments.algorithms:
    RUN_DIR, summary = colab.run_notebook_stage(cfg, [EXP], RUN_DIR, arm="llm", algorithms=["deap"])
    print(summary)
else:
    print("DEAP fora de ALGORITMOS: nada a executar")
results = colab.load_results(RUN_DIR)
display(nv.status_table(results[results["llm"] != "none"]))
display(nv.error_rows(results))
""")

md(r"""
### 5b — gplearn + LLM
""")

code(r"""
# Passo 5b — gplearn com LLM (retomável)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

if "gplearn" in cfg.experiments.algorithms:
    RUN_DIR, summary = colab.run_notebook_stage(cfg, [EXP], RUN_DIR, arm="llm", algorithms=["gplearn"])
    print(summary)
else:
    print("gplearn fora de ALGORITMOS: nada a executar")
results = colab.load_results(RUN_DIR)
display(nv.status_table(results[results["llm"] != "none"]))
display(nv.error_rows(results))
""")

md(r"""
### 5c — PySR + LLM (reduzido, desvio declarado; a primeira importação instala o Julia)
""")

code(r"""
# Passo 5c — PySR com LLM (retomável; o mais lento)
from IPython.display import display
from physics_llm_sr import colab
from physics_llm_sr.analysis import notebook_views as nv

if "pysr" in cfg.experiments.algorithms:
    RUN_DIR, summary = colab.run_notebook_stage(cfg, [EXP], RUN_DIR, arm="llm", algorithms=["pysr"])
    print(summary)
else:
    print("PySR fora de ALGORITMOS: nada a executar")
results = colab.load_results(RUN_DIR)
display(nv.status_table(results[results["llm"] != "none"]))
display(nv.error_rows(results))
""")

md(r"""
## Passo 6 — Comparação pareada: com × sem LLM

Cada repetição com LLM é comparada com o baseline **da mesma repetição** (mesmos dados e semente). A diferença é **`sem_llm − com_llm`**: **positiva = o LLM reduziu** o erro (ou a distância estrutural).
O teste de Wilcoxon pareado e o IC95 usam as repetições como amostras; **com poucas repetições o teste não tem poder** (com `n = 3` o menor p-valor bilateral exato é 0,25; com `n = 5`, 0,0625): a tabela traz `n` e uma `note`.

**O que verificar:**

- `n_expr_changed`: em quantas repetições o LLM **mudou a equação final**. Se for 0, o LLM não alterou a busca (esperado no braço `literal`, GAP-02).
- Sinal e tamanho de `mean_diff` por métrica; `tree_distance` (estrutura) e MAE/MSE (ajuste) podem divergir.
- Nas figuras: cada linha cinza liga o baseline ao LLM **da mesma repetição**; o painel F6 mostra a diferença média com IC95 (a linha 0 = sem efeito).
""")

code(r"""
# Passo 6 — comparação pareada
import pandas as pd
from IPython.display import Image, display
from physics_llm_sr.analysis import figures, loading, tables

data = loading.load_runs([RUN_DIR])
effect = tables.paired_effect(data)
pd.set_option("display.max_colwidth", 80)
display(nv.paired_repetitions(results))
display(effect[["sr", "llm", "prompt", "noise_type", "noise_location", "noise_level", "metric", "n", "mean_base", "mean_llm", "mean_diff", "ci_low", "ci_high", "p_value", "n_expr_changed", "note"]])
FIG_DIR = ANALYSIS_DIR / "figuras"
for artifacts in (figures.figure5(data, "mae", FIG_DIR), figures.figure5(data, "tree_distance", FIG_DIR), figures.figure6(data, effect, FIG_DIR)):
    display(Image(filename=str(next(a.path for a in artifacts if a.path.suffix == ".png"))))
""")

md(r"""
## Passo 7 — Equações descobertas lado a lado e a avaliação do LLM sobre elas

Para cada repetição: a equação **sem** LLM e a equação **com** LLM, com a distância à GT. Em seguida, os escores `c1`, `c2`, `c3`, a penalidade `c` e o feedback que o LLM deu à **equação final de cada célula**, lidos do **cache de avaliações do run** (`.cache/llm_nb02/evaluations.jsonl`), sem chamar o modelo de novo.
As equações do baseline **não passam pelo LLM** durante a busca: ficam `found = False` (nada é inventado).

**O que verificar:**

- `mudou = True` mostra onde o LLM trocou a equação; compare `tree_distance_com_llm` com `tree_distance_sem_llm`.
- A equação final com LLM tem `c` **baixo**? (a busca a escolheu em parte por isso). `found = False` com LLM indicaria cache não persistido (ex.: `cache_enabled = false`).
- O feedback textual explica **por que** o LLM aprovou ou reprovou: leia-o para equações que melhoraram e que pioraram.
""")

code(r"""
# Passo 7 — equações lado a lado + avaliações do LLM (do cache do run)
from IPython.display import display

records = llm_inspection.read_cache_records(llm_inspection.cache_file(cfg))
checkpoints = {m.name: m.checkpoint for m in cfg.llm.models}
display(nv.side_by_side_equations(results))
evaluations = nv.evaluations_of_discovered(results, records, checkpoints, significant_digits=cfg.llm.cache_significant_digits, key_mode=cfg.llm.cache_key_mode)
display(evaluations)
print(f"avaliações válidas no cache: {len(records)} ({llm_inspection.cache_file(cfg)})")
""")

md(r"""
## Passo 8 — Decomposição da perda: `w1·e`, `w2·s`, `w3·c`

Fração média de cada termo na **perda final das células com LLM**. É a evidência direta do GAP-02: se `w3·c` é uma fração minúscula de `L`, o LLM quase não pesa na seleção, independentemente de quão bem ele julga as equações.

**O que verificar:**

- `frac_w3c`: no braço `literal` espere um valor pequeno (o erro `e` em SI domina); no `normalized`, uma fração visível.
- Se `w3·c` for desprezível, o Passo 6 deve mostrar `n_expr_changed` baixo: as duas leituras se confirmam.
""")

code(r"""
# Passo 8 — decomposição da perda
loss = tables.loss_decomposition(data)
display(loss)
display(Image(filename=str(next(a.path for a in figures.figure7(data, loss, FIG_DIR) if a.path.suffix == ".png"))))
""")

md(r"""
## Passo 9 — Convergência com × sem LLM

Melhor perda por geração (fino = repetição, grosso = média), por algoritmo, para o baseline (`sem LLM`) e para o modelo. **Atenção:** o baseline usa `w3 = 0`, então os **valores absolutos de `L` não são comparáveis** entre os dois (a perda com LLM tem o termo extra `w3·c`); compare a **forma** das curvas e a parada antecipada.

**O que verificar:** o LLM muda a velocidade de convergência ou o número de gerações até a parada antecipada? Curvas idênticas confirmam um LLM que não influencia a busca.
""")

code(r"""
# Passo 9 — convergência com/sem LLM
display(Image(filename=str(next(a.path for a in figures.figure8(data, FIG_DIR) if a.path.suffix == ".png"))))
""")

md(r"""
## Passo 10 — Custo: chamadas, cache, tempo LLM × SR e falhas

Quanto o LLM custou: pedidos de avaliação (`llm_requested`), equações únicas (`llm_unique`), chamadas reais ao modelo (`llm_calls`), acertos de cache, respostas inválidas, tempo total do LLM e do SR (`sr_seconds = train_seconds − llm_seconds`) e células com erro.

**O que verificar:**

- `llm_calls > 0` (o LLM participou) e `cache_hit_rate` crescendo entre repetições (a mesma equação não é reavaliada).
- `llm_seconds` ≫ `sr_seconds`: o custo do LLM domina (esperado; no PySR é ainda maior).
- `llm_invalid_responses` e `cells_error` baixos: muitas respostas inválidas indicam um modelo que não segue o formato JSON.
""")

code(r"""
# Passo 10 — custo
cost = nv.cost_summary(results)
display(cost)
""")

md(r"""
## Passo 11 — Exportação no formato legado

Gera os CSV no formato da reprodução anterior (`resultados_benchmark.csv`, `resultados_prompts.csv`, `resultados_ruido.csv`, `llm_cache.jsonl` com o cache de avaliações) em `results/nb02_.../<hash>/legado/` e as tabelas/figuras artigo × local em `analise/comparacao/`.
Só as tabelas do experimento escolhido têm conteúdo.

**O que verificar:** `exportação legada em ...` sem erro; `FORMATO_LEGADO.md` descreve os arquivos.
""")

code(r"""
# Passo 11 — export legado + análise
import subprocess
import sys

LEGACY_DIR = RUN_DIR.parent / "legado"
subprocess.run([sys.executable, "-m", "physics_llm_sr.analysis", "--runs", str(RUN_DIR), "--out", str(ANALYSIS_DIR / "comparacao")], check=True)
subprocess.run([sys.executable, "-m", "physics_llm_sr.analysis.legacy_export", "--runs", str(RUN_DIR), "--out", str(LEGACY_DIR)], check=True)
print(sorted(p.name for p in LEGACY_DIR.iterdir()))
""")

md(r"""
## Passo 12 — Resumo final e o que observar

A célula abaixo resume a execução **a partir dos arquivos** (nenhum número é digitado aqui).

**O que observar** (leia nesta ordem):

1. **O LLM discrimina?** (Passo 3) `c(GT)` menor que `c(absurda)`. Sem isso, não há o que esperar da busca.
2. **O LLM mudou a busca?** (Passos 6 e 8) `n_expr_changed > 0` e `frac_w3c` não desprezível. Se a fração é minúscula, o resultado é GAP-02 (escala), não "o LLM é inútil".
3. **O impacto é a favor?** (Passo 6) `mean_diff > 0` em `tree_distance` (estrutura mais próxima da GT) e/ou em MAE/MSE (melhor ajuste). Cuidado: com `n` pequeno o Wilcoxon não tem poder; 25 repetições é o número do artigo.
4. **A que custo?** (Passo 10) tempo de LLM × SR e taxa de falhas.
5. **O resultado é reprodução do artigo?** **Não**: o braço `normalized` é um desvio declarado, o PySR é reduzido e há poucas repetições; os rótulos nas figuras e tabelas vêm dos dados.
6. **Para adaptar a outro cenário**: mude só os parâmetros do topo (`CENARIO`, `EXPERIMENTO`, `LLM`, `PROMPT`...) e rode de novo; cada combinação grava em sua própria pasta.
""")

code(r"""
# Passo 12 — resumo final
from IPython.display import Markdown, display

ok = nv.ok_rows(results)
display(
    Markdown(
        f"**Execução:** `{RUN_DIR}`  \n"
        f"Caso: **{CENARIO}**, experimento **{EXPERIMENTO}**, LLM **{LLM}**, prompt {PROMPT}, braço **{BRACO}** ({cfg.track.value}).  \n"
        f"Células ok: **{len(ok)}** de {len(results)}; com erro: **{len(results) - len(ok)}**; repetições: {cfg.experiments.repetitions}; modo: {cfg.mode.value}."
    )
)
display(effect[effect["metric"] == "tree_distance"][["sr", "llm", "prompt", "n", "mean_base", "mean_llm", "mean_diff", "p_value", "n_expr_changed"]])
display(loss[["scenario", "sr", "n", "frac_w1e", "frac_w2s", "frac_w3c"]])
""")

if __name__ == "__main__":
    main()
