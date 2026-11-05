# physics-llm-sr

Reprodução científica de **regressão simbólica informada por física com LLMs**: um LLM avalia, durante a busca,
a plausibilidade física de cada equação candidata, e essa avaliação entra na função de perda
`L = w1·e + w2·s + w3·c` (erro de ajuste, tamanho da árvore e penalidade do LLM). O projeto implementa três
sistemas físicos (queda livre, movimento harmônico simples, onda amortecida), três algoritmos de regressão
simbólica (DEAP, gplearn, PySR), três LLMs (Mistral 7B, Llama 2 7B, Falcon 7B), oito variantes de prompt (A a H),
os experimentos 1 a 7 do artigo e a comparação com os resultados publicados.

## Aviso: clean-room

A implementação é independente. **O código dos autores do artigo não foi consultado** (nem o repositório
público deles). Tudo o que o artigo não informa é uma hipótese declarada, com identificador (GAP-xx, H-xx,
INC-xx) em [`references/scientific_assumptions.md`](references/scientific_assumptions.md). Os prompts foram
**reconstruídos** a partir do texto do artigo (GAP-13) e estão marcados `reconstructed: true`; o suplemento não os traz
([`references/prompts_source.md`](references/prompts_source.md)). Não use este repositório como substituto dos
resultados do artigo.

## Estrutura

```
src/physics_llm_sr/
  config/          configuração tipada (Pydantic), caminhos
  physics/         simuladores (free_fall, harmonic_motion, damped_wave)
  data/            geração, ruído, partição por simulação, validação
  symbolic/        AST comum, operadores protegidos, parser seguro, canonicalização, distância entre árvores
  sr/              adaptadores DEAP, gplearn e PySR (LLM dentro do laço de busca)
  llm/             backends (fake, Ollama, transformers), prompts A–H, parser de resposta, cache
  optimization/    função de perda combinada, parada antecipada, histórico da busca
  evaluation/      métricas, estatística, comparação com o artigo, gráficos
  experiments/     runner dos experimentos 1–7 (células, checkpoint, retomada, relatório)
  reproducibility/ sementes, manifestos, artefatos
  colab.py         preparo de ambiente (local/Colab) e escolha de configs para o notebook
  cli.py           linha de comando `physics-llm-sr`
configs/           smoke, pilot, pilot_extended, colab_pilot, colab_pilot_extended, colab_hoje_queda_livre(+_literal), replication, extended, nb01_baseline, nb02_um_llm
prompts/           prompt-base e variantes A–H (reconstruídos)
references/        rastreabilidade, hipóteses, tabelas publicadas transcritas
notebooks/         01_baseline_sem_llm, 02_um_llm_um_caso, 03_pipeline_completo (etapas 00–18), 04_comparacao_com_artigo (+ build_*.py)
tests/             unit, integration, scientific, e2e
```

## Instalação local

Requer Python >= 3.10 (desenvolvido com 3.12) e [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # instala o pacote e o grupo de desenvolvimento
make check         # ruff + mypy + testes (sem Julia, sem LLM real, sem testes lentos)
make smoke         # experimento mínimo com avaliador simulado (segundos)
```

O PySR precisa de Julia, instalado automaticamente na primeira importação (leva vários minutos e exige rede).
O backend de piloto local usa [Ollama](https://ollama.com) com `mistral:7b-instruct`, `llama2:7b-chat` e `falcon:7b-instruct`
(quantização Q4, um desvio declarado).

## Notebooks (Google Colab)

Quatro notebooks, **sem interseção entre si e fechados em si**: cada um tem a própria célula de preparo (monta o Drive, clona/atualiza o GitHub,
`pip install -e`), a própria pasta de resultados e a própria config derivada, e **não lê saídas dos outros**. Parâmetros no topo (nada fixo no meio),
passo a passo com "O que verificar" em cada etapa, e geradores versionados em `notebooks/build_*.py`.

| Notebook | Abrir no Colab | Ambiente | Tempo estimado* | O que responde |
| --- | --- | --- | --- | --- |
| **01** `01_baseline_sem_llm` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/notebooks/01_baseline_sem_llm.ipynb) | **CPU** basta (sem transformers/torch/HF) | padrão (3 cenários x 3 SR x 3 repetições, ruído no alvo 1/3/5%): 162 células, cerca de 2,3 h (PySR domina) | Quão bem a SR funciona **sem LLM** (Exp. 1 + baseline de Exp. 4/5): métricas, equações vs GT, convergência, robustez a ruído, comparação com as linhas "sem LLM" publicadas (Tabelas 3, 4, 5). Saída: `results/nb01_baseline/<hash>/`. |
| **02** `02_um_llm_um_caso` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/notebooks/02_um_llm_um_caso.ipynb) | **GPU A100 (High-RAM)**; Llama 2 exige `HF_TOKEN` | queda livre, Mistral, 3 repetições: `benchmark` ~0,7 h de LLM, `prompts` ~3 h, `ruido` e `ruido_enviesado` ~1,5 h cada (mais a busca do SR) | **Qual o impacto de inserir o LLM** em um cenário, um experimento e um LLM: baseline pareado próprio vs com LLM (Wilcoxon), teste do LLM em 3 expressões, equações lado a lado com as avaliações do LLM, decomposição da perda, convergência, custo. Saída: `results/nb02_<cenario>_<experimento>_<llm>/<hash>/`. |
| **03** `03_pipeline_completo` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/notebooks/03_pipeline_completo.ipynb) | **GPU A100 (High-RAM)** | preset `"hoje"`: cerca de 4 a 6 h; `"completo"`: dias | O pipeline do artigo inteiro (etapas 00 a 18, `PRESET`s e seletores), Exp. 1 a 7. Saída: `results/nb03_<config>/`. |
| **04** `04_comparacao_com_artigo` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/notebooks/04_comparacao_com_artigo.ipynb) | CPU | minutos (só lê execuções existentes) | Como os resultados locais se comparam às Tabelas 2 a 6 do artigo (tabelas e figuras F1–F8 com leitura guiada). |

\* Estimativas por hipótese (a medir); só tempo de LLM nos notebooks 02 e 03 (veja "Escolhendo o que rodar").

**Parâmetros principais.** 01: `CENARIOS`, `ALGORITMOS`, `REPETICOES` (3; 25 = artigo), `NIVEIS_RUIDO`, `LOCALIZACOES`, `BRACO` (`literal` | `normalized`).
02: `CENARIO` (um), `EXPERIMENTO` (`benchmark` | `prompts` | `ruido` | `ruido_enviesado`), `LLM` (`mistral` | `llama2` | `falcon`), `ALGORITMOS`, `REPETICOES`, `PROMPT`, `PROMPTS`, `BRACO` (padrão `normalized`:
no `literal` o termo do LLM é dominado pelo erro, GAP-02). Parâmetros inválidos falham com mensagem clara **antes** de rodar.

**Reprodutibilidade entre 01 e 02 (checagem opcional, sem um ler o outro).** Com as mesmas sementes, o baseline do 02 para um caso coincide com o do 01 para o mesmo caso e a mesma repetição
(chaves de célula e sementes derivadas idênticas; use o mesmo `BRACO`). Rode o 01 com `CENARIOS=[<caso>]` e compare a coluna `expression` e as métricas com o Passo 4 do 02.
Um teste automatizado (`tests/integration/test_nb_baseline_reproducibility.py`) confere isso em smoke.

**Smoke (`make notebook-smoke`).** Executa 01, 02 e 03 de ponta a ponta com a variável `PHYSICS_LLM_SR_NOTEBOOK_SMOKE=1` (backend `fake`, tamanhos mínimos, sem PySR/GPU);
alvos por notebook: `make notebook-smoke-01`, `-02`, `-03`. Os notebooks executados e os resultados do smoke ficam em `results/notebook_smoke/`.

### Passo a passo no Colab (notebooks 02 e 03: A100)

1. **Abrir pelo badge** do notebook na tabela acima.
2. **GPU A100 + High-RAM** (02 e 03) ou CPU (01 e 04): *Ambiente de execução, Alterar tipo de ambiente de execução*.
3. **Executar tudo.** A primeira célula é autossuficiente (não importa o pacote), na ordem: monta o Drive (`MOUNT_DRIVE = True`),
   clona `REPO_URL` (`git clone --depth 1 -b REPO_BRANCH`) ou, se `REPO_DIR` já existe, `git pull --ff-only`, verifica
   `REPO_DIR`, roda `pip install -e <repo>[hf]` (o 01 instala sem o extra `hf`) e ajusta `sys.path`; só depois o notebook importa `physics_llm_sr`.
   A primeira importação do PySR instala o Julia (minutos).
4. **Llama 2** (só fora da execução de hoje): o checkpoint `meta-llama/Llama-2-7b-chat-hf` é restrito; aceite a licença no
   Hugging Face e crie o segredo `HF_TOKEN` no Colab. O Mistral não precisa de token.
5. **Pesos no disco local (`HF_HOME_DIR = "/content/hf_home"`):** o cache do Hugging Face fica no disco do Colab (Mistral 7B fp16 ocupa cerca de
   15 GB). Drive só com ≥ 20 GB livres por modelo, e não é recomendado. Antes de carregar, o pacote baixa só `*.json`, `*.safetensors` e
   tokenizer (sem os `.bin`), confere o tamanho de cada `.safetensors`, rebaixa uma vez o que estiver truncado e falha com mensagem clara.
   **Erro `SafetensorError: ... header too large`:** pesos truncados (cota de 15 GB do Drive ou escrita via FUSE). Apague o cache antigo
   (`rm -rf /content/drive/MyDrive/physics-llm-sr-runs/hf_home`), reinicie o ambiente de execução e reexecute com o padrão local.
6. *Alternativa (sem GitHub):* envie a pasta do projeto com `.git` (ou um `git bundle`) ao Drive e use `REPO_DIR` (ou
   `REPO_URL` apontando para o `.bundle`). No Drive, `git config core.fileMode false` evita `git_dirty` por modos de arquivo.

## Escolhendo o que rodar

Na primeira célula do **notebook 03**, `PRESET` escolhe a base e `MODELOS`, `CENARIOS`, `EXPERIMENTOS`, `ALGORITMOS`, `REPETICOES` e
`PROMPTS` (quando diferentes de `None`) a sobrescrevem, sem editar YAML. `colab.build_run_config` valida o resultado pelo schema
(combinação inválida falha com o campo e o motivo) e grava `<PROJECT_ROOT>/configs_derivadas/nb03/<hash>.yaml`; com sobrescritas, o
diretório de resultados (`results/nb03_<config>`) ganha o sufixo `_<hash>`, então execuções diferentes não se misturam e a retomada usa a config certa.

| `PRESET` | O que roda | Células | LLM estimado* |
| --- | --- | --- | --- |
| `"hoje"` (padrão) | queda livre, Mistral, DEAP/gplearn/PySR, 3 repetições, P1 a P4 (extended; PySR reduzido) | 117 | 5,9 h |
| `"hoje_literal"` | controle literal (`paper_replication`), só P1 | 18 | 0,7 h |
| `"3_modelos_queda_livre"` | hoje com Mistral, Llama 2 (exige `HF_TOKEN`) e Falcon | 189 | 11,9 h |
| `"completo"` | artigo: 3 cenários x 3 SR x 3 LLMs x 25 repetições, prompts A a H, ruído completo (literal) | 18 675 | cerca de 983 h (41 dias) |
| `"completo_extended"` | idem, braço extended (perda normalizada) | 18 675 | cerca de 983 h |
| `"smoke"` | validação mínima com avaliador simulado | 20 | n/a |

\* Só o tempo de LLM, com as hipóteses de 0,3 s/chamada em lote (DEAP/gplearn) e 2,5 s em série (PySR) e 300/1000/200
chamadas únicas por célula (DEAP/gplearn/PySR); não inclui a busca do SR. A célula "Plano resolvido" do notebook mede a latência
real e recalcula. Em `"hoje"`, P1 0,7 h, P2 2,2 h, P3 1,5 h, P4 1,5 h: por isso a ordem de prioridade e o corte em P3/P4 se faltar tempo.

Exemplos:

```python
PRESET = "hoje"  # padrão
PRESET = "hoje"
MODELOS = ["mistral", "falcon"]  # hoje com dois modelos
PRESET = "hoje"
EXPERIMENTOS = ["exp1", "exp2"]  # só P1
PRESET = "hoje"
CENARIOS = ["free_fall", "harmonic_motion"]
REPETICOES = 5
PRESET = "completo"  # artigo inteiro (dias; retomável)
```

Notas: os três modelos transformers (revisões fixadas) existem em `llm.models` de todas as configs Colab. Só um modelo fica
residente por vez (o runner descarrega o anterior) e cada um ocupa cerca de 15 GB em `HF_HOME` (disco local do Colab). O Llama 2 é restrito:
se `"llama2"` estiver em `MODELOS`, a célula de setup exige o segredo `HF_TOKEN` e para com instruções (peça acesso em
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf); Mistral e Falcon não exigem token.

## Execução de hoje no Colab

Preset `"hoje"` (padrão do notebook): pipeline Exp1 a Exp6, análise e
exportação legada **só em queda livre**, Mistral 7B fp16 (`configs/colab_hoje_queda_livre.yaml`, `extended_analysis`, perda
normalizada), DEAP e gplearn com 100 x 50, 3 repetições, `batch_size: 16`, `max_cell_seconds: 1800`.
**Desvio declarado:** PySR reduzido (1 população de 50, 50 ciclos, 10 iterações; early stopping ativo).

1. Abra pelo badge, escolha **A100 + High-RAM**, monte o Drive quando pedido. O HF token **não** é necessário para o Mistral.
2. **Executar tudo.** Logo após o setup, uma célula mede a latência real do Mistral (10 avaliações em lote, 5 individuais) e
   imprime o tempo projetado de LLM por etapa. Depois rodam, nesta ordem de prioridade, cada uma persistida e retomável:
   **P1** Exp1 baseline + Exp2 (prompt E), 18 células; **P2** Exp3 prompts A, C, E, H, 27 células novas;
   **P3** Exp4 ruído de média zero só no alvo (1, 3, 5%), 36 novas; **P4** Exp5 ruído enviesado só no alvo (1, 5%), 36 novas.
   Exp6 (distância estrutural) está nas métricas de cada célula. Total: 117 células.
3. **Quanto esperar:** o orçamento é de cerca de 4 h de A100. Com as hipóteses (0,3 s/chamada em lote, 2,5 s em série;
   DEAP 300, gplearn 1000 e PySR 200 chamadas únicas por célula com LLM) só o tempo de LLM soma cerca de 0,7 h (P1), 2,2 h (P2),
   1,5 h (P3) e 1,5 h (P4), fora o tempo de busca; por isso a ordem é por prioridade e P3/P4 são os primeiros a cortar.
   Célula acima de 1800 s vira `error` registrado e a execução segue.
4. **Retomar após desconexão:** reabra o notebook (mesmos parâmetros) e execute tudo de novo; o runner pula as células
   concluídas, que estão no Drive.
5. **Onde ficam os resultados:** execução em `<DRIVE_PROJECT_DIR>/results/colab_hoje_queda_livre/notebook_run/`; tabelas e figuras
   da análise em `<DRIVE_PROJECT_DIR>/results/analise_colab_hoje/` (`INDEX.md`, `tables/`, `figures/`); formato legado em
   `<DRIVE_PROJECT_DIR>/results/legado_colab_hoje/`. A última célula mostra o `INDEX.md` e três figuras.
6. **Baixar:** pelo Drive (pasta `physics-llm-sr-runs/results`) ou, numa célula extra,
   `shutil.make_archive("/content/resultados", "zip", "<DRIVE_PROJECT_DIR>/results")` e `google.colab.files.download`.
7. **Controle literal (opcional, só se sobrar tempo):** com `PRESET = "hoje_literal"` roda `configs/colab_hoje_queda_livre_literal.yaml`
   (`paper_replication`, perda literal, só P1). A análise final o inclui como segundo braço, rotulado à parte.

Sem Colab, a CLI faz o mesmo: `make pilot-colab` (veja abaixo) com `PHYSICS_LLM_SR_PROJECT_ROOT` apontando para o Drive.

## Comandos

| Comando | O que faz |
| --- | --- |
| `make install` | `uv sync`. |
| `make check` | `ruff check`, `ruff format --check`, `mypy` e `pytest -m "not slow and not julia and not llm_real"`. |
| `make smoke` | Config `smoke`: dados mínimos, avaliador simulado, DEAP e gplearn (segundos). |
| `make pilot` | Piloto local (Ollama, Mistral): braço `paper_replication` (perda literal) e braço `extended_analysis` (normalizada), em diretórios separados. |
| `make pilot-colab` | Piloto no Colab A100 (transformers fp16, população 100, 50 gerações), dois braços. |
| `make report RUN_DIR=results/<cfg>/<run_id>` | Reconstrói tabelas, comparação com o artigo e figuras a partir dos arquivos persistidos (sem treinar). |
| `make test-julia` | Testes que exigem PySR/Julia (opt-in). |
| `make test-llm-real` | Testes com Ollama/transformers reais (opt-in). |
| `make notebook-smoke` | Executa os notebooks 01, 02 e 03 em modo smoke (variável `PHYSICS_LLM_SR_NOTEBOOK_SMOKE=1`: backend `fake`, tamanhos mínimos, sem PySR) de ponta a ponta (nbconvert); saída em `results/notebook_smoke/`. Um alvo por notebook: `notebook-smoke-01`, `-02`, `-03`. |

CLI direta: `uv run physics-llm-sr run --config configs/smoke.yaml [--only exp1,exp2] [--sr deap,gplearn] [--resume]` e
`uv run physics-llm-sr report <run_dir>`. Variáveis de ambiente: `PHYSICS_LLM_SR_PROJECT_ROOT` (raiz de `results/` e `.cache/`),
`PHYSICS_LLM_SR_PROMPTS_DIR` e `PHYSICS_LLM_SR_REFERENCES_DIR` (o notebook as define).

## Notebooks

- `notebooks/01_baseline_sem_llm.ipynb`, `02_um_llm_um_caso.ipynb` e `03_pipeline_completo.ipynb`: veja a tabela "Notebooks (Google Colab)". Cada passo tem explicação, chamada ao módulo Python,
  visualização e "O que verificar"; as células só importam e chamam `physics_llm_sr` (nenhuma lógica científica nas células).
- O **03** percorre as etapas 00 a 18 do PRD (seção 10), cada uma com explicação teórica, referência ao artigo, chamada ao módulo e interpretação. Parâmetros no topo:
  `PRESET` (e sobrescritas `MODELOS`, `CENARIOS`, `EXPERIMENTOS`, `ALGORITMOS`, `REPETICOES`, `PROMPTS`; veja "Escolhendo o que rodar") e `LOAD_EXISTING_RUN` (opcional: caminho de um `run_dir` existente para **apresentar** os resultados sem executar).
- Os `.ipynb` são gerados por `notebooks/build_01_baseline.py`, `build_02_um_llm.py` e `build_03_pipeline_completo.py` (`uv run python notebooks/build_0X_*.py`); `tests/e2e/test_notebook.py` confere que cada `.ipynb` corresponde ao seu gerador.

## Comparação artigo × experimentos locais

`make compare RUNS="results/pilot/<id> results/pilot_extended/<id>" OUT=results/analysis/x` (ou
`python -m physics_llm_sr.analysis --runs ... --out ...`) gera, a partir dos arquivos persistidos, tabelas
(CSV, Markdown, LaTeX booktabs e `tabelas.tex` mestre), figuras F1–F8 (PNG 200 dpi e PDF) e um `INDEX.md` com a
tabela do artigo correspondente e o rótulo de execução de cada artefato. Aceita qualquer modo/trilha. Célula sem
dado local aparece como `n/e` (não executado), nunca como zero. O notebook `notebooks/04_comparacao_com_artigo.ipynb`
(gerado por `notebooks/build_04_comparacao_com_artigo.py`) apresenta tudo com perguntas de leitura guiada.

## Exportação no formato legado

`make export-legado RUNS="results/pilot/<id>" OUT=results/legado/literal` (ou
`python -m physics_llm_sr.analysis.legacy_export --runs ... --out ...`) grava os resultados persistidos nos
nomes de arquivo e colunas de uma reprodução anterior (`resultados_{benchmark,prompts,ruido}.csv`,
`llm_cache.jsonl`, `commit.txt`, `tabelas_<modo>/` com tabelas 2–6, overhead e figuras) para comparação direta.
Use uma saída por braço (literal/normalizada). `FORMATO_LEGADO.md` documenta cada coluna e o rótulo de execução
reduzida; campos sem origem (ex.: `n_evaluated`) ficam `nan`.

## Modos e trilhas: o que pode ser apresentado como reprodução

| Modo / trilha | Para que serve | Pode ser apresentado como reprodução? |
| --- | --- | --- |
| `smoke` (qualquer trilha) | Verificar o encanamento do pipeline; avaliador simulado. | **Não.** Execução reduzida. |
| `pilot` / `colab_pilot` (`paper_replication`) | Validar a matriz e medir custo: queda livre, 3 repetições, população e gerações reduzidas no local. | **Não.** Execução reduzida; os rótulos vêm do `report.json`. |
| qualquer modo, trilha `extended_analysis` | Análises de engenharia com desvios (perda normalizada, cache por estrutura, top-k). | **Não.** Nunca entra nas tabelas de reprodução. |
| `full` + `paper_replication` (`configs/replication.yaml`) | Matriz completa do artigo, 25 repetições, LLMs reais em fp16 com revisão fixada. | **Sim**, e só se concluir sem falhas de célula (a execução aborta na primeira falha). |

O `report` sempre grava o rótulo no `report.json` (campos `label`, `reduced_run`, `complete_result`). Células sem dados locais ou
sem o número mínimo de repetições são classificadas `inconclusiva` na comparação com o artigo.

## Principais hipóteses e desvios

Lista completa em [`references/scientific_assumptions.md`](references/scientific_assumptions.md); a matriz artigo, requisito, módulo e teste está em
[`references/methodology_traceability.md`](references/methodology_traceability.md).

- **GAP-01 (pesos)**: o artigo não informa `w1, w2, w3`. Padrão: `(1/3, 1/3, 1/3)` [hipótese]; baseline com `w3 = 0` e pesos `(0,5, 0,5, 0)`.
- **GAP-02 (escalas)**: modo `literal` (artigo; MSE em SI, `s` = nós, `c ∈ [0, 1]`) e modo `normalized` (desvio, só em `extended_analysis`). Na escala literal o termo `c` tende a ser dominado, e o LLM quase não altera a busca; isso é um achado a reportar.
- **GAP-03 / GAP-04 (simuladores e tempo)**: constantes k, φ, E0, k_onda e ω são hipóteses; tempo amostrado no intervalo físico.
- **GAP-05 (PySR)**: Huber no termo de erro, conforme o artigo; DEAP e gplearn usam MSE.
- **GAP-07 (LLMs)**: Colab usa transformers fp16 com revisão por SHA; o piloto local usa Ollama Q4 (desvio). Temperatura 0.
- **GAP-10 / GAP-11 (distância)**: definição recursiva normalizada; distância 0 = equivalência. A métrica secundária (`tree_distance_equivalent_min`) nunca substitui a primária.
- **GAP-12**: LLM e prompt das tabelas de ruído não informados; usamos Mistral e prompt E [hipótese].
- **GAP-13 (prompts)**: reconstruídos; o suplemento não os contém.
- **GAP-14 (custo)**: avaliar todo candidato com cache; PySR com LLM no laço é a configuração mais cara.
- **INC-01 a INC-15**: inconsistências do artigo e do suplemento, transcritas com nota (por exemplo, INC-01: linhas de F, G e H idênticas às de E na Tabela 2).

## Estimativas de custo

Ordens de grandeza do GAP-14, **a medir no piloto** (latências são estimativas):

| Item | Estimativa |
| --- | --- |
| Chamadas LLM por treino (DEAP/gplearn, pop 100, 50 gerações, sem cache global) | cerca de 2 000 |
| Latência por chamada, A100 fp16 | cerca de 1,0 s sem lote; cerca de 0,15 s efetivos em lote de 32 |
| Latência por chamada, GTX 1060 com Ollama Q4 | cerca de 3 a 6 s |
| Piloto local (`pilot.yaml`) | cerca de 2,8 h mais o PySR |
| Piloto Colab (`colab_pilot.yaml`) | cerca de 45 min com lote (cerca de 5 h sem lote) mais o PySR |
| PySR com LLM no laço, por treino | de 0,5 a 3 h na A100; `experiments.max_cell_seconds` limita cada célula |
| Matriz completa (12 825 treinos com LLM) | da ordem de 10³ h de GPU com lote, antes do cache global; exige reduzir repetições ou configurações (registrar como desvio) |
| `make smoke` / `make notebook-smoke` | segundos / alguns minutos, sem LLM real |

## Como retomar execuções

O runner grava cada célula (cenário × algoritmo × LLM × prompt × ruído × repetição) de forma atômica em `<run_dir>/cells/`. Uma execução interrompida
(queda do Colab, falta de memória, Ctrl+C) é retomada sem recalcular o que já terminou:

```bash
make smoke RESUME=1
make pilot-colab RESUME=1
uv run physics-llm-sr run --config configs/colab_pilot.yaml --resume            # execução mais recente com a mesma config
uv run physics-llm-sr run --config configs/colab_pilot.yaml --resume results/colab_pilot/<run_id>
uv run physics-llm-sr report results/colab_pilot/<run_id>                       # tabelas e figuras, sem treinar
```

Células com `status = ok` são puladas; células com erro são tentadas de novo. A retomada exige a mesma configuração (hash). O cache de avaliações do LLM
(`.cache/llm`) persiste entre execuções; no Colab, defina `MOUNT_DRIVE = True` para que ele sobreviva à sessão. No notebook, rodar as etapas de novo retoma o mesmo
diretório (`<output_dir>/notebook_run`) e mostra os resultados já calculados.

## Reprodutibilidade

Cada execução grava `run.json` (config e hash), `manifest.json` (SHA do git, versões de bibliotecas, Julia, hardware, backends dos LLMs),
`results.parquet`, `history.parquet`, relatórios CSV e `figures/`. As sementes são derivadas por nome a partir de uma semente mestra. O determinismo entre
ambientes não é presumido (INC-11); a saída do LLM é validada por parser estrito e nunca substituída por nota padrão.

## Roteiro Colab A100

1. **Runtime:** *Ambiente de execução, Alterar tipo*, GPU **A100** com **High-RAM**.
2. **Segredo:** crie `HF_TOKEN` em *Segredos* (com "Acesso ao notebook" ligado), depois de aceitar a licença do Llama 2 no Hugging Face.
3. **Parâmetros** da primeira célula: `RUN_PROFILE = "colab_pilot"`, `MOUNT_DRIVE = True`, `REPO_DIR` (e `REPO_URL` se usar o bundle); confira cerca de 40 GB livres no disco local para `HF_HOME_DIR`.
4. **Executar** as etapas em ordem. Estimativas do `colab_pilot` (a medir, não garantidas): cerca de 45 min com lote e cerca de 5 h sem lote, mais o PySR (0,5 a 3 h por treino com LLM).
5. **Resultados:** `<DRIVE_PROJECT_DIR>/results/colab_pilot/notebook_run/` (`results.parquet`, `manifest.json`, `figures/`); cache do LLM em `<DRIVE_PROJECT_DIR>/.cache/llm`.
6. **Retomar após desconexão:** abra o notebook, repita os parâmetros e reexecute a primeira célula e as etapas; as células concluídas são puladas.
7. **Baixar:** pelo Drive, ou compacte `notebook_run/` (`shutil.make_archive`) e baixe com `google.colab.files.download`.
