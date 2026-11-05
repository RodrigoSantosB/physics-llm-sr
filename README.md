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
configs/           smoke, pilot, pilot_extended, colab_pilot, colab_pilot_extended, replication, extended
prompts/           prompt-base e variantes A–H (reconstruídos)
references/        rastreabilidade, hipóteses, tabelas publicadas transcritas
notebooks/         01_full_paper_reproduction.ipynb (etapas 00–18)
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

## Execução no Google Colab (A100)

O repositório ainda não tem remoto. Passo a passo:

1. **Subir o repositório ao Google Drive**: compacte a pasta do projeto (sem `.venv`, `results/` e `.cache/`) e envie
   para `MyDrive/physics-llm-sr`. Quando houver um remoto, basta preencher `REPO_URL` no notebook (clonagem automática).
2. **Abrir o notebook** `notebooks/01_full_paper_reproduction.ipynb` no Colab (Drive, "Abrir com", Google Colaboratory).
3. **Escolher a GPU A100**: *Ambiente de execução, Alterar tipo de ambiente de execução, A100* (com High-RAM).
4. **Configurar o Llama 2**: o checkpoint `meta-llama/Llama-2-7b-chat-hf` é restrito. Aceite a licença no Hugging Face e crie o segredo
   `HF_TOKEN` no Colab (ícone de chave, "Acesso ao notebook" ligado). O notebook faz o login sem imprimir o token.
5. **Ajustar os parâmetros** da primeira célula: `RUN_PROFILE = "colab_pilot"` (ou `"replication"`), `ARM`, `MOUNT_DRIVE = True`
   (resultados e cache persistem em `DRIVE_PROJECT_DIR`) e `REPO_DIR` (pasta no Drive, por exemplo
   `/content/drive/MyDrive/physics-llm-sr`).
6. **Executar tudo**. A etapa 01 instala o pacote (`pip install -e <repo>[hf]`), monta o Drive e define
   `PHYSICS_LLM_SR_PROJECT_ROOT`. A primeira importação do PySR instala o Julia (minutos).

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
| `make notebook-smoke` | Executa o notebook em `RUN_PROFILE = "smoke"` de ponta a ponta (nbconvert); saída em `results/notebook_smoke/`. |

CLI direta: `uv run physics-llm-sr run --config configs/smoke.yaml [--only exp1,exp2] [--sr deap,gplearn] [--resume]` e
`uv run physics-llm-sr report <run_dir>`. Variáveis de ambiente: `PHYSICS_LLM_SR_PROJECT_ROOT` (raiz de `results/` e `.cache/`),
`PHYSICS_LLM_SR_PROMPTS_DIR` e `PHYSICS_LLM_SR_REFERENCES_DIR` (o notebook as define).

## Notebook

`notebooks/01_full_paper_reproduction.ipynb` percorre as etapas 00 a 18 do PRD (seção 10), cada uma com explicação teórica,
referência ao artigo, chamada ao módulo Python e visualização com interpretação. As células só importam e chamam `physics_llm_sr`. Parâmetros no topo:

- `RUN_PROFILE`: `smoke`, `pilot`, `colab_pilot` ou `replication`.
- `ARM`: `replication` (perda literal) ou `extended` (normalizada, desvio).
- `LOAD_EXISTING_RUN` (opcional): caminho de um `run_dir` existente (por exemplo, o do piloto) para **apresentar** os resultados sem executar.

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
