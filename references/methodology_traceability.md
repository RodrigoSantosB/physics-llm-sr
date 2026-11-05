# Matriz de rastreabilidade: artigo → requisito → módulo → teste

Requisitos R1..R13 e testes `T:*` conforme o grafo SDD
(`physics-llm-sr/.git/grafo-sdd/physics-llm-sr/grafo.json`). Módulos conforme o PRD §9.1.
Páginas do periódico; "S1" é o suplemento. "PRD §x" marca requisito de engenharia sem
correspondência direta no artigo. Os GAPs estão em `references/scientific_assumptions.md`.

## Requisitos

| ID | Requisito | Testes |
| --- | --- | --- |
| R1 | Fundação: pacote, config tipada, sementes, manifestos, persistência, qualidade | T:config, T:manifest, T:make-check |
| R2 | Três simuladores físicos | T:fisica, T:imports |
| R3 | Dados, ruído gaussiano/enviesado e particionamento sem vazamento | T:dados, T:ruido, T:split |
| R4 | AST comum, operadores protegidos e parser seguro | T:ast, T:operadores, T:parser-neg |
| R5 | Distância estrutural normalizada | T:tree-distance, T:tree-distance-neg |
| R6 | Avaliador LLM: contrato, parser, cache, prompts A–H | T:llm-parser, T:llm-parser-neg, T:llm-cache, T:prompts, T:llm-real |
| R7 | Perda combinada e parada antecipada | T:perda, T:early-stop |
| R8 | Adaptadores DEAP, gplearn e PySR com LLM na seleção | T:baseline-sr, T:inloop-deap, T:inloop-gplearn, T:inloop-pysr |
| R9 | Métricas, estatística e comparação com o artigo | T:metricas, T:comparacao |
| R10 | Runner dos experimentos 1–7, modos smoke/pilot/full, retomada | T:runner, T:retomada, T:e2e, T:piloto |
| R11 | Notebooks 01_baseline_sem_llm, 02_um_llm_um_caso e 03_pipeline_completo (etapas 00–18) | T:notebook |
| R12 | Auditoria metodológica (este diretório) | T:docs |
| R13 | Execução no Google Colab (A100) com backend transformers | T:colab, T:llm-hf, T:llm-hf-neg |

## Matriz

| Seção do artigo (pág.) | Conteúdo | Requisito | Módulo | Teste | GAP |
| --- | --- | --- | --- | --- | --- |
| LLM-based knowledge integration, Eq. (1) (p.4–5) | `L = w1·e + w2·s + w3·c`, pesos em [0,1] com soma 1 | R7 | `src/physics_llm_sr/optimization/objective.py`, `src/physics_llm_sr/config/schemas.py` | T:perda, T:config | GAP-01, GAP-02 |
| Idem (p.5) | `c = 1 − (c1 + c2 + c3)/3`; c1 dimensional, c2 simplicidade, c3 realismo | R7, R6 | `src/physics_llm_sr/optimization/objective.py`, `src/physics_llm_sr/llm/scoring.py` | T:perda, T:llm-parser | GAP-02 |
| Idem (p.5) | Prompt com `### ROLE`, `### METRICS`, exemplos e saída `[dim corr, simp, sim, feedback]` | R6 | `prompts/base/`, `src/physics_llm_sr/llm/prompting.py`, `src/physics_llm_sr/llm/response_parser.py` | T:prompts, T:llm-parser, T:llm-parser-neg | GAP-13 |
| Fig. 1 (p.2) e p.4 | LLM dentro do processo de busca, não pós-processamento | R8 | `src/physics_llm_sr/sr/deap_adapter.py`, `src/physics_llm_sr/sr/gplearn_adapter.py`, `src/physics_llm_sr/sr/pysr_adapter.py` | T:inloop-deap, T:inloop-gplearn, T:inloop-pysr | GAP-08 |
| p.4 ("represented as an expression tree") | Expressões como árvores; conversão entre bibliotecas | R4 | `src/physics_llm_sr/symbolic/expression.py`, `src/physics_llm_sr/symbolic/canonicalization.py` | T:ast, T:parser-neg | GAP-06, GAP-10 |
| In silico physical experiments, Eq. (2) (p.5) | Queda livre `v = √(2gh)`, g = 9.81 | R2 | `src/physics_llm_sr/physics/free_fall.py` | T:fisica | GAP-03 |
| Idem, Eq. (3) (p.6) | MHS `x = A cos(√(k/m) t + φ)` | R2 | `src/physics_llm_sr/physics/harmonic_motion.py` | T:fisica | GAP-03, INC-06 |
| Idem, Eq. (4) (p.6) | Onda amortecida `E = E0 e^(−αt/2) cos(kx − ωt)` | R2 | `src/physics_llm_sr/physics/damped_wave.py` | T:fisica | GAP-03, INC-07 |
| PRD §9.2 | Física independente de LLM, SR e otimização | R2 | `src/physics_llm_sr/physics/`, `src/physics_llm_sr/data/` | T:imports | — |
| Idem, geração de dados (p.6) | N = 500, cinco parâmetros, cerca de 100 passos, cerca de 50 000 amostras, SI | R3 | `src/physics_llm_sr/data/generator.py`, `src/physics_llm_sr/data/sampling.py`, `src/physics_llm_sr/data/validation.py` | T:dados | GAP-03, GAP-04 |
| Idem, ruído padrão (p.6) | Gaussiano de média zero, 1% da variabilidade, cerca de 40 dB | R3 | `src/physics_llm_sr/data/noise.py` | T:ruido | GAP-12 |
| Não informado (PRD §3.3) | Split por simulação, sem vazamento | R3 | `src/physics_llm_sr/data/splitting.py` | T:split | GAP-09 |
| Large language models (p.6) | Mistral 7B, Llama 2 7B, Falcon 7B, temperatura 0, uso local | R6, R13 | `src/physics_llm_sr/llm/inference.py`, `src/physics_llm_sr/llm/interfaces.py` | T:llm-real, T:llm-hf, T:llm-hf-neg | GAP-07 |
| p.11 e S1 (Apêndice) | Execução em Google Colab Premium | R13 | `notebooks/03_pipeline_completo.ipynb`, `configs/colab_pilot.yaml` | T:colab | GAP-07, GAP-14 |
| Symbolic regression models (p.6) e Tabela 1 (p.7) | Pop. 100, 50 gerações; DEAP cx 0.05, mut 0.01, torneio 3, prof. 8; Huber no PySR; parcimônia | R8 | `src/physics_llm_sr/sr/interfaces.py`, `src/physics_llm_sr/sr/deap_adapter.py`, `src/physics_llm_sr/sr/gplearn_adapter.py`, `src/physics_llm_sr/sr/pysr_adapter.py` | T:baseline-sr | GAP-05, H-02, H-03 |
| Idem, operadores (p.6, Tabela 1) | `+ − × ÷ exp log sin cos`; `neg`, `^`, divisão protegida | R4 | `src/physics_llm_sr/symbolic/operators.py` | T:operadores | GAP-06, INC-05 |
| Idem, parada antecipada (p.6) | Melhoria < 0.1% em três gerações | R7 | `src/physics_llm_sr/optimization/early_stopping.py`, `src/physics_llm_sr/optimization/search_history.py` | T:early-stop | H-01 |
| PRD §15 (custo); S1 Tabela 6 | Cache por expressão canônica, contagem de chamadas | R6 | `src/physics_llm_sr/llm/cache.py` | T:llm-cache | GAP-14 |
| Experiments (p.5) | MAE, MSE, R² (reportado 1−R²), distância | R9 | `src/physics_llm_sr/evaluation/metrics.py` | T:metricas | GAP-09 |
| Expression tree distance metric (p.7) | Definição recursiva, α, comutatividade, normalização | R5 | `src/physics_llm_sr/symbolic/tree_distance.py`, `src/physics_llm_sr/symbolic/canonicalization.py` | T:tree-distance, T:tree-distance-neg | GAP-10, GAP-11, INC-02 |
| Experiment types, Benchmarking (p.6) e S1 Tabela 5 | 27 configurações + baseline sem LLM | R10, R9 | `src/physics_llm_sr/experiments/baseline.py`, `src/physics_llm_sr/experiments/llm_integration.py`, `references/paper_results/` | T:runner, T:comparacao | GAP-12, INC-08 |
| LLM prompt knowledge integration (p.6–7) e Tabela 2 (p.9–10) | Variantes A–H; D, F, G e H revelam a GT | R6, R10 | `prompts/variants/`, `src/physics_llm_sr/experiments/prompt_sensitivity.py` | T:prompts, T:runner | GAP-13, INC-01 |
| Noise robustness experiment (p.7) e Tabela 3 (p.10) | Ruído de média zero 1–5% nas entradas, no alvo e em ambos | R3, R10 | `src/physics_llm_sr/data/noise.py`, `src/physics_llm_sr/experiments/noise_robustness.py` | T:ruido, T:runner | GAP-12, INC-03 |
| Results (p.8) e Tabela 4 (p.10) | Ruído enviesado, μ = 1% da média | R3, R10 | `src/physics_llm_sr/data/noise.py`, `src/physics_llm_sr/experiments/biased_noise.py` | T:ruido, T:runner | GAP-12 |
| Experiments (p.5) | n = 25 repetições, média | R10, R9 | `src/physics_llm_sr/experiments/runner.py`, `src/physics_llm_sr/evaluation/statistics.py` | T:retomada, T:comparacao | GAP-09, H-05 |
| Results e Discussion (p.7–8, p.11) | Comparação com as Tabelas 2–5 e afirmações A1–A7 | R9 | `src/physics_llm_sr/evaluation/paper_comparison.py`, `src/physics_llm_sr/evaluation/plots.py`, `references/paper_results/` | T:comparacao, T:docs | Critérios pré-registrados |
| Discussion (p.8) e S1 Tabela 6 | Razão de tempo LLM/SR, custo | R10, R13 | `src/physics_llm_sr/experiments/runner.py`, `configs/colab_pilot.yaml` | T:runner, T:piloto | GAP-14, INC-09 |
| PRD §14 / §16 Fase 0 | Lacunas, hipóteses, inconsistências | R12 | `references/scientific_assumptions.md`, `references/methodology_traceability.md`, `references/prompts_source.md` | T:docs | GAP-01..14 |
| PRD §11 | Manifesto, hashes, sementes, versões | R1 | `src/physics_llm_sr/reproducibility/manifests.py`, `src/physics_llm_sr/reproducibility/seeds.py`, `src/physics_llm_sr/reproducibility/artifacts.py` | T:manifest | GAP-09 |
| PRD §2, §17 | Config tipada, modos e qualidade | R1 | `src/physics_llm_sr/config/schemas.py`, `src/physics_llm_sr/config/loader.py`, `Makefile`, `pyproject.toml` | T:config, T:make-check | — |
| PRD §12.4 | Pipeline mínimo de ponta a ponta | R10 | `src/physics_llm_sr/experiments/runner.py` | T:e2e | — |
| PRD §10 | Notebook com etapas 00–18 | R11 | `notebooks/03_pipeline_completo.ipynb` | T:notebook | — |

## Matriz de experimentos

| Exp. | Fonte | Fatores | Configurações | Repetições | Saída comparada |
| --- | --- | --- | --- | --- | --- |
| 1 Baseline | p.6 | 3 cenários × 3 SR, `w3 = 0` | 9 | 25 | S1 Tabela 5 (linhas Baseline) |
| 2 SR + LLM | p.6 | 3 cenários × 3 SR × 3 LLM | 27 | 25 | S1 Tabela 5 |
| 3 Prompts | p.6–7 | 3 cenários × 3 SR × 3 LLM × 8 prompts | 216 | 25 | Tabela 2 (média dos cenários) |
| 4 Ruído média zero | p.7 | 3 cenários × 3 SR × 3 localizações × 5 níveis (+ baseline) | 135 (+135) | 25 | Tabela 3 |
| 5 Ruído enviesado | p.8 | idem | 135 (+135) | 25 | Tabela 4 |
| 6 Estrutural | p.7 | distância de cada equação final à GT | todas | — | coluna de distância das Tabelas 2–5 |
| 7 Comparação | PRD §7 | critérios pré-registrados | — | — | relatório por célula |
