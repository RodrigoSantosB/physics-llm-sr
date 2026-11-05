# Resultados publicados (referência de comparação)

Transcrição das tabelas de resultados de Taskin, Xie & Lazebnik, Sci. Rep. 16:1614 (2026),
DOI 10.1038/s41598-026-35327-6, e do seu suplemento (`41598_2026_35327_MOESM1_ESM.docx`).
São **valores publicados**, não resultados nossos. Servem de entrada para
`src/physics_llm_sr/evaluation/paper_comparison.py` e para os critérios pré-registrados em
`references/scientific_assumptions.md`.

## Arquivos

| Arquivo | Origem | Conteúdo | Células |
| --- | --- | --- | --- |
| `table2_prompt_sensitivity.csv` | Tabela 2, p.9–10 | Prompts A–H × LLM × SR; MAE, MSE, 1−R², distância; média dos três cenários | 72 linhas × 4 métricas = 288 |
| `table3_noise_zero_mean.csv` | Tabela 3, p.10 | Distância sob ruído de média zero, 3 cenários × 3 SR × 3 localizações × 5 níveis | 135 |
| `table4_noise_biased.csv` | Tabela 4, p.10 | Idem, ruído enviesado | 135 |
| `table5_benchmark_supplement.csv` | Suplemento, Tabela 5 | Benchmark: 3 cenários × (baseline + 3 LLM) × 3 SR; 4 métricas | 36 linhas × 4 = 144 |
| `table6_runtime_supplement.csv` | Suplemento, Tabela 6 | Tempos SR e LLM (s) por cenário × SR × LLM (esquema próprio) | 27 linhas |

**A Tabela 5 não está no artigo.** O texto (p.7, p.8) remete ao "supplementary material". Ela
foi encontrada no suplemento e transcrita aqui. O suplemento **não contém os prompts A–H**,
apesar da remissão de p.7 (ver `references/prompts_source.md`).

## Esquema (Tabelas 2–5, formato longo: uma linha por célula publicada)

| Coluna | Tipo / domínio | Significado |
| --- | --- | --- |
| `table` | `2`, `3`, `4`, `5` | número da tabela no artigo ou suplemento |
| `row_idx` | inteiro ≥ 1 | linha na ordem impressa da tabela original |
| `scenario` | `free_fall`, `harmonic_motion`, `damped_wave`, `average_3_scenarios` | cenário. Rótulos originais (INC-04) em `notes` |
| `sr` | `deap`, `gplearn`, `pysr` | algoritmo |
| `llm` | `mistral_7b`, `llama2_7b`, `falcon_7b`, `none` (baseline), `unspecified` | LLM |
| `prompt` | `A`–`H`, `none` (baseline), `unspecified` | variante de prompt |
| `noise_type` | `zero_mean`, `biased` | tipo de ruído |
| `noise_location` | `inputs`, `target`, `both` | onde o ruído é aplicado |
| `noise_level` | `0.01`–`0.05` | fração do desvio-padrão |
| `metric` | `mae`, `mse`, `one_minus_r2`, `tree_distance` | métrica (o artigo reporta 1−R²) |
| `value` | decimal em texto | valor publicado **com as casas publicadas** (3 para MAE e MSE, 2 para 1−R² e distância). A precisão faz parte do critério de comparação. |
| `source_page` | inteiro (página do periódico) ou `S1` (suplemento) | origem |
| `notes` | texto | condições inferidas, rótulo original, inconsistências (INC-xx) |

As condições das Tabelas 2 e 5 (ruído de média zero, 1%, no alvo) vêm do padrão declarado em
p.6 ("if not stated otherwise") e são hipótese. Nas Tabelas 3 e 4, LLM e prompt não são
informados (`unspecified`, GAP-12). Na Tabela 2 os prompts D, F, G e H revelam a GT.

## Como foi transcrito e conferido

- Tabelas 2–4: extraídas da camada de texto do PDF (`pdftotext -layout`, páginas 9–10) por
  script determinístico com asserções de estrutura (72 linhas na Tabela 2; 9 × 15 nas Tabelas
  3 e 4; ordem DEAP/PySR/gplearn por bloco), e conferidas visualmente contra as páginas 9 e 10
  renderizadas.
- Tabelas 5 e 6: extraídas célula a célula de `word/document.xml` do suplemento. As células
  mescladas (cenário e LLM) foram reconstruídas pela posição no bloco, e os rótulos presentes
  foram conferidos por asserção. O docx não foi renderizado visualmente.
- `tests/unit/test_references.py` valida esquema, domínios, precisão, contagens, ausência de
  duplicatas, a desigualdade MSE ≥ MAE² (todas as linhas passam, considerando o arredondamento)
  e a transcrição fiel de INC-01.

## Inconsistências conhecidas (detalhes em `references/scientific_assumptions.md`)

- **INC-01**: na Tabela 2, F, G e H são idênticos a E.
- **INC-02**: o texto (p.8) fala em "distance of 1.00 = perfect", e a tabela usa 0.00.
- **INC-08**: as linhas com LLM de "Dropping ball" na Tabela 5 são idênticas ao prompt A da
  Tabela 2, embora esta seja descrita como média dos três cenários.
- **INC-09**: na Tabela 6, as linhas Falcon têm razão, tempo por chamada e tempo por iteração
  incompatíveis com os totais. A razão "< 0.4" (p.8) só vale para o PySR.
- Os valores da Tabela 2 são muito regulares (MSE quase proporcional a MAE). Use-os como alvo
  quantitativo com a precisão publicada, sem presumir dispersão. O artigo não publica
  desvio-padrão nem IC.
