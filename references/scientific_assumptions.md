# Hipóteses científicas de reprodução (GAP-01..GAP-14)

Artigo: Taskin, Xie & Lazebnik, *Knowledge integration for physics-informed symbolic regression
using pre-trained large language models*, Sci. Rep. 16:1614 (2026), DOI 10.1038/s41598-026-35327-6.
Suplemento: `41598_2026_35327_MOESM1_ESM.docx` (Apêndice, Tabela 5 e Tabela 6).

Este documento é a Fase 0 do PRD (§14, §16). Ele formaliza as decisões registradas pelo
orquestrador em `CONVENCOES.md`; não as substitui. Páginas citadas são as do periódico (o PDF
tem 14 páginas, numeradas igual ao periódico). "S1" é o suplemento.

Status permitidos:

- **hipótese**: o artigo não define; escolhemos um valor plausível e registrado.
- **desvio**: escolhemos conscientemente algo diferente do que o artigo descreve.
- **reconstruído**: o artefato existia, mas não foi publicado; reconstruímos a partir do texto.
- **limitação**: não é possível reproduzir fielmente; o efeito na comparabilidade fica declarado.

As hipóteses ficam congeladas em `configs/replication.yaml` antes de qualquer execução final
(PRD §5.2, §14). Toda mudança posterior exige nova versão de configuração e novo `config_hash`.

---

## Inconsistências encontradas no artigo e no suplemento

| ID | Onde | Inconsistência | Tratamento |
| --- | --- | --- | --- |
| INC-01 | Tabela 2, p.9–10 | As 9 linhas de F, G e H são numericamente idênticas às de E (MAE, MSE, 1−R² e distância). | Transcritas fielmente; comparação de F/G/H marcada com essa nota. O teste `test_table2_prompt_blocks_and_inc01` trava a transcrição. |
| INC-02 | p.8 vs p.7 e Tabela 2 | p.8: "for the cases where the GT equation is provided to the LLM (Cases D, F, G, and H), an expression tree distance of 1.00 is obtained, meaning a perfect reconstruction". A definição (p.7) diz "A zero distance signifies structural equivalence"; a Tabela 2 mostra 0.00 em E–H e valores de 0.00 a 0.10 em D. | Distância 0 = equivalência (GAP-11). A afirmação de que D é perfeito não é sustentada pela tabela. |
| INC-03 | p.8 vs Tabela 3 | Texto: "DEAP ... from 0.07 at 1% noise to 0.18 at 5% noise"; "DEAP and gplearn start at lower baselines – 0.80 and 0.90 at 1% noise"; "as small as 0.60 to 0.40 at the 5% noise level". Nenhum desses números aparece na Tabela 3 (DEAP 1% entradas: 0.17/0.12/0.10; 5%: 0.30/0.25/0.22). Além disso p.8 ("as indicated by Table 3") se refere ao estudo de prompts, que está na Tabela 2. | A comparação usa somente as tabelas. As frases de p.8 sobre a Tabela 3 não viram critério. O texto sobre a Tabela 4 (p.8) confere com as médias da tabela (DEAP entradas 0.16→0.32; ambos 0.44→0.60; PySR 0.10→0.23). |
| INC-04 | Resumo, Fig. 2, tabelas, S1 | Nomes dos cenários: "dropping ball / simple harmonic motion / electromagnetic wave" (resumo, Tabelas 3–5), "free fall / harmonic motion / damped wave" (Fig. 2, p.5), "Ball Drop / SHM / Wave" (Tabela 6). | Identificadores internos: `free_fall`, `harmonic_motion`, `damped_wave`. O rótulo original fica na coluna `notes` dos CSVs. |
| INC-05 | p.6 vs Tabela 1 (p.7) | Texto: "Across all three SR models, we utilized the following operator set: +, −, ×, ÷, exp, log, sin, and cos". A Tabela 1 acrescenta `neg` aos três, `^` ao PySR e "protected div" ao DEAP, e a lista unária do DEAP não traz `exp`. | Conjunto por algoritmo definido no GAP-06. |
| INC-06 | Eq. (3) e texto, p.6 | A Eq. (3) renderizada aparece como `x(t) = A cos (k/m) t + φ` (sem raiz). O texto da mesma página escreve `x = A cos(√(k/m t + φ))`, com a fase dentro da raiz. | Adotada a forma física correta `x(t) = A cos(√(k/m)·t + φ)`, igual ao PRD §3.1. |
| INC-07 | p.6, Eq. (4) | O coeficiente de amortecimento é amostrado em "0 to 1 kg s⁻¹" (unidade de arrasto linear), mas na Eq. (4) α multiplica t no expoente (exige s⁻¹). A Eq. (4) escreve `cos kx − wt` sem parênteses e usa `w` onde o texto usa ω. | α em s⁻¹ com a mesma faixa numérica [0, 1]; argumento do cosseno `(k·x − ω·t)` (GAP-03). |
| INC-08 | S1 Tabela 5 vs Tabela 2 | As 9 linhas com LLM de "Dropping ball" na Tabela 5 são idênticas às linhas do prompt A da Tabela 2, embora a Tabela 2 seja descrita como média sobre os três cenários (p.8: "average performance ... in three sets of physical experiments"). | Transcrito fielmente com nota. Indício de que a Tabela 5 usa o prompt A ou de que há erro de edição; o prompt da Tabela 5 fica "não informado" (GAP-12). |
| INC-09 | S1 Tabela 6 e p.8 | Nas 9 linhas com Falcon, a razão LLM/SR é 0.0 ou 0.1, o tempo médio de LLM por chamada é 0.0 e o tempo por iteração é 0.1, o que não fecha com Total LLM de 26 a 49 s. O texto (p.8) diz "the LLM-to-SR runtime ratio typically remains below 0.4", mas a Tabela 6 mostra razões de 5 a 19 (DEAP) e de 25 a 77 (gplearn). Só o PySR fica abaixo de 0.4. | Transcrito com nota por linha. A razão "< 0.4" não é usada como critério de aceitação (GAP-14). |
| INC-10 | p.5 | A frase final da seção está truncada ("(i) dimensional consistency (dimcorr)and(ii)physicalrealism/plausibility(sim), whichoperationa.") e cita só dois dos três critérios. | Seguimos os três critérios explicitados no parágrafo anterior (GAP-13). |
| INC-11 | p.6 vs p.11 | Temperatura zero "to guarantee deterministic, repeatable outputs" (p.6) vs "the non-deterministic nature of LLMs led to occasional invalid or syntactically incorrect suggestions" (p.11). | Temperatura 0 / `do_sample=False`. Determinismo não é presumido entre ambientes (PRD §6.2); respostas inválidas seguem a política H-04. |
| INC-12 | p.8 (Discussão) | Os LLMs aparecem como "(DEAP, Falcon, and Mistral)". | Erro de digitação evidente (Llama 2). Sem efeito. |
| INC-13 | Legendas das Tabelas 2–4 | Na Tabela 2 a legenda diz "expression-tree score" e a coluna diz "Expression tree distance". A Tabela 3 diz "score" na legenda e na nota. A Tabela 4 diz "Values are expression-tree distances". | Tratamos todas como a mesma distância (menor = melhor). |
| INC-14 | S1, Apêndice | Hardware descrito como "32 GB with 2 virtual cores and 100 GB of RAM" (sem GPU identificada). | Registro apenas. GPU do Colab desconhecida (GAP-14). |
| INC-15 | p.5 | "each LLM returns a concise, three-element list", mas o formato descrito é `[dim corr, simp, sim, feedback]`, com quatro elementos. | Quatro elementos: três escores e o feedback (`references/prompts_source.md`). |

---

## GAP-01 a GAP-14

### GAP-01 — Pesos w1, w2, w3

- **Questão:** Quais são os valores de `w1`, `w2` e `w3` em `L = w1·e + w2·s + w3·c`, e quais pesos usar no baseline sem LLM?
- **Evidência:** p.4, Eq. (1): "where w1, w2, w3 ∈ [0, 1] ∧ w1 + w2 + w3 = 1". Não há valores numéricos no artigo nem no suplemento. p.6: o baseline é a configuração "in which no LLM model is used".
- **Hipótese adotada:** No modo replication, o padrão literal é `w = (1/3, 1/3, 1/3)`. A grade de sensibilidade é {(0.8, 0.1, 0.1), (0.6, 0.1, 0.3), (0.4, 0.1, 0.5)}. Baseline: `w3 = 0`, com `w1` e `w2` renormalizados para manter a soma 1, o que dá `(0.5, 0.5, 0)` para o padrão. A renormalização preserva a ordem dos candidatos, então a seleção por torneio não muda. Com `w3 = 0` o avaliador LLM não é chamado.
- **Justificativa:** Pesos iguais são a escolha menos informativa compatível com a restrição publicada. A grade mostra o quanto a conclusão depende dessa escolha. A renormalização mantém a validação "soma = 1" do esquema de configuração (C:config-tipada) também no baseline.
- **Chave de configuração:** `loss.w1`, `loss.w2`, `loss.w3`, `loss.sensitivity_grid`, `loss.baseline_renormalize`
- **Impacto na comparabilidade:** Alto. Na escala literal (GAP-02), `s` (contagem de nós, cerca de 5 a 30) domina `c ∈ [0, 1]`. Com pesos iguais, um nó a mais pesa tanto quanto toda a faixa de `c`. Diferenças em relação às Tabelas 2–5 podem vir só dos pesos. Os pesos ficam congelados antes das execuções finais.
- **Status:** hipótese

### GAP-02 — Escalas dos componentes da perda

- **Questão:** Como combinar `e` (MSE em unidades SI ao quadrado), `s` (número de nós) e `c ∈ [0, 1]`, que têm escalas diferentes?
- **Evidência:** p.4–5: "e is the mean square error between the prediction of the provided analytical function and the provided dataset, s is the size of the analytical equation in terms of nodes in the expression tree". p.6: "The data were retained in their original SI units, relying on the regression models to manage any required scaling". Nenhuma normalização é descrita.
- **Hipótese adotada:** Há dois modos. `literal` (padrão do replication): `e` = MSE em SI, `s` = número de nós da AST comum e `c = 1 − (c1 + c2 + c3)/3`. `normalized` (padrão só no extended): `e = MSE / Var(y_train)` e `s = nós / max_nodes`. O modo `normalized` é desvio.
- **Justificativa:** O texto descreve a soma literal, e reproduzir significa usá-la. A versão normalizada fica disponível como análise de engenharia separada (PRD §2.2).
- **Chave de configuração:** `loss.scale_mode`, `loss.max_nodes`
- **Impacto na comparabilidade:** Alto. Na queda livre, `v` chega a 44 m/s, e um MSE ruim (dezenas a centenas) domina `L`. Nos cenários oscilatórios o MSE fica abaixo de 1 e `s` domina. Resultados do modo normalized nunca entram nas tabelas de reprodução.
- **Status:** hipótese + desvio

### GAP-03 — Parâmetros dos simuladores

- **Questão:** Como os cinco parâmetros amostrados (massa, comprimento característico, altura, amortecimento e tempo) se mapeiam nas variáveis de cada GT, e quais são os valores de k, φ, E0, k_onda e ω?
- **Evidência:** p.6: "(1) mass, randomly selected from 0.1 to 10 kg; (2) a characteristic length (e.g. drop height, spring amplitude, or radius) between 0.01 and 0.5 m; (3) initial height/displacement, varying from 1 to 100 m; (4) a linear drag/damping coefficient from 0 to 1 kg s⁻¹; and (5) time, sampled randomly between zero and the physically realistic maximum of √(2h/g), where g = 9.81 m s⁻²". Eqs. (2)–(4) nas p.5–6. k, φ, E0, k_onda e ω não têm valores no artigo.
- **Hipótese adotada:** `X` tem sempre as 5 colunas `mass`, `length`, `height`, `damping` e `t`. As colunas que não entram na GT ficam como distratores.
  - `free_fall`: `h0 ~ U(1, 100)` m; `t ~ U(0, √(2·h0/g))`; a coluna `height` é a distância percorrida `h(t) = g·t²/2`; alvo `v = √(2·g·height)`; `g = 9.81`; sem arrasto na GT.
  - `harmonic_motion`: `A = length ~ U(0.01, 0.5)` m; `m = mass ~ U(0.1, 10)` kg; `k = 10` N/m fixo; `φ = 0` (soltura do repouso na amplitude); `t ~ U(0, T)` com `T = 2π·√(m/k)`.
  - `damped_wave`: `α = damping ~ U(0, 1)` s⁻¹ (INC-07); `x = length` (posição); `E0 = 1.0` V/m, `k = 2π` rad/m, `ω = 2π` rad/s; `t ~ U(0, T)` com `T = 2π/ω`.
  - Formas de referência para a distância (após a canonicalização do GAP-10): `free_fall`: `pow(19.62*height, 0.5)`; `harmonic_motion`: `length*cos(pow(10/mass, 0.5)*t)`; `damped_wave`: `exp(-0.5*(damping*t))*cos(6.283185*length - 6.283185*t)`, com `E0 = 1` omitido.
- **Justificativa:** As faixas publicadas são mantidas. Constantes não publicadas recebem valores simples e fisicamente válidos, ficam na configuração e entram no manifesto. `φ = 0` e `E0 = 1` evitam constantes que só o SR teria de descobrir. O comprimento característico do texto ("spring amplitude") mapeia naturalmente em `A` e em `x`.
- **Chave de configuração:** `physics.free_fall.g`, `physics.free_fall.h0_range`, `physics.harmonic_motion.spring_constant`, `physics.harmonic_motion.phase`, `physics.damped_wave.e0`, `physics.damped_wave.wave_number`, `physics.damped_wave.angular_frequency`, `data.parameter_ranges`
- **Impacto na comparabilidade:** Médio a alto. **Identificabilidade na queda livre:** com `height = g·t²/2`, tanto `v = √(2·g·height)` quanto `v = g·t` são exatos nos dados. `9.81*t` tem 3 nós e é mais barato em `s` que a GT. Uma busca que minimiza `e + s` pode preferir `g·t`, que tem distância alta para a GT apesar de ser fisicamente correta. O relatório deve registrar a fração de soluções equivalentes a `g·t` (ver "Discordâncias e riscos"). Com `k` fixo, a GT harmônica contém a constante 10, que o SR precisa descobrir.
- **Status:** hipótese + limitação

### GAP-04 — Processo temporal, malha e parada

- **Questão:** Como interpretar "1000 steps", "equilibrium", "state observed twice" e "~100 steps" para fórmulas analíticas?
- **Evidência:** p.6: "Each experiment is run until reaching equilibrium or 1000 steps in time"; "these calculations occur until an equilibrium is achieved or the first time a state is observed twice (for period dynamics). On average, this process took around 100 steps in time, resulting in around 50,000 samples for each case."
- **Hipótese adotada:** Cada simulação tem 100 instantes amostrados uniformemente, de forma independente, e ordenados, no intervalo físico. Na queda livre o intervalo vai até o impacto no solo. Nos casos oscilatórios cobre um período ("estado observado duas vezes"). O teto é de 1000 instantes por simulação. São 500 simulações, cerca de 50 000 amostras por cenário. Não há integração numérica: a resposta é analítica em cada instante.
- **Justificativa:** O número médio publicado (cerca de 100 passos e cerca de 50 000 amostras) é preservado. Para fórmulas fechadas, "equilíbrio" e "estado repetido" equivalem aos limites físicos acima.
- **Chave de configuração:** `data.n_simulations`, `data.steps_per_simulation`, `data.max_steps`, `data.time_sampling`
- **Impacto na comparabilidade:** Baixo a médio. A distribuição de `t` muda a dificuldade relativa das regiões do domínio, mas o tamanho total confere com o artigo.
- **Status:** hipótese

### GAP-05 — MSE geral vs Huber no PySR

- **Questão:** Qual função de erro entra em `e` para cada algoritmo?
- **Evidência:** p.4: `e` é o MSE (Eq. 1). p.6: "for PySR, we used the Huber loss function and enabled a complexity penalty that softly discourages extra operators".
- **Hipótese adotada:** PySR: `e = Huber_δ(y, ŷ)` com `δ = 1.0`, dentro da perda composta. DEAP e gplearn: `e = MSE`. A penalidade de complexidade de cada biblioteca é o próprio termo `w2·s`, sem dupla contagem: `parsimony = 0` no PySR e `parsimony_coefficient = 0` no gplearn quando `L` é calculada pelo adaptador. As métricas reportadas (MAE, MSE, 1−R²) são calculadas sempre fora da busca, iguais para os três algoritmos.
- **Justificativa:** O artigo afirma Huber para o PySR, e seguimos o texto. `δ` não é informado, e 1.0 é o valor usual. Unificar a penalidade de complexidade em `w2·s` dá significado à Eq. (1) para os três algoritmos.
- **Chave de configuração:** `sr.pysr.elementwise_loss`, `sr.pysr.huber_delta`, `sr.pysr.parsimony`, `sr.gplearn.parsimony_coefficient`
- **Impacto na comparabilidade:** Médio. A Huber reduz a influência de grandes resíduos (queda livre em SI), o que muda o equilíbrio entre `e` e `c` só no PySR.
- **Status:** hipótese

### GAP-06 — Conjunto de operadores por algoritmo

- **Questão:** Quais operadores cada algoritmo usa, dada a divergência entre o texto e a Tabela 1 (INC-05)?
- **Evidência:** p.6: "+, −, ×, ÷, exp, log, sin, and cos" para os três. Tabela 1 (p.7): DEAP binários "+, −, *, /, protected div" e unários "neg, log, sin, cos"; PySR binários "+, −, *, /, ^" e unários "neg, exp, log, sin, cos"; gplearn binários "add, sub, mul, div" e unários "neg, exp, log, sin, cos".
- **Hipótese adotada:** União do texto com a Tabela 1, sem `sqrt`:
  - `operators.deap`: `add, sub, mul, protected_div, neg, exp, log, sin, cos`. O `exp` vem do texto e é necessário para a GT da onda. "/" e "protected div" da Tabela 1 são tratados como o mesmo operador protegido.
  - `operators.gplearn`: `add, sub, mul, div, neg, log, sin, cos` nativos (o `div` e o `log` do gplearn já são protegidos) mais `exp` protegido via `gplearn.functions.make_function`.
  - `operators.pysr`: binários `+, -, *, /, ^`; unários `neg, exp, log, sin, cos`, com a semântica segura nativa do PySR.
  - Semântica protegida na AST comum (avaliação, métricas e conversões), inspirada na do gplearn: `div(a, b) = 1` se `|b| ≤ 0.001`; `log(x) = log|x|` se `|x| > 0.001`, senão 0; `exp(x)` com argumento limitado a 100; `pow(a, b)` só com base `|a|` quando o expoente não é inteiro. O resultado nunca é NaN ou inf (C:operadores).
  - A AST comum inclui `pow` para representar `^` do PySR e `sqrt` canonicalizado (GAP-10).
- **Justificativa:** O texto diz que o conjunto é comum aos três. A Tabela 1 detalha nomes e proteções. A união sem `sqrt` respeita as duas fontes. Sem `exp` no DEAP, a GT da onda amortecida seria inalcançável.
- **Chave de configuração:** `operators.deap`, `operators.gplearn`, `operators.pysr`, `operators.protection.div_threshold`, `operators.protection.log_threshold`, `operators.protection.exp_clip`
- **Impacto na comparabilidade:** Alto para a distância estrutural. Sem `sqrt`, as GTs de queda livre e harmônica só são representáveis como `x^0.5` (PySR) ou `exp(0.5·log(x))` (DEAP e gplearn). A distância 0 em DEAP e gplearn depende da canonicalização do GAP-10. Incluir `sqrt` fica para o extended (`operators.*.extra`), como desvio.
- **Status:** hipótese

### GAP-07 — Versões exatas dos modelos de linguagem

- **Questão:** Quais checkpoints, revisões e quantizações correspondem a "Mistral 7B, Llama 2 7B, and Falcon 7B"?
- **Evidência:** p.5–6: "three open-source and locally-deployable LLMs—Mistral 7B, Llama 2 7B, and Falcon 7B"; p.6: "we set the sampling temperature to zero"; p.11: "Google Colab Premium environment". Sem IDs, revisões ou quantização.
- **Hipótese adotada:** No Colab, com transformers em fp16: `mistralai/Mistral-7B-Instruct-v0.1`, `meta-llama/Llama-2-7b-chat-hf` (gated, exige token HF) e `tiiuae/falcon-7b-instruct`, cada um com revisão fixada por SHA no primeiro download. Sem `trust_remote_code`, porque o Falcon é suportado nativamente pelo transformers atual. No ambiente local, com Ollama: `mistral:7b-instruct`, `llama2:7b-chat` e `falcon:7b-instruct` em Q4, o que é desvio de quantização. Em todos: temperatura 0 / `do_sample=False`.
- **Justificativa:** Usamos as variantes instruct/chat porque o prompt tem papel, critérios e formato de saída e exige seguir instruções. Fixar por SHA permite reexecução. Q4 só serve para validação local.
- **Chave de configuração:** `llm.models.mistral.checkpoint`, `llm.models.mistral.revision`, `llm.models.llama2.checkpoint`, `llm.models.falcon.checkpoint`, `llm.backend`, `llm.dtype`, `llm.temperature`, `llm.max_new_tokens`
- **Impacto na comparabilidade:** Médio. Variante base vs instruct e versão (Mistral v0.1/v0.2) mudam os escores. As execuções locais em Q4 nunca são rotuladas como reprodução.
- **Status:** hipótese + desvio

### GAP-08 — Integração in-loop por biblioteca

- **Questão:** Como fazer o termo `w3·c` participar da seleção e otimização em cada biblioteca, e não só da reordenação final?
- **Evidência:** p.4: "In this study, we aim to integrate LLM into the SR's search process itself". Fig. 1 (p.2): o LLM alimenta a "Loss function" do SR ("Update"). p.6: a parada antecipada usa "the best composite loss". O artigo não diz como cada biblioteca foi adaptada.
- **Hipótese adotada:** Hipótese técnica, a comprovar na onda 5 por T:inloop-deap, T:inloop-gplearn e T:inloop-pysr. Prova exigida: com o mesmo `e` e o mesmo `s`, mudar os escores do avaliador simulado muda o candidato selecionado quando `w3 > 0` e não muda quando `w3 = 0`.
  - **DEAP**: `toolbox.register("evaluate", f)`, com `f(individual)` devolvendo `(L,)` e `creator.FitnessMin`. `L` usa `e` (MSE do indivíduo compilado pela AST, nunca `gp.compile`/`eval` sobre texto), `s = len(individual)` e `c` do avaliador. Laço geracional próprio (equivalente a `eaSimple`, com `cxpb = 0.05`, `mutpb = 0.01` e `selTournament(tournsize = 3)`) para permitir a parada antecipada e o histórico. Os escores LLM são pré-calculados em lote para os indivíduos inválidos de cada geração, deduplicados pela forma canônica.
  - **gplearn**: `make_fitness(function, greater_is_better)` recebe só `(y, y_pred, sample_weight)` e não dá acesso ao programa. Pela API pública, o termo LLM não entra no fitness. Estratégia: subclasse do programa (`gplearn._program._Program`) sobrescrevendo `raw_fitness(X, y, sample_weight)` para devolver `L` com acesso a `self.program` e `self.length_`, com `parsimony_coefficient = 0` e `n_jobs = 1` (o callback LLM e o cache ficam no processo principal). A evolução avança uma geração por vez (`warm_start = True` e `generations` incrementado) para aplicar a parada antecipada. Isso usa API privada: a versão do gplearn fica fixada e o uso é registrado. Alternativa, se a substituição não for possível: limitação declarada.
  - **PySR**: perda customizada em Julia via `loss_function` (assinatura `(tree, dataset, options)`, que dá acesso à árvore). A função Julia calcula `w1·Huber + w2·complexidade` e chama de volta um callable Python registrado em `Main` via juliacall/PythonCall para obter `c` (string canônica → cache → avaliador). Exige execução serial (`parallelism = "serial"`, sem multithreading) por causa do GIL, e `populations = 1` com `population_size = 100` para a hipótese "população 100". Se o callback Julia→Python não funcionar de forma estável, o adaptador não é rotulado como integrado (C:inloop-pysr), o GAP vira limitação e o PySR com LLM fica fora das tabelas de reprodução.
- **Justificativa:** O PRD §4.3 proíbe chamar de integração a reordenação pós-busca. Cada estratégia coloca `c` no ponto em que a biblioteca escolhe sobreviventes.
- **Chave de configuração:** `sr.deap.evaluation_mode`, `sr.gplearn.fitness_hook`, `sr.pysr.loss_mode`, `sr.pysr.parallelism`, `sr.pysr.populations`, `llm.evaluation_scope`
- **Impacto na comparabilidade:** Alto. O PySR faz muito mais avaliações de perda por iteração que DEAP e gplearn (GAP-14), então a mesma "população 100 × 50 iterações" não equivale entre bibliotecas. Uma integração inviável no PySR invalida as linhas PySR com LLM das Tabelas 2–5.
- **Status:** hipótese

### GAP-09 — Validação e particionamento

- **Questão:** Como dividir treino, validação e teste, em quais dados calcular as métricas e como tratar as 25 repetições?
- **Evidência:** Não há split nem sementes no artigo. p.5: "the results are shown as the mean value of n = 25 repetitions". p.8: "each experiment had slightly different samples".
- **Hipótese adotada:** Split 70/15/15 por `simulation_id` (sem trajetória compartilhada), com semente da configuração. A busca usa o treino; a validação serve à parada antecipada e ao diagnóstico; as métricas principais são calculadas no teste, contra `y_observed`, e também reportadas no treino e contra `y_true`. Cada repetição `r` regenera os dados com semente derivada `(seed_base, r)`. Baseline e LLM compartilham os dados e a semente do SR da mesma repetição (pareamento).
- **Justificativa:** O split por simulação evita vazamento (PRD §3.3). O texto de p.8 sugere amostras diferentes por repetição. O pareamento reduz a variância da diferença baseline × LLM.
- **Chave de configuração:** `split.train`, `split.validation`, `split.test`, `split.group_by`, `seeds.base`, `seeds.regenerate_data_per_repetition`, `experiment.repetitions`
- **Impacto na comparabilidade:** Médio. Se o artigo reportou métricas no treino, nossos valores ficam sistematicamente maiores. Por isso as métricas de treino também são persistidas.
- **Status:** hipótese

### GAP-10 — Distância entre árvores: definição e normalização

- **Questão:** Como formalizar a distância descrita na p.7: custo dos nós, constantes, comutatividade e "normalized by the number of node comparisons"?
- **Evidência:** p.7 (literal): "Identical leaf node contents yield a distance of zero; differing variables yield one. For differing numbers, the distance is α|v1 − v2| for α ∈ [0, 1], capped at one. A leaf/internal node mismatch yields a distance of one. ... mismatched operators at matched internal nodes contribute a distance of one. For identical, commutative operators (e.g., '+' or '*'), the subtree matches with the minimum total distance (direct or cross) is selected. For non-commutative operators (e.g. '-', '/'), subtree matches are strictly left-to-left and right-to-right, and subdistances are summed. These local distances are summed and normalized by the number of node comparisons, yielding a distance in [0, 1]". O valor de α não é dado.
- **Hipótese adotada:** Definição recursiva abaixo, com `α = 1.0` e grade {0.1, 0.5, 1.0}. A comparação usa as ASTs após a canonicalização mínima `tree_distance.canonicalize = true`, que faz só três coisas: (i) dobra subárvores só de constantes (`2*9.81 → 19.62`); (ii) reescreve `sqrt(u) → pow(u, 0.5)`; (iii) reescreve `exp(c*log(u)) → pow(u, c)` (inclusive `exp(log(u)*c)`). Não há simplificação algébrica: `x + 0` continua diferente de `x`. Por padrão, operadores diferentes com a mesma aridade continuam a descida (`descend_on_operator_mismatch = true`).
- **Justificativa:** Cada comparação de nós contribui com um custo local em [0, 1] e conta 1 no denominador, então `d ∈ [0, 1]`, e `d = 0` só ocorre com equivalência estrutural (com α > 0). A descida em operadores diferentes dá a granularidade observada nas tabelas (valores como 0.05 e 0.12). Sem ela, qualquer raiz diferente daria 1. A canonicalização (ii)–(iii) é necessária para que DEAP e gplearn, sem `sqrt` (GAP-06), possam atingir distância 0, como a Tabela 2 relata.
- **Chave de configuração:** `tree_distance.alpha`, `tree_distance.alpha_grid`, `tree_distance.canonicalize`, `tree_distance.descend_on_operator_mismatch`, `tree_distance.commutative_operators`
- **Impacto na comparabilidade:** Alto. α, a regra de descida e a canonicalização mudam todos os valores das Tabelas 2–5. A grade de α é reportada.
- **Status:** reconstruído + hipótese

**Definição.** Seja `pair(a, b) → (S, N)`, em que `S` é a soma dos custos locais e `N` é o número de comparações de nós. A distância é `d(T1, T2) = S / N`, calculada na raiz.

1. Folha × folha (`N = 1`):
   - variável × variável: `S = 0` se o nome é igual, senão `S = 1`;
   - constante × constante: `S = min(1, α·|v1 − v2|)`;
   - variável × constante: `S = 1`.
2. Folha × nó interno (em qualquer ordem): `S = 1`, `N = 1`, sem descida.
3. Nó interno × nó interno com o mesmo operador:
   - não comutativo ou unário: `S = 0 + Σ S_i`, `N = 1 + Σ N_i`, com filhos pareados por posição;
   - comutativo binário (`add`, `mul`): calcula-se o pareamento direto `(a1,b1),(a2,b2)` e o cruzado `(a1,b2),(a2,b1)` e escolhe-se o de menor `S`. Em empate, fica o de maior `N`; persistindo o empate, o direto. Então `S = 0 + S_escolhido` e `N = 1 + N_escolhido`.
4. Nó interno × nó interno com operadores diferentes:
   - mesma aridade e `descend_on_operator_mismatch = true`: `S = 1 + Σ S_i`, `N = 1 + Σ N_i`, com filhos por posição (sem melhor correspondência, pois o operador não coincide);
   - aridades diferentes, ou `descend_on_operator_mismatch = false`: `S = 1`, `N = 1`.

Propriedades que os testes T:tree-distance e T:tree-distance-neg devem verificar: `d ∈ [0, 1]`; `d(T, T) = 0`; simetria `d(T1, T2) = d(T2, T1)`; troca de operandos comutativos dá 0; troca em operador não comutativo dá valor > 0; `x + 0` vs `x` dá valor > 0; nó desconhecido gera erro.

**Exemplos calculados à mão** (`α = 1.0`, `descend_on_operator_mismatch = true` e ASTs sem canonicalização, salvo indicação):

| ID | T1 | T2 | Cálculo | S | N | d |
| --- | --- | --- | --- | --- | --- | --- |
| TD-01 | `x` | `x` | folha igual | 0 | 1 | 0 |
| TD-02 | `x` | `y` | variáveis diferentes | 1 | 1 | 1 |
| TD-03 | `2.0` | `2.5` | min(1, 1·0.5) | 0.5 | 1 | 0.5 |
| TD-04 | `2.0` | `5.0` | min(1, 1·3) | 1 | 1 | 1 |
| TD-05 | `2.0` | `5.0` (α = 0.1) | min(1, 0.1·3) | 0.3 | 1 | 0.3 |
| TD-06 | `x + y` | `y + x` | `+` comutativo: direto S=2, cruzado S=0 → cruzado; N = 1+1+1 | 0 | 3 | 0 |
| TD-07 | `x - y` | `y - x` | não comutativo: 0 + 1 + 1 | 2 | 3 | 0.6667 |
| TD-08 | `x + 0` | `x` | nó interno × folha | 1 | 1 | 1 |
| TD-09 | `x * y` | `x + y` | operador diferente (1), descida por posição 0 + 0 | 1 | 3 | 0.3333 |
| TD-10 | `sin(x)` | `cos(x)` | operador diferente (1) + folha igual (0) | 1 | 2 | 0.5 |
| TD-11 | `sin(x)` | `x * y` | aridades diferentes, sem descida | 1 | 1 | 1 |
| TD-12 | `x * y` | `x + y` (`descend_on_operator_mismatch = false`) | sem descida | 1 | 1 | 1 |
| TD-13 | `x * (y + 2)` | `(y + 2.5) * x` | raiz `*`: direto S=2 (N=2); cruzado `x~x` (0,1) + `(y+2)~(y+2.5)` direto (0 + 0 + 0.5, N=3) → S=0.5; N = 1+1+3 | 0.5 | 5 | 0.1 |
| TD-14 | `x + x` | `x + y` | empate direto/cruzado (S=1, N=2) → direto; N = 1+2 | 1 | 3 | 0.3333 |
| TD-15 | `exp(-0.5 * (a * t))` | `exp(-0.4 * (t * a))` | `exp` (0); `*`: direto `-0.5~-0.4` (0.1) + `(a*t)~(t*a)` (0, N=3) → S=0.1, N=4; N total 1+1+4 | 0.1 | 6 | 0.0167 |
| TD-16 | `pow(19.62 * h, 0.5)` | `sqrt(19.6 * h)` (canonicalize → `pow(19.6 * h, 0.5)`) | `pow` (0); `*`: `19.62~19.6` (0.02) + `h~h` (0) → N=3; `0.5~0.5` (0, N=1); N = 1+3+1 | 0.02 | 5 | 0.004 |
| TD-17 | `pow(19.62 * h, 0.5)` | `exp(0.5 * log(19.62 * h))` (canonicalize → `pow(19.62 * h, 0.5)`) | idênticas após canonicalização | 0 | 5 | 0 |
| TD-18 | `pow(19.62 * h, 0.5)` | `9.81 * t` | raiz `pow` × `*`, mesma aridade: 1 + (`19.62*h`~`9.81`: interno × folha = 1) + (`0.5`~`t`: constante × variável = 1) | 3 | 3 | 1 |

Observação sobre TD-18: `v = g·t` é exata nos dados da queda livre (GAP-03), mas tem distância máxima para a GT.

### GAP-11 — Distância 0 vs 1 como equivalência

- **Questão:** A equivalência estrutural corresponde a distância 0 ou 1?
- **Evidência:** p.7: "A zero distance signifies structural equivalence (accounting for commutativity); higher values indicate structural deviations". p.8: "an expression tree distance of 1.00 is obtained, meaning a perfect reconstruction of the GT equation" (INC-02). A Tabela 2 mostra 0.00 nas linhas descritas como perfeitas (E–H).
- **Hipótese adotada:** Distância 0 = correspondência estrutural; quanto maior, pior. A frase de p.8 é tratada como erro de redação.
- **Justificativa:** A definição formal e as tabelas concordam entre si. Só a frase de p.8 diverge.
- **Chave de configuração:** `tree_distance.zero_means_equivalent`
- **Impacto na comparabilidade:** Nenhum nos números, porque as tabelas já usam 0 = equivalência. Evita inverter a interpretação na comparação.
- **Status:** hipótese

### GAP-12 — Condições das tabelas de ruído (e da Tabela 5)

- **Questão:** Quais LLM, prompt, pesos e baseline correspondem às células das Tabelas 3 e 4, como o ruído enviesado é parametrizado e qual prompt foi usado na Tabela 5?
- **Evidência:** p.7: "we first followed the regular data generation procedure ... but without adding noise. We then added zero-mean Gaussian noise at five levels— 1%, 2%, 3%, 4%, and 5% of each signal's standard deviation—applying it separately to the input features and to the target variable". p.8: "ε ∼ N(μ, σ²) with μ = 1% of the mean value of each feature of the data". As Tabelas 3 e 4 têm só linhas por SR, sem LLM, prompt ou baseline. p.8 afirma "LLM-integrated variants often degrade less than their corresponding non-LLM baselines", sem números de baseline. A Tabela 5 (S1) não informa o prompt (INC-08). p.6: ruído padrão de 1% e "SNR of approximately 40 dB", coerente com σ = 0.01·std (20·log10(100) = 40 dB).
- **Hipótese adotada:** Matriz de correspondência:

  | Tabela | Cenário | SR | LLM | Prompt | Ruído | Localização | Nível | Baseline local |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | 2 | média dos 3 | 3 | 3 | A–H | média zero, σ = 0.01·std(y) | alvo | 1% | prompt A da mesma célula LLM×SR (referência de direção) |
  | 3 | cada um | 3 | Mistral 7B | E | média zero, σ = nível·std(col) por coluna perturbada | entradas / alvo / ambos | 1–5% | mesmo SR sem LLM (`w3 = 0`), mesmas condições |
  | 4 | cada um | 3 | Mistral 7B | E | μ = 0.01·mean(col), σ = nível·std(col) | entradas / alvo / ambos | 1–5% | mesmo SR sem LLM, mesmas condições, e a célula correspondente da Tabela 3 |
  | 5 | cada um | 3 | 3 + baseline | A (INC-08) | média zero, σ = 0.01·std(y) | alvo | 1% | linhas "Baseline" publicadas |

  Nas Tabelas 3 e 4 o ruído parte do mesmo conjunto limpo, com sementes derivadas por (repetição, localização, nível). Em "ambos", entradas e alvo recebem ruído independente no mesmo nível. O ruído enviesado se aplica também ao alvo quando a localização inclui o alvo, e μ não escala com o nível. O ruído é aplicado ao conjunto inteiro antes do split.
- **Justificativa:** Mistral com o prompt E é a configuração que o artigo recomenda (p.11) e a de melhor desempenho. O baseline sem LLM é necessário para testar a afirmação de p.8. O prompt A na Tabela 5 segue a única evidência disponível (INC-08).
- **Chave de configuração:** `experiments.noise.llm`, `experiments.noise.prompt`, `noise.default_level`, `noise.levels`, `noise.locations`, `noise.biased.mean_fraction`, `experiments.benchmark.prompt`
- **Impacto na comparabilidade:** Alto para as Tabelas 3 e 4. Se os autores usaram outro LLM ou prompt, a comparação de nível absoluto é inconclusiva. A direção (aumento com o nível; enviesado ≥ média zero) continua comparável.
- **Status:** hipótese + limitação

### GAP-13 — Prompts e Tabela 5 dependentes do suplemento

- **Questão:** Qual o texto exato do prompt-base e das variantes A–H, e onde está a Tabela 5?
- **Evidência:** p.5: "To be exact, we used the following prompt:", seguido de nada na página renderizada. p.7: "The prompt variant (A–H) used for the three physical scenarios is provided as supplementary material". O suplemento (S1) contém só o Apêndice com o texto da Tabela 6, a Tabela 5 e a Tabela 6. Não há prompts.
- **Hipótese adotada:** O prompt-base e as variantes A–H são reconstruídos a partir das descrições do artigo (`references/prompts_source.md`): seções `### ROLE` e `### METRICS`, três critérios com escala numérica, três exemplos (equação cinemática clássica, incompatibilidade dimensional, trigonometria aninhada desnecessária) e saída em lista Python `[dim corr, simp, sim, feedback]` com instrução "ONLY". Os arquivos levam `reconstructed: true`. A Tabela 5 foi transcrita do suplemento (`references/paper_results/table5_benchmark_supplement.csv`).
- **Justificativa:** Não existe fonte primária para o texto. A reconstrução segue cada elemento descrito, sem inventar critérios novos.
- **Chave de configuração:** `prompts.base_path`, `prompts.variants_path`, `prompts.reconstructed`, `prompts.score_scale`
- **Impacto na comparabilidade:** Alto para a Tabela 2. O texto do prompt muda os escores. A Tabela 5 está resolvida.
- **Status:** reconstruído + limitação

### GAP-14 — Custo da avaliação LLM por geração

- **Questão:** Quantas avaliações LLM cada treino exige, e isso cabe em A100 (Colab) e em GTX 1060 (local)?
- **Evidência:** p.8: "the LLM-to-SR runtime ratio typically remains below 0.4" (contradito pela Tabela 6, INC-09). Tabela 6 (S1): tempo médio por chamada de LLM de 1.8 a 2.4 s (Mistral) e de 3.0 a 5.9 s (LLaMA). Os totais implicam cerca de 3 iterações e cerca de 24 chamadas de LLM por treino: Total/Avg por iteração ≈ 3; Total/Avg por chamada ≈ 24; ou seja, cerca de 8 chamadas por iteração, longe de 100 por geração. p.11: Colab Premium, GPU não identificada.
- **Hipótese adotada:** Avaliamos todo candidato cuja perda é calculada (`llm.evaluation_scope = all_candidates`), com cache por (expressão canônica, modelo, revisão, prompt, contexto). O escore não depende dos dados, então o cache vale entre repetições, níveis de ruído e localizações. Avaliar só os top-k por geração (`llm.evaluation_scope = top_k`), o que a Tabela 6 sugere, é desvio do texto (a perda de todo candidato inclui `c`). Fica disponível só no extended, salvo decisão em contrário. Estimativas a medir no piloto (T:piloto):

  | Item | Valor estimado |
  | --- | --- |
  | Avaliações de perda por treino (DEAP/gplearn) | ≤ 100 × (50 + 1) = 5 100 |
  | Chamadas LLM por treino (fração única u = 0.4, sem cache global) | ≈ 2 000 |
  | Avaliações de perda por treino (PySR, `populations = 1`, ciclos padrão) | ordem de 10⁵–10⁶; chamadas únicas ordem de 10⁴ |
  | Latência por chamada, A100 fp16 transformers (≈ 700 tokens de entrada, ≈ 30 de saída) | ≈ 1.0 s sem lote; ≈ 0.15 s efetivos em lote de 32 |
  | Latência por chamada, GTX 1060 6 GB, Ollama Q4 | ≈ 3–6 s (prefixo do prompt em cache); usamos 4 s |
  | Treino DEAP/gplearn com LLM, A100 | ≈ 0.56 h sem lote; ≈ 5 min com lote |
  | Treino DEAP/gplearn com LLM, GTX 1060 | ≈ 2.2 h |
  | Matriz completa (treinos com LLM): Exp2 675 + Exp3 5 400 + Exp4 3 375 + Exp5 3 375 | 12 825 treinos ≈ 2.6·10⁷ chamadas sem cache global |
  | Matriz completa, A100 com lote (sem PySR inflado) | ≈ 1 070 h GPU (≈ 45 dias); a redução vem do cache global (a medir) |
  | Piloto local (`pilot.yaml`: pop 50, 10 gerações, 3 SR × 3 reps, Mistral, prompt E) | ≈ 275 chamadas/treino × 4 s ≈ 18 min/treino; ≈ 2.8 h, mais o PySR |
  | Piloto Colab (`colab_pilot.yaml`: pop 100, 50 gerações) | ≈ 2 000 chamadas/treino; ≈ 5 h sem lote, ≈ 45 min com lote, mais o PySR |

  Conta da linha com lote: 2.6·10⁷ × 0.15 s ≈ 3.9·10⁶ s ≈ 1 070 h. Sem lote, cerca de 7 100 h.
- **Justificativa:** Avaliar todo candidato é a leitura literal da Eq. (1). O custo só fica viável com lote por geração, cache global e parada antecipada. As latências são estimativas de ordem de grandeza e precisam ser medidas antes de congelar a matriz.
- **Chave de configuração:** `llm.evaluation_scope`, `llm.top_k`, `llm.batch_size`, `llm.cache.enabled`, `llm.cache.scope`, `sr.pysr.ncycles_per_iteration`, `runtime.budget_hours`
- **Impacto na comparabilidade:** Alto. A execução completa provavelmente exige reduzir repetições ou configurações (Q1 do grafo), o que precisa ser registrado. A Tabela 6 sugere que os autores fizeram muito menos chamadas por treino que a leitura literal. Se adotarmos o top-k, isso vira desvio declarado.
- **Status:** hipótese + limitação

---

## Outras hipóteses operacionais

| ID | Tema | Evidência | Hipótese | Chave |
| --- | --- | --- | --- | --- |
| H-01 | Parada antecipada | p.6: "Optimization stopped if the best composite loss improved by less than 0.1% over three generations" | Para quando a melhoria relativa da melhor perda composta, `(L_{g-1} − L_g) / abs(L_{g-1})`, fica abaixo de 0.001 em 3 gerações consecutivas. Calculada sobre a perda de treino. | `early_stopping.min_rel_improvement`, `early_stopping.patience` |
| H-02 | Hiperparâmetros sem valor | Tabela 1: "—" para PySR e gplearn | gplearn: torneio 3, profundidade inicial (1, 8), demais valores padrão da versão fixada. PySR: `populations = 1`, `population_size = 100`, `niterations = 50`, demais padrões. DEAP: profundidade inicial 1 (Tabela 1), `genHalfAndHalf(min=1, max=2)`. | `sr.gplearn.*`, `sr.pysr.*`, `sr.deap.*` |
| H-03 | Constantes nas árvores | não informado | DEAP: constante efêmera U(−1, 1); gplearn: `const_range = (−1, 1)`; PySR: otimização de constantes nativa. | `sr.deap.ephemeral_range`, `sr.gplearn.const_range` |
| H-04 | Resposta LLM inválida | p.11: "occasional invalid or syntactically incorrect suggestions" | Até 2 novas tentativas com o mesmo prompt. Persistindo a falha, o erro é registrado e o candidato recebe `c = 1` (pior escore), marcado `llm_invalid = true` e contado na taxa de inválidas. A execução primária é interrompida se a taxa passar de 5%. Nunca há escore substituto silencioso. | `llm.max_retries`, `llm.invalid_policy`, `llm.max_invalid_rate` |
| H-05 | Repetições | p.5: "n = 25 repetitions" | 25 repetições pareadas (GAP-09). Em smoke e pilot, menos repetições, rotuladas como tal. | `experiment.repetitions` |

---

## Critérios de comparação pré-registrados

Registrados antes de qualquer execução final (PRD §13.1). Não há tolerância arbitrária: a
proximidade é dada pela precisão publicada e pela incerteza amostral das 25 repetições.

**Entradas por célula** (`references/paper_results/*.csv`):

- `p`: valor publicado. `k`: casas decimais publicadas, ou seja, a **precisão publicada**: 3 para MAE e MSE; 2 para 1−R² e distância (o briefing fala em 3 casas, mas as tabelas usam 2 nessas métricas). Meia unidade de arredondamento: `r = 0.5·10^(−k)`. Intervalo de arredondamento: `P = [p − r, p + r]`, truncado em 0.
- `m`, `sd`, `n`: média, desvio-padrão e número de repetições válidas locais. **IC95** da média: `m ± t(0.975, n−1)·sd/√n` (t de Student; n = 25 → t = 2.064).
- Célula de referência (para a direção do efeito), definida por tabela:
  - Tabela 2: o prompt A do mesmo LLM×SR (o artigo afirma que mais informação melhora o resultado).
  - Tabela 5: a linha **baseline** do mesmo cenário×SR.
  - Tabela 3: o nível 1% da mesma localização (o artigo afirma aumento com o nível) e o baseline local sem LLM nas mesmas condições (o artigo afirma degradação menor com LLM).
  - Tabela 4: a célula correspondente da Tabela 3 (o enviesado deve ser pior ou igual) e o nível 1%.
- Efeito publicado: `Δp = p − p_ref`. Efeito local: `Δm = m − m_ref`, com IC95 da diferença pareada por repetição. O efeito publicado é "nulo" se `|Δp| ≤ 2r`.

**Regras, aplicadas nesta ordem:**

1. **inconclusiva**: (a) a célula depende de um GAP com status `limitação` que afeta o nível do valor e não há evidência para fixar a condição (por padrão, nível absoluto das Tabelas 3 e 4 e linhas PySR com LLM se o GAP-08 virar limitação); ou (b) `n < 25` repetições válidas no modo full; ou (c) informação metodológica ausente impede definir a célula de referência. A classificação de direção ainda é reportada como informação, sem classe.
2. **numericamente próxima**: o IC95 local intersecta `P` **e** a direção do efeito concorda (regra 6) ou a célula não tem referência.
3. **qualitativa**: o IC95 não intersecta `P`, mas a direção concorda **com significância**: o IC95 de `Δm` exclui 0 com o mesmo sinal de `Δp`, ou, quando `Δp` é nulo, o IC95 de `Δm` contém 0.
4. **parcial**: (a) o IC95 intersecta `P`, mas a direção não concorda; ou (b) o sinal de `Δm` concorda com o de `Δp`, mas o IC95 de `Δm` contém 0.
5. **divergente**: nenhuma das anteriores. Isso cobre efeito local com sinal oposto ao publicado, ou efeito local significativo quando o publicado é nulo.
6. Concordância de direção: `sign(Δm) = sign(Δp)` com IC95 de `Δm` excluindo 0, ou `Δp` nulo e IC95 de `Δm` contendo 0.

**Relatório por célula** (C:comparacao): `p`, `P`, `m`, `sd`, IC95, diferença absoluta `m − p`, diferença relativa `(m − p)/p` (não definida se `p = 0`, reportada como `NA`), baseline local e publicado, `Δp`, `Δm` com IC95 e a classe.

**Afirmações do artigo verificadas no nível da tabela** (mesmas regras de sinal e IC95, sobre médias de células):

- A1: Mistral < LLaMA < Falcon em MAE e distância (p.7, p.8).
- A2: PySR < DEAP < gplearn (p.7).
- A3: Tabela 2: E–H ≤ D ≤ {B, C} ≤ A (mais contexto melhora).
- A4: Tabela 5: LLM < baseline em todas as métricas.
- A5: Tabelas 3 e 4: distância monotonicamente não decrescente com o nível, e "ambos" ≥ "alvo" ≥ "entradas".
- A6: Tabela 4 ≥ Tabela 3, célula a célula.
- A7: com LLM, a degradação é menor que sem LLM (Tabelas 3 e 4, baseline local).

Resultados do modo `extended_analysis`, smoke ou pilot nunca recebem classe de reprodução.
Sem correção de múltiplas comparações na classificação. Contagens por classe e p-valores
ajustados por Holm são reportados à parte.

---

## Discordâncias e riscos levados ao orquestrador

1. **Identificabilidade na queda livre (GAP-03).** Com `height = g·t²/2` e `t` em `X`, a expressão `v = 9.81·t` é exata e menor que a GT. Esperamos que as buscas encontrem `g·t` com frequência, e que a distância para a GT fique alta (TD-18). Opções, a decidir pelo orquestrador: manter (fidelidade às 5 colunas, risco registrado), remover `t` de `X` só na queda livre (desvio) ou reportar também a distância à forma equivalente `g·t`.
2. **Sem `sqrt` em nenhum conjunto de operadores (GAP-06).** Distância 0 em DEAP e gplearn só é alcançável com a canonicalização `exp(c·log u) → pow(u, c)`. O T-simbolico precisa incluir `pow` na AST.
3. **Baseline e soma dos pesos (GAP-01).** Com `w3 = 0`, a configuração precisa usar `(0.5, 0.5, 0)` para não violar a validação "soma = 1".
4. **Custo (GAP-14).** A leitura literal (LLM em todo candidato) contradiz a ordem de grandeza da Tabela 6 (cerca de 24 chamadas por treino). PySR com LLM in-loop é de 1 a 2 ordens de grandeza mais caro que DEAP e gplearn.
