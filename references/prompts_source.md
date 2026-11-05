# Fontes para os prompts (prompt-base e variantes A–H)

Insumo da PR de prompts (`prompts/base/`, `prompts/variants/`, `src/physics_llm_sr/llm/prompting.py`).
Este arquivo reúne **tudo** o que o artigo e o suplemento dizem sobre os prompts, com citações
literais em inglês, tal como aparecem na camada de texto do PDF. Os textos dos prompts **não
foram publicados**: os arquivos versionados devem ser reconstruídos e marcados com
`reconstructed: true` (GAP-13 em `references/scientific_assumptions.md`).

## 1. Situação das fontes

| Fonte | O que contém sobre prompts |
| --- | --- |
| Artigo, p.5 | Princípios de desenho, seções `### ROLE` e `### METRICS`, três critérios, três exemplos (só descritos), formato de saída. **O texto do prompt está ausente**: depois de "To be exact, we used the following prompt:" a página renderizada não traz nada (sem caixa, figura ou listagem). |
| Artigo, p.6 | Temperatura zero. |
| Artigo, p.6–7 | Definições textuais das variantes A–H e a remissão "provided as supplementary material". |
| Artigo, Tabela 2 (p.9–10) | Rótulos curtos das variantes. |
| Artigo, p.11 | O prompt E é a recomendação; limitações (zero-shot, saídas inválidas, revisões de prompt). |
| Suplemento (MOESM1) | **Nada sobre prompts.** Contém só o Apêndice (texto da Tabela 6), a Tabela 5 (benchmark) e a Tabela 6 (tempos). A remissão de p.7 não se cumpre. |

## 2. Prompt-base: citações literais

p.4: "the SR component receives the dataset and a loss function L that expects a dataset, an
analytical function (represented as an expression tree) and returns a value in R+".

p.5: "c is the LLM-based score of how well the analytical equation fulfills a pre-defined list of
physical constraints."

p.5: "Importantly, in order to ensure the reliability of the LLM-based score, which is not
necessarily a floating point value but rather any string, we carefully design the LLM's prompt.
To be exact, we used the following prompt:" *(nada a seguir)*

p.5: "To ensure that the prompt would work consistently across different models, we adopted a
few key design principles. First, we set the temperature to zero to guarantee deterministic
outputs—ensuring that models like Mistral, Llama, and Falcon would always produce the same
response for the same input. We introduced a "### ROLE" section to immediately frame the
assistant's focus on scientific reasoning, helping the model interpret the task correctly from
the start. In the "### METRICS" section, we clearly defined three core evaluation criteria—
dimensional consistency, simplicity, and physical realism—using well-spaced numerical scales so
that all models could interpret them in the same way. We included a few carefully chosen examples
(a classic kinematics equation, a dimensional mismatch, and an unnecessarily nested trigonometric
form) to anchor the scoring logic. Finally, we enforced a strict Python-style list format for the
outputs and emphasized the "ONLY" instruction to minimize the chance of extra text generation.
With these measures in place, the [dim corr, simp, sim, feedback] lists returned by Mistral,
Llama, and Falcon were not only structurally consistent but also reliably comparable."

p.5: "Importantly, the prompt included three metrics: dim corr, simp, and sim, which stand for the
dimensional correction of the equation, its simplicity, and its physical realism, respectively."

p.5: "To be exact, upon receiving the prompt, each LLM returns a concise, three-element list,
including a score about physical dimensionality (c1) consistency, syntactic and structural
simplicity (c2), and physical realism (c3). The LLM's overall score is computed as follows:
c := 1 − (c1 + c2 + c3)/3."

p.5: "Notably, the physics prior enters the optimization through the auxiliary term c, which
evaluates whether a candidate symbolic expression respects pre-defined physical criteria. [...]
Specifically, the LLM is prompted to return (i) dimensional consistency (dimcorr)and(ii)
physicalrealism/plausibility(sim), whichoperationa." *(frase truncada no artigo, INC-10)*

p.6: "Importantly, we set the sampling temperature to zero to guarantee deterministic, repeatable
outputs and to ensure that differences in plausibility scores arise solely from model knowledge
rather than random sampling noise."

### Critérios c1, c2 e c3

| Escore | Chave na saída | Nome no artigo | Termo em `c` |
| --- | --- | --- | --- |
| c1 | `dim corr` | "dimensional consistency" / "dimensional correction of the equation" / "physical dimensionality (c1) consistency" | `c = 1 − (c1 + c2 + c3)/3` |
| c2 | `simp` | "simplicity" / "syntactic and structural simplicity" | idem |
| c3 | `sim` | "physical realism" / "physical realism/plausibility" | idem |

Formato de saída descrito: lista Python `[dim corr, simp, sim, feedback]`, instrução "ONLY".
Há uma contradição menor: o artigo fala em "three-element list", mas o formato tem quatro
elementos, com `feedback`. Adotamos quatro elementos, três escores e o feedback.

O que falta e precisa ser reconstruído (com `reconstructed: true`):

- A escala numérica ("well-spaced numerical scales"). Como `c` assume `c_i ∈ [0, 1]`
  (c = 1 com escores 0 e c = 0 com escores 1), a escala apresentada ao LLM precisa ser
  [0, 1] ou mapeada para [0, 1] de forma documentada (`prompts.score_scale`).
- O texto das seções `### ROLE` e `### METRICS`, a seção de formato e a de exemplos.
- Os três exemplos e seus escores: (1) "a classic kinematics equation", que deve ter escores
  altos e não pode coincidir com nenhuma GT dos três cenários, para não vazar a resposta nos
  prompts A, B, C e E; (2) "a dimensional mismatch", com c1 baixo; (3) "an unnecessarily nested
  trigonometric form", com c2 baixo.
- A sintaxe com que a equação candidata é apresentada (infixo da AST comum) e os nomes das
  variáveis (`mass`, `length`, `height`, `damping`, `t`).

## 3. Variantes A–H: citações literais

p.6–7: "Due to the fact that the exact prompt used to query the LLM response can produce highly
different responses, and therefore numerical values for the SR optimization process, we explored
the influence of different types of knowledge in prompts on the LLM's influence on the SR's
produced equation. To this end, we consider eight configurations."

| Variante | Rótulo na Tabela 2 | Definição literal (p.7) | Contém a GT |
| --- | --- | --- | --- |
| A | No context | "prompt A provided no additional information, acting as a baseline." | não |
| B | Variable descriptions | "Prompt B included data-column descriptions only, testing if this reminder of physical quantities improved dimensional reasoning." | não |
| C | Experiment description | "Prompt C added a concise, natural-language "Experiment description", exploring whether a broader physical context improved plausibility judgments." | não |
| D | Formula at end | "Prompt D appended the GT equation, investigating whether this reiteration of the ground truth results in the SR optimization process to converge to it." | **sim** |
| E | B+C | "Prompt E merged B and C to assess whether pairing raw data details with contextual framing yields improved performance." | não |
| F | B+D | "Prompt F merged B and D to test whether coupling data reminders with the true formula enhances dimensional correction, potentially at the risk of over-guiding the LLM." | **sim** |
| G | C+D | "Prompt G merged C and D to evaluate if contextual description combined with ground-truth disclosure facilitates reconciling intuition with explicit answer keys." | **sim** |
| H | B+C+D | "Finally, Prompt H fused B, C, and D, creating a "kitchen-sink" template to determine whether information overload hinders performance or if redundancy reinforces consistent scoring." | **sim** |

p.7: "In addition to the single-cue prompts, we explored combinations to investigate synergistic
or competing effects." / "The prompt variant (A–H) used for the three physical scenarios is
provided as supplementary material." *(o suplemento não contém os prompts)*

p.11 (prompt E): "First, adopt a fully informative prompt that includes clear variable
definitions with their physical dimensions and a detailed experiment description. This
configuration—corresponding to Prompt E in Table 2, which show promising results, as it both
fitted the data well and find the GT equation exactly for all LLM and SR combinations."

Leitura operacional para a reconstrução:

- B ("data-column descriptions"; p.11: "clear variable definitions with their physical
  dimensions"): uma linha por coluna de `X` e pelo alvo, com nome, significado físico e unidade
  SI. As colunas distratoras também são descritas, porque o texto fala em descrever as colunas
  de dados.
- C ("concise, natural-language 'Experiment description'"; p.11: "detailed experiment
  description"): parágrafo curto sobre o fenômeno, sem fórmula.
- D ("appended the GT equation"; rótulo "Formula at end"): a GT é **anexada no fim** do prompt.
- Combinações (E–H): concatenação na ordem B, C, D, com D sempre no fim.
- Controle obrigatório (PRD §6.4): D, F, G e H marcadas `gt_disclosed: true` e analisadas à parte.

## 4. Limitações relatadas que afetam o avaliador

p.11: "The models were used in a zero-shot setting without fine-tuning, which likely limited
their effectiveness. Although structured prompting was applied to encourage more reliable
outputs, the non-deterministic nature of LLMs led to occasional invalid or syntactically
incorrect suggestions, requiring prompt revisions and increasing training time."

p.11: "the quality of LLM score was not manually or systematically evaluated by human experts.
Without human verification, it is challenging to gauge whether performance gains were due to
genuinely helpful scores in terms of physical realism or the score captures more technical
properties, such as the equation's length."

Consequências: zero-shot, sem fine-tuning. "prompt revisions" indica que o prompt mudou durante
o estudo e que a versão final é desconhecida. A política para saídas inválidas está em H-04
(`references/scientific_assumptions.md`).
