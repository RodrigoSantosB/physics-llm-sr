# Spike GAP-08: termo LLM dentro da seleção do PySR

Protótipo descartável. Não é o adaptador final (o contrato `sr/interfaces` ainda está em ajuste).

**Decisão recomendada: `integrated`**, com as restrições da seção 5. O callback Julia → Python
funciona, o termo `w3·c` entra na perda que o PySR usa para selecionar, e a prova do GAP-08 passa:
com `w3 = 0`, trocar o avaliador não muda nada; com `w3 > 0`, muda.

Ambiente: PySR 2.7.2, SymbolicRegression.jl (instalado pelo juliacall em `.venv/julia_env` e
`~/.julia`), PythonCall 0.9.36, Julia 1.13, 12 núcleos.

## Comandos

```bash
uv sync
uv run python spikes/pysr_inloop/inloop_spike.py influence   # prova de influência + determinismo (~40 s)
uv run python spikes/pysr_inloop/inloop_spike.py cost        # chamadas por iteração e cache (~30 s)
PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python spikes/pysr_inloop/inloop_spike.py threads
```

Os resultados vão para `out/{influence,cost,threads}.json`, com o hall of fame completo e uma
auditoria (`L_recomputed`) por equação. Na primeira execução, a instalação e a pré-compilação do
Julia levam alguns minutos.

## 1. Mecanismo (funciona)

Em Julia, o objetivo completo (`loss_function`) lê o callable Python de uma global de `Main`:

```julia
using PythonCall
const INLOOP_CB = Ref{Any}(nothing)        # recebe um callable Python (vira PythonCall.Py)
set_inloop_cb!(f) = (INLOOP_CB[] = f; nothing)

function inloop_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    pred, ok = eval_tree_array(tree, dataset.X, options); !ok && return L(Inf)
    e = mean(Huber_δ=1(pred - y))                       # GAP-05
    s = count_nodes(tree)
    txt = string_tree(tree, options)                    # "cos(x1) * 0.53", "x1 ^ 0.5", "neg(x2)"
    c = PythonCall.GIL.@lock pyconvert(Float64, INLOOP_CB[](txt))   # @lock é OBRIGATÓRIO
    return L(w1*e + w2*s + w3*c)
end
```

Do lado Python:

```python
import pysr                       # importar antes de juliacall
from juliacall import Main as jl  # o mesmo Main do PySR
jl.seval(PRELUDE); jl.seval("set_inloop_cb!")(callback)
PySRRegressor(loss_function=LOSS_FUNCTION, parsimony=0.0, parallelism="serial",
              deterministic=True, random_state=seed, batching=False, ...)
# callback(txt): parse_infix(txt.replace("^","**"), variables) -> simplify -> canonical_key
#                -> cache -> evaluator.evaluate -> c = 1 - mean(scores)
```

- **GIL:** o PySR chama `equation_search` com `_jl_call_nogil`, então o GIL está **liberado**
  dentro da perda. Sem `PythonCall.GIL.@lock` (PyGILState_Ensure), a primeira chamada dá segfault em
  `PyTuple_New`. Com o lock, o modo serial roda estável (dezenas de execuções, cerca de 250 mil
  chamadas, nenhuma falha).
- **Parsing:** em todas as execuções, nenhum texto de `string_tree` falhou em `parse_infix` depois
  de trocar `^` por `**`. `neg(...)` e as variáveis `x1..xn` são aceitos.
- **Auditoria:** para cada equação do hall of fame, `L` foi recalculada em Python (Huber via
  `model.predict`, `s` = complexidade, `c` pelo callback). A diferença em relação a
  `equations_.loss` é 0.0 em todas as execuções seriais com o escore calculado sobre a forma
  simplificada.

## 2. Prova de influência (queda livre, n = 200)

Dados: `x1 = h ∈ U(1,100)`, `x2 = massa` (distratora), `x3 = N(0,1)` (distratora),
`y = sqrt(2·9.81·h)`. Semente 7, `populations = 2`, `population_size = 30`, `niterations = 8`,
`maxsize = 20`. Avaliador A (`FakeEquationEvaluator.from_function`): escores 0.9 (c = 0.1) se a
árvore contém `cos`, senão 0.1 (c = 0.9). Avaliador B: c = 0.9 para tudo.

| w = (w1, w2, w3) | avaliador | chamadas | HoF (complexidade: equação) | % do HoF com cos | melhor tem cos |
|---|---|---|---|---|---|
| (½, ½, 0) | A | 22851 | 1: `31.94`; 3: `x1*0.531`; 4: `log(x1)/0.120` | 0 % | não |
| (½, ½, 0) | B | 22851 | **idêntico ao de A** | 0 % | não |
| (⅓, ⅓, ⅓) | A | 27290 | 1: `31.94`; 3: `x1*0.531` | 0 % | não |
| (⅓, ⅓, ⅓) | B | 22826 | 1: `31.94`; 3: `x1*0.531`; 5: `x1*0.324+13.97` | 0 % | não |
| (⅓, ⅓, ⅓) | A (repetida) | 27290 | **idêntico à 1ª execução A** | 0 % | não |
| (⅓, ⅓, 5) | A | 18290 | 1: `x1`; 2: `cos(x3)`; 3: `x1/1.39`; 4: `x1*cos(x3)`; 5: `31.26-cos(log(x1))`; 11: `x1^0.574/0.336-(cos(log(x1))-2.44)` | **67 %** | **sim** |
| (⅓, ⅓, 5) | B | 23379 | 1: `31.94`; 3: `x1*0.531`; 7: `(x1+7.86)*0.324+11.44` | 0 % | não |

Leitura:
- Com **w3 = 0**, trocar o avaliador não altera nada: o HoF e a contagem de chamadas são
  idênticos. Com **w3 > 0**, a trajetória muda (outro número de chamadas, outro HoF). É o critério
  do GAP-08.
- Com os **pesos do piloto (⅓, ⅓, ⅓)**, o avaliador muda a busca, mas o `cos` não entra no HoF.
  Cada nó custa `w2 = 0.33`, e o ganho máximo de `c` é `w3·0.8 = 0.27`, menor que o custo do nó
  extra que um `cos` exige. Isso confirma o impacto previsto em GAP-01/GAP-02: com `s` em número
  de nós e sem normalização, `w3·c` quase não pesa diante de `w2·s`.
- Com w3 = 5, o operador favorecido domina o HoF (67 %), inclusive na melhor equação.
- **Determinismo:** a mesma configuração executada duas vezes dá o mesmo HoF e o mesmo número de
  chamadas, com o callback ativo.

### Armadilha encontrada: o escore precisa ser invariante à simplificação do PySR

| variante (w3 = 5, A) | HoF | auditoria máx \|loss − L_recalc\| |
|---|---|---|
| escore sobre a árvore crua, `should_simplify=True` | `24.47`, `x1*0.548` (sem cos) | **3.0** (perdas inconsistentes) |
| escore sobre a árvore crua, `should_simplify=False` | `x1 ^ cos(-13.19)` (cos de constante) | 0.0 |
| escore sobre `simplify(expr)` (padrão do spike) | ver tabela acima | 0.0 |

O avaliador é "enganado" por `cos(constante)`, porque o PySR depois dobra constantes
(`simplify_tree!`/`combine_operators`) e o HoF guarda uma perda que não corresponde à árvore
exibida. **Regra para o adaptador:** calcular `c` sobre a forma com constantes dobradas
(`symbolic.canonicalization.simplify`, ou desligar `should_simplify`) e auditar `loss` contra o
`L` recalculado no fim.

## 3. Custo

Medido com `w = (⅓, ⅓, ⅓)`, avaliador A e uma chamada de callback por avaliação de perda (o
contador do Julia bate com o do Python).

| configuração | chamadas | textos únicos | chaves `canonical_key` únicas (= chamadas ao LLM) | chaves com 3 alg. signif. | estruturas (constantes mascaradas) | acerto do cache |
|---|---|---|---|---|---|---|
| 2×30, 1 iteração | 4621 | 2800 | 2434 | 1947 | 489 | 47 % |
| 2×30, 2 iterações | 7497 | 4640 | 4001 | 3275 | 864 | 47 % |
| 2×30, 4 iterações | 14025 | 8340 | 7423 | 6047 | 1646 | 47 % |
| 2×30, 1 iteração, sem otimização de constantes | 2590 | 762 | 634 | 634 | 445 | 76 % |
| 2×30, 4 iterações, sem otimização de constantes | 7316 | 2880 | 2482 | 2477 | 1820 | 66 % |
| **1×100, 1 iteração (hipótese GAP-08)** | 4574 | 1594 | **1356** | 988 | 241 | 70 % |
| 1×100, `ncycles_per_iteration=100` | 2832 | 1464 | 1287 | 768 | 162 | 55 % |
| 1×100, `ncycles_per_iteration=50` | 2359 | 1500 | 1312 | 840 | 134 | 44 % |

- **Sobrecarga do callback** (parse, canonicalização, cache e avaliador falso): 60 a 90 µs por
  chamada. Quatro iterações 2×30 levam 0.17 s sem callback e 1.5 s com ele. O custo real é o
  LLM: cerca de 1.3 a 2.4 mil chaves novas por iteração.
- A maioria das chaves únicas vem da **otimização de constantes** (BFGS sobre a perda completa,
  cada passo gera constantes novas). Reduzir `ncycles_per_iteration` quase não muda o número de
  chaves.
- Estimativa com LLM real (Ollama Q4, GTX 1060, de 2 a 3 s por resposta): cerca de 1.3 mil chaves
  por iteração dá **de 45 a 70 min por iteração** só na primeira, e o crescimento é aproximadamente
  linear (cerca de 1.8 mil chaves novas por iteração na configuração 2×30). 50 iterações × 25
  repetições ficam inviáveis sem outro recorte.
- O cache por `canonical_key` funciona do lado Python e é obrigatório. Ele derruba de 30 % a 75 %
  das chamadas. Com 3 algarismos significativos, cai mais 20 a 35 %. Uma chave só de estrutura
  (constantes mascaradas) reduz 5 a 6×, mas aí o LLM não vê constantes, o que seria um desvio a
  declarar.
- `evaluate_many` (lote) não se aplica: a perda é chamada por árvore, de forma síncrona. O lote só
  seria possível com pré-avaliação (fora do escopo deste spike).

## 4. Paralelismo e determinismo

- `parallelism="serial"` + `deterministic=True` + `random_state`: **determinístico** com o
  callback (seção 2).
- `parallelism="multithreading"` (12 threads Julia, com `GIL.@lock`): sem
  `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, deu **segfault** (exit 139) em 1 de 2 execuções. Com a
  variável, 3 de 3 terminaram, mas os resultados são não determinísticos (o PySR exige o modo
  serial para `deterministic`), uma das três teve auditoria ≠ 0 (0.33), e o callback serializa no
  GIL. **Não usar.**
- `multiprocessing`/`procs`: não testado. Os workers são processos Julia separados, sem o
  interpretador Python do processo principal, então o callable não existe lá. Inviável por
  construção.

## 5. Recomendação para o adaptador final

1. `sr.pysr.loss_mode = integrated`: `loss_function` em Julia, `c` via callback com
   `PythonCall.GIL.@lock`, `parallelism="serial"`, `deterministic=True`, `random_state` da
   repetição, `batching=False`, `parsimony=0`, `procs` ausente.
2. Escore sobre a forma com constantes dobradas (`simplify`) → `canonical_key` → cache persistente
   (o `llm.cache` existente) → `evaluate`. Auditoria pós-busca: `|loss − L_recalc| < tol` em todas
   as linhas do HoF. Se falhar, a execução fica marcada como inválida.
3. `EvaluationFailedError` dentro do callback não pode virar nota substituta (PRD). A exceção deve
   propagar para abortar a busca, ou o callback deve devolver `L = Inf` para a árvore, que não é
   selecionada. Decidir no contrato. A propagação de exceções Python através do PythonCall (como
   `PyException`) não foi testada neste spike.
4. Decisão de custo pendente, a levar ao PRD/GAP-14: (a) limitar `optimize_probability` e
   `optimizer_iterations`, (b) arredondar constantes da chave para 3 algarismos, ou (c) chave
   estrutural, que é desvio. Sem uma delas, o PySR com LLM real no piloto local não cabe no tempo.
5. Normalização de `s` (GAP-02): com `s` = número de nós e pesos ⅓, o `w3·c` raramente muda o
   vencedor. Isso não invalida a integração, mas explica efeitos pequenos.

## 6. Alternativas avaliadas (caso o item 1 tivesse falhado)

- **(a) Laço externo com `warm_start`, 1 iteração por vez, e re-score:** **não conta como
  in-loop**. As populações retomadas pelo `warm_start` guardam a perda calculada no Julia. Um
  re-score em Python só reordena o HoF exibido e não altera torneio, substituição nem migração
  entre populações. Para influenciar a seleção seria preciso reescrever `cost` dos membros em
  `julia_state_`, ou seja, voltar para dentro do Julia com acoplamento maior que o callback.
  Desnecessário, já que (1) funciona.
- **(b) Tabela de escores pré-calculada em Julia** (`Dict{String,Float64}` consultado na perda,
  com o Python preenchendo entre iterações): elimina o GIL, mas árvores novas não teriam escore
  dentro da iteração. Na prática, `c` fica defasado em uma iteração. Serve de reserva se o callback
  ficar instável.
- **(c) `loss_function_expression`/`TemplateExpression`:** não muda o problema (o callback continua
  necessário). Não testado.
