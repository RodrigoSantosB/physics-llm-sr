"""Gera ``03_pipeline_completo.ipynb`` (``uv run python notebooks/build_03_pipeline_completo.py``).

O .ipynb versionado é a saída deste script; edite as células aqui e regenere. Os IDs das
células são determinísticos (hash do índice e do conteúdo) para que o diff seja estável.
O teste ``tests/e2e/test_notebook.py`` confere que o arquivo corresponde ao gerador.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_common import format_code  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parent / "03_pipeline_completo.ipynb"
NBFORMAT_MINOR = 5
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Acrescenta uma célula markdown."""
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    """Acrescenta uma célula de código (formatada com ruff, como o repositório formata os .ipynb)."""
    CELLS.append(("code", format_code(text.strip("\n"))))


def build() -> nbformat.NotebookNode:
    """Monta o notebook a partir de ``CELLS``."""
    nb = nbformat.v4.new_notebook()
    nb.nbformat_minor = NBFORMAT_MINOR
    nb.metadata = {"kernelspec": KERNELSPEC, "language_info": {"name": "python"}}
    for index, (kind, source) in enumerate(CELLS):
        cell = (nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell)(
            source
        )
        cell["id"] = hashlib.sha1(f"{index}:{source}".encode()).hexdigest()[:8]
        nb.cells.append(cell)
    nbformat.validate(nb)
    return nb


def main() -> None:
    """Grava o notebook."""
    nbformat.write(build(), NOTEBOOK)
    print(f"gravado {NOTEBOOK} ({len(CELLS)} células)")


md(r"""
# Reprodução de regressão simbólica informada por física com LLMs: pipeline completo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RodrigoSantosB/physics-llm-sr/blob/main/notebooks/03_pipeline_completo.ipynb)

**Notebook 03 de 4** do projeto `physics-llm-sr` (os outros: 01 baseline sem LLM, 02 um LLM em um caso, 04 comparação com o artigo). É independente dos demais: tem a própria célula de preparo, grava em `<PROJECT_ROOT>/results/nb03_...` (e `configs_derivadas/nb03`, cache `.cache/llm_nb03`) e não lê saídas de outros notebooks. Percorre as etapas 00 a 18 (PRD, seção 10). Cada etapa tem
quatro componentes: **1. Explicação teórica**, **2. Referência ao artigo**, **3. Chamada ao módulo** e
**4. Visualização e interpretação**.

Regras deste notebook:

- As células só importam e chamam `physics_llm_sr`. Não há implementação de algoritmo, simulação, perda ou métrica aqui.
- Não há estado oculto: cada etapa relê a configuração e recarrega do disco o que já foi calculado (o runner
  retoma células concluídas em vez de recalculá-las).
- Os rótulos de reprodução vêm dos **dados** (`report.json`), não do texto. **`smoke` e `pilot` são execuções reduzidas e não
  são reprodução do artigo.** Só `replication` com os três LLMs reais e 25 repetições pode ser apresentada como tentativa de reprodução.
- Projeto clean-room: o código dos autores não foi consultado. Prompts reconstruídos a partir do artigo (GAP-13).
""")

md(r"""
### Parâmetros

- `PRESET`: `"hoje"` (padrão: queda livre no Colab A100, Mistral, ~4 h, etapas P1 a P4), `"hoje_literal"`, `"3_modelos_queda_livre"`, `"completo"` (artigo inteiro, dias de GPU), `"completo_extended"` ou `"smoke"` (minutos, avaliador simulado). Os braços literal e extended nunca se misturam nas tabelas de reprodução.
- `MODELOS`, `CENARIOS`, `EXPERIMENTOS`, `ALGORITMOS`, `REPETICOES`, `PROMPTS`: sobrescrevem o preset quando não são `None`. A configuração derivada é validada pelo schema e gravada em `<PROJECT_ROOT>/configs_derivadas/nb03/<hash>.yaml`; com sobrescritas, o diretório de resultados ganha o sufixo do hash (retomada correta, sem misturar execuções). Llama 2 é restrito: exige o segredo `HF_TOKEN` do Colab.
- `LOAD_EXISTING_RUN` (opcional, vazio por padrão): caminho de um `run_dir` já gerado (por exemplo, o do piloto). Com ele, as etapas só **apresentam** os resultados existentes, sem executar nada.
- `REPO_URL`, `REPO_DIR`, `MOUNT_DRIVE`, `DRIVE_PROJECT_DIR`, `HF_HOME_DIR`: usados só no Colab. A **primeira célula de código** é autossuficiente: monta o Drive, clona (se `REPO_URL`), verifica `REPO_DIR` (erro claro se faltar), instala o pacote e só então a etapa 01 importa `physics_llm_sr`. `HF_HOME_DIR` (padrão `/content/hf_home`, disco local) guarda os pesos (cerca de 15 GB por modelo) e sobrescreve `HF_HOME`; Drive só com ≥ 20 GB livres por modelo (não recomendado: trunca os pesos e causa `header too large`). Se `transformers`/`huggingface_hub` já foi importado com outro `HF_HOME`, reinicie o ambiente.
""")

code(r"""
# Parâmetros do notebook ----------------------------------------------------------------
import os as _os  # só para a sobrescrita do PRESET por ambiente (make notebook-smoke / testes)

# O QUE RODAR (sem editar YAML). O preset define a base; as demais variáveis, quando diferentes
# de None, sobrescrevem o preset (a config derivada é validada e gravada em configs_derivadas/).
PRESET = _os.environ.get("PHYSICS_LLM_SR_NOTEBOOK_PRESET", "hoje")
# "hoje"                  queda livre, Mistral, P1-P4 (cerca de 4 h no A100)   <- padrão
# "hoje_literal"          controle literal (paper_replication) só com P1
# "3_modelos_queda_livre" hoje com Mistral + Llama 2 (exige HF_TOKEN) + Falcon
# "completo"              artigo inteiro: 3 cenários x 3 SR x 3 LLMs x 25 repetições (DIAS de GPU)
# "completo_extended"     idem, braço extended (perda normalizada)
# "smoke"                 validação mínima, sem LLM real (minutos)
MODELOS = None  # None = do preset; ex.: ["mistral"], ["mistral", "llama2"], ["mistral", "llama2", "falcon"]
CENARIOS = None  # None = do preset; ex.: ["free_fall"], ["free_fall", "harmonic_motion"]
EXPERIMENTOS = None  # None = do preset; ex.: ["exp1","exp2"], ["exp1","exp2","exp3","exp4","exp5"]
ALGORITMOS = None  # ex.: ["deap","gplearn","pysr"]
REPETICOES = None  # ex.: 3 (hoje) | 25 (artigo)
PROMPTS = None  # ex.: ["A","C","E","H"] | ["A","B","C","D","E","F","G","H"]

LOAD_EXISTING_RUN = ""  # ex.: "results/pilot/20260101T000000-abcd1234" (apenas apresenta)

# Somente Colab. O código vem do GitHub (disco local do Colab, rápido); o Drive só PERSISTE
# resultados, cache e pesos (PHYSICS_LLM_SR_PROJECT_ROOT = DRIVE_PROJECT_DIR).
REPO_URL = "https://github.com/RodrigoSantosB/physics-llm-sr.git"  # alternativa: .bundle no Drive
REPO_BRANCH = "main"
REPO_DIR = "/content/physics-llm-sr"  # alternativa: pasta com .git no Drive (REPO_URL = "")
MOUNT_DRIVE = True  # monta o Drive para persistir resultados/cache/HF_HOME (não para o código)
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/physics-llm-sr-runs"
# Pesos (~15 GB por modelo) no disco LOCAL do Colab. Drive só com >= 20 GB livres por modelo (não
# recomendado: cota de 15 GB e escrita via FUSE truncam o arquivo -> "header too large").
HF_HOME_DIR = "/content/hf_home"

# Preparo autossuficiente (não importa o pacote do projeto): Drive -> clone/pull -> verificação -> pip -> path.
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
_repo = Path(REPO_DIR or "/content/physics-llm-sr")
if IN_COLAB:
    if MOUNT_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
    if REPO_URL and not _repo.is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(_repo)], check=True
        )
    elif REPO_URL and (_repo / ".git").exists():
        subprocess.run(["git", "-C", str(_repo), "pull", "--ff-only"], check=True)
    if not (_repo / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"repositório não encontrado em {_repo} (falta pyproject.toml). Preencha REPO_URL "
            "(GitHub) para clonar, ou aponte REPO_DIR para uma pasta com o código (Drive, com .git)."
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{_repo}[hf]"], check=True)
    os.environ.setdefault("PHYSICS_LLM_SR_REPO_DIR", str(_repo))
    if HF_HOME_DIR:  # sobrescreve; se transformers/huggingface_hub já foi importado, reinicie o ambiente
        os.environ["HF_HOME"] = HF_HOME_DIR
# O .pth do pip -e não vale no kernel já aberto: garante o src no sys.path.
if (_repo / "src").is_dir() and str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")  # figuras dentro do notebook
""")

md(r"""
## Etapa 00 — Introdução ao artigo, hipótese e metodologia

**1. Explicação teórica.** A hipótese do artigo é que a regressão simbólica (SR) melhora quando um LLM avalia, durante a busca, a plausibilidade física de cada equação candidata. A função de perda combina o erro de ajuste `e`, o tamanho da árvore `s` e uma penalidade `c` derivada dos escores do LLM: `L = w1·e + w2·s + w3·c`, com `c = 1 − (c1 + c2 + c3)/3` (consistência dimensional, simplicidade e realismo físico). O LLM participa da seleção, não é pós-processamento. A metodologia compara três algoritmos de SR (DEAP, gplearn, PySR), três LLMs (Mistral 7B, Llama 2 7B, Falcon 7B), oito prompts (A a H) e ruído de medição, em três sistemas físicos.

**2. Referência ao artigo.** Seção *LLM-based knowledge integration*, Eq. (1) (p.4–5); Fig. 1 (p.2); seções *Experiment types* e *Benchmarking* (p.6). Rastreabilidade completa em `references/methodology_traceability.md`; lacunas e hipóteses em `references/scientific_assumptions.md` (GAP-01 a GAP-14, INC-01 a INC-15).

**3. Chamada ao módulo.** `physics_llm_sr.colab` (perfis e configs) e `physics_llm_sr.config` (configuração tipada). Os experimentos 1 a 5 são executados por `physics_llm_sr.experiments.runner`; o Exp. 6 (distância estrutural) está nas métricas de cada célula e o Exp. 7 (comparação com o artigo) no `report`.
""")

code(r"""
# Etapa 00 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr.colab import PRESETS, preset_profile

RUN_PROFILE, ARM = preset_profile(PRESET)  # perfil/braço legados, usados na comparação de braços
display(
    pd.DataFrame(
        [{"PRESET": k, "config base": p.config, "descrição": p.description} for k, p in PRESETS.items()]
    )
)
display(Markdown(f"Selecionado: **{PRESET}** (perfil `{RUN_PROFILE}`, braço `{ARM}`)."))
""")

code(r"""
# Etapa 00 — 4. Visualização e interpretação

display(
    pd.DataFrame(
        {
            "experimento": ["Exp. 1", "Exp. 2", "Exp. 3", "Exp. 4", "Exp. 5", "Exp. 6", "Exp. 7"],
            "conteúdo": [
                "baseline sem LLM (w3 = 0)",
                "SR + LLM",
                "sensibilidade aos prompts A–H",
                "ruído gaussiano de média zero",
                "ruído enviesado",
                "distância estrutural à equação verdadeira",
                "comparação com as Tabelas 2–5 do artigo",
            ],
            "etapas do notebook": ["07", "11", "12", "13", "14", "15", "16–17"],
        }
    )
)
""")

md(r"""
**4. Visualização e interpretação.** A tabela mapeia os experimentos do artigo para as etapas. A execução escolhida em `PRESET` define a escala: `smoke` verifica o encanamento do pipeline; só `replication` com LLMs reais tem escala de reprodução.
""")

md(r"""
## Etapa 01 — Ambiente e verificação das dependências

**1. Explicação teórica.** A reprodução depende de versões fixas de bibliotecas, de Julia (para o PySR) e de GPU para os LLMs em fp16. No Google Colab o pacote precisa ser instalado (`pip install -e <repo>[hf]`), o Drive pode ser montado para que resultados e cache sobrevivam à sessão (`PHYSICS_LLM_SR_PROJECT_ROOT`), e o Llama 2 é um modelo restrito: exige a licença aceita no Hugging Face e um segredo `HF_TOKEN` no Colab (o token nunca é impresso). A primeira importação do PySR instala o Julia e leva vários minutos. Localmente, a etapa só verifica versões.

**2. Referência ao artigo.** Seção *Large language models* (p.6): modelos 7B em precisão local; apêndice S1 e p.11: execução em Google Colab Premium (GPU não identificada, INC-14). Decisão GAP-07 (checkpoints e revisões fixadas por SHA) em `references/scientific_assumptions.md`.

**3. Chamada ao módulo.** `physics_llm_sr.colab.setup_environment` (plano e execução do preparo), `colab.environment_report` e `colab.login_huggingface`.
""")

code(r"""
# Etapa 01 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
import importlib.util
from physics_llm_sr import colab

plan = colab.setup_environment(
    repo_dir=(REPO_DIR or None) if IN_COLAB else None,
    repo_url=REPO_URL,
    repo_branch=REPO_BRANCH,
    mount_drive=MOUNT_DRIVE,
    drive_dir=DRIVE_PROJECT_DIR,
    hf_home=HF_HOME_DIR,
)
display(
    Markdown(
        f"Ambiente: **{'Google Colab' if plan.in_colab else 'local'}**; repositório: `{plan.repo_dir}`; preparo executado: {plan.executed}"
    )
)
display(
    pd.DataFrame(
        [
            {"ação": a.kind, "descrição": a.description, "nome": a.name, "valor": a.value}
            for a in plan.actions
        ]
    )
)
""")

code(r"""
from physics_llm_sr.config.paths import project_root

CONFIG_PATH = colab.build_run_config(
    PRESET,
    models=MODELOS,
    scenarios=CENARIOS,
    experiments=EXPERIMENTOS,
    algorithms=ALGORITMOS,
    repetitions=REPETICOES,
    prompts=PROMPTS,
    out_dir=project_root(),
    output_prefix="nb03_",  # resultados em results/nb03_...: não colide com os notebooks 01 e 02
    derived_dir_name="configs_derivadas/nb03",
    patch={"llm": {"cache_dir": ".cache/llm_nb03"}},  # cache de avaliações só deste notebook
)
cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
print("configuração derivada:", CONFIG_PATH)
print("diretório da execução:", RUN_DIR)
gated = colab.ensure_hf_access(cfg)  # Llama 2 exige HF_TOKEN; erro claro antes de rodar
if cfg.llm.backend == "transformers":
    print("login no Hugging Face:", "ok" if gated else "não necessário (Mistral/Falcon)")
else:
    print(f"backend '{cfg.llm.backend}': login no Hugging Face não é necessário")
""")

code(r"""
# Etapa 01 — 4. Visualização e interpretação

info = colab.environment_report()
display(pd.Series(info["libraries"], name="versão").to_frame())
display(pd.Series({k: v for k, v in info.items() if k != "libraries"}, name="valor").to_frame())
print("PySR instalado:", importlib.util.find_spec("pysr") is not None, "| Julia:", info["julia"])
""")

md(r"""
**4. Visualização e interpretação.** Confira a tabela de versões (bibliotecas ausentes aparecem como `None`), a GPU e a versão do Julia. Sem Julia, as células PySR são gravadas como erro pelo runner e não impedem as demais. No Colab, escolha a GPU A100 (Ambiente de execução, Alterar tipo) antes de rodar os perfis `colab_pilot`/`replication`.
""")

md(r"""
## Execução de hoje (perfil `colab_hoje`): latência, estimativa e etapas P1 a P4

Bloco operacional da execução de hoje (queda livre, Mistral 7B fp16, A100). Não é uma etapa do artigo: mede a latência real do LLM, projeta o tempo e roda as etapas por prioridade, cada uma persistida e retomável (o runner pula células concluídas): **P1** Exp. 1 (baseline) + Exp. 2 (prompt E); **P2** Exp. 3 (prompts A, C, E, H); **P3** Exp. 4 (ruído de média zero só no alvo, 1/3/5%); **P4** Exp. 5 (ruído enviesado só no alvo, 1/5%). O Exp. 6 (distância estrutural) já está nas métricas de cada célula. Com outro `PRESET` ou sobrescritas, estas células rodam só os experimentos que estão na configuração resolvida. O PySR é reduzido hoje (desvio declarado no YAML).
""")

code(r"""
# Plano resolvido, latência real do LLM e tempo projetado (ANTES de executar)
import pandas as pd
from IPython.display import Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
display(Markdown(f"**Plano resolvido** (`{CONFIG_PATH.name}`): modelos {[m.name for m in cfg.llm.models if m.name in (cfg.experiments.models or [m.name for m in cfg.llm.models])]}, cenários {cfg.experiments.scenarios}, algoritmos {cfg.experiments.algorithms}, {cfg.experiments.repetitions} repetições."))
display(colab.plan_table(cfg))
latency = colab.measure_llm_latency(cfg)
print(
    f"latência medida: {latency['batched_s_per_call']:.3f} s/chamada em lote (10 avaliações), "
    f"{latency['serial_s_per_call']:.3f} s/chamada individual (5 avaliações)"
)
if PRESET == "smoke":
    print("smoke: a estimativa usa a matriz do smoke; os tempos não são representativos")
estimate = colab.estimate_runtime(cfg, latency["batched_s_per_call"], latency["serial_s_per_call"])
display(estimate)
total_h = estimate["llm_hours"].sum()
display(
    Markdown(
        f"Tempo projetado só de LLM (sem o tempo de busca do SR): **{total_h:.2f} h** (~{total_h / 24:.1f} dias). "
        f"Hipóteses de chamadas por célula: {colab.DEFAULT_CALLS_PER_CELL}. Se passar do orçamento, "
        "interrompa após P1/P2: as etapas seguintes são opcionais e retomáveis."
    )
)
""")

code(r"""
# P1 - Exp. 1 (baseline) + Exp. 2 (llm_integration); só o que está em experiments.run
from physics_llm_sr import colab

RUN_DIR, summary = colab.run_notebook_stage(
    cfg, colab.stage_experiments(cfg, "P1"), RUN_DIR, execute=not LOAD_EXISTING_RUN
)
print(summary if summary else "nada executado (execução existente ou fora da config)")
""")

code(r"""
# P2 - Exp. 3 (prompt_sensitivity, prompts A, C, E, H)
RUN_DIR, summary = colab.run_notebook_stage(
    cfg, colab.stage_experiments(cfg, "P2"), RUN_DIR, execute=not LOAD_EXISTING_RUN
)
print(summary if summary else "nada executado (execução existente ou fora da config)")
""")

code(r"""
# P3 - Exp. 4 (ruído de média zero só no alvo)
RUN_DIR, summary = colab.run_notebook_stage(
    cfg, colab.stage_experiments(cfg, "P3"), RUN_DIR, execute=not LOAD_EXISTING_RUN
)
print(summary if summary else "nada executado (execução existente ou fora da config)")
""")

code(r"""
# P4 - Exp. 5 (ruído enviesado só no alvo)
RUN_DIR, summary = colab.run_notebook_stage(
    cfg, colab.stage_experiments(cfg, "P4"), RUN_DIR, execute=not LOAD_EXISTING_RUN
)
print(summary if summary else "nada executado (execução existente ou fora da config)")
""")

md(r"""
## Etapa 02 — Configuração e sementes experimentais

**1. Explicação teórica.** Toda execução é descrita por uma configuração tipada (Pydantic) e uma semente mestra. As sementes de dados, ruído, partição e busca são derivadas por nome (`derive_seed(mestra, nome)`), de modo que o mesmo conjunto limpo é reaproveitado entre condições de ruído e a execução é reproduzível. Os pesos de `L` e a escala dos termos são hipóteses (GAP-01, GAP-02): modo `literal` (artigo) ou `normalized` (desvio, só no braço extended).

**2. Referência ao artigo.** Eq. (1) (p.4–5): pesos em [0, 1] somando 1 (valores não informados, GAP-01); p.6: dados em unidades SI sem escalonamento (GAP-02); Tabela 1 (p.7): hiperparâmetros (população 100, 50 gerações).

**3. Chamada ao módulo.** `physics_llm_sr.config` (`load_config`, `ExperimentConfig`), `physics_llm_sr.reproducibility.seeds` e `reproducibility.manifests.hash_config`.
""")

code(r"""
# Etapa 02 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.reproducibility.manifests import hash_config
from physics_llm_sr.reproducibility.seeds import derive_seed, seed_everything

seed_everything(cfg.seeds.master)
names = [
    "data/free_fall",
    "data/harmonic_motion",
    "data/damped_wave",
    "noise/data/free_fall",
    "notebook/split",
]
seeds = pd.DataFrame(
    {"nome": names, "semente derivada": [derive_seed(cfg.seeds.master, n) for n in names]}
)
""")

code(r"""
# Etapa 02 — 4. Visualização e interpretação

display(pd.Series(colab.profile_summary(cfg), name="configuração ativa").to_frame())
print("hash da config:", hash_config(cfg))
display(seeds)
display(
    pd.DataFrame(
        {
            "parâmetro": ["w1", "w2", "w3", "loss.normalization", "baseline_weights"],
            "valor": [
                cfg.loss.w1,
                cfg.loss.w2,
                cfg.loss.w3,
                cfg.loss.normalization,
                str(cfg.loss.baseline_weights),
            ],
            "status": ["hipótese GAP-01"] * 3 + ["GAP-02"] + ["GAP-01 (w3 = 0)"],
        }
    )
)
""")

md(r"""
**4. Visualização e interpretação.** O hash identifica a configuração: a retomada de uma execução exige o mesmo hash. As sementes derivadas são fixas para um mesmo nome. Pesos iguais e a escala `literal` são hipóteses documentadas; em `ARM = "extended"` a escala vira `normalized`, um desvio rotulado.
""")

md(r"""
## Etapa 03 — Definição dos sistemas físicos

**1. Explicação teórica.** Três sistemas têm solução analítica (a equação verdadeira, GT): queda livre `v = √(2gh)`, movimento harmônico simples `x = A·cos(√(k/m)·t + φ)` e onda amortecida `E = E0·e^(−αt/2)·cos(kx − ωt)`. Cada simulação amostra cinco atributos (massa, comprimento, altura, amortecimento e tempo); colunas que não entram na GT são distratoras. O tempo é amostrado no intervalo físico (queda até o solo; um período nos oscilatórios, GAP-04).

**2. Referência ao artigo.** Seção *In silico physical experiments*, Eqs. (2) a (4) (p.5–6); INC-06 (forma da Eq. 3) e INC-07 (unidade de α); parâmetros k, φ, E0, k_onda, ω em GAP-03.

**3. Chamada ao módulo.** `physics_llm_sr.physics.registry.get_scenario` e `scenario.simulate` / `physical_context` / `equivalent_ground_truths`.
""")

code(r"""
# Etapa 03 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.physics.registry import SCENARIO_NAMES, get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng

rows, curves = [], {}
for name in SCENARIO_NAMES:
    scenario = get_scenario(name, cfg.physics)
    ctx = scenario.physical_context()
    rows.append(
        {
            "cenário": name,
            "alvo": scenario.target_name,
            "GT (infixo)": ctx.ground_truth,
            "formas equivalentes": scenario.equivalent_ground_truths(),
            "unidade do alvo": ctx.units[scenario.target_name],
        }
    )
    rng = derive_rng(cfg.seeds.master, f"notebook/physics/{name}")
    curves[name] = [
        scenario.simulate(rng, cfg.physics.sampling, cfg.data.n_time_points, cfg.data.max_steps)
        for _ in range(3)
    ]
""")

code(r"""
# Etapa 03 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

display(pd.DataFrame(rows))
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, name in zip(axes, SCENARIO_NAMES):
    for k, traj in enumerate(curves[name]):
        ax.plot(traj.t, traj.y_true, ".", ms=3, label=f"simulação {k + 1}")
    ax.set_title(name)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(get_scenario(name, cfg.physics).target_name)
    ax.legend()
plt.tight_layout()
plt.show()
""")

md(r"""
**4. Visualização e interpretação.** Cada curva é uma simulação com parâmetros próprios. Na queda livre a velocidade cresce até o fim do intervalo físico; nos cenários oscilatórios o intervalo cobre um período. A tabela mostra a GT que o artigo usa e as formas fisicamente equivalentes (usadas só na métrica secundária da etapa 15).
""")

md(r"""
## Etapa 04 — Geração e exploração dos dados

**1. Explicação teórica.** Para cada cenário, N simulações independentes geram cerca de 100 instantes aleatórios ordenados (teto de 1000 por simulação), ~50 000 amostras no total na escala do artigo. A partição treino/validação/teste (70/15/15) é feita por `simulation_id`, nunca por linha, para evitar vazamento entre simulações (GAP-09). O conjunto limpo guarda `y_true`; `y_observed` recebe o ruído (etapa 05).

**2. Referência ao artigo.** Seção *In silico physical experiments* (p.6): N = 500 simulações, cinco parâmetros, ~100 passos, ~50 000 amostras em unidades SI; partição não informada (GAP-09).

**3. Chamada ao módulo.** `physics_llm_sr.data.generator.generate_dataset`, `data.validation.validate_dataset`, `data.splitting.split_by_simulation`.
""")

code(r"""
# Etapa 04 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.data.validation import validate_dataset
from physics_llm_sr.physics.registry import SCENARIO_NAMES
from physics_llm_sr.reproducibility.seeds import derive_rng

datasets = {name: generate_dataset(cfg, name, noisy=False) for name in SCENARIO_NAMES}
summary = pd.DataFrame(
    [
        {
            "cenário": n,
            "amostras": ds.X.shape[0],
            "simulações": np.unique(ds.simulation_ids).size,
            "problemas de validação": validate_dataset(ds, cfg.physics),
            "hash": ds.metadata["data_hash"][:12],
        }
        for n, ds in datasets.items()
    ]
)
parts = split_by_simulation(
    datasets["free_fall"], cfg.data.split, derive_rng(cfg.seeds.master, "notebook/split")
)
overlap = pd.DataFrame(
    [
        {
            "par": f"{a} x {b}",
            "simulações em comum": np.intersect1d(
                parts[a].simulation_ids, parts[b].simulation_ids
            ).size,
        }
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ]
)
""")

code(r"""
# Etapa 04 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

display(summary)
display(pd.DataFrame({k: v.X.shape[0] for k, v in parts.items()}, index=["amostras"]))
display(overlap)
ff = datasets["free_fall"]
display(
    pd.DataFrame(ff.X, columns=ff.feature_names).assign(**{ff.target_name: ff.y_true}).describe().T
)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].scatter(ff.X[:, ff.feature_names.index("height")], ff.y_true, s=2)
axes[0].set(
    xlabel="height (m, distância percorrida)",
    ylabel="velocity (m/s)",
    title="free_fall: alvo x height",
)
axes[1].hist(ff.y_true, bins=40)
axes[1].set(xlabel="velocity (m/s)", title="free_fall: distribuição do alvo")
for name in SCENARIO_NAMES[1:]:
    d = datasets[name]
    axes[2].scatter(d.t, d.y_true, s=2, label=name)
axes[2].set(xlabel="t (s)", title="cenários oscilatórios: alvo x t")
axes[2].legend()
plt.tight_layout()
plt.show()
""")

md(r"""
**4. Visualização e interpretação.** A validação deve listar zero problemas e a coluna de sobreposição deve ser zero: nenhuma simulação aparece em duas partições. O gráfico da queda livre mostra a relação `v = √(2gh)` entre o alvo e a coluna `height`; as demais colunas (massa, comprimento, amortecimento) são distratoras para esse cenário.
""")

md(r"""
## Etapa 05 — Introdução controlada de ruído

**1. Explicação teórica.** O ruído padrão é gaussiano de média zero com desvio de 1% do desvio-padrão do sinal. Nos experimentos de robustez, o ruído varia de 1% a 5% e é aplicado nas entradas, no alvo ou em ambos, sempre sobre o **mesmo** conjunto limpo (sementes derivadas por condição). O ruído enviesado soma uma média de 1% da média do atributo (Exp. 5).

**2. Referência ao artigo.** Seção *In silico physical experiments* (p.6: ruído gaussiano de 1%); *Noise robustness experiment* (p.7) e Tabela 3 (p.10); ruído enviesado (p.8) e Tabela 4 (p.10); GAP-12 (condições não informadas).

**3. Chamada ao módulo.** `physics_llm_sr.data.noise` (`NoiseSpec`, `apply_noise`) e `data.generator.generate_dataset`.
""")

code(r"""
# Etapa 05 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.noise import NoiseSpec, apply_noise
from physics_llm_sr.reproducibility.seeds import derive_rng

clean = generate_dataset(cfg, "free_fall", noisy=False)
conditions = [
    ("zero_mean", 0.01, "target"),
    ("zero_mean", 0.05, "target"),
    ("zero_mean", 0.05, "inputs"),
    ("biased", 0.05, "target"),
    ("biased", 0.05, "both"),
]
noisy, rows = {}, []
for kind, level, location in conditions:
    spec = NoiseSpec(kind, level, location, bias_fraction=cfg.data.exp5.mean_fraction)
    ds = apply_noise(
        clean, spec, derive_rng(cfg.seeds.master, f"notebook/noise/{kind}/{location}/{level}")
    )
    noisy[(kind, level, location)] = ds
    resid = ds.y_observed - clean.y_true
    rows.append(
        {
            "tipo": kind,
            "nível": level,
            "local": location,
            "média do resíduo do alvo": resid.mean(),
            "dp do resíduo do alvo": resid.std(),
            "|ΔX| médio": np.abs(ds.X - clean.X).mean(),
        }
    )
""")

code(r"""
# Etapa 05 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

display(pd.DataFrame(rows))
display(
    Markdown(
        f"Níveis do Exp. 4: `{cfg.data.exp4.levels}`, locais `{cfg.data.exp4.locations}`; Exp. 5: viés = {cfg.data.exp5.mean_fraction} x média, níveis `{cfg.data.exp5.levels}`."
    )
)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
h = clean.X[:, clean.feature_names.index("height")]
for ax, key in zip(axes, [("zero_mean", 0.01, "target"), ("zero_mean", 0.05, "target")]):
    ax.scatter(h, noisy[key].y_observed, s=2, label="observado")
    ax.scatter(h, clean.y_true, s=2, label="limpo")
    ax.set(title=f"ruído {key[0]} nível {key[1]} em {key[2]}", xlabel="height (m)")
axes[0].set_ylabel("velocity (m/s)")
axes[0].legend()
plt.tight_layout()
plt.show()
""")

md(r"""
**4. Visualização e interpretação.** O resíduo do ruído de média zero tem média próxima de zero e desvio crescente com o nível; o ruído enviesado desloca a média. Ruído nas entradas perturba as colunas de `X` (coluna |ΔX| diferente de zero), enquanto no alvo só `y_observed` muda. O conjunto limpo é o mesmo em todas as condições.
""")

md(r"""
## Etapa 06 — Construção e validação das expressões

**1. Explicação teórica.** Todos os algoritmos e o avaliador LLM compartilham uma AST comum (constantes, variáveis e operadores protegidos: divisão por quase zero vale 1, `log|x|` com |x| < 0,001 vale 0, `exp` com argumento limitado). Expressões de texto entram apenas por um parser por lista de permissão, sem `eval`; entradas maliciosas são rejeitadas. A forma canônica ordena operandos comutativos e arredonda constantes (3 algarismos significativos), e define a chave de cache do LLM.

**2. Referência ao artigo.** Seção *Symbolic regression models* (p.6) e Tabela 1 (p.7): conjunto de operadores (GAP-06, INC-05); p.4: expressões como árvores.

**3. Chamada ao módulo.** `physics_llm_sr.symbolic.expression` (`parse_infix`, `evaluate`, `to_prefix`, `from_prefix`, `to_sympy`), `symbolic.canonicalization` e `evaluation.metrics.regression_metrics`.
""")

code(r"""
# Etapa 06 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.evaluation.metrics import regression_metrics
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.canonicalization import canonical_key, simplify
from physics_llm_sr.symbolic.expression import (
    ExpressionError,
    depth,
    evaluate,
    from_prefix,
    parse_infix,
    size,
    to_infix,
    to_prefix,
    to_sympy,
)

ds = generate_dataset(cfg, "free_fall", noisy=False)
features = ds.feature_names
gt_text = get_scenario("free_fall", cfg.physics).physical_context().ground_truth
texts = [gt_text, "9.81*t", "sqrt(height*9.81*2)", "exp(log(height)) + 0*mass", "height / (t - t)"]
rows = []
for text in texts:
    expr = parse_infix(text, features)
    pred = evaluate(expr, ds.X, features)
    m = regression_metrics(ds.y_true, pred)
    rows.append(
        {
            "entrada": text,
            "infixo": to_infix(expr),
            "nós": size(expr),
            "profundidade": depth(expr),
            "ida e volta (prefixo)": from_prefix(to_prefix(expr), features) == expr,
            "chave canônica": canonical_key(expr),
            "simplificada": to_infix(simplify(expr)),
            "MAE vs y_true": m["mae"],
            "R²": m["r2"],
        }
    )
try:
    parse_infix('__import__("os").system("ls")', features)
    rejected = "NÃO rejeitada"
except ExpressionError as exc:
    rejected = type(exc).__name__
""")

code(r"""
# Etapa 06 — 4. Visualização e interpretação

display(pd.DataFrame(rows))
print("entrada maliciosa:", rejected)
display(to_sympy(parse_infix(gt_text, features)))
""")

md(r"""
**4. Visualização e interpretação.** A GT avaliada nos dados limpos deve reproduzir `y_true` (MAE ~ 0, R² ~ 1): isso valida simulador, parser e avaliador. A forma com operandos trocados tem a mesma chave canônica; a divisão por zero é protegida (resultado finito) e a entrada com `__import__` é rejeitada pelo parser seguro.
""")

md(r"""
## Etapa 07 — Baselines DEAP, gplearn e PySR

**1. Explicação teórica.** O baseline é a SR sem LLM: a perda usa só erro e tamanho (`w3 = 0`, pesos renormalizados 0,5/0,5, GAP-01). DEAP e gplearn são algoritmos genéticos; o PySR é uma busca evolutiva em Julia (erro Huber, GAP-05). Todos retornam a mesma estrutura de resultado e passam pela mesma parada antecipada (melhoria relativa < 0,1% por 3 gerações).

**2. Referência ao artigo.** Seção *Symbolic regression models* (p.6) e Tabela 1 (p.7): população 100, 50 gerações, DEAP com cruzamento 0,05, mutação 0,01, torneio 3, profundidade 8; Experiment 1 (baseline) e S1 Tabela 5; GAP-05, GAP-06, H-02, H-03.

**3. Chamada ao módulo.** `physics_llm_sr.experiments.runner` via `colab.run_notebook_stage` (Exp. 1, `baseline`), adaptadores em `physics_llm_sr.sr.{deap,gplearn,pysr}_adapter`.
""")

code(r"""
# Etapa 07 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
RUN_DIR, summary = colab.run_notebook_stage(cfg, ["exp1"], RUN_DIR, execute=not LOAD_EXISTING_RUN)
print(
    summary
    if summary
    else "nada executado nesta etapa (execução existente ou experimento fora da config)"
)
results = colab.load_results(RUN_DIR)
""")

code(r"""
# Etapa 07 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

if results.empty:
    print("sem resultados em", RUN_DIR)
else:
    base = results[results["llm"] == "none"]
    cols = [
        "scenario",
        "sr",
        "repetition",
        "status",
        "expression",
        "mae",
        "mse",
        "one_minus_r2",
        "tree_distance",
        "n_generations",
        "stop_reason",
    ]
    display(base[[c for c in cols if c in base.columns]])
    display(base[base["status"] == "error"][["cell_key", "error"]])
    hist = pd.read_parquet(RUN_DIR / "history.parquet")
    hb = hist[hist["llm"] == "none"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (sr, scen), g in hb.groupby(["sr", "scenario"]):
        ax.plot(
            *zip(*g.groupby("generation")["best_loss"].mean().items()),
            marker="o",
            label=f"{sr} / {scen}",
        )
    ax.set(
        xlabel="geração",
        ylabel="melhor perda (e e s)",
        title="baseline: perda por geração",
        yscale="log",
    )
    ax.legend()
    plt.show()
""")

md(r"""
**4. Visualização e interpretação.** A tabela traz a melhor expressão de cada algoritmo e as métricas no conjunto de teste; a curva mostra a melhor perda por geração. Células com `status = error` (por exemplo, PySR sem Julia) são listadas e não interrompem o notebook. Resultados de `smoke`/`pilot` são reduzidos (população e gerações menores, poucas repetições) e não equivalem aos do artigo.
""")

md(r"""
## Etapa 08 — Configuração dos modelos de linguagem

**1. Explicação teórica.** O LLM é usado como avaliador, com temperatura 0 (`do_sample=False`) e saída em JSON validada por parser estrito. O backend é escolhido pela configuração: `fake` (avaliador simulado determinístico, smoke e testes), `ollama` (piloto local, quantização Q4, desvio) ou `transformers` (Colab, fp16, revisão fixada por SHA). Os prompts A a H variam o contexto fornecido; D, F, G e H revelam a GT. Os textos dos prompts são reconstruídos (GAP-13) e marcados como tal.

**2. Referência ao artigo.** Seção *Large language models* (p.6); *LLM prompt knowledge integration* (p.6–7) e Tabela 2 (p.9–10); `references/prompts_source.md`; GAP-07, GAP-13, INC-01, INC-15.

**3. Chamada ao módulo.** `physics_llm_sr.llm.prompting.PromptLibrary`, `llm.inference.build_backend` e a lista `cfg.llm.models`.
""")

code(r"""
# Etapa 08 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.llm.inference import build_backend
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import parse_infix

models = pd.DataFrame([m.model_dump() for m in cfg.llm.models])
library = PromptLibrary.from_config(cfg.llm)
variants = pd.DataFrame(
    [
        {
            "prompt": v,
            "descrição": library.variant(v).description,
            "seções": library.variant(v).includes,
            "revela a GT": library.variant(v).gt_disclosed,
            "reconstruído": library.variant(v).reconstructed,
            "hash": library.prompt_hash(v)[:12],
        }
        for v in library.variants
    ]
)
ctx = get_scenario("free_fall", cfg.physics).physical_context()
rendered = library.render(
    parse_infix("9.81*t", ("mass", "length", "height", "damping", "t")),
    ctx,
    cfg.experiments.benchmark_prompt,
)
metadata = None
if cfg.llm.backend == "fake":
    metadata = build_backend(cfg.llm, cfg.llm.models[0].name, track=cfg.track).model_metadata()
""")

code(r"""
# Etapa 08 — 4. Visualização e interpretação

display(
    Markdown(
        f"Backend: **{cfg.llm.backend}**, temperatura {cfg.llm.temperature}, `max_new_tokens` = {cfg.llm.max_new_tokens}, cache em `{cfg.llm.cache_dir}`"
    )
)
display(models)
display(variants)
print(
    metadata
    if metadata
    else "pesos do modelo são carregados na etapa 09 (backend real); nada é baixado aqui"
)
print("--- SYSTEM ---\n" + rendered.system)
print("--- USER (prompt " + rendered.variant + ") ---\n" + rendered.user)
""")

md(r"""
**4. Visualização e interpretação.** Verifique que, no perfil Colab, cada modelo tem `revision` fixada (SHA) e quantização `fp16`; no perfil local, os checkpoints do Ollama são Q4 (desvio declarado). A coluna `reconstruído` mostra que os textos de prompt não vêm do artigo (GAP-13); a coluna `revela a GT` identifica D, F, G e H. O prompt renderizado é exatamente o que o LLM recebe.
""")

md(r"""
## Etapa 09 — Avaliação física de equações pelos LLMs

**1. Explicação teórica.** Para cada equação candidata, o LLM devolve três escores em [0, 1]: consistência dimensional (c1), simplicidade (c2) e realismo físico (c3), e um texto de feedback. Respostas inválidas são tentadas de novo (política H-04); não há nota substituta. O avaliador deduplica expressões pela forma canônica e consulta um cache persistente, o que reduz o custo (GAP-14). Com o backend `fake` os escores são determinísticos e **não** representam um julgamento de LLM.

**2. Referência ao artigo.** Seção *LLM-based knowledge integration* (p.4–5): `c = 1 − (c1 + c2 + c3)/3` e formato de saída `[dim corr, simp, sim, feedback]` (INC-15); GAP-13, GAP-14.

**3. Chamada ao módulo.** `physics_llm_sr.colab.build_evaluator` (monta `LLMEquationEvaluator` da config) e `evaluator.evaluate_many`; penalidade via `llm.scoring.penalty`.
""")

code(r"""
# Etapa 09 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.llm.scoring import penalty
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import parse_infix

evaluator = colab.build_evaluator(cfg)
ctx = get_scenario("free_fall", cfg.physics).physical_context()
features = ("mass", "length", "height", "damping", "t")
texts = [
    "sqrt(2*9.81*height)",
    "9.81*t",
    "height",
    "sqrt(2*9.81*height) + mass",
    "exp(height)",
    "sin(damping)*length",
]
exprs = [parse_infix(t, features) for t in texts]
rows = []
for variant in cfg.llm.prompt_variants:
    for text, ev in zip(texts, evaluator.evaluate_many(exprs, ctx, variant)):
        rows.append(
            {
                "prompt": variant,
                "expressão": text,
                "c1 dimensional": ev.dimensional_consistency,
                "c2 simplicidade": ev.simplicity,
                "c3 realismo": ev.physical_realism,
                "c (penalidade)": penalty(ev),
                "feedback": ev.feedback[:80],
            }
        )
table = pd.DataFrame(rows)
""")

code(r"""
# Etapa 09 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

display(table)
first = table[table["prompt"] == table["prompt"].iloc[0]].set_index("expressão")
first[["c1 dimensional", "c2 simplicidade", "c3 realismo"]].plot.bar(
    figsize=(9, 3.8), title=f"escores do LLM (prompt {table['prompt'].iloc[0]})"
)
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
display(pd.Series(evaluator.metrics.as_dict(), name="contadores do avaliador").to_frame())
""")

md(r"""
**4. Visualização e interpretação.** Os contadores mostram quantas chamadas chegaram ao backend e quantas vieram do cache ou da deduplicação. Com um LLM real, espera-se que a GT receba escores altos e expressões dimensionalmente erradas (`sqrt(...) + mass`) escores menores; com o backend `fake` esta leitura **não** se aplica, pois os escores são sintéticos. Pergunte-se sempre se a diferença entre os prompts vem do LLM ou só do contexto adicional.
""")

md(r"""
## Etapa 10 — Construção da função de perda combinada

**1. Explicação teórica.** `L = w1·e + w2·s + w3·c`: `e` é o erro (MSE; Huber no PySR), `s` o número de nós da AST e `c` a penalidade do LLM. Na escala `literal` do artigo `e` está em unidades SI ao quadrado, `s` é uma contagem e `c ∈ [0, 1]`: as escalas diferem muito e o termo dominante depende do cenário. Na escala `normalized` (desvio, GAP-02) `e = MSE/Var(y)` e `s = nós/max_nodes`. O baseline usa `w3 = 0`. Esta etapa mostra o peso de cada termo para as mesmas candidatas nas duas escalas.

**2. Referência ao artigo.** Eq. (1) (p.4–5) e definição de `e`, `s` e `c` (p.5); p.6: dados em unidades SI sem normalização; GAP-01 (pesos), GAP-02 (escalas), GAP-05 (Huber no PySR).

**3. Chamada ao módulo.** `physics_llm_sr.optimization.objective.build_objective` e `CombinedObjective.evaluate_population`.
""")

code(r"""
# Etapa 10 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.data.generator import generate_dataset
from physics_llm_sr.data.splitting import split_by_simulation
from physics_llm_sr.optimization.objective import build_objective
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.reproducibility.seeds import derive_rng
from physics_llm_sr.symbolic.expression import parse_infix

ds = generate_dataset(cfg, "free_fall")
train = split_by_simulation(ds, cfg.data.split, derive_rng(cfg.seeds.master, "notebook/split"))[
    "train"
]
ctx = get_scenario("free_fall", cfg.physics).physical_context()
evaluator = colab.build_evaluator(cfg)
texts = [
    "sqrt(2*9.81*height)",
    "9.81*t",
    "height",
    "sqrt(2*9.81*height) + mass",
    "exp(log(height)) / (1 + t)",
]
exprs = [parse_infix(t, ds.feature_names) for t in texts]
rows = []
for label, c, baseline in [
    ("literal", colab.notebook_config(RUN_PROFILE, "replication"), False),
    ("normalizada (desvio)", colab.notebook_config(RUN_PROFILE, "extended"), False),
    ("baseline (w3 = 0)", colab.notebook_config(RUN_PROFILE, "replication"), True),
]:
    obj = build_objective(
        c, evaluator, ctx, "deap", baseline, ds.feature_names, cfg.experiments.benchmark_prompt
    )
    w1, w2, w3 = obj.weights
    for text, b in zip(texts, obj.evaluate_population(exprs, train.X, train.y_observed)):
        rows.append(
            {
                "escala": label,
                "expressão": text,
                "L": b.total,
                "e": b.e,
                "s": b.s,
                "c": b.c,
                "w1·e": w1 * b.e,
                "w2·s": w2 * b.s,
                "w3·c": w3 * b.c,
            }
        )
table = pd.DataFrame(rows)
""")

code(r"""
# Etapa 10 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

display(table)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, (label, g) in zip(axes, table.groupby("escala", sort=False)):
    g.set_index("expressão")[["w1·e", "w2·s", "w3·c"]].plot.bar(stacked=True, ax=ax, title=label)
    ax.tick_params(axis="x", rotation=40)
plt.tight_layout()
plt.show()
""")

md(r"""
**4. Visualização e interpretação.** Compare as barras empilhadas: na escala `literal` o termo `w1·e` (ou `w2·s`, conforme a candidata) pode ser muito maior que `w3·c`, que está limitado a [0, w3]; nesse caso o LLM quase não altera a ordenação das candidatas. Esse é o achado de escala (GAP-02): reportá-lo é um resultado científico, não um defeito. Na escala normalizada os três termos têm magnitude comparável, mas esse é um desvio do artigo e fica no braço `extended`.
""")

md(r"""
## Etapa 11 — Regressão simbólica integrada ao LLM

**1. Explicação teórica.** No Exp. 2, o LLM avalia candidatas **dentro** do laço de busca: cada candidata cuja perda é calculada passa pelo avaliador (com cache), e a penalidade `c` entra em `L` e portanto na seleção. DEAP e gplearn avaliam a população a cada geração; o PySR chama o avaliador por um callback no laço Julia. Falha do LLM não gera nota substituta: a célula é registrada como erro. A comparação é pareada com o baseline (mesma semente, mesmos dados).

**2. Referência ao artigo.** Seção *Experiment types*, Benchmarking (p.6): 27 configurações (3 cenários × 3 SR × 3 LLMs) + baseline, 25 repetições; S1 Tabela 5 (INC-08, GAP-12); Fig. 1 (p.2); GAP-08, GAP-14.

**3. Chamada ao módulo.** `physics_llm_sr.experiments.runner` via `colab.run_notebook_stage` (Exp. 2, `llm_integration`) e `experiments.reporting` para as tabelas pareadas.
""")

code(r"""
# Etapa 11 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.experiments.reporting import paired_table

RUN_DIR, summary = colab.run_notebook_stage(cfg, ["exp2"], RUN_DIR, execute=not LOAD_EXISTING_RUN)
print(
    summary
    if summary
    else "nada executado nesta etapa (execução existente ou experimento fora da config)"
)
results = colab.load_results(RUN_DIR)
ok = results[results["status"] == "ok"] if not results.empty else results
""")

code(r"""
# Etapa 11 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

if results.empty:
    print("sem resultados em", RUN_DIR)
else:
    noise = (results["noise_type"] == "zero_mean") & (
        results["noise_level"] == cfg.data.noise.sigma_fraction
    )
    arm = results[
        noise
        & (results["status"] == "ok")
        & (results["prompt"].isin([cfg.experiments.benchmark_prompt, "none"]))
    ]
    cols = [
        "scenario",
        "sr",
        "llm",
        "llm_in_loop",
        "expression",
        "mae",
        "tree_distance",
        "llm_requested",
        "llm_unique",
        "llm_calls",
        "llm_cache_hits",
        "llm_seconds",
        "train_seconds",
    ]
    display(arm[[c for c in cols if c in arm.columns]])
    display(results[results["status"] == "error"][["cell_key", "error"]])
    hist = pd.read_parquet(RUN_DIR / "history.parquet")
    hist = hist[
        hist["prompt"].isin([cfg.experiments.benchmark_prompt, "none"])
        & (hist["noise_level"] == cfg.data.noise.sigma_fraction)
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (sr, llm), g in hist.groupby(["sr", "llm"]):
        ax.plot(
            *zip(*g.groupby("generation")["best_loss"].mean().items()),
            marker="o",
            ls="--" if llm == "none" else "-",
            label=f"{sr} / {llm}",
        )
    ax.set(
        xlabel="geração",
        ylabel="melhor perda",
        yscale="log",
        title="perda por geração: baseline (tracejado) x com LLM",
    )
    ax.legend()
    plt.show()
    paired = paired_table(ok)
    if not paired.empty:
        display(paired[["sr", "llm", "metric", "n", "mean_diff", "p_value", "method", "note"]])
""")

md(r"""
**4. Visualização e interpretação.** A coluna `llm_in_loop` comprova a participação do LLM na busca; `llm_calls` e `llm_cache_hits` mostram o custo e a economia do cache. Na tabela pareada, a diferença é baseline menos LLM (positiva = LLM melhor em métricas de erro); com poucas repetições o teste de Wilcoxon não tem poder, e a própria tabela traz a nota. Em `smoke` o LLM é simulado: o resultado valida o encanamento, não a hipótese do artigo.
""")

md(r"""
## Etapa 12 — Sensibilidade aos prompts A–H

**1. Explicação teórica.** As variantes de prompt acrescentam contexto progressivamente (nomes de variáveis, descrição do experimento, a GT). Nas variantes D, F, G e H a equação verdadeira é informada ao LLM, o que pode tornar a avaliação trivial; a análise separa prompts que revelam a GT dos que não revelam. O artigo reporta média sobre os três cenários, por algoritmo e LLM.

**2. Referência ao artigo.** Seção *LLM prompt knowledge integration* (p.6–7) e Tabela 2 (p.9–10); INC-01 (linhas de F, G e H idênticas às de E na tabela); INC-02 (leitura de "distância 1,00"); GAP-13.

**3. Chamada ao módulo.** `physics_llm_sr.experiments.runner` via `colab.run_notebook_stage` (Exp. 3, `prompt_sensitivity`); análise em `experiments.prompt_sensitivity`.
""")

code(r"""
# Etapa 12 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.experiments import prompt_sensitivity
from physics_llm_sr.experiments.reporting import replot

RUN_DIR, summary = colab.run_notebook_stage(cfg, ["exp3"], RUN_DIR, execute=not LOAD_EXISTING_RUN)
print(
    summary
    if summary
    else "Exp. 3 não faz parte deste perfil (ou execução existente): nada executado"
)
results = colab.load_results(RUN_DIR)
""")

code(r"""
# Etapa 12 — 4. Visualização e interpretação

ok = results[results["status"] == "ok"] if not results.empty else results
by_prompt = ok[(ok["llm"] != "none") & ok["prompt"].isin(list("ABCDEFGH"))] if not ok.empty else ok
many = (not by_prompt.empty) and by_prompt["prompt"].nunique() > 1
if not many:
    print("menos de dois prompts nos resultados: sem sensibilidade a mostrar neste perfil")
else:
    display(
        by_prompt.pivot_table(
            index=["prompt", "gt_disclosed"],
            columns="sr",
            values=["mae", "tree_distance"],
            aggfunc="mean",
        )
    )
    display(prompt_sensitivity.analyze(by_prompt).head(30))
    replot(RUN_DIR)
    for fig in sorted((RUN_DIR / "figures").glob("prompt_*.png")):
        display(Image(filename=str(fig)))
""")

md(r"""
**4. Visualização e interpretação.** Compare as linhas com `gt_disclosed = True` com as demais: se revelar a GT melhora muito a distância, o ganho vem da informação fornecida e não do conhecimento físico do LLM. Em `smoke` o avaliador é simulado e os prompts não têm efeito real; com LLMs reais, lembre-se de que o artigo apresenta F, G e H numericamente idênticos a E (INC-01), e que os textos dos prompts aqui são reconstruídos (GAP-13).
""")

md(r"""
## Etapa 13 — Robustez a ruído gaussiano

**1. Explicação teórica.** Ruído gaussiano de média zero entre 1% e 5% do desvio-padrão é aplicado nas entradas, no alvo ou em ambos, sobre o mesmo conjunto limpo. Para cada condição o runner treina o baseline pareado e a versão com LLM (o artigo não informa qual LLM e qual prompt: usamos o modelo e o prompt de `experiments.noise_model` e `noise_prompt`, hipótese GAP-12). A métrica de interesse é a distância estrutural à GT em função do nível de ruído.

**2. Referência ao artigo.** Seção *Noise robustness experiment* (p.7) e Tabela 3 (p.10); INC-03 (números do texto que não aparecem na Tabela 3); GAP-12.

**3. Chamada ao módulo.** `physics_llm_sr.experiments.runner` via `colab.run_notebook_stage` (Exp. 4, `noise_robustness`), `data.noise` e `experiments.noise_robustness`.
""")

code(r"""
# Etapa 13 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.experiments import noise_robustness

RUN_DIR, summary = colab.run_notebook_stage(cfg, ["exp4"], RUN_DIR, execute=not LOAD_EXISTING_RUN)
print(
    summary
    if summary
    else "Exp. 4 não faz parte deste perfil (ou execução existente): nada executado"
)
results = colab.load_results(RUN_DIR)
""")

code(r"""
# Etapa 13 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

ok = results[results["status"] == "ok"] if not results.empty else results
sub = ok[ok["noise_type"] == "zero_mean"] if not ok.empty else ok
if sub.empty or sub["noise_level"].nunique() < 2:
    print("sem curva de ruído para mostrar neste perfil")
else:
    sub = sub.assign(braço=np.where(sub["llm"] == "none", "baseline", "com LLM"))
    curves = (
        sub.groupby(["noise_location", "braço", "noise_level"])["tree_distance"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (loc, arm), g in curves.groupby(["noise_location", "braço"]):
        ax.plot(
            g["noise_level"],
            g["tree_distance"],
            marker="o",
            ls="--" if arm == "baseline" else "-",
            label=f"{loc} / {arm}",
        )
    ax.set(
        xlabel="nível de ruído (fração do desvio-padrão)",
        ylabel="distância estrutural média",
        title="Exp. 4: ruído de média zero",
    )
    ax.legend()
    plt.show()
    display(noise_robustness.analyze(sub).head(30))
""")

md(r"""
**4. Visualização e interpretação.** A inclinação positiva da curva indica que o ruído degrada a recuperação estrutural; compare a curva do baseline (tracejada) com a do LLM para ver se o conhecimento físico reduz a degradação. A comparação numérica com a Tabela 3 é feita na etapa 16, onde o nível absoluto é marcado como inconclusivo (LLM e prompt não informados) e só a direção é avaliada.
""")

md(r"""
## Etapa 14 — Robustez a ruído enviesado

**1. Explicação teórica.** No ruído enviesado a média da perturbação é 1% da média do atributo (um viés sistemático, como um erro de calibração), com desvio igual ao nível × desvio-padrão. Ao contrário do ruído de média zero, ele não se cancela na média e tende a degradar mais a recuperação da estrutura.

**2. Referência ao artigo.** Seção *Results* (p.8) e Tabela 4 (p.10); a descrição do texto sobre a Tabela 4 confere com as médias da tabela (INC-03); GAP-12.

**3. Chamada ao módulo.** `physics_llm_sr.experiments.runner` via `colab.run_notebook_stage` (Exp. 5, `biased_noise`), `data.noise` com `NoiseSpec(kind="biased")` e `experiments.biased_noise`.
""")

code(r"""
# Etapa 14 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.experiments import noise_robustness

RUN_DIR, summary = colab.run_notebook_stage(cfg, ["exp5"], RUN_DIR, execute=not LOAD_EXISTING_RUN)
print(
    summary
    if summary
    else "Exp. 5 não faz parte deste perfil (ou execução existente): nada executado"
)
results = colab.load_results(RUN_DIR)
""")

code(r"""
# Etapa 14 — 4. Visualização e interpretação
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")  # physics_llm_sr.evaluation.plots usa Agg

ok = results[results["status"] == "ok"] if not results.empty else results
sub = ok[ok["noise_type"] == "biased"] if not ok.empty else ok
if sub.empty or sub["noise_level"].nunique() < 2:
    print("sem curva de ruído para mostrar neste perfil")
else:
    sub = sub.assign(braço=np.where(sub["llm"] == "none", "baseline", "com LLM"))
    curves = (
        sub.groupby(["noise_location", "braço", "noise_level"])["tree_distance"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (loc, arm), g in curves.groupby(["noise_location", "braço"]):
        ax.plot(
            g["noise_level"],
            g["tree_distance"],
            marker="o",
            ls="--" if arm == "baseline" else "-",
            label=f"{loc} / {arm}",
        )
    ax.set(
        xlabel="nível de ruído (fração do desvio-padrão)",
        ylabel="distância estrutural média",
        title="Exp. 5: ruído enviesado",
    )
    ax.legend()
    plt.show()
    display(noise_robustness.analyze(sub).head(30))
""")

md(r"""
**4. Visualização e interpretação.** Compare esta curva com a da etapa 13 nas mesmas localizações: diferenças entre os dois tipos de ruído indicam sensibilidade a viés sistemático. Em execuções reduzidas (poucas repetições, poucos níveis) as curvas são apenas ilustrativas.
""")

md(r"""
## Etapa 15 — Métricas e distância estrutural

**1. Explicação teórica.** As métricas numéricas são MAE, MSE e `1 − R²`, calculadas no conjunto de teste. A distância estrutural entre árvores é recursiva e normalizada em [0, 1]: 0 significa estrutura equivalente (GAP-11), constantes diferentes custam `α = 1`, operadores comutativos são comparados em forma canônica e operadores diferentes de mesma aridade descem pela árvore (GAP-10). A métrica primária usa a GT do artigo; a secundária (`tree_distance_equivalent_min`) usa o mínimo sobre formas fisicamente equivalentes declaradas e **nunca** a substitui.

**2. Referência ao artigo.** Seção *Expression tree distance metric* (p.7) e *Experiments* (p.5); INC-02 ("distância 1,00" vs definição), INC-13 (nomes da métrica nas legendas), GAP-10, GAP-11.

**3. Chamada ao módulo.** `physics_llm_sr.symbolic.tree_distance` (`tree_distance`, `tree_distance_terms`), `evaluation.metrics` (`structural_metrics`, `exact_recovery`) e `experiments.structural`.
""")

code(r"""
# Etapa 15 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.evaluation.metrics import exact_recovery, structural_metrics
from physics_llm_sr.experiments import structural
from physics_llm_sr.physics.registry import get_scenario
from physics_llm_sr.symbolic.expression import parse_infix
from physics_llm_sr.symbolic.tree_distance import tree_distance_terms

scenario = get_scenario("free_fall", cfg.physics)
features = ("mass", "length", "height", "damping", "t")
gt = parse_infix(scenario.physical_context().ground_truth, features)
equivalents = [parse_infix(t, features) for t in scenario.equivalent_ground_truths()]
texts = [
    "sqrt(2*9.81*height)",
    "sqrt(height*2*9.81)",
    "9.81*t",
    "sqrt(2*9.8*height)",
    "height + length",
    "exp(height)",
]
rows = []
for text in texts:
    expr = parse_infix(text, features)
    s, n = tree_distance_terms(expr, gt)
    sm = structural_metrics(expr, gt, equivalents)
    rows.append(
        {
            "candidata": text,
            "S": s,
            "N": n,
            "distância primária": sm["tree_distance"],
            "distância equivalente (mín.)": sm["tree_distance_equivalent_min"],
            "recuperação exata (primária)": exact_recovery(sm["tree_distance"]),
        }
    )
RUN_DIR, _ = colab.run_notebook_stage(cfg, ["exp2"], RUN_DIR, execute=not LOAD_EXISTING_RUN)
results = colab.load_results(RUN_DIR)
""")

code(r"""
# Etapa 15 — 4. Visualização e interpretação

display(pd.DataFrame(rows))
if not results.empty:
    ok = results[results["status"] == "ok"]
    cols = [
        "scenario",
        "sr",
        "llm",
        "expression",
        "mae",
        "mse",
        "one_minus_r2",
        "tree_distance",
        "tree_distance_equivalent_min",
        "exact_recovery",
    ]
    display(ok[[c for c in cols if c in ok.columns]].head(20))
    display(structural.summarize(ok))
""")

md(r"""
**4. Visualização e interpretação.** Operandos trocados e constantes iguais dão distância 0 (mesma estrutura); `9.81*t` é fisicamente equivalente na queda livre, mas só a métrica secundária o reconhece (a primária o compara com a GT do artigo). Constantes diferentes aumentam a distância mesmo com a forma correta, e estruturas sem relação chegam perto de 1. As tabelas das execuções trazem as duas métricas lado a lado, sempre rotuladas.
""")

md(r"""
## Etapa 16 — Comparação com os resultados publicados

**1. Explicação teórica.** Cada célula comparável das Tabelas 2 a 5 do artigo é confrontada com os resultados locais segundo critérios **pré-registrados** (sem tolerância arbitrária): numericamente próxima, qualitativa, parcial, divergente ou inconclusiva. Uma execução só pode ser apresentada como reprodução se for `full` na trilha `paper_replication` com o número mínimo de repetições. **Smoke e pilot são execuções reduzidas e não são reprodução do artigo**: o rótulo abaixo vem do `report.json` gerado a partir dos dados.

**2. Referência ao artigo.** Tabelas 2 a 5 (p.9–10 e S1) transcritas em `references/paper_results/`; critérios em `references/scientific_assumptions.md` ("Critérios de comparação pré-registrados"); INC-01, INC-02, INC-03, INC-08, INC-09 e GAP-12.

**3. Chamada ao módulo.** `physics_llm_sr.experiments.reporting.report` (reconstrói tabelas e figuras dos arquivos, sem treinar) e `evaluation.paper_comparison` (`compare_with_paper`, `load_paper_results`).
""")

code(r"""
# Etapa 16 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.evaluation.paper_comparison import load_paper_results
from physics_llm_sr.experiments.reporting import report

info = report(RUN_DIR) if (RUN_DIR / "run.json").is_file() else None
paper = load_paper_results()
""")

code(r"""
# Etapa 16 — 4. Visualização e interpretação

if info is None:
    print("sem execução em", RUN_DIR, "- rode as etapas 07 e 11 antes")
else:
    display(Markdown(f"### Rótulo da execução (vindo dos dados)\n\n> **{info['label']}**"))
    display(
        pd.Series(
            {
                "mode": info["mode"],
                "track": info["track"],
                "execução reduzida": info["reduced_run"],
                "resultado completo": info["complete_result"],
                "células ok": info["n_ok"],
                "células com erro": info["n_error"],
            }
        ).to_frame("valor")
    )
    if "comparison_error" in info:
        print("comparação indisponível:", info["comparison_error"])
    comp_path = RUN_DIR / "comparison.csv"
    if comp_path.is_file():
        comp = pd.read_csv(comp_path)
        display(comp["classe"].value_counts().rename("células comparáveis por classe").to_frame())
        shown = [
            "table",
            "scenario",
            "sr",
            "llm",
            "prompt",
            "noise_location",
            "noise_level",
            "metric",
            "published",
            "n",
            "mean",
            "classe",
            "reason",
        ]
        display(comp[comp["n"] > 0][[c for c in shown if c in comp.columns]].head(40))
display(paper.groupby("table").size().rename("valores publicados transcritos").to_frame())
""")

md(r"""
**4. Visualização e interpretação.** Leia primeiro o rótulo: se a execução é reduzida, as classes servem para verificar o encanamento da comparação e **não** para afirmar que o artigo foi reproduzido. Células sem dados locais saem como `inconclusiva` (com o motivo na coluna `reason`); nas Tabelas 3 e 4 o nível absoluto é inconclusivo por padrão (LLM e prompt não informados, GAP-12) e só a direção é avaliada.
""")

md(r"""
## Etapa 17 — Análise estatística das 25 repetições

**1. Explicação teórica.** O artigo reporta médias de 25 repetições por configuração. Para cada condição calculamos média, desvio-padrão amostral e IC95% (t de Student) e comparamos baseline e LLM em pares (mesma repetição, mesmos dados) com o teste de Wilcoxon e tamanhos de efeito (`d_z` e correlação rank-biserial). Com poucas repetições o IC é indefinido e o teste não tem poder: o número `n` de cada condição é sempre exibido.

**2. Referência ao artigo.** Seção *Experiments* (p.5): n = 25 repetições, média; H-05 (mínimo de repetições válidas no modo full); critérios pré-registrados em `references/scientific_assumptions.md`.

**3. Chamada ao módulo.** `physics_llm_sr.experiments.runner` via `colab.run_notebook_stage` (retoma e consolida os experimentos da config), `evaluation.statistics` (`summarize_groups`) e `experiments.reporting.paired_table`.
""")

code(r"""
# Etapa 17 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.evaluation.paper_comparison import MIN_REPETITIONS_FULL
from physics_llm_sr.evaluation.statistics import summarize_groups
from physics_llm_sr.experiments.reporting import CONDITION, PRIMARY_METRICS, paired_table, replot

RUN_DIR, summary = colab.run_notebook_stage(
    cfg, cfg.experiments.run, RUN_DIR, execute=not LOAD_EXISTING_RUN
)
print(summary if summary else "nada executado nesta etapa (execução existente)")
results = colab.load_results(RUN_DIR)
ok = results[results["status"] == "ok"] if not results.empty else results
stats = (
    summarize_groups(ok, list(CONDITION), list(PRIMARY_METRICS)) if not ok.empty else pd.DataFrame()
)
paired = paired_table(ok) if not ok.empty else pd.DataFrame()
""")

code(r"""
# Etapa 17 — 4. Visualização e interpretação

display(
    Markdown(
        f"Repetições configuradas: **{cfg.experiments.repetitions}**; mínimo para o modo full: **{MIN_REPETITIONS_FULL}**."
    )
)
if stats.empty:
    print("sem resultados ok em", RUN_DIR)
else:
    display(stats[stats["metric"] == "tree_distance"].head(30))
    if not paired.empty:
        display(
            paired[paired["metric"] == "tree_distance"][
                [
                    "sr",
                    "llm",
                    "prompt",
                    "noise_type",
                    "noise_location",
                    "noise_level",
                    "n",
                    "n_nonzero",
                    "mean_diff",
                    "p_value",
                    "method",
                    "note",
                ]
            ].head(30)
        )
    replot(RUN_DIR)
    for fig in sorted((RUN_DIR / "figures").glob("*_bars.png"))[:2]:
        display(Image(filename=str(fig)))
""")

md(r"""
**4. Visualização e interpretação.** Onde `ci_defined = False` (n < 2) não há intervalo de confiança; onde `n` é menor que o mínimo, não se pode afirmar significância. O p-valor exato bilateral mínimo com poucos pares já supera 0,05, e a coluna `note` registra isso. Só uma execução com as 25 repetições sustenta conclusões estatísticas sobre a diferença entre baseline e LLM.
""")

md(r"""
## Etapa 18 — Discussão, limitações e conclusões

**1. Explicação teórica.** Esta etapa reúne o que os resultados dizem e o que eles não permitem concluir. Qualquer número citado aqui vem das células de código abaixo, que leem os arquivos da execução; o texto não contém resultados.

**Limitações e lacunas a considerar.** (1) *Escala literal domina o termo LLM (GAP-02):* com `e` em unidades SI ao quadrado e `s` contagem de nós, o termo `w3·c ≤ w3` pode ser pequeno frente aos demais, e o LLM praticamente não altera a busca; é um resultado a reportar, não um erro, e o modo `normalized` é um desvio rotulado. (2) *Pesos (GAP-01) e simuladores (GAP-03) são hipóteses.* (3) *Custo do PySR com o LLM no laço (GAP-14):* o PySR avalia a perda ordens de grandeza mais vezes que DEAP e gplearn, o que torna o PySR com LLM caro; o piloto usa configuração reduzida e o cache é essencial. (4) *Prompts reconstruídos (GAP-13):* o suplemento não traz os textos; a sensibilidade aos prompts aqui não é diretamente comparável à Tabela 2. (5) *Inconsistências do artigo:* INC-01 (F, G e H idênticos a E), INC-02, INC-03, INC-08 e INC-09 limitam comparações pontuais. (6) *LLMs locais:* o piloto Ollama usa quantização Q4 (desvio); o Llama 2 é restrito e o determinismo entre ambientes não é presumido (INC-11).

**2. Referência ao artigo.** Seção *Discussion* (p.8–11) e conclusões do artigo; afirmações A1 a A7 avaliadas pelos critérios pré-registrados (`references/scientific_assumptions.md`); registro completo de lacunas e inconsistências em `references/scientific_assumptions.md`.

**3. Chamada ao módulo.** `physics_llm_sr.colab.load_results` (lê `results.parquet`), `llm.prompting.PromptLibrary` e `experiments.reporting.report` e `loss_term_contributions`. Os números abaixo são calculados dos arquivos, nunca digitados.
""")

code(r"""
# Etapa 18 — 3. Chamada ao módulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from physics_llm_sr import colab

cfg, RUN_DIR = colab.resolve_stage(RUN_PROFILE, ARM, LOAD_EXISTING_RUN, CONFIG_PATH)
from physics_llm_sr.experiments.reporting import REPORT_FILE, loss_term_contributions, report
from physics_llm_sr.llm.prompting import PromptLibrary
from physics_llm_sr.reproducibility.artifacts import load_json

results = colab.load_results(RUN_DIR)
ok = results[results["status"] == "ok"] if not results.empty else results
if (RUN_DIR / REPORT_FILE).is_file():
    info = load_json(RUN_DIR / REPORT_FILE)  # gerado na etapa 16
else:
    info = report(RUN_DIR) if (RUN_DIR / "run.json").is_file() else None
library = PromptLibrary.from_config(cfg.llm)
""")

code(r"""
# Etapa 18 — 4. Visualização e interpretação

if info is not None:
    display(Markdown(f"> **{info['label']}**"))
if ok.empty:
    print("sem resultados ok para discutir em", RUN_DIR)
else:
    display(
        Markdown(
            "**GAP-02 - contribuição média de cada termo à perda final (escala `"
            + cfg.loss.normalization
            + "`)**"
        )
    )
    display(loss_term_contributions(ok, cfg.loss.w1, cfg.loss.w2, cfg.loss.w3))
    display(Markdown("**GAP-14 - custo do LLM por algoritmo**"))
    display(
        ok.groupby("sr")[
            [
                "llm_requested",
                "llm_unique",
                "llm_calls",
                "llm_cache_hits",
                "llm_seconds",
                "train_seconds",
            ]
        ].agg(["sum", "mean"])
    )
    base = (
        ok[ok["llm"] == "none"]
        .groupby(["scenario", "sr"])["tree_distance"]
        .mean()
        .rename("distância (baseline)")
    )
    llm = ok[ok["llm"] != "none"]
    withllm = llm.groupby(["scenario", "sr"])["tree_distance"].mean().rename("distância (com LLM)")
    display(pd.concat([base, withllm], axis=1))
display(
    Markdown(
        "**GAP-13 - prompts reconstruídos:** "
        + str(sum(library.variant(v).reconstructed for v in library.variants))
        + " de "
        + str(len(library.variants))
        + " variantes carregadas estão marcadas `reconstructed`."
    )
)
if info is not None and info["errors"]:
    display(pd.DataFrame(info["errors"]))
""")

md(r"""
**4. Visualização e interpretação.** Leia as tabelas acima à luz das limitações: (a) se a fração do termo LLM em `L` é pequena, a escala literal reduz a influência do LLM (GAP-02); (b) as colunas de custo indicam o peso do LLM e do cache no tempo de execução (GAP-14), e o PySR deve ser avaliado à parte; (c) a diferença entre baseline e LLM só é conclusiva com repetições suficientes (etapa 17); (d) o rótulo no topo da discussão diz se este resultado é reduzido. Com `LOAD_EXISTING_RUN` apontando para o piloto, esta etapa discute os resultados do piloto sem reexecutar. A conclusão científica sobre a reprodução do artigo exige a execução `replication` completa, com LLMs reais e 25 repetições.
""")


md(r"""
## Execução de hoje (final): análise e exportação no formato legado

Roda `python -m physics_llm_sr.analysis` (tabelas e figuras artigo × execução, com `INDEX.md`) e `python -m physics_llm_sr.analysis.legacy_export` (CSV, cache e tabelas no formato da reprodução anterior). Se existir a execução do controle literal (`colab_hoje_queda_livre_literal`, rodada em outra sessão com `PRESET = "hoje_literal"`), ela entra na análise como segundo braço, rotulada à parte.
""")

code(r"""
# Execução de hoje - análise e exportação legada (final)
import subprocess
import sys
from pathlib import Path

from IPython.display import Image, Markdown, display
from physics_llm_sr import colab
from physics_llm_sr.config.paths import project_root

ROOT = project_root()
runs = [RUN_DIR]
literal = ROOT / "results" / "nb03_colab_hoje_queda_livre_literal" / colab.NOTEBOOK_RUN_NAME
if PRESET == "hoje" and (literal / "run.json").is_file():
    runs.append(literal)
TAG = {"hoje": "colab_hoje", "smoke": "smoke"}.get(PRESET, RUN_DIR.parent.name)
ANALYSIS_DIR = ROOT / "results" / f"nb03_analise_{TAG}"
LEGACY_DIR = ROOT / "results" / f"nb03_legado_{TAG}"
run_args = [str(r) for r in runs]
subprocess.run(
    [sys.executable, "-m", "physics_llm_sr.analysis", "--runs", *run_args, "--out", str(ANALYSIS_DIR)],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "physics_llm_sr.analysis.legacy_export", "--runs", *run_args, "--out", str(LEGACY_DIR)],
    check=True,
)
display(Markdown((ANALYSIS_DIR / "INDEX.md").read_text(encoding="utf-8")))
for pattern in ("F3_bars_t5_*.png", "F2_noise_*.png", "F5_paired_*.png"):
    for fig in sorted((ANALYSIS_DIR / "figures").glob(pattern))[:1]:
        display(Image(filename=str(fig)))
print("análise:", ANALYSIS_DIR, "| legado:", LEGACY_DIR)
""")

md(r"""
## Roteiro Colab A100

1. **Runtime:** *Ambiente de execução, Alterar tipo*, GPU **A100** com **High-RAM**.
2. **Segredo:** em *Segredos* (ícone de chave) crie `HF_TOKEN` com "Acesso ao notebook" ligado; aceite antes a licença do Llama 2 no Hugging Face.
3. **Parâmetros:** na primeira célula (o padrão `PRESET = "hoje"` já serve; veja o README, "Escolhendo o que rodar"; `REPO_URL` já aponta para o GitHub). Os pesos dos modelos ocupam cerca de 40 GB: `HF_HOME_DIR` (padrão `<DRIVE_PROJECT_DIR>/hf_home`) precisa dessa folga no Drive.
4. **Executar** as etapas em ordem (Ambiente de execução, Executar tudo). Estimativas do `colab_pilot` (a medir, não garantidas): cerca de 45 min com lote, cerca de 5 h sem lote, mais o PySR (0,5 a 3 h por treino com LLM; `experiments.max_cell_seconds` limita cada célula).
5. **Onde ficam os resultados:** `<DRIVE_PROJECT_DIR>/results/nb03_<config>/notebook_run/` (`results.parquet`, `manifest.json`, `figures/`; no preset `hoje`: `nb03_colab_hoje_queda_livre`) e o cache do LLM em `<DRIVE_PROJECT_DIR>/.cache/llm_nb03`.
6. **Retomar após desconexão:** abra o notebook, repita os parâmetros e execute de novo a primeira célula e as etapas; o runner pula as células já concluídas e recarrega os resultados do Drive.
7. **Baixar os resultados:** pelo Drive (pasta acima) ou compactando `notebook_run/` numa célula extra (`shutil.make_archive`) e baixando com `google.colab.files.download`.
""")


if __name__ == "__main__":
    main()
