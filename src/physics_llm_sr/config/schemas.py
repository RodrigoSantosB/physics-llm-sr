"""Esquema Pydantic v2 completo da configuração experimental.

Todos os valores de experimento vivem aqui (defaults) ou em ``configs/*.yaml``;
nenhum módulo científico deve ter constantes de experimento embutidas.
Valores que o artigo não informa são marcados como hipótese com o GAP do PRD.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from physics_llm_sr.core_types import Mode, Track

#: Tolerância numérica para aceitar ``w1+w2+w3 == 1`` e ``train+val+test == 1``.
WEIGHT_SUM_TOLERANCE = 1e-6

#: Backends que nunca produzem reprodução completa (Q4 é desvio; fake é simulado).
FULL_REPLICATION_FORBIDDEN_BACKENDS = ("fake", "ollama")

ScenarioName = Literal["free_fall", "harmonic_motion", "damped_wave"]
AlgorithmName = Literal["deap", "gplearn", "pysr"]
OperatorName = Literal["add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos", "pow", "sqrt"]
PromptVariant = Literal["A", "B", "C", "D", "E", "F", "G", "H"]
ExperimentName = Literal[
    "baseline", "llm_integration", "prompt_sensitivity", "noise_robustness", "biased_noise"
]
NoiseLocation = Literal["inputs", "target", "both"]

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


class _Strict(BaseModel):
    """Base: rejeita campos desconhecidos."""

    model_config = ConfigDict(extra="forbid")


class SeedsConfig(_Strict):
    """Semente mestre; as demais são derivadas por nome."""

    master: int = Field(default=42, ge=0)


class Range(_Strict):
    """Intervalo fechado ``[low, high]``."""

    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.low > self.high:
            raise ValueError("low deve ser <= high")
        return self


class SamplingRanges(_Strict):
    """Faixas amostradas por simulação (PRD 3.2). Todas as 5 colunas de X são amostradas."""

    mass: Range = Range(low=0.1, high=10.0)  # kg
    length: Range = Range(low=0.01, high=0.5)  # m
    height: Range = Range(low=1.0, high=100.0)  # m
    damping: Range = Range(low=0.0, high=1.0)  # 1/s


class FreeFallConfig(_Strict):
    """Queda livre (GAP-03): v = sqrt(2 g h), sem arrasto."""

    gravity: PositiveFloat = 9.81  # m/s^2


class HarmonicMotionConfig(_Strict):
    """MHS (GAP-03): x = A cos(sqrt(k/m) t + phi), solto do repouso."""

    spring_constant: PositiveFloat = 10.0  # N/m [hipótese GAP-03]
    phase: float = 0.0  # rad [hipótese GAP-03]


class DampedWaveConfig(_Strict):
    """Onda amortecida (GAP-03): E = E0 exp(-alpha t/2) cos(kx - omega t)."""

    amplitude: PositiveFloat = 1.0  # V/m [hipótese GAP-03]
    wave_number: PositiveFloat = 2 * math.pi  # rad/m [hipótese GAP-03]
    angular_frequency: PositiveFloat = 2 * math.pi  # rad/s [hipótese GAP-03]


class PhysicsConfig(_Strict):
    """Parâmetros físicos por cenário."""

    sampling: SamplingRanges = SamplingRanges()
    free_fall: FreeFallConfig = FreeFallConfig()
    harmonic_motion: HarmonicMotionConfig = HarmonicMotionConfig()
    damped_wave: DampedWaveConfig = DampedWaveConfig()


class SplitConfig(_Strict):
    """Proporções do split por simulation_id (GAP-09)."""

    train: Probability = 0.70
    validation: Probability = 0.15
    test: Probability = 0.15

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitConfig:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"train+validation+test = {total}; esperado 1")
        return self


class NoiseConfig(_Strict):
    """Ruído gaussiano de média zero: sigma = sigma_fraction * std(sinal)."""

    sigma_fraction: Annotated[float, Field(ge=0)] = 0.01
    location: NoiseLocation = "target"


class Exp4Config(_Strict):
    """Experimento 4: robustez a ruído de média zero."""

    levels: list[PositiveFloat] = [0.01, 0.02, 0.03, 0.04, 0.05]
    locations: list[NoiseLocation] = ["inputs", "target", "both"]


class Exp5Config(_Strict):
    """Experimento 5: ruído enviesado (média = mean_fraction * média do atributo)."""

    mean_fraction: float = 0.01
    levels: list[PositiveFloat] = [0.01, 0.02, 0.03, 0.04, 0.05]
    locations: list[NoiseLocation] = ["inputs", "target", "both"]


class DataConfig(_Strict):
    """Geração, split e ruído dos dados."""

    n_simulations: PositiveInt = 500
    n_time_points: PositiveInt = 100  # instantes aleatórios ordenados por simulação
    max_steps: PositiveInt = 1000  # cap por simulação
    split: SplitConfig = SplitConfig()
    noise: NoiseConfig = NoiseConfig()
    exp4: Exp4Config = Exp4Config()
    exp5: Exp5Config = Exp5Config()

    @model_validator(mode="after")
    def _points_within_cap(self) -> DataConfig:
        if self.n_time_points > self.max_steps:
            raise ValueError("n_time_points não pode exceder max_steps")
        return self


class TreeDistanceConfig(_Strict):
    """Opções da distância entre árvores (GAP-10)."""

    descend_on_operator_mismatch: bool = True  # operadores diferentes, mesma aridade: desce
    canonicalize: bool = True  # forma normal estrutural antes de comparar


class SymbolicConfig(_Strict):
    """Semântica protegida de operadores (GAP-06) e distância entre árvores (GAP-10)."""

    division_epsilon: PositiveFloat = 0.001  # GAP-06: |den| < eps -> resultado protegido
    division_fallback: float = 1.0  # valor devolvido na divisão protegida
    log_epsilon: PositiveFloat = 0.001  # GAP-06: |x| < eps -> log = 0; senão log|x|
    exp_clip: PositiveFloat = 100.0  # GAP-06: exp(clip(x, -c, c)) evita overflow
    power_clip: PositiveFloat = 1e6  # magnitude máxima após potência
    tree_distance_constant_penalty: Probability = (
        1.0  # GAP-10: alpha (custo por |dv| de constantes)
    )
    tree_distance: TreeDistanceConfig = TreeDistanceConfig()


class EarlyStoppingConfig(_Strict):
    """Parada antecipada: melhoria relativa < min_relative_improvement por patience gerações."""

    enabled: bool = True
    min_relative_improvement: PositiveFloat = 0.001  # 0,1%
    patience: PositiveInt = 3


class DeapConfig(_Strict):
    """Hiperparâmetros DEAP (PRD 4.1)."""

    crossover_prob: Probability = 0.05
    mutation_prob: Probability = 0.01
    tournament_size: PositiveInt = 3
    max_depth: PositiveInt = 8
    init_min_depth: PositiveInt = 1  # [hipótese]
    init_max_depth: PositiveInt = 3  # [hipótese]

    @model_validator(mode="after")
    def _depths(self) -> DeapConfig:
        if self.init_min_depth > self.init_max_depth:
            raise ValueError("init_min_depth deve ser <= init_max_depth")
        return self


class GplearnConfig(_Strict):
    """Hiperparâmetros gplearn (artigo só informa população e gerações).

    Defaults = gplearn 0.4.2, todos [hipótese] (decisão da onda 5; difere de H-02, que
    sugeria torneio 3 e profundidade (1, 8)).
    """

    parsimony_coefficient: Annotated[float, Field(ge=0)] = 0.001  # [hipótese]
    tournament_size: PositiveInt = 20  # [hipótese] default gplearn 0.4.2
    p_crossover: Probability = 0.9  # [hipótese] default gplearn 0.4.2
    p_subtree_mutation: Probability = 0.01  # [hipótese] default gplearn 0.4.2
    p_hoist_mutation: Probability = 0.01  # [hipótese] default gplearn 0.4.2
    p_point_mutation: Probability = 0.01  # [hipótese] default gplearn 0.4.2
    init_depth: tuple[PositiveInt, PositiveInt] = (2, 6)  # [hipótese] default gplearn 0.4.2
    const_range: tuple[float, float] = (-1.0, 1.0)  # [hipótese] H-03 = default gplearn 0.4.2

    @model_validator(mode="after")
    def _consistent(self) -> GplearnConfig:
        total = self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation
        total += self.p_point_mutation
        if total > 1.0 + WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"soma das probabilidades genéticas = {total}; deve ser <= 1")
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError("init_depth deve ser (min, max) com min <= max")
        if self.const_range[0] > self.const_range[1]:
            raise ValueError("const_range deve ser (low, high) com low <= high")
        return self


class PysrConfig(_Strict):
    """Hiperparâmetros PySR. Huber no termo e (GAP-05, conforme artigo)."""

    loss: Literal["huber", "mse"] = "huber"
    huber_delta: PositiveFloat = 1.0  # [hipótese GAP-05]
    #: H-02: uma população de ``sr.population`` indivíduos; ``niterations = None`` usa
    #: ``sr.generations`` (artigo: 50). Pilotos locais reduzem (validação, não reprodução).
    niterations: PositiveInt | None = None  # [hipótese H-02]
    populations: PositiveInt = 1  # [hipótese H-02]
    population_size: PositiveInt | None = None  # None = sr.population [hipótese H-02]
    ncycles_per_iteration: PositiveInt = 380  # [hipótese] default PySR 2.x
    maxsize: PositiveInt = 30  # [hipótese] default PySR 2.x
    parsimony: Annotated[float, Field(ge=0)] = 0.0  # [hipótese] default PySR 2.x
    timeout_in_seconds: PositiveFloat | None = None  # limite nativo do PySR (opcional)


def _default_operators() -> dict[AlgorithmName, list[OperatorName]]:
    """Conjuntos do GAP-06 (sem ``sqrt``); ``div`` é protegido; PySR acrescenta ``pow``."""
    base: list[OperatorName] = ["add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos"]
    return {
        "deap": list(base),
        "gplearn": list(base),
        "pysr": [*base[:4], "pow", *base[4:]],
    }


class SRConfig(_Strict):
    """Regressão simbólica: população, gerações e operadores por algoritmo."""

    population: PositiveInt = 100
    generations: PositiveInt = 50
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    deap: DeapConfig = DeapConfig()
    gplearn: GplearnConfig = GplearnConfig()
    pysr: PysrConfig = PysrConfig()
    operators: dict[AlgorithmName, list[OperatorName]] = Field(default_factory=_default_operators)


class LossConfig(_Strict):
    """L = w1*e + w2*s + w3*c (PRD 5). Pesos não informados no artigo (GAP-01)."""

    w1: Annotated[float, Field(ge=0)] = 1 / 3  # [hipótese GAP-01]
    w2: Annotated[float, Field(ge=0)] = 1 / 3  # [hipótese GAP-01]
    w3: Annotated[float, Field(ge=0)] = 1 / 3  # [hipótese GAP-01]
    normalization: Literal["literal", "normalized"] = "literal"  # GAP-02
    #: Baseline sem LLM (GAP-01): w3 = 0, w1:w2 preservada e renormalizada.
    baseline_weights: tuple[float, float, float] = (0.5, 0.5, 0.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> LossConfig:
        total = self.w1 + self.w2 + self.w3
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"pesos w1+w2+w3 somam {total}; esperado 1")
        return self

    @model_validator(mode="after")
    def _baseline_weights_valid(self) -> LossConfig:
        w = self.baseline_weights
        if any(x < 0 for x in w) or abs(sum(w) - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"baseline_weights {w} devem ser >= 0 e somar 1")
        if w[2] != 0.0:
            raise ValueError(f"baseline_weights {w}: w3 deve ser 0 (sem avaliador LLM)")
        return self


class LLMModelConfig(_Strict):
    """Identificação reprodutível de um modelo (GAP-07)."""

    name: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    revision: str | None = None  # SHA fixado no Colab
    quantization: str | None = None
    infra: str = "unspecified"


class LLMConfig(_Strict):
    """Backend e política de inferência (temperatura 0)."""

    backend: Literal["fake", "ollama", "transformers"] = "fake"
    models: list[LLMModelConfig] = Field(default_factory=list)
    temperature: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0  # artigo: temperatura 0
    max_new_tokens: PositiveInt = 256
    max_retries: Annotated[int, Field(ge=0)] = 3
    failure_policy: Literal["retry", "abort"] = "retry"
    prompt_variants: list[PromptVariant] = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cache_enabled: bool = True
    cache_dir: str = ".cache/llm"  # diretório do cache JSONL de avaliações
    prompts_dir: str | None = None  # None = prompts/ do repositório
    #: Algarismos significativos das constantes na chave do cache (GAP-14, precisão publicada).
    cache_significant_digits: Annotated[int, Field(ge=1)] = 3
    #: ``exact`` = estrutura + constantes arredondadas; ``structure`` (só extended) = desvio.
    cache_key_mode: Literal["exact", "structure"] = "exact"
    #: Endereço do servidor Ollama (backend ``ollama``).
    ollama_endpoint: str = "http://localhost:11434"
    batch_size: PositiveInt = 1  # lote do backend transformers
    request_timeout_s: PositiveFloat = 120.0  # timeout HTTP do backend ollama
    seed: Annotated[int, Field(ge=0)] = 42  # semente de inferência
    #: ``all`` = todos os candidatos únicos, com cache (replication); ``top_k`` só em extended.
    evaluation_scope: Literal["all", "top_k"] = "all"
    top_k: PositiveInt | None = None

    @model_validator(mode="after")
    def _scope_consistent(self) -> LLMConfig:
        if self.evaluation_scope == "top_k" and self.top_k is None:
            raise ValueError("llm.top_k é obrigatório quando evaluation_scope='top_k'")
        if self.evaluation_scope == "all" and self.top_k is not None:
            raise ValueError("llm.top_k só é válido com evaluation_scope='top_k'")
        return self


class ExperimentsConfig(_Strict):
    """Quais experimentos rodar e a matriz de cenários, algoritmos e modelos."""

    run: list[ExperimentName] = ["baseline"]
    repetitions: PositiveInt = 25
    scenarios: list[ScenarioName] = ["free_fall", "harmonic_motion", "damped_wave"]
    algorithms: list[AlgorithmName] = ["deap", "gplearn", "pysr"]
    models: list[str] = Field(default_factory=list)  # vazio = todos os de llm.models
    #: Prompt do Exp2 (Tabela 5): A, por INC-08/GAP-12 (o artigo não informa).
    benchmark_prompt: PromptVariant = "A"
    #: LLM e prompt dos Exp4/Exp5 (Tabelas 3 e 4 não informam): Mistral + E [hipótese GAP-12].
    noise_model: str = "mistral"
    noise_prompt: PromptVariant = "E"
    #: Limite opcional de tempo de parede por célula (s); excedido -> célula ``error``.
    max_cell_seconds: PositiveFloat | None = None


class ExperimentConfig(_Strict):
    """Configuração raiz de uma execução."""

    mode: Mode
    track: Track
    seeds: SeedsConfig = SeedsConfig()
    physics: PhysicsConfig = PhysicsConfig()
    data: DataConfig = DataConfig()
    symbolic: SymbolicConfig = SymbolicConfig()
    sr: SRConfig = SRConfig()
    loss: LossConfig = LossConfig()
    llm: LLMConfig = LLMConfig()
    experiments: ExperimentsConfig = ExperimentsConfig()
    output_dir: str = "results"

    @model_validator(mode="after")
    def _cross_checks(self) -> ExperimentConfig:
        if (
            self.mode is Mode.FULL
            and self.track is Track.PAPER_REPLICATION
            and self.llm.backend in FULL_REPLICATION_FORBIDDEN_BACKENDS
        ):
            raise ValueError(
                f"mode='full' com track='paper_replication' exige llm.backend='transformers' "
                f"(fp16, revisão fixada); backend {self.llm.backend!r} é desvio (Q4 no Ollama / "
                "avaliador simulado): use mode pilot/smoke ou track extended_analysis"
            )
        if self.track is Track.PAPER_REPLICATION and self.loss.normalization == "normalized":
            raise ValueError(
                "loss.normalization='normalized' é desvio experimental; "
                "permitido apenas em extended_analysis"
            )
        if self.track is Track.PAPER_REPLICATION and (
            self.llm.evaluation_scope == "top_k" or self.llm.top_k is not None
        ):
            raise ValueError(
                "llm.top_k/evaluation_scope='top_k' é desvio; permitido apenas em extended_analysis"
            )
        if self.track is Track.PAPER_REPLICATION and self.llm.cache_key_mode == "structure":
            raise ValueError(
                "llm.cache_key_mode='structure' é desvio; permitido apenas em extended_analysis"
            )
        if self.track is Track.PAPER_REPLICATION and self.llm.backend == "transformers":
            unpinned = [m.name for m in self.llm.models if not (m.revision or "").strip()]
            if unpinned:
                raise ValueError(
                    f"llm.models[].revision (SHA) é obrigatório em paper_replication: {unpinned}"
                )
        declared = {m.name for m in self.llm.models}
        unknown = [m for m in self.experiments.models if m not in declared]
        if unknown:
            raise ValueError(f"experiments.models referencia modelos não declarados: {unknown}")
        noise_runs = {"noise_robustness", "biased_noise"} & set(self.experiments.run)
        if noise_runs and self.experiments.noise_model not in declared:
            raise ValueError(
                f"experiments.noise_model={self.experiments.noise_model!r} não declarado em "
                f"llm.models (exigido por {sorted(noise_runs)})"
            )
        return self
