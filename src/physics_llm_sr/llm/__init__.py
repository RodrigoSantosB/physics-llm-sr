"""Avaliador LLM: interfaces, inferência, prompts, parser e cache.

Contrato para os consumidores (função de perda, adaptadores SR):

- ``interfaces``: ``EquationEvaluation`` (frozen; três escores em [0, 1] + ``feedback``),
  ``PhysicsEquationEvaluator`` (``evaluate`` e ``evaluate_many`` com ordem preservada),
  ``TextGenerationBackend`` (``generate(list[(system, user)]) -> list[str]`` e
  ``model_metadata()``) e erros (``EvaluationFailedError``, ``BackendError``,
  ``BackendConfigurationError``).
- ``scoring``: ``penalty(ev) = 1 - (c1 + c2 + c3) / 3``; ``LLMEquationEvaluator`` (cache,
  deduplicação canônica, retry só para resposta inválida, política ``retry|abort``, métricas);
  ``FakeEquationEvaluator`` (tabela por ``canonical_key`` + default explícito, ou função).
- ``prompting``: ``PromptLibrary.load(dir)`` e ``render`` (variantes A-H, ``gt_disclosed``,
  ``prompt_hash``); ``resolve_prompts_dir``.
- ``response_parser``: ``parse_evaluation`` (JSON validado, nunca executado).
- ``cache``: ``EvaluationCache`` (memória + JSONL), ``cache_key``, ``context_hash``.
- ``inference``: ``FakeBackend``, ``OllamaBackend``, ``TransformersBackend``, ``build_backend``.

Nenhum componente atribui nota substituta: falha levanta ``EvaluationFailedError``.
"""
