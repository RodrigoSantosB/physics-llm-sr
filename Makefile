.PHONY: install check test test-julia test-llm-real notebook-smoke lint format \
	smoke pilot pilot-colab report compare export-legado

NOT_OPT_IN = not slow and not julia and not llm_real
NOTEBOOK = notebooks/01_full_paper_reproduction.ipynb

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -m "$(NOT_OPT_IN)"

check: lint test

test-julia:
	uv run pytest -m julia

test-llm-real:
	uv run pytest -m llm_real

# Executa o notebook principal em RUN_PROFILE="smoke" (valor padrão do notebook), de ponta a ponta,
# sem LLM real. O notebook executado vai para results/notebook_smoke/.
notebook-smoke:
	uv run jupyter nbconvert --to notebook --execute $(NOTEBOOK) \
		--ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 \
		--output-dir results/notebook_smoke --output 01_full_paper_reproduction.executed.ipynb

# ---------------------------------------------------------------- execução de experimentos
# RESUME=1 retoma a execução mais recente com a mesma config (células concluídas são puladas).
CLI = uv run physics-llm-sr
RESUME_FLAG = $(if $(RESUME),--resume,)

# Smoke: avaliador simulado, segundos.
smoke:
	$(CLI) run --config configs/smoke.yaml $(RESUME_FLAG)

# Piloto local (Ollama mistral:7b-instruct), DOIS braços: paper_replication (literal) e
# extended_analysis (normalized). Diretórios separados; nunca misturar nas tabelas.
# Células com erro (ex.: PySR/Julia indisponível) não impedem o segundo braço.
pilot:
	-$(CLI) run --config configs/pilot.yaml $(RESUME_FLAG)
	-$(CLI) run --config configs/pilot_extended.yaml $(RESUME_FLAG)

# Piloto Colab A100 (transformers fp16), dois braços.
pilot-colab:
	-$(CLI) run --config configs/colab_pilot.yaml $(RESUME_FLAG)
	-$(CLI) run --config configs/colab_pilot_extended.yaml $(RESUME_FLAG)

# Relatório/figuras a partir dos arquivos persistidos: make report RUN_DIR=results/smoke/<id>
report:
	@test -n "$(RUN_DIR)" || (echo "uso: make report RUN_DIR=results/<cfg>/<run_id>" && exit 2)
	$(CLI) report $(RUN_DIR)

# Tabelas e figuras artigo x local: make compare RUNS="results/pilot/<id> results/pilot_extended/<id>" [OUT=dir]
compare:
	@test -n "$(RUNS)" || (echo 'uso: make compare RUNS="<run_dir1> <run_dir2>" [OUT=<dir>]' && exit 2)
	uv run python -m physics_llm_sr.analysis --runs $(RUNS) $(if $(OUT),--out $(OUT),)

# Exporta no formato legado (comparação direta): make export-legado RUNS="results/pilot/<id>" OUT=results/legado/literal
export-legado:
	@test -n "$(RUNS)" -a -n "$(OUT)" || (echo 'uso: make export-legado RUNS="<run_dir>..." OUT=<dir>' && exit 2)
	uv run python -m physics_llm_sr.analysis.legacy_export --runs $(RUNS) --out $(OUT)
