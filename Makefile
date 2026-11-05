.PHONY: install check test test-julia test-llm-real notebook-smoke lint format

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

# Executa o notebook em modo smoke (o notebook é entregue por outra PR).
notebook-smoke:
	PHYSICS_LLM_SR_MODE=smoke uv run jupyter nbconvert --to notebook --execute $(NOTEBOOK) \
		--output-dir results/notebook_smoke
