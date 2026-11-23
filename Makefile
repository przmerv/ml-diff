.PHONY: setup test lint format train serve

setup:
	uv sync --frozen
	uv run pre-commit install

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

format:
	uv run ruff check --fix .
	uv run ruff format .

train:
	uv run python -m ml_diff.train

serve:
	uv run uvicorn ml_diff.api:app --reload --no-access-log

docker-build:
	docker build -t ml-diff:local .

docker-run:
	docker run --rm -p 8000:8000 --name ml-diff ml-diff:local