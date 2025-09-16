.PHONY: setup lint test run-train run-eval

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip
	pip install -r requirements.txt
	pre-commit install

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests

test:
	pytest -q

run-train:
	python scripts/train.py --config src/ml_portfolio/configs/config.yaml

run-eval:
	python scripts/evaluate.py --config src/ml_portfolio/configs/config.yaml
