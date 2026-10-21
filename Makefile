.PHONY: install lint format test check

install:
	pip install -r requirements-dev.txt

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:
	pytest -q --cov=ai_share_of_voice --cov-report=term-missing

check: lint test
