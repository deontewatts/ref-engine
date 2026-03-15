.PHONY: install test lint clean

install:
	pip install -e .
	pip install pytest flake8

test:
	pytest tests/

lint:
	flake8 src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
