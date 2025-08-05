.PHONY: help install test clean format lint

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package in development model_type
	pip install -e .

test: ## Run tests
	pytest tests/ -v

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

format: ## Format code with black
	black src/ tests/

lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/

train-pretrain: ## Run pretraining
	python src/transaction_transformer/models/pretrain.py --config configs/pretrain.yaml

train-finetune: ## Run finetuning
	python src/transaction_transformer/models/finetune.py --config configs/finetune.yaml

run-eda: ## Run exploratory data analysis
	python src/transaction_transformer/visualization/eda.py

evaluate: ## Run model evaluation
	python src/transaction_transformer/models/evaluate.py 