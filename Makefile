#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ieee-cis-fraud-detection
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
.DEFAULT_GOAL := help

#################################################################################
# ENVIRONMENT & SETUP                                                           #
#################################################################################

## Create conda environment (Python 3.10)
.PHONY: create_environment
create_environment:
conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
@echo ">>> Conda environment created. Activate with: conda activate $(PROJECT_NAME)"

## Install Python dependencies from requirements.txt
.PHONY: requirements
requirements:
$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install dependencies and pre-commit hooks
.PHONY: install
install: requirements
pre-commit install

#################################################################################
# CODE QUALITY                                                                  #
#################################################################################

## Format source code with ruff
.PHONY: format
format:
ruff check --fix src/ tests/
ruff format src/ tests/

## Lint code with ruff (non-destructive)
.PHONY: lint
lint:
ruff format --check src/ tests/
ruff check src/ tests/

## Check code quality (lint + format check)
.PHONY: check
check: lint
@echo "✓ Code quality checks passed"

## Run pre-commit hooks on all files
.PHONY: pre-commit
pre-commit:
pre-commit run --all-files

#################################################################################
# TESTING                                                                       #
#################################################################################

## Run all unit tests
.PHONY: test
test:
$(PYTHON_INTERPRETER) -m pytest tests/unit/ -v --tb=short

## Run tests with coverage report
.PHONY: test-cov
test-cov:
$(PYTHON_INTERPRETER) -m pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term-missing

## Run dataset preparation tests only
.PHONY: test-dataset
test-dataset:
$(PYTHON_INTERPRETER) -m pytest tests/unit/prepare_dataset/ -v --tb=short

## Run dataset processor unit tests
.PHONY: test-processor
test-processor:
$(PYTHON_INTERPRETER) -m pytest tests/unit/prepare_dataset/test_processor.py -v --tb=short

## Run dataset processor integration tests
.PHONY: test-integration
test-integration:
$(PYTHON_INTERPRETER) -m pytest tests/unit/prepare_dataset/test_processor_integration.py -v --tb=short

## Run hyperparameter tuning tests
.PHONY: test-tune
test-tune:
$(PYTHON_INTERPRETER) -m pytest tests/unit/tune_hyperparameter/ -v --tb=short

## Run tests with detailed output (full traceback)
.PHONY: test-verbose
test-verbose:
$(PYTHON_INTERPRETER) -m pytest tests/unit/ -vv

## Run specific test file (usage: make test-file FILE=tests/unit/path/to/test.py)
.PHONY: test-file
test-file:
$(PYTHON_INTERPRETER) -m pytest $(FILE) -v --tb=short

#################################################################################
# DATA PROCESSING                                                               #
#################################################################################

## Prepare dataset: Convert raw CSV to interim Parquet format
.PHONY: data
data:
$(PYTHON_INTERPRETER) src/pipeline/prepare_dataset.py

## Prepare dataset (overwrite existing files)
.PHONY: data-overwrite
data-overwrite:
$(PYTHON_INTERPRETER) src/pipeline/prepare_dataset.py --overwrite

## Prepare dataset with verbose output
.PHONY: data-verbose
data-verbose:
$(PYTHON_INTERPRETER) src/pipeline/prepare_dataset.py --verbose

## Prepare dataset and run all tests
.PHONY: data-test
data-test: data test

#################################################################################
# PIPELINE EXECUTION                                                            #
#################################################################################

## Run feature engineering pipeline
.PHONY: features
features:
$(PYTHON_INTERPRETER) src/pipeline/run_feature_engineering.py

## Run training pipeline
.PHONY: train
train:
$(PYTHON_INTERPRETER) src/pipeline/run_training.py

## Run complete pipeline: data → features → train
.PHONY: pipeline
pipeline: data features train

#################################################################################
# CLEANUP                                                                       #
#################################################################################

## Delete all compiled Python files and cache directories
.PHONY: clean
clean:
find . -type f -name "*.py[co]" -delete
find . -type d -name "__pycache__" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

## Clean and remove coverage reports
.PHONY: clean-cov
clean-cov:
rm -rf htmlcov/ .coverage

## Deep clean (compiled files, cache, coverage, models)
.PHONY: clean-all
clean-all: clean clean-cov
rm -rf models/*.pkl models/*.joblib 2>/dev/null || true
rm -rf data/interim/* data/processed/* 2>/dev/null || true
@echo "✓ Deep clean completed"

#################################################################################
# UTILITY & DOCUMENTATION                                                       #
#################################################################################

## Show this help message
.PHONY: help
help:
@$(PYTHON_INTERPRETER) -c "import re, sys; lines = '\n'.join([line for line in open('Makefile')]); matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); print('Available commands:\n'); print('\n'.join(['{:20} {}'.format(*reversed(match)) for match in matches]))"
