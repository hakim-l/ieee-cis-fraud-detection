# Copilot Instructions for IEEE CIS Fraud Detection

## Project Overview
This is a machine learning project for IEEE-CIS fraud detection. It uses the Cookiecutter Data Science template and follows MLOps best practices with modular code organization, testing, and CI/CD support.

- **Language**: Python 3.10
- **Framework**: scikit-learn, LightGBM, Dask/Distributed
- **Testing**: pytest with coverage
- **Linting**: ruff
- **Package Manager**: pip with flit

## Project Structure
```
src/
├── config.py                 # Configuration and constants
├── dataset/                  # Data loading and handling
├── feature_engineering/      # Feature creation and transformation
├── preprocessing/            # Data preprocessing utilities
├── metrics/                  # Evaluation metrics
├── models/                   # Model training and inference
├── train/                    # Training pipeline
├── tune_hyperparameter/      # Hyperparameter optimization
└── utils/                    # Utility functions

tests/
└── unit/
    ├── prepare_dataset/      # Dataset preparation tests
    └── tune_hyperparameter/  # Tuning tests

notebooks/                    # Jupyter notebooks (1.0-initials-description.ipynb)
data/
├── raw/                      # Original data
├── interim/                  # Transformed data
└── processed/                # Final datasets for modeling
```

## Key Commands
```bash
make help                    # Show all available commands
make requirements            # Install dependencies
make format                  # Format code with ruff
make lint                    # Lint code with ruff
make test                    # Run dataset tests
make test-cov               # Run tests with coverage report
make test-processor         # Run processor-specific tests
make test-integration       # Run integration tests
make test-tune              # Run hyperparameter tuning tests
make data                   # Prepare dataset from raw CSV
make data-overwrite         # Re-prepare dataset
make data-verbose           # Prepare with verbose output
make clean                  # Remove compiled Python files
```

## Code Standards & Style

### Python Conventions
- **Line length**: 99 characters (enforced by ruff)
- **Import sorting**: Enabled with isort rules
- **Formatting**: Automated with ruff format
- **Type hints**: Encouraged where practical

### Linting Rules
- Use `make format` before committing
- Run `make lint` to check for violations
- Project uses ruff for both linting and formatting
- First-party imports from `ieee_cis_fraud_detection` are auto-sorted

### Testing
- All tests use pytest
- Tests are in `tests/unit/` directory
- Use `--cov` flag for coverage reports
- HTML coverage reports generated in `htmlcov/`

## Development Workflow

1. **Before making changes**: Run `make lint` to understand current code style
2. **Code changes**: Follow PEP 8 and ruff conventions
3. **Testing**: Write tests in appropriate `tests/unit/` directories
4. **Before committing**: 
   - Run `make format` to format code
   - Run `make lint` to verify style
   - Run `make test` to ensure tests pass

## Common Patterns

### Adding Dependencies
- Add to `dependencies` list in `pyproject.toml` for main dependencies
- Add to `[project.optional-dependencies] dev` for development-only dependencies
- Run `make requirements` to install

### Creating Feature Engineering Code
- Add new features in `src/feature_engineering/`
- Use modules within the package for organization
- Reference `src/config.py` for constants

### Writing Tests
- Mirror source structure in `tests/unit/`
- Use pytest fixtures for setup
- Coverage should be maximized with `--cov` reporting

### Data Processing
- Raw data goes in `data/raw/`
- Transformed data goes in `data/interim/`
- Final datasets go in `data/processed/`
- Use `prepare_dataset.py` for ETL pipeline

## Important Files to Review
- `pyproject.toml` - Project metadata and tool configuration
- `Makefile` - All available commands
- `src/config.py` - Central configuration
- `requirements.txt` - Pinned dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks

## Tips for Contributions
1. Keep modules focused and single-purpose
2. Use descriptive variable and function names
3. Add docstrings to public functions
4. Write tests for new functionality
5. Use Dask/Distributed only when needed for large datasets
6. Reference the notebook naming convention (e.g., `1.0-jqp-initial-exploration.ipynb`)
7. Store analysis results in `reports/` and figures in `reports/figures/`

## Pre-commit Hooks
This project uses pre-commit hooks (see `.pre-commit-config.yaml`). Run hooks manually with:
```bash
pre-commit run --all-files
```

## Troubleshooting
- **Import errors**: Run `make requirements` to ensure dependencies are installed
- **Test failures**: Check that data files exist in expected locations (`data/raw/`)
- **Linting issues**: Run `make format` to auto-fix most issues
- **Python version mismatch**: Ensure Python 3.10 is being used
