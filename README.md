# IEEE CIS Fraud Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A machine learning project for detecting fraudulent transactions using the IEEE CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203).

## Project Organization

```
├── LICENSE                  <- MIT License
├── Makefile                 <- Convenience commands
├── README.md                <- Project documentation
├── pyproject.toml           <- Project config & dependencies
├── requirements.txt         <- Pinned dependencies
├── .pre-commit-config.yaml  <- Pre-commit hooks
│
├── data/
│   ├── raw/                 <- Original immutable data (from Kaggle)
│   ├── interim/             <- Intermediate transformed data (Parquet)
│   └── processed/           <- Final datasets for modeling
│
├── src/                     <- Source code for the project
│   ├── __init__.py
│   ├── config.py            <- Configuration constants
│   ├── dataset/             <- Data loading and handling
│   ├── preprocessing/       <- Data preprocessing utilities
│   ├── feature_engineering/ <- Feature creation and transformation
│   ├── metrics/             <- Evaluation metrics
│   ├── models/              <- Model training and inference
│   ├── train/               <- Training pipeline
│   ├── tune_hyperparameter/ <- Hyperparameter optimization
│   ├── utils/               <- Utility functions
│   └── pipeline/            <- End-to-end pipeline scripts
│
├── tests/
│   └── unit/
│       ├── prepare_dataset/            <- Dataset tests
│       └── tune_hyperparameter/        <- Tuning tests
│
├── notebooks/               <- Jupyter notebooks
│   └── [1.0-initials-description.ipynb]
│
├── models/                  <- Trained models and predictions
│
├── docs/                    <- Documentation (mkdocs)
│
├── references/              <- Data dictionaries and manuals
│
└── reports/                 <- Generated analysis and reports
    └── figures/             <- Generated graphics and figures
```

## Quick Start

### Prerequisites
- Python 3.10
- pip or conda

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ieee-cis-fraud-detection
```

2. Create conda environment:
```bash
make create_environment
conda activate ieee-cis-fraud-detection
```

3. Install dependencies:
```bash
make install
```

### Running the Pipeline

1. Download the IEEE CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/) and place the CSV files in `data/raw/`:
```bash
# Place these files in data/raw/:
# - train_transaction.csv
# - train_identity.csv
# - test_transaction.csv
# - test_identity.csv
```

2. Run the complete pipeline:
```bash
make pipeline
```

This will execute:
- `make data` - Convert raw CSV to Parquet format
- `make features` - Generate engineered features
- `make train` - Train models

Or run individual steps:
```bash
make data                # Prepare dataset
make features            # Run feature engineering
make train               # Train models
```

## Available Commands

### Code Quality
```bash
make format      # Format code with ruff
make lint        # Lint code (non-destructive check)
make check       # Run format + lint checks
make pre-commit  # Run pre-commit hooks
```

### Testing
```bash
make test           # Run all unit tests
make test-cov       # Run tests with coverage report (HTML + terminal)
make test-dataset   # Run dataset preparation tests
make test-processor # Run processor-specific tests
make test-tune      # Run hyperparameter tuning tests
make test-verbose   # Run tests with full traceback
make test-file FILE=<path>  # Run specific test file
```

### Data & Pipeline
```bash
make data           # Prepare dataset (convert CSV to Parquet)
make data-overwrite # Re-prepare dataset (overwrite existing)
make data-verbose   # Prepare with verbose output
make features       # Run feature engineering
make train          # Run training pipeline
make pipeline       # Run complete pipeline: data → features → train
```

### Cleanup
```bash
make clean          # Remove Python cache and compiled files
make clean-cov      # Remove coverage reports
make clean-all      # Deep clean (cache, coverage, models, data)
```

### Documentation
```bash
make help           # Show all available commands
```

## Key Technologies

- **Data Processing**: pandas, pyarrow, Dask
- **ML Models**: scikit-learn, LightGBM
- **Hyperparameter Tuning**: Optuna
- **Distributed Computing**: Dask, Distributed
- **Testing**: pytest, pytest-cov
- **Code Quality**: ruff (formatting + linting)
- **Configuration**: Python-dotenv, Typer

## Development Workflow

1. **Before making changes**: Run linting to understand code style
   ```bash
   make lint
   ```

2. **Write your code** following PEP 8 conventions (99 char line limit)

3. **Write tests** in `tests/unit/` matching the source structure

4. **Before committing**:
   ```bash
   make format    # Auto-format code
   make lint      # Verify style
   make test      # Run tests
   ```

5. **Commit with pre-commit hooks**:
   ```bash
   make pre-commit  # Optional: run hooks manually
   git add .
   git commit -m "Your message"
   ```

## Code Standards

- **Line length**: 99 characters (enforced by ruff)
- **Import sorting**: Enabled with isort
- **Formatting**: Automated with ruff format
- **Type hints**: Encouraged where practical
- **Python version**: 3.10+

## Testing

All tests use pytest and are located in `tests/unit/`:

```bash
# Run all tests with coverage
make test-cov

# View HTML coverage report
open htmlcov/index.html
```

Coverage reports are generated in `htmlcov/` directory.

## Data Download

1. Download the IEEE CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/)
2. Extract the CSV files and place them in `data/raw/`:
   - `train_transaction.csv`
   - `train_identity.csv`
   - `test_transaction.csv` (optional)
   - `test_identity.csv` (optional)
3. Run `make data` to process the data

## Contributing

- Follow the code standards outlined above
- Write tests for new functionality
- Keep modules focused and single-purpose
- Use descriptive variable and function names
- Add docstrings to public functions

## References

- [IEEE-CIS Fraud Detection Challenge](https://www.kaggle.com/c/ieee-fraud-detection/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [Project Discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)