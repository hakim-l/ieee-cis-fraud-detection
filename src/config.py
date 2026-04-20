"""Configuration module for IEEE CIS Fraud Detection project.

Stores project paths, model settings, and other configuration parameters.
"""

from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model and training settings
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TRAIN_CHUNKSIZE = 1_000_000

# LightGBM parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": DEFAULT_RANDOM_STATE,
    "verbose": -1,
}

# Evaluation metrics
METRICS = ["auc", "accuracy", "precision", "recall", "f1"]

# Preprocessing settings
STANDARD_SCALING_FEATURES: Optional[list] = None
CATEGORICAL_FEATURES: Optional[list] = None

# Parquet compression
PARQUET_COMPRESSION = "snappy"

# Dask settings
DASK_PARTITION_SIZE = "256MB"


def ensure_directories_exist() -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
