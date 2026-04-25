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

# DATA INDEX
TABLE_INDEX = "TransactionID"

# CATEGORICAL FEATURES
CATEGORICAL_FEATURES_LIST = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    'id_12',
    'id_13',
    'id_14',
    'id_15',
    'id_16',
    'id_17',
    'id_18',
    'id_19',
    'id_20',
    'id_21',
    'id_22',
    'id_23',
    'id_24',
    'id_25',
    'id_26',
    'id_27',
    'id_28',
    'id_29',
    'id_30',
    'id_31',
    'id_32',
    'id_33',
    'id_34',
    'id_35',
    'id_36',
    'id_37',
    'id_38'
]

# CATEGORICAL FEATURE GROUPINGS
CATEGORICAL_FEATURE_DEFAULT_VALUE= 'Other'


CATEGORICAL_FEATURE_MAPPING_FILES= [
    INTERIM_DATA_DIR / "device_info_category_mapping.json",
    INTERIM_DATA_DIR / "id_30_category_mapping.json",
    INTERIM_DATA_DIR / "id_31_category_mapping.json",
    INTERIM_DATA_DIR / "id_33_category_mapping.json",
    INTERIM_DATA_DIR / "card1_category_mapping.json",
    INTERIM_DATA_DIR / "card2_category_mapping.json",
    INTERIM_DATA_DIR / "card3_category_mapping.json",
    INTERIM_DATA_DIR / "card4_category_mapping.json",
    INTERIM_DATA_DIR / "card5_category_mapping.json",
    INTERIM_DATA_DIR / "card6_category_mapping.json",
    INTERIM_DATA_DIR / "ProductCD_category_mapping.json",
]

# used features for modeling after feature engineering

USED_FEATURE_NAMES= [
    # 'TransactionID',
    'TransactionDT',
    'TransactionAmt',
    'ProductCD',
    "DeviceType",
    "DeviceInfo",
    'card1',
    'card2',
    'card3',
    'card4',
    'card5',
    'card6',
    # 'addr1',
    # 'addr2',
    'dist1',
    'dist2',
    # 'P_emaildomain',
    # 'R_emaildomain',
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'C6',
    'C7',
    'C8',
    'C9',

    'id_01',
    'id_02',
    'id_03',
    'id_04',
    'id_05',
    'id_06',
    'id_07',
    'id_08',
    'id_09',
    'id_10',
    'id_11',
    'id_12',
    'id_13',
    'id_14',
    'id_15',
    'id_16',
    'id_17',
    'id_18',
    'id_19',
    'id_20',
    'id_21',
    'id_22',
    'id_23',
    'id_24',
    'id_25',
    'id_26',
    'id_27',
    'id_28',
    'id_29',
    'id_30',
    'id_31',
    'id_32',
    'id_33',
    'id_34',
    'id_35',
    'id_36',
    'id_37',
    'id_38'
]

TARGET_COLUMN = "isFraud"

# Model and training settings
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TRAIN_CHUNKSIZE = 5_0_000

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
