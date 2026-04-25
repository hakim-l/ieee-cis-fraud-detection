from src.dataset.dask_dataset import DaskDataset
from src.feature_engineering.group_categorical_features import GroupCategoricalFeatures
from src.feature_engineering.standard_scaling import StandardScaling
from src.config import CATEGORICAL_FEATURE_MAPPING_FILES, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, USED_FEATURE_NAMES, TARGET_COLUMN, CATEGORICAL_FEATURES_LIST
import dask.dataframe as dd
from src.utils.dask import create_dask_client
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_features(dask_dataframe, feature_names):
    """Select specific features from the Dask DataFrame."""
    # if TARGET_COLUMN not in dask_dataframe.columns:

    return dask_dataframe[feature_names]

def get_categorical_feature_mappings():
    """Load categorical feature mappings from JSON files."""
    feature_mapping_list = []
    for mapping_file in CATEGORICAL_FEATURE_MAPPING_FILES:
        with open(mapping_file, 'r') as f:
            feature_mapping = json.load(f)
            feature_mapping_list.append(feature_mapping)
    return feature_mapping_list

def force_categorical_to_string(dask_dataframe, categorical_columns):
    """Force categorical columns to string type to preserve categories during merge."""
    for col in categorical_columns:
        if col in dask_dataframe.columns:
            dask_dataframe[col] = dask_dataframe[col].astype(str)
    return dask_dataframe

def run_feature_engineering(dask_dataset: DaskDataset, output_dir: str, used_feature_names: list):
    """Run the feature engineering pipeline on the given DaskDataset."""
    # Load categorical feature mappings
    feature_mapping_list = get_categorical_feature_mappings()
    
    input_data_schema = dask_dataset.dataframe.head(1)  # Get the schema from the Dask DataFrame
    # Initialize feature engineering steps
    group_categorical_features = GroupCategoricalFeatures(input_data_schema, feature_mapping_list)
    standard_scaling = StandardScaling(
        input_data_schema=input_data_schema,
        columns_to_scale=dask_dataset.numeric_columns
        )
    
    dask_dataframe= dask_dataset.dataframe
    
    # Add feature engineering steps to the dataset pipeline
    feature_engineering_job=(
        dask_dataframe
        .pipe(group_categorical_features.compute_features)
        .pipe(standard_scaling.compute_features)
        .pipe(force_categorical_to_string, categorical_columns=CATEGORICAL_FEATURES_LIST)
        .pipe(select_features, used_feature_names + [TARGET_COLUMN])
        .to_parquet(output_dir)
    ) 

    return True

if __name__ == "__main__":
    client = create_dask_client(n_workers=2, threads_per_worker=1, memory_limit='2GB')
    print(f"Dask cluster started: {client}")
    
    with client:
        # Load the merged dataset
        train_merged_dir = INTERIM_DATA_DIR / "train" / "merged"
        test_merged_dir = INTERIM_DATA_DIR / "test" / "merged"

        train_dask_dataset = DaskDataset(train_merged_dir)
        test_dask_dataset = DaskDataset(test_merged_dir)
        
        # Run feature engineering and save to processed directory
        # train_processed_dir = PROCESSED_DATA_DIR / "train"
        test_processed_dir = PROCESSED_DATA_DIR / "test"

        # print("Running feature engineering on training data...")
        # run_feature_engineering(train_dask_dataset, train_processed_dir)
     
        # print("Running feature engineering on test data...")
        run_feature_engineering(
            test_dask_dataset, 
            test_processed_dir,
            USED_FEATURE_NAMES
            )
