from src.models import LGBMDaskClassifier
from src.dataset.dask_dataset import DaskDataset
from src.utils.dask import create_dask_client
from src.config import (
    PROCESSED_DATA_DIR,
    CATEGORICAL_FEATURES_LIST,
    USED_FEATURE_NAMES,
    TARGET_COLUMN,
    MODELS_DIR
)

def create_model_dir_if_not_exists(model_dir=MODELS_DIR):
    """Create the model directory if it does not exist."""
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

def force_categorical_to_string(dask_dataframe, categorical_columns):
    """Force categorical columns to string type to preserve categories during merge."""
    for col in categorical_columns:
        if col in dask_dataframe.columns:
            dask_dataframe[col] = dask_dataframe[col].astype('category')
    dask_dataframe_categorized= dask_dataframe.categorize(columns=categorical_columns)
    return dask_dataframe_categorized

if __name__ == "__main__":
    # Create Dask client
    client = create_dask_client()
    
    # Run feature engineering and save to processed directory
    train_processed_dir = PROCESSED_DATA_DIR / "train"
    test_processed_dir = PROCESSED_DATA_DIR / "test"

    # Load processed datasets
    train_dask_dataset = DaskDataset(train_processed_dir)
    train_dataframe= train_dask_dataset.dataframe
    test_dask_dataset = DaskDataset(test_processed_dir)
    test_dataframe= test_dask_dataset.dataframe
    # Force categorical features to string type
    train_dataframe[TARGET_COLUMN] = train_dataframe[TARGET_COLUMN].astype(int)  # Ensure target is integer type
    
    for col in CATEGORICAL_FEATURES_LIST:
        if col in train_dataframe.columns:
            train_dataframe[col] = train_dataframe[col].astype('category')
        if col in test_dataframe.columns:
            test_dataframe[col] = test_dataframe[col].astype('category')
    train_dataframe_categorized= train_dataframe.categorize(columns=CATEGORICAL_FEATURES_LIST).persist()
    test_dataframe_categorized= test_dataframe.categorize(columns=CATEGORICAL_FEATURES_LIST).persist()
    # train_dataframe= force_categorical_to_string(train_dask_dataset.dataframe, CATEGORICAL_FEATURES_LIST)
    # test_dataframe= force_categorical_to_string(test_dask_dataset.dataframe, CATEGORICAL_FEATURES_LIST)

    # Initialize and train the model
    model = LGBMDaskClassifier(
        categorical_feature= [col for col in CATEGORICAL_FEATURES_LIST if col in train_dataframe.columns],
    )

    x= train_dataframe_categorized[USED_FEATURE_NAMES]
    y= train_dataframe_categorized[TARGET_COLUMN]


    with client:
        model.fit(
            X=x,
            y=y,
        )

    create_model_dir_if_not_exists()
    model.save(
        MODELS_DIR / "lgbm_dask_classifier.pkl"
    )
