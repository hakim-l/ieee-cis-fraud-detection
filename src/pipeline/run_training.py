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
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import dask.dataframe as dd
import gc
import optuna
from sklearn.isotonic import IsotonicRegression
import joblib
import typer

app = typer.Typer()

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

def datasplit_on_partition(pandas_dataframe, frac, random_state=None):
    """Split a pandas DataFrame into two parts based on a fraction."""
    rng = np.random.default_rng(random_state)
    results= pandas_dataframe.copy()
    # rng= rng.seed(random_state)
    mask=  pd.Series(
        rng.uniform(0,1,size=(results.shape[0])) 
        )<= frac
    results['set'] = np.where(mask, 'train', 'validation')
    gc.collect()
    return results

def run_negative_sampling(dask_dataframe, target_column, random_state=None):
    """Run negative sampling on the given Dask DataFrame."""
    # Separate positive and negative samples
    positive_samples = dask_dataframe[dask_dataframe[target_column] == 1]
    negative_samples = dask_dataframe[dask_dataframe[target_column] == 0]
    
    # Sample negative samples to balance the dataset
    rng = np.random.default_rng(random_state)
    negative_sampled = negative_samples.sample(
        frac= len(positive_samples) / len(negative_samples),
        random_state=rng.integers(0, 1e6)
    )
    
    # Combine positive samples with sampled negative samples
    balanced_dataset = dd.concat([positive_samples, negative_sampled])
    
    return balanced_dataset

def compute_metrics(y_true, x, model, isotonic_model):
    y_model_pred_proba= model.predict_proba(x)[:,1]
    # y_model_pred_proba= y_model_pred_proba_dask.compute()
    isotonic_proba= isotonic_model.predict(y_model_pred_proba)
    y_pred= (isotonic_proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, isotonic_proba),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

def main(do_negative_sampling: bool=False):
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
    train_dataframe_meta= train_dataframe_categorized.head(1)
    train_dataframe_meta['set']= 'train'
    
    train_dataframe_split_job= train_dataframe_categorized.map_partitions(
        datasplit_on_partition,
        0.8,
        42,
        meta=train_dataframe_meta
    )

    train_dataframe_splited= train_dataframe_split_job.persist()


    train_set= train_dataframe_splited.query(
        "set=='train'"
    )

    validation_set= train_dataframe_splited.query(
        "set=='validation'"
    )

    # print train size and validation size
    print(
        "Train set size: ", train_set.shape[0].compute()
    )
    print(
        "Validation set size: ", validation_set.shape[0].compute()
    )

    test_dataframe_categorized= test_dataframe.categorize(columns=CATEGORICAL_FEATURES_LIST).persist()
    # train_dataframe= force_categorical_to_string(train_dask_dataset.dataframe, CATEGORICAL_FEATURES_LIST)
    # test_dataframe= force_categorical_to_string(test_dask_dataset.dataframe, CATEGORICAL_FEATURES_LIST)

    # Initialize and train the model
    model = LGBMDaskClassifier(
        n_estimators= 1000,
        categorical_feature= [col for col in CATEGORICAL_FEATURES_LIST if col in train_dataframe.columns],
        # focal_loss_alpha=0.25,
        # focal_loss_gamma=2.0,
        class_weight= "balanced",
    ) 

    x_train= train_set[USED_FEATURE_NAMES]
    y_train= train_set[TARGET_COLUMN]

    x_validation= validation_set[USED_FEATURE_NAMES]
    y_validation= validation_set[TARGET_COLUMN]

    with client:
        model.fit(
            X=x_train,
            y=y_train,
        )

        lgbm_train_pred_proba= model.predict_proba(x_train)[:,1]
        lgbm_val_pred_proba= model.predict_proba(x_validation)[:,1]

        y_train_numpy= y_train.compute()
        y_val_numpy= y_validation.compute()
    
        isotonic_model= IsotonicRegression(out_of_bounds="clip")
        isotonic_model.fit(
            lgbm_train_pred_proba,
            y_train_numpy.astype(int)
        )

        train_metrics= compute_metrics(y_train_numpy, x_train, model, isotonic_model)
        train_metrics['set']= 'train'

        val_metrics= compute_metrics(y_val_numpy, x_validation, model, isotonic_model)
        val_metrics['set']= 'validation'

    metrics= pd.DataFrame([train_metrics, val_metrics])
    metrics.to_csv(MODELS_DIR / "training_metrics.csv", index=False)

    print("Training metrics:")
    print(metrics)

    # train_pred= isotonic_model.predict(lgbm_train_pred_proba)
    
    # val_pred= isotonic_model.predict(lgbm_val_pred_proba)

    # print('Training set performance:')
    # print(
    #     classification_report(
    #         y_train_numpy.astype(int),
    #         train_pred.astype(int)
    #     )
    # )

    # print('Validation set performance:')
    # print(
    #     classification_report(
    #         y_val_numpy.astype(int),
    #         val_pred.astype(int)
    #     )
    # )

    create_model_dir_if_not_exists()
    model.save(
        MODELS_DIR / "lgbm_dask_classifier.pkl"
    )

    with open(MODELS_DIR / "isotonic_model.pkl", "wb") as f:
        joblib.dump(isotonic_model, f)


if __name__ == "__main__":
    typer.run(main)