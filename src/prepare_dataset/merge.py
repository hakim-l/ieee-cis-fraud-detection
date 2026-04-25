"""Dask-based merger for interim identity and transaction data.

Reads partitioned Parquet files for identity and transactions from interim directories,
merges them on 'TransactionID', and writes merged output to interim/train/merged/ and interim/test/merged/.
"""

from pathlib import Path
from turtle import pd
import dask.dataframe as dd
from loguru import logger
import pandas as pd
import os
from src.utils.dask import create_dask_client

def make_output_meta(transaction_sample, identity_sample):
    """
    Create an empty Dask DataFrame with the correct schema for merged output.

    Parameters
    ----------
    transaction_sample : pd.DataFrame
        Sample DataFrame from transactions to infer dtypes.
    identity_sample : pd.DataFrame
        Sample DataFrame from identity to infer dtypes.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with combined columns and correct dtypes.
    """
    merged_columns = list(transaction_sample.columns) + [col for col in identity_sample.columns if col not in transaction_sample.columns]
    merged_dtypes = {**transaction_sample.dtypes.to_dict(), **{col: identity_sample[col].dtype for col in identity_sample.columns if col not in transaction_sample.columns}}
    
    meta = pd.DataFrame(columns=merged_columns).astype(merged_dtypes)
    return meta

def merge_identity_transactions(
    interim_dir: Path, dataset_type: str, merge_on: str = "TransactionID"
) -> None:
    """
    Merge identity and transaction Parquet files using Dask and save to merged directory.

    Parameters
    ----------
    interim_dir : Path
        Base interim directory (e.g., data/interim)
    dataset_type : str
        Either 'train' or 'test'
    merge_on : str
        Column to merge on (default: 'TransactionID')
    """

    identity_dir = interim_dir / dataset_type / "identity"
    transactions_dir = interim_dir / dataset_type / "transactions"
    merged_dir = interim_dir / dataset_type / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading {dataset_type} identity from {identity_dir}")
    identity_ddf = dd.read_parquet(str(identity_dir) + "/*.parquet")
   
    # identity_df = pd.read_parquet(
    #     [os.path.join(str(identity_dir), f) for f in os.listdir(str(identity_dir)) if f.endswith(".parquet")]
    # )
   
    logger.info(f"Reading {dataset_type} transactions from {transactions_dir}")
    transactions_ddf = dd.read_parquet(str(transactions_dir) + "/*.parquet")

    # get transaction and identity samples to create meta
    # transaction_sample = transactions_ddf.head(1)
    # identity_sample = identity_df.head(1)
    # meta = make_output_meta(transaction_sample, identity_sample)

    logger.info(f"Merging {dataset_type} identity and transactions on '{merge_on}'")
    merged_ddf = transactions_ddf.merge(identity_ddf, how="left")

    logger.info(f"Writing merged {dataset_type} data to {merged_dir}")
    merged_ddf.to_parquet(str(merged_dir), write_index=True, overwrite=True)
    logger.info(f"Done: merged {dataset_type} written to {merged_dir}")


if __name__ == "__main__":
    interim_dir = Path("data/interim")
    
    client = create_dask_client(n_workers=4, threads_per_worker=1, memory_limit='1GB')
    logger.info(f"Dask cluster started: {client}")
    with client:
        merge_identity_transactions(interim_dir, "train")
        merge_identity_transactions(interim_dir, "test")
