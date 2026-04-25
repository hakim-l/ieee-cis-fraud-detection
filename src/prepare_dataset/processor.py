"""Dataset processor for converting raw IEEE CIS fraud detection data to interim format.

Handles:
- Reading train/test identity and transaction CSVs from raw directory
- Partitioning them into Parquet files for efficient processing
- Organizing output into interim/train/ and interim/test/ directories
"""

from pathlib import Path
from typing import Optional
import gc
import pandas as pd
from loguru import logger

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    DEFAULT_TRAIN_CHUNKSIZE,
    # PARQUET_COMPRESSION,
    TABLE_INDEX,
    TARGET_COLUMN
)
from src.utils import list_files

def add_target_column_if_not_exists(df, target_column=TARGET_COLUMN):
    """Add a dummy target column to the DataFrame if it does not exist."""
    if target_column not in df.columns:
        logger.info(f"Target column '{target_column}' not found. Adding dummy target column with default value 0.")
        df[target_column] = pd.NA
    return df

class DatasetProcessor:
    """Process raw IEEE CIS fraud detection data and save to interim format."""

    def __init__(
        self,
        raw_data_dir: Path = RAW_DATA_DIR,
        interim_data_dir: Path = INTERIM_DATA_DIR,
        chunksize: int = DEFAULT_TRAIN_CHUNKSIZE,
        # compression: str = PARQUET_COMPRESSION,
    ):
        """Initialize the processor.

        Parameters
        ----------
        raw_data_dir : Path
            Directory containing raw CSV files.
        interim_data_dir : Path
            Base directory for interim output.
        chunksize : int
            Number of rows to process at a time when converting CSV to Parquet.
        compression : str
            Compression algorithm for Parquet files.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.interim_data_dir = Path(interim_data_dir)
        self.chunksize = chunksize
        # self.compression = compression

        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create required interim directory structure."""
        directories = [
            self.interim_data_dir / "train" / "identity",
            self.interim_data_dir / "train" / "transactions",
            self.interim_data_dir / "test" / "identity",
            self.interim_data_dir / "test" / "transactions",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

    def process_all(self, overwrite: bool = False) -> None:
        """Process all raw data files to interim format.

        Parameters
        ----------
        overwrite : bool
            If True, overwrite existing interim files. If False, skip if output exists.
        """
        logger.info("Starting dataset processing...")

        files_to_process = [
            ("train_identity.csv", "train/identity"),
            ("train_transaction.csv", "train/transactions"),
            ("test_identity.csv", "test/identity"),
            ("test_transaction.csv", "test/transactions"),
        ]

        for csv_file, output_subdir in files_to_process:
            self._process_file(csv_file, output_subdir, overwrite)

        logger.info("Dataset processing completed!")
        gc.collect()  # Final cleanup after processing all files

    def _process_file(
        self, csv_file: str, output_subdir: str, overwrite: bool = False
    ) -> None:
        """Process a single CSV file and save to Parquet.

        Parameters
        ----------
        csv_file : str
            Name of the CSV file in raw directory.
        output_subdir : str
            Output subdirectory within interim (e.g., "train/identity").
        overwrite : bool
            If True, overwrite existing files.
        """
        csv_path = self.raw_data_dir / csv_file
        output_dir = self.interim_data_dir / output_subdir

        if not csv_path.exists():
            logger.warning(f"Raw file not found: {csv_path}")
            return

        # Check if output already exists
        existing_files = list_files(output_dir, extension=".parquet", recursive=False)
        if existing_files and not overwrite:
            logger.info(
                f"Output already exists for {csv_file}. Use overwrite=True to replace."
            )
            return

        logger.info(f"Processing {csv_file}...")
        self._partition_csv_to_parquet(csv_path, output_dir)
        logger.info(f"Completed: {csv_file} -> {output_subdir}")
        gc.collect()  # Clean up memory after processing each file

    def _partition_csv_to_parquet(
        self, csv_path: Path, output_dir: Path
    ) -> None:
        """Read CSV in chunks and write Parquet files.

        Parameters
        ----------
        csv_path : Path
            Path to the input CSV file.
        output_dir : Path
            Directory to save Parquet files.
        """
        chunk_num = 0
        total_rows = 0

        for chunk in pd.read_csv(csv_path, chunksize=self.chunksize):
            output_file = output_dir / f"chunk_{chunk_num:06d}.parquet"
            chunk= chunk.set_index(TABLE_INDEX, drop=True)
            chunk.columns= [
                col.replace('-', '_') for col in chunk.columns
            ]

            chunk= add_target_column_if_not_exists(chunk, target_column=TARGET_COLUMN)
            chunk.to_parquet(
                output_file,
                # compression=self.compression,
            )
            chunk_num += 1
            total_rows += len(chunk)
            logger.debug(f"  Wrote chunk {chunk_num}: {len(chunk)} rows")

        logger.info(
            f"Wrote {chunk_num} Parquet files ({total_rows} total rows) to {output_dir}"
        )

        gc.collect()  # Clean up memory after writing files

    def load_interim_data(
        self, dataset_type: str = "train"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed interim data as DataFrames.

        Parameters
        ----------
        dataset_type : str
            Either "train" or "test".

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (identity_df, transactions_df)
        """
        if dataset_type not in ("train", "test"):
            raise ValueError(f"dataset_type must be 'train' or 'test', got {dataset_type}")

        identity_dir = self.interim_data_dir / dataset_type / "identity"
        transactions_dir = self.interim_data_dir / dataset_type / "transactions"

        logger.info(f"Loading {dataset_type} identity data...")
        identity_df = self._load_parquet_directory(identity_dir)

        logger.info(f"Loading {dataset_type} transactions data...")
        transactions_df = self._load_parquet_directory(transactions_dir)
        gc.collect()  # Clean up memory after loading data

        return identity_df, transactions_df

    def _load_parquet_directory(self, directory: Path) -> pd.DataFrame:
        """Load all Parquet files from a directory and concatenate.

        Parameters
        ----------
        directory : Path
            Directory containing Parquet files.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame from all Parquet files.
        """
        parquet_files = sorted(list_files(directory, extension=".parquet"))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {directory}")
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        result = pd.concat(dfs, ignore_index=True)

        logger.info(f"Loaded {len(parquet_files)} files: {len(result)} total rows")
        gc.collect()  # Clean up memory after concatenation
        return result

if __name__ == "__main__":
    processor = DatasetProcessor()
    processor.process_all(overwrite=False)