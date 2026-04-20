"""Unit tests for DatasetProcessor class."""

import pytest
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from src.prepare_dataset import DatasetProcessor


class TestDatasetProcessorInitialization:
    """Test DatasetProcessor initialization and setup."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        processor = DatasetProcessor()
        assert processor.raw_data_dir is not None
        assert processor.interim_data_dir is not None
        assert processor.chunksize > 0
        assert processor.compression is not None

    def test_init_with_custom_paths(self, temp_dir):
        """Test initialization with custom paths."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()
        interim_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=5000
        )

        assert processor.raw_data_dir == raw_dir
        assert processor.interim_data_dir == interim_dir
        assert processor.chunksize == 5000

    def test_init_custom_compression(self, temp_dir):
        """Test initialization with custom compression."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()
        interim_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, compression="gzip"
        )

        assert processor.compression == "gzip"

    def test_interim_directories_created(self, temp_dir):
        """Test that interim directory structure is created on init."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        # Check that all required directories were created
        expected_dirs = [
            interim_dir / "train" / "identity",
            interim_dir / "train" / "transactions",
            interim_dir / "test" / "identity",
            interim_dir / "test" / "transactions",
        ]

        for expected_dir in expected_dirs:
            assert expected_dir.exists()


class TestDatasetProcessorFileProcessing:
    """Test CSV to Parquet conversion and partitioning."""

    @pytest.fixture
    def sample_csv_data(self, temp_dir):
        """Create sample CSV files for testing."""
        # Create sample data
        df_identity = pd.DataFrame(
            {
                "TransactionID": range(1, 11),
                "id_01": [float(i) for i in range(10)],
                "id_02": [f"cat_{i}" for i in range(10)],
            }
        )

        df_transactions = pd.DataFrame(
            {
                "TransactionID": range(1, 11),
                "TransactionAmt": [100.0 * i for i in range(1, 11)],
                "ProductCD": ["W", "H", "S", "C", "R"] * 2,
            }
        )

        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        csv_files = {
            raw_dir / "train_identity.csv": df_identity,
            raw_dir / "train_transaction.csv": df_transactions,
            raw_dir / "test_identity.csv": df_identity,
            raw_dir / "test_transaction.csv": df_transactions,
        }

        for csv_file, df in csv_files.items():
            df.to_csv(csv_file, index=False)

        return temp_dir, raw_dir, csv_files

    def test_process_single_file(self, sample_csv_data):
        """Test processing a single CSV file."""
        temp_dir, raw_dir, csv_files = sample_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=5
        )

        processor._process_file("train_identity.csv", "train/identity", overwrite=False)

        # Check that parquet file was created
        output_dir = interim_dir / "train" / "identity"
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

    def test_partition_csv_to_parquet(self, sample_csv_data):
        """Test CSV partitioning with chunksize."""
        temp_dir, raw_dir, csv_files = sample_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=3
        )

        output_dir = interim_dir / "train" / "identity"
        csv_path = raw_dir / "train_identity.csv"

        processor._partition_csv_to_parquet(csv_path, output_dir)

        # With 10 rows and chunksize=3, should create 4 files (3+3+3+1)
        parquet_files = sorted(output_dir.glob("*.parquet"))
        assert len(parquet_files) == 4

        # Check file naming convention
        for i, f in enumerate(parquet_files):
            assert f.name == f"chunk_{i:06d}.parquet"

    def test_process_file_not_found(self, sample_csv_data):
        """Test processing non-existent file gracefully."""
        temp_dir, raw_dir, csv_files = sample_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        # Should not raise error, just log warning
        processor._process_file("nonexistent.csv", "train/identity", overwrite=False)

    def test_process_all_files(self, sample_csv_data):
        """Test processing all files."""
        temp_dir, raw_dir, csv_files = sample_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=5
        )

        processor.process_all(overwrite=False)

        # Check that all output directories have files
        expected_files = {
            "train/identity": "chunk_000000.parquet",
            "train/transactions": "chunk_000000.parquet",
            "test/identity": "chunk_000000.parquet",
            "test/transactions": "chunk_000000.parquet",
        }

        for subdir, expected_file in expected_files.items():
            path = interim_dir / subdir
            files = list(path.glob("*.parquet"))
            assert len(files) > 0
            assert any(f.name == expected_file for f in files)


class TestDatasetProcessorOverwriteProtection:
    """Test overwrite protection and idempotency."""

    @pytest.fixture
    def sample_csv_data(self, temp_dir):
        """Create sample CSV files for testing."""
        df = pd.DataFrame({"id": range(10), "value": range(10, 20)})

        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        df.to_csv(raw_dir / "train_identity.csv", index=False)
        df.to_csv(raw_dir / "train_transaction.csv", index=False)
        df.to_csv(raw_dir / "test_identity.csv", index=False)
        df.to_csv(raw_dir / "test_transaction.csv", index=False)

        return temp_dir, raw_dir

    def test_no_overwrite_by_default(self, sample_csv_data):
        """Test that files are not overwritten by default."""
        temp_dir, raw_dir = sample_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=5
        )

        # Process first time
        processor._process_file("train_identity.csv", "train/identity", overwrite=False)
        output_dir = interim_dir / "train" / "identity"
        files_first_run = set(f.name for f in output_dir.glob("*.parquet"))
        mtime_first_run = (output_dir / "chunk_000000.parquet").stat().st_mtime

        # Wait a bit and try to process again
        import time

        time.sleep(0.1)

        processor._process_file("train_identity.csv", "train/identity", overwrite=False)
        mtime_second_run = (output_dir / "chunk_000000.parquet").stat().st_mtime

        # File should not be modified
        assert mtime_first_run == mtime_second_run

    def test_overwrite_replaces_files(self, sample_csv_data):
        """Test that overwrite=True replaces existing files."""
        temp_dir, raw_dir = sample_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=5
        )

        # Process first time
        processor._process_file("train_identity.csv", "train/identity", overwrite=False)
        output_dir = interim_dir / "train" / "identity"

        # Get first file size
        first_file = output_dir / "chunk_000000.parquet"
        size_first_run = first_file.stat().st_size
        mtime_first_run = first_file.stat().st_mtime

        import time

        time.sleep(0.1)

        # Process with overwrite
        processor._process_file("train_identity.csv", "train/identity", overwrite=True)

        # File should be different
        mtime_second_run = first_file.stat().st_mtime
        assert mtime_first_run < mtime_second_run


class TestDatasetProcessorDataLoading:
    """Test loading interim data."""

    @pytest.fixture
    def processed_data(self, temp_dir):
        """Create processed parquet data."""
        interim_dir = temp_dir / "interim"

        # Create sample data
        df = pd.DataFrame(
            {
                "id": range(20),
                "value": range(20, 40),
            }
        )

        # Create train identity
        train_id_dir = interim_dir / "train" / "identity"
        train_id_dir.mkdir(parents=True)
        df.iloc[:10].to_parquet(train_id_dir / "chunk_000000.parquet", index=False)

        # Create train transactions
        train_tx_dir = interim_dir / "train" / "transactions"
        train_tx_dir.mkdir(parents=True)
        df.iloc[:8].to_parquet(train_tx_dir / "chunk_000000.parquet", index=False)
        df.iloc[8:10].to_parquet(train_tx_dir / "chunk_000001.parquet", index=False)

        # Create test identity
        test_id_dir = interim_dir / "test" / "identity"
        test_id_dir.mkdir(parents=True)
        df.iloc[10:15].to_parquet(test_id_dir / "chunk_000000.parquet", index=False)

        # Create test transactions
        test_tx_dir = interim_dir / "test" / "transactions"
        test_tx_dir.mkdir(parents=True)
        df.iloc[10:20].to_parquet(test_tx_dir / "chunk_000000.parquet", index=False)

        return temp_dir, interim_dir

    def test_load_train_data(self, processed_data):
        """Test loading train dataset."""
        temp_dir, interim_dir = processed_data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        identity, transactions = processor.load_interim_data("train")

        assert isinstance(identity, pd.DataFrame)
        assert isinstance(transactions, pd.DataFrame)
        assert len(identity) == 10
        assert len(transactions) == 10

    def test_load_test_data(self, processed_data):
        """Test loading test dataset."""
        temp_dir, interim_dir = processed_data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        identity, transactions = processor.load_interim_data("test")

        assert isinstance(identity, pd.DataFrame)
        assert isinstance(transactions, pd.DataFrame)
        assert len(identity) == 5
        assert len(transactions) == 10

    def test_load_invalid_dataset_type(self, processed_data):
        """Test error handling for invalid dataset type."""
        temp_dir, interim_dir = processed_data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        with pytest.raises(ValueError, match="dataset_type must be 'train' or 'test'"):
            processor.load_interim_data("invalid")

    def test_load_parquet_directory_multiple_files(self, processed_data):
        """Test concatenation of multiple parquet files."""
        temp_dir, interim_dir = processed_data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        # Load transactions which has 2 files
        transactions_dir = interim_dir / "train" / "transactions"
        df = processor._load_parquet_directory(transactions_dir)

        # Should combine both chunks
        assert len(df) == 10
        assert "id" in df.columns
        assert "value" in df.columns

    def test_load_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()
        empty_dir = interim_dir / "empty"
        empty_dir.mkdir(parents=True)

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        df = processor._load_parquet_directory(empty_dir)

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestDatasetProcessorDataIntegrity:
    """Test data integrity during processing."""

    @pytest.fixture
    def large_csv_data(self, temp_dir):
        """Create larger CSV file for testing."""
        df = pd.DataFrame(
            {
                "id": range(1000),
                "value": range(1000, 2000),
                "category": [f"cat_{i % 10}" for i in range(1000)],
            }
        )

        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        df.to_csv(raw_dir / "train_identity.csv", index=False)
        df.to_csv(raw_dir / "train_transaction.csv", index=False)
        df.to_csv(raw_dir / "test_identity.csv", index=False)
        df.to_csv(raw_dir / "test_transaction.csv", index=False)

        return temp_dir, raw_dir, df

    def test_row_count_preserved(self, large_csv_data):
        """Test that all rows are preserved during conversion."""
        temp_dir, raw_dir, original_df = large_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=100
        )

        processor._process_file("train_identity.csv", "train/identity", overwrite=False)

        # Load back
        identity, _ = processor.load_interim_data("train")

        assert len(identity) == len(original_df)

    def test_column_preservation(self, large_csv_data):
        """Test that all columns are preserved."""
        temp_dir, raw_dir, original_df = large_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=100
        )

        processor.process_all(overwrite=False)

        identity, transactions = processor.load_interim_data("train")

        assert set(identity.columns) == set(original_df.columns)
        assert set(transactions.columns) == set(original_df.columns)

    def test_data_values_preserved(self, large_csv_data):
        """Test that data values are preserved correctly."""
        temp_dir, raw_dir, original_df = large_csv_data
        interim_dir = temp_dir / "interim"

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=100
        )

        processor._process_file("train_identity.csv", "train/identity", overwrite=False)

        identity, _ = processor.load_interim_data("train")

        # Check specific values
        pd.testing.assert_frame_equal(
            original_df.reset_index(drop=True), identity.reset_index(drop=True)
        )


class TestDatasetProcessorEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_row_csv(self, temp_dir):
        """Test processing CSV with single row."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()

        df = pd.DataFrame({"id": [1], "value": [100]})
        df.to_csv(raw_dir / "train_identity.csv", index=False)
        df.to_csv(raw_dir / "train_transaction.csv", index=False)
        df.to_csv(raw_dir / "test_identity.csv", index=False)
        df.to_csv(raw_dir / "test_transaction.csv", index=False)

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=1
        )

        processor.process_all(overwrite=False)

        identity, transactions = processor.load_interim_data("train")
        assert len(identity) == 1
        assert len(transactions) == 1

    def test_large_chunksize(self, temp_dir):
        """Test with chunksize larger than file."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()

        df = pd.DataFrame({"id": range(10), "value": range(10, 20)})
        df.to_csv(raw_dir / "train_identity.csv", index=False)
        df.to_csv(raw_dir / "train_transaction.csv", index=False)
        df.to_csv(raw_dir / "test_identity.csv", index=False)
        df.to_csv(raw_dir / "test_transaction.csv", index=False)

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir, chunksize=1000000
        )

        processor._process_file("train_identity.csv", "train/identity", overwrite=False)

        output_dir = interim_dir / "train" / "identity"
        parquet_files = list(output_dir.glob("*.parquet"))

        # Should create only 1 file
        assert len(parquet_files) == 1

    def test_csv_with_special_characters(self, temp_dir):
        """Test CSV with special characters and unicode."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()

        df = pd.DataFrame(
            {
                "id": range(5),
                "text": ["hello", "café", "日本語", "émojis", "test"],
            }
        )
        df.to_csv(raw_dir / "train_identity.csv", index=False, encoding="utf-8")
        df.to_csv(raw_dir / "train_transaction.csv", index=False, encoding="utf-8")
        df.to_csv(raw_dir / "test_identity.csv", index=False, encoding="utf-8")
        df.to_csv(raw_dir / "test_transaction.csv", index=False, encoding="utf-8")

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        processor.process_all(overwrite=False)

        identity, _ = processor.load_interim_data("train")

        # Check that special characters are preserved
        assert identity["text"].iloc[1] == "café"
        assert identity["text"].iloc[2] == "日本語"


class TestDatasetProcessorDirectoryHandling:
    """Test directory creation and handling."""

    def test_creates_all_required_directories(self, temp_dir):
        """Test that all required interim directories are created."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        required_dirs = [
            interim_dir / "train" / "identity",
            interim_dir / "train" / "transactions",
            interim_dir / "test" / "identity",
            interim_dir / "test" / "transactions",
        ]

        for directory in required_dirs:
            assert directory.exists()
            assert directory.is_dir()

    def test_handles_existing_directories(self, temp_dir):
        """Test that processor handles existing directories correctly."""
        raw_dir = temp_dir / "raw"
        interim_dir = temp_dir / "interim"
        raw_dir.mkdir()

        # Pre-create directories
        for subdir in [
            "train/identity",
            "train/transactions",
            "test/identity",
            "test/transactions",
        ]:
            (interim_dir / subdir).mkdir(parents=True)

        # Should not raise error
        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        # Check all directories still exist
        assert (interim_dir / "train" / "identity").exists()


class TestDatasetProcessorReturnTypes:
    """Test return types and values."""

    @pytest.fixture
    def sample_processed_data(self, temp_dir):
        """Create sample processed data."""
        interim_dir = temp_dir / "interim"

        df = pd.DataFrame({"id": range(10), "value": range(10, 20)})

        for split in ["train", "test"]:
            for dtype in ["identity", "transactions"]:
                output_dir = interim_dir / split / dtype
                output_dir.mkdir(parents=True)
                df.to_parquet(output_dir / "chunk_000000.parquet", index=False)

        return temp_dir, interim_dir

    def test_load_interim_data_returns_tuple(self, sample_processed_data):
        """Test that load_interim_data returns tuple."""
        temp_dir, interim_dir = sample_processed_data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        result = processor.load_interim_data("train")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_load_interim_data_returns_dataframes(self, sample_processed_data):
        """Test that load_interim_data returns DataFrames."""
        temp_dir, interim_dir = sample_processed_data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        processor = DatasetProcessor(
            raw_data_dir=raw_dir, interim_data_dir=interim_dir
        )

        identity, transactions = processor.load_interim_data("train")

        assert isinstance(identity, pd.DataFrame)
        assert isinstance(transactions, pd.DataFrame)
