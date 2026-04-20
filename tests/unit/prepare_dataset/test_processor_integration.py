"""Integration tests for DatasetProcessor with real project data."""

import pytest
from pathlib import Path

from src.prepare_dataset import DatasetProcessor
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR


class TestDatasetProcessorWithRealData:
    """Integration tests using actual project data."""

    def test_real_data_processing(self):
        """Test processing real project data from data/raw."""
        if not RAW_DATA_DIR.exists():
            pytest.skip("Raw data directory not found")

        # Check if raw files exist
        required_files = [
            RAW_DATA_DIR / "train_identity.csv",
            RAW_DATA_DIR / "train_transaction.csv",
            RAW_DATA_DIR / "test_identity.csv",
            RAW_DATA_DIR / "test_transaction.csv",
        ]

        for file in required_files:
            if not file.exists():
                pytest.skip(f"Required file not found: {file}")

        processor = DatasetProcessor()

        # Process without overwrite
        processor.process_all(overwrite=False)

        # Verify files were created
        interim_files = list(INTERIM_DATA_DIR.glob("**/*.parquet"))
        assert len(interim_files) > 0, "No parquet files created"

    def test_real_data_loading(self):
        """Test loading actual processed data."""
        if not INTERIM_DATA_DIR.exists():
            pytest.skip("Interim data directory not found")

        processor = DatasetProcessor()

        # Try loading train data
        try:
            train_identity, train_transactions = processor.load_interim_data("train")
            assert not train_identity.empty, "Train identity is empty"
            assert not train_transactions.empty, "Train transactions is empty"
        except FileNotFoundError:
            pytest.skip("Processed train data not found")

        # Try loading test data
        try:
            test_identity, test_transactions = processor.load_interim_data("test")
            assert not test_identity.empty, "Test identity is empty"
            assert not test_transactions.empty, "Test transactions is empty"
        except FileNotFoundError:
            pytest.skip("Processed test data not found")

    def test_interim_structure_matches_expected(self):
        """Test that interim directory structure matches specification."""
        if not INTERIM_DATA_DIR.exists():
            pytest.skip("Interim data directory not found")

        expected_structure = {
            "train/identity": INTERIM_DATA_DIR / "train" / "identity",
            "train/transactions": INTERIM_DATA_DIR / "train" / "transactions",
            "test/identity": INTERIM_DATA_DIR / "test" / "identity",
            "test/transactions": INTERIM_DATA_DIR / "test" / "transactions",
        }

        for name, path in expected_structure.items():
            assert path.exists(), f"Expected directory not found: {name}"
            assert path.is_dir(), f"Path is not a directory: {name}"

    def test_parquet_files_readable(self):
        """Test that all created parquet files are readable."""
        import pandas as pd

        if not INTERIM_DATA_DIR.exists():
            pytest.skip("Interim data directory not found")

        parquet_files = list(INTERIM_DATA_DIR.glob("**/*.parquet"))

        if not parquet_files:
            pytest.skip("No parquet files found")

        for file in parquet_files:
            # Should be readable without errors
            df = pd.read_parquet(file)
            assert isinstance(df, pd.DataFrame), f"File not readable: {file}"
            assert len(df) > 0, f"File contains no data: {file}"
