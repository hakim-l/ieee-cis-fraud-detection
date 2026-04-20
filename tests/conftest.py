"""pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file_structure(temp_dir):
    """Create a test file structure."""
    # Create some files
    (temp_dir / "file1.csv").touch()
    (temp_dir / "file2.csv").touch()
    (temp_dir / "file3.parquet").touch()
    (temp_dir / "file4.txt").touch()

    # Create subdirectory with files
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.csv").touch()
    (subdir / "nested.parquet").touch()

    return temp_dir
