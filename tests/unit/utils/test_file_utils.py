"""Unit tests for file_utils.list_files()."""

import pytest
from pathlib import Path
from src.utils import list_files


class TestListFilesBasic:
    """Test basic list_files functionality."""

    def test_list_all_files(self, temp_file_structure):
        """Test listing all files without filter."""
        files = list_files(temp_file_structure)
        # Should return 4 files in root (no subdirs)
        assert len(files) == 4
        names = [f.name for f in files]
        assert "file1.csv" in names
        assert "file3.parquet" in names

    def test_list_single_extension(self, temp_file_structure):
        """Test filtering by single extension."""
        files = list_files(temp_file_structure, extension=".csv")
        assert len(files) == 2
        assert all(f.suffix == ".csv" for f in files)

    def test_list_multiple_extensions(self, temp_file_structure):
        """Test filtering by multiple extensions."""
        files = list_files(temp_file_structure, extension=[".csv", ".parquet"])
        assert len(files) == 3
        assert all(f.suffix in [".csv", ".parquet"] for f in files)

    def test_list_extension_without_dot(self, temp_file_structure):
        """Test extension filter without dot notation."""
        files = list_files(temp_file_structure, extension="csv")
        assert len(files) == 2
        assert all(f.suffix == ".csv" for f in files)

    def test_list_extension_with_glob(self, temp_file_structure):
        """Test extension filter with glob notation."""
        files = list_files(temp_file_structure, extension="*.txt")
        assert len(files) == 1
        assert files[0].suffix == ".txt"

    def test_case_insensitive_matching(self, temp_file_structure):
        """Test that extension matching is case-insensitive."""
        files_lower = list_files(temp_file_structure, extension=".csv")
        files_upper = list_files(temp_file_structure, extension=".CSV")
        files_mixed = list_files(temp_file_structure, extension=".CsV")
        assert len(files_lower) == len(files_upper) == len(files_mixed)

    def test_sorted_output(self, temp_file_structure):
        """Test that output is sorted alphabetically."""
        files = list_files(temp_file_structure)
        names = [f.name for f in files]
        assert names == sorted(names)


class TestListFilesRecursive:
    """Test recursive search functionality."""

    def test_recursive_search(self, temp_file_structure):
        """Test recursive search through subdirectories."""
        files = list_files(temp_file_structure, recursive=True)
        # 4 files in root + 2 in subdir = 6 total
        assert len(files) == 6

    def test_recursive_with_extension(self, temp_file_structure):
        """Test recursive search with extension filter."""
        files = list_files(temp_file_structure, extension=".csv", recursive=True)
        # file1.csv, file2.csv in root + nested.csv in subdir = 3
        assert len(files) == 3

    def test_non_recursive_excludes_subdirs(self, temp_file_structure):
        """Test that non-recursive search excludes subdirectories."""
        files = list_files(temp_file_structure, recursive=False)
        # Only 4 files in root directory
        assert len(files) == 4
        assert all(f.parent == temp_file_structure for f in files)


class TestListFilesErrorHandling:
    """Test error handling."""

    def test_nonexistent_directory(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            list_files("nonexistent_dir_12345/")

    def test_file_path_raises_error(self, temp_file_structure):
        """Test error when path points to a file, not directory."""
        file_path = temp_file_structure / "file1.csv"
        with pytest.raises(NotADirectoryError):
            list_files(file_path)


class TestListFilesReturnType:
    """Test return types and values."""

    def test_returns_list(self, temp_file_structure):
        """Test that function returns a list."""
        result = list_files(temp_file_structure)
        assert isinstance(result, list)

    def test_returns_path_objects(self, temp_file_structure):
        """Test that returned items are Path objects."""
        files = list_files(temp_file_structure)
        assert all(isinstance(f, Path) for f in files)

    def test_empty_directory(self, temp_dir):
        """Test listing files in empty directory."""
        files = list_files(temp_dir)
        assert files == []

    def test_empty_result_with_filter(self, temp_file_structure):
        """Test empty result when no files match filter."""
        files = list_files(temp_file_structure, extension=".xyz")
        assert files == []


class TestListFilesRealData:
    """Test with real project data."""

    def test_list_notebooks(self):
        """Test listing Jupyter notebooks."""
        notebooks_dir = Path("notebooks")
        if notebooks_dir.exists():
            files = list_files(notebooks_dir, extension=".ipynb")
            assert len(files) > 0
            assert all(f.suffix == ".ipynb" for f in files)

    def test_list_parquet_files(self):
        """Test listing Parquet files in data/interim."""
        interim_dir = Path("data/interim")
        if interim_dir.exists():
            files = list_files(interim_dir, extension=".parquet", recursive=True)
            assert len(files) > 0
            assert all(f.suffix == ".parquet" for f in files)

    def test_list_python_files(self):
        """Test listing Python source files."""
        src_dir = Path("src")
        if src_dir.exists():
            files = list_files(src_dir, extension=".py", recursive=True)
            assert len(files) > 0
            assert all(f.suffix == ".py" for f in files)
