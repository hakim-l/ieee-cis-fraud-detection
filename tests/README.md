# Test Suite

## Structure

```
tests/
├── __init__.py
├── conftest.py                 # pytest fixtures and configuration
├── unit/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── test_file_utils.py # File utilities tests
```

## Running Tests

### Using pytest (recommended)

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/utils/test_file_utils.py

# Run with verbose output
pytest tests/unit/utils/test_file_utils.py -v

# Run specific test class
pytest tests/unit/utils/test_file_utils.py::TestListFilesBasic

# Run specific test
pytest tests/unit/utils/test_file_utils.py::TestListFilesBasic::test_list_all_files
```

### Using unittest

```bash
python -m unittest discover tests/
```

### Manual execution

```bash
python tests/unit/utils/test_file_utils.py
```

## Test Coverage

### `test_file_utils.py`

Tests for `src.utils.list_files()`:

**TestListFilesBasic**
- `test_list_all_files` - List all files without filter
- `test_list_single_extension` - Filter by single extension
- `test_list_multiple_extensions` - Filter by multiple extensions
- `test_list_extension_without_dot` - Extension without dot notation (e.g., `csv`)
- `test_list_extension_with_glob` - Extension with glob notation (e.g., `*.txt`)
- `test_case_insensitive_matching` - Case-insensitive extension matching
- `test_sorted_output` - Verify output is sorted alphabetically

**TestListFilesRecursive**
- `test_recursive_search` - Recursive search through subdirectories
- `test_recursive_with_extension` - Recursive search with extension filter
- `test_non_recursive_excludes_subdirs` - Non-recursive excludes subdirectories

**TestListFilesErrorHandling**
- `test_nonexistent_directory` - Error handling for missing directory
- `test_file_path_raises_error` - Error handling for file path instead of directory

**TestListFilesReturnType**
- `test_returns_list` - Verify function returns a list
- `test_returns_path_objects` - Verify items are Path objects
- `test_empty_directory` - Handling empty directory
- `test_empty_result_with_filter` - Handling empty filter results

**TestListFilesRealData**
- `test_list_notebooks` - Real test: list .ipynb files
- `test_list_parquet_files` - Real test: list Parquet files
- `test_list_python_files` - Real test: list Python source files

## Fixtures

### `temp_dir`
Creates a temporary directory cleaned up after test.

```python
def test_something(temp_dir):
    # temp_dir is a Path object
    file_path = temp_dir / "test.txt"
```

### `temp_file_structure`
Creates a test file structure:
- Root: 4 files (file1.csv, file2.csv, file3.parquet, file4.txt)
- Subdir: 2 files (nested.csv, nested.parquet)

```python
def test_recursive(temp_file_structure):
    # Use predefined test structure
```

## Adding New Tests

1. Create test class inheriting from existing pattern
2. Use fixtures for temporary files
3. Follow naming: `test_<function_to_test>_<scenario>`
4. Use descriptive docstrings

Example:

```python
def test_my_feature(temp_file_structure):
    """Test my new feature."""
    result = list_files(temp_file_structure, extension=".txt")
    assert len(result) == 1
```

## CI/CD Integration

Add to your CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest tests/ -v --cov=src/
```
