# File Utilities Module

This module provides utility functions for working with files and directories.

## Functions

### `list_files(directory, extension=None, recursive=False)`

List files in a directory with optional filtering by file extension.

#### Parameters

- **directory** (`str` or `Path`): Path to the directory to search.
- **extension** (`str`, `list`, or `None`, optional): File extension(s) to filter by.
  - `None` (default): Return all files
  - `str`: Single extension (e.g., `'.csv'`, `'csv'`, or `'*.csv'`)
  - `list`: Multiple extensions (e.g., `['.csv', '.parquet']`)
  - Extensions are case-insensitive
- **recursive** (`bool`, optional): If `True`, search subdirectories recursively. Default is `False`.

#### Returns

`list of Path`: List of `pathlib.Path` objects for matching files, sorted alphabetically.

#### Raises

- `FileNotFoundError`: If the directory does not exist
- `NotADirectoryError`: If the path is not a directory

#### Examples

```python
from src.utils import list_files

# List all files in a directory
all_files = list_files('data/')

# List only CSV files
csv_files = list_files('data/', extension='.csv')

# List multiple file types
data_files = list_files('data/', extension=['.csv', '.parquet'])

# Recursive search through subdirectories
all_csvs = list_files('data/', extension='.csv', recursive=True)

# Extension without dot notation also works
parquet_files = list_files('data/', extension='parquet')

# Case-insensitive matching
all_images = list_files('images/', extension=['.jpg', '.png', '.gif'])
```

## Use Cases

### Data Pipeline
```python
from src.utils import list_files
import pandas as pd

# Find all training data files
train_files = list_files('data/processed/', extension=['.csv', '.parquet'])

# Load and concatenate
dfs = [pd.read_csv(f) if f.suffix == '.csv' else pd.read_parquet(f) 
       for f in train_files]
data = pd.concat(dfs, ignore_index=True)
```

### Model Checkpoints
```python
from src.utils import list_files

# Find all saved model checkpoints
checkpoints = list_files('models/', extension='.pkl', recursive=True)

# Get the latest checkpoint
latest = sorted(checkpoints)[-1]
```

### Feature Engineering
```python
from src.utils import list_files

# Find all feature engineering notebooks
notebooks = list_files('notebooks/', extension='.ipynb')

# Filter by pattern
feature_nb = [nb for nb in notebooks if 'feature' in nb.name.lower()]
```

## Notes

- File extensions are normalized to lowercase and include leading dot (e.g., `.csv`)
- The function returns `pathlib.Path` objects for easy path manipulation
- Results are automatically sorted alphabetically
- Both forward slashes (`/`) and backslashes (`\`) work for path separators
