from pathlib import Path
from typing import List, Optional, Union


def list_files(
    directory: Union[str, Path],
    extension: Optional[Union[str, List[str]]] = None,
    recursive: bool = False,
) -> List[Path]:
    """List files in a directory, optionally filtered by extension.

    Parameters
    ----------
    directory : str or Path
        Path to the directory to search.
    extension : str, list of str, or None, optional
        File extension(s) to filter by. Can be:
        - None (default): return all files
        - str: single extension (e.g., '.csv', 'csv', '*.csv')
        - list: multiple extensions (e.g., ['.csv', '.parquet'])
        Extensions are case-insensitive.
    recursive : bool, optional
        If True, search subdirectories recursively. Default is False.

    Returns
    -------
    list of Path
        List of Path objects for matching files, sorted alphabetically.

    Examples
    --------
    >>> from pathlib import Path
    >>> from src.utils import list_files
    >>>
    >>> # List all files
    >>> all_files = list_files('data/')
    >>>
    >>> # List only CSV files
    >>> csv_files = list_files('data/', extension='.csv')
    >>>
    >>> # List multiple file types
    >>> data_files = list_files('data/', extension=['.csv', '.parquet'])
    >>>
    >>> # Recursive search
    >>> all_csvs = list_files('data/', extension='.csv', recursive=True)
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Normalize extension format
    if extension is not None:
        if isinstance(extension, str):
            ext_list = [extension]
        else:
            ext_list = list(extension)

        # Ensure all extensions start with a dot and are lowercase
        normalized_exts = []
        for ext in ext_list:
            ext = ext.lower().lstrip("*")
            if not ext.startswith("."):
                ext = "." + ext
            normalized_exts.append(ext)
    else:
        normalized_exts = None

    # Search for files
    pattern = "**/*" if recursive else "*"
    files = [p for p in directory.glob(pattern) if p.is_file()]

    # Filter by extension if specified
    if normalized_exts is not None:
        files = [p for p in files if p.suffix.lower() in normalized_exts]

    return sorted(files)
