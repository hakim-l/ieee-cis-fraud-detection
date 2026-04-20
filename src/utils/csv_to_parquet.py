"""Utilities to read large CSVs in chunks and write partitioned Parquet files.

Provides partition_csv_to_parquet which streams a CSV and writes parquet files
optionally partitioned by specified columns.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional
    def tqdm(x, **kwargs):
        return x


def partition_csv_to_parquet(
    csv_path: str,
    out_dir: str,
    chunksize: int = 1_000_000,
    partition_cols: Optional[List[str]] = None,
    compression: str = "snappy",
    index: bool = False,
) -> int:
    """Read a CSV in chunks and write out Parquet files.

    Args:
        csv_path: Path to the input CSV file.
        out_dir: Directory to write parquet files into. Created if missing.
        chunksize: Number of rows per chunk to read from CSV.
        partition_cols: Optional list of columns to partition by. If provided, the
            function will create subdirectories named like "col=value" for each
            unique combination in a chunk.
        compression: Parquet compression (e.g., 'snappy', 'gzip', 'none').
        index: If True, preserve the dataframe index in the parquet files.

    Returns:
        The number of parquet files written.

    Notes:
        - Requires pandas and pyarrow to be installed.
        - This function streams the CSV and should be memory efficient as long
          as `chunksize` is reasonable.
    """
    csv_path = str(csv_path)
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    files_written = 0
    reader = pd.read_csv(csv_path, chunksize=chunksize, iterator=True)

    for chunk in tqdm(reader, desc="Reading CSV chunks"):
        if partition_cols:
            # Ensure partition columns exist
            missing = [c for c in partition_cols if c not in chunk.columns]
            if missing:
                raise ValueError(f"Partition columns not in CSV: {missing}")

            # Group by partition columns and write each group to its partition dir
            grouped = chunk.groupby(partition_cols, sort=False)
            for keys, group in grouped:
                if not isinstance(keys, tuple):
                    keys = (keys,)

                # Build partition subpath like col1=val1/col2=val2
                parts = []
                for col, val in zip(partition_cols, keys):
                    # sanitize value to be safe in a filename
                    sval = str(val)
                    sval = sval.replace(os.sep, "_")
                    parts.append(f"{col}={sval}")

                part_dir = os.path.join(out_dir, *parts)
                Path(part_dir).mkdir(parents=True, exist_ok=True)
                out_path = os.path.join(part_dir, f"part-{files_written:06d}.parquet")
                group.to_parquet(out_path, index=index, compression=compression, engine="pyarrow")
                files_written += 1
        else:
            out_path = os.path.join(out_dir, f"part-{files_written:06d}.parquet")
            chunk.to_parquet(out_path, index=index, compression=compression, engine="pyarrow")
            files_written += 1

    return files_written


if __name__ == "__main__":  # pragma: no cover - simple CLI for manual runs
    import argparse

    parser = argparse.ArgumentParser(description="Partition large CSV into Parquet files.")
    parser.add_argument("csv", help="Input CSV file path")
    parser.add_argument("out", help="Output directory for parquet files")
    parser.add_argument("--chunksize", type=int, default=1000000)
    parser.add_argument("--partition-cols", nargs="*", help="Columns to partition by")
    parser.add_argument("--compression", default="snappy")
    args = parser.parse_args()

    n = partition_csv_to_parquet(args.csv, args.out, chunksize=args.chunksize, partition_cols=args.partition_cols, compression=args.compression)
    print(f"Wrote {n} parquet files to {args.out}")
