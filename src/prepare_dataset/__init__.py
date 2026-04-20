"""Dataset preparation module for IEEE CIS Fraud Detection project.

Handles converting raw CSV data to partitioned Parquet format in interim directory.
"""

from .processor import DatasetProcessor

__all__ = ["DatasetProcessor"]
