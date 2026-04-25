"""Dataset preparation module for IEEE CIS Fraud Detection project.

Handles converting raw CSV data to partitioned Parquet format in interim directory.
"""

from .processor import DatasetProcessor

from .merge import merge_identity_transactions

__all__ = ["DatasetProcessor", "merge_identity_transactions"]
