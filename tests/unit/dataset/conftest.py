"""Shared pytest fixtures for dataset tests."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


@pytest.fixture
def temp_dir():
    """Create a temporary directory that persists for the test."""
    with TemporaryDirectory() as tmp:
        yield Path(tmp)
