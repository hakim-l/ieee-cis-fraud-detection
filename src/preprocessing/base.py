from abc import ABC, abstractmethod


class PreprocessorBase(ABC):
    def __init__(self, columns=):
        super().__init__()
    
    def _compute_preprocessing_pandas(self, df):
        """Compute preprocessing for the given pandas DataFrame."""
        pass

    def _compute_preprocessing_dask(self, df):
        """Compute preprocessing for the given Dask DataFrame."""
        pass