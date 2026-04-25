from abc import ABC, abstractmethod
import pandas as pd
import dask.dataframe as dd
import gc

class BaseFeatureEngineering(ABC):
    def __init__(self, output_meta: pd.DataFrame= pd.DataFrame(), **kwargs):
        super().__init__()
        self.output_meta = output_meta

    def is_meta_exist(self) -> bool:
        """Check if the output metadata is not empty."""
        return not self.output_meta.empty
    
    @abstractmethod
    def _compute_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for the given DataFrame."""
        pass

    def _compute_features_dask(self, df: dd.DataFrame) -> dd.DataFrame:
        """Compute features for the given Dask DataFrame."""
        if not self.is_meta_exist():
            raise ValueError("Output metadata is required to compute features for Dask DataFrame.") 
        
        results= df.map_partitions(
            self._compute_features_pandas, 
            meta=self.output_meta
            )
        gc.collect()  # Force garbage collection to free memory after processing
        return results

    def compute_features(self, df):
        """Compute features for the given DataFrame, handling both pandas and Dask DataFrames."""
        if isinstance(df, pd.DataFrame):
            return self._compute_features_pandas(df)
        elif isinstance(df, dd.DataFrame):
            return self._compute_features_dask(df)
        else:
            raise TypeError("Input must be a pandas DataFrame or a Dask DataFrame.")