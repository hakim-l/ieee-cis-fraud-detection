from src.dataset import columns

from .base import BaseFeatureEngineering
import pandas as pd
import dask.dataframe as dd
from typing import List
from src.dataset.columns import NumericColumn


class StandardScaling(BaseFeatureEngineering):
    def __init__(self, input_data_schema: pd.DataFrame, columns_to_scale: List[NumericColumn]):
        self.columns_to_scale = columns_to_scale
        self.input_data_schema = input_data_schema
        self.set_output_meta()
        
    def set_output_meta(self):
        """Set the output metadata based on the input DataFrame and the columns to be scaled."""
        # meta_dict = {col.column_name: 'float64' for col in self.columns_to_scale}
        output_data_schema= self.input_data_schema.copy()
        for col in self.columns_to_scale:
            output_data_schema[col.column_name]= output_data_schema[col.column_name].astype('float32')
        self.output_meta = output_data_schema
    
    def _compute_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute standard scaled features for the given pandas DataFrame."""
        for col in self.columns_to_scale:
            df[col.column_name] = (df[col.column_name] - col.mean) / col.std
        return df
