from src.feature_engineering.base import BaseFeatureEngineering
from dask.dataframe import DataFrame as dd
import pandas as pd
from src.config import CATEGORICAL_FEATURE_DEFAULT_VALUE
from typing import List


class GroupCategoricalFeatures(BaseFeatureEngineering):
    def __init__(self, input_data_schema: pd.DataFrame, feature_mapping_list: List[dict]):
        self.feature_mapping_list = feature_mapping_list
        self.output_meta = self.set_output_meta(input_data_schema)
    
    def set_output_meta(self, input_data_schema):
        """Set the output metadata based on the input DataFrame and the columns to be extracted."""
        
        output_meta= input_data_schema.copy()
        return output_meta
    
    def categorize_value(self, value, mapping_dict):
        """Categorize a value based on the provided mapping dictionary."""
        mapped= mapping_dict.get(value)
        if mapped is not None:
            return mapped['category']
        else:
            return CATEGORICAL_FEATURE_DEFAULT_VALUE
        
    def _compute_features_pandas(self, df):
        # Extract device brand and device info from id_30
        
        results= df.copy()

        for feature_mapping in self.feature_mapping_list:
            column = feature_mapping['column']
            mapping_dict = feature_mapping['mapping']
            # new_column_name = f"{column}Category"
            results[column] = results[column].map(lambda x: self.categorize_value(x, mapping_dict))
        return results