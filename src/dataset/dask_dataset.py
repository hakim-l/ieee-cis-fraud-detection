from .columns import NumericColumn, CategoricalColumn, FreeTextColumn
from src.utils.file_utils import list_files
import dask.dataframe as dd
import pandas as pd


class DaskDataset:
    def __init__(self, data_folder, free_text_threshold=1000):
        self.free_text_threshold = free_text_threshold
        self.data_folder= data_folder
        self.parquet_files = self.list_parquet_files()
        self.dataframe= self.load_data()
        self.columns, self.numeric_columns, self.categorical_columns, self.free_text_columns = self.identify_column_types()

    def list_parquet_files(self):
        parquet_files = list_files(self.data_folder, extension=".parquet")
        return parquet_files
    
    def load_data(self):
        if not self.parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {self.data_folder}")
        
        # Load all Parquet files into a single Dask DataFrame
        dask_df = dd.read_parquet(self.parquet_files)
        return dask_df
    
    def identify_column_types(self):
        
        columns= []
        numeric_columns= []
        categorical_columns= []
        free_text_columns= []

        for col in self.dataframe.columns:
            dtype = self.dataframe[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                mean = self.dataframe[col].mean().compute()
                std = self.dataframe[col].std().compute()
                column= NumericColumn(column_name=col, mean=mean, std=std)
                columns.append(column)
                numeric_columns.append(column)
            elif (pd.api.types.is_string_dtype(dtype) and 
                  self.dataframe[col].nunique().compute() < self.free_text_threshold
                  ) or pd.api.types.is_categorical_dtype(dtype):
                unique_values = self.dataframe[col].dropna().unique().compute()
                column= CategoricalColumn(column_name=col, categories=unique_values)
                columns.append(column)
                categorical_columns.append(column)
            else:
                column= FreeTextColumn(column_name=col)
                columns.append(column)
                free_text_columns.append(column)
        return columns, numeric_columns, categorical_columns, free_text_columns