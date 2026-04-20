from .columns import NumericColumn, CategoricalColumn, FreeTextColumn
from src.utils.file_utils import list_files
import dask.dataframe as dd
import pandas as pd


class DaskDataset:
    def __init__(self, data_folder):
        self.data_folder= data_folder
        self.parquet_files = self.list_parquet_files()
        self.dataframe= self.load_data()
        self.columns= self.identify_column_types()

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
        if not self.dataframe:
            raise ValueError("Dataframe is not loaded. Cannot identify column types.")
        
        columns= []

        for col in self.dataframe.columns:
            dtype = self.dataframe[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                mean = self.dataframe[col].mean().compute()
                std = self.dataframe[col].std().compute()
                columns.append(NumericColumn(column_name=col, mean=mean, std=std))
            elif pd.api.types.is_string_dtype(dtype):
                columns.append(CategoricalColumn(column_name=col))
            else:
                columns.append(FreeTextColumn(column_name=col))
        return columns