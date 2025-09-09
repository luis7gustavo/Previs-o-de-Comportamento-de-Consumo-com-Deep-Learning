import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataLoader:
    """Class for loading and basic preprocessing of sales data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration"""
        from config.model_config import load_config, get_paths
        self.config = load_config(config_path)
        self.paths = get_paths()
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw sales data from CSV file
        
        Returns:
            pd.DataFrame: Raw sales data
        """
        try:
            data_path = self.paths.get("raw_data", "data/raw/vendas_online.csv")
            logger.info(f"Loading raw data from {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data for required columns and formats
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            bool: Whether data is valid
        """
        from config.model_config import get_preprocessing_params
        
        preproc_params = get_preprocessing_params()
        target_column = preproc_params.get("target_column")
        date_column = preproc_params.get("date_column")
        
        # Check required columns exist
        if target_column not in df.columns or date_column not in df.columns:
            logger.error(f"Missing required columns. Need {target_column} and {date_column}")
            return False
            
        # Check for null values in critical columns
        if df[target_column].isnull().sum() > 0:
            logger.warning(f"Found {df[target_column].isnull().sum()} null values in target column")
        
        # Attempt to convert date column
        try:
            pd.to_datetime(df[date_column])
        except:
            logger.error(f"Could not convert {date_column} to datetime")
            return False
            
        return True
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df (pd.DataFrame): Preprocessed data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test dataframes
        """
        from config.model_config import get_preprocessing_params
        
        params = get_preprocessing_params()
        test_size = params.get("test_size", 0.2)
        val_size = params.get("validation_size", 0.2)
        
        # Time-based splitting for time series
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        train = df.iloc[:val_idx].copy()
        val = df.iloc[val_idx:test_idx].copy()
        test = df.iloc[test_idx:].copy()
        
        logger.info(f"Split data: train={train.shape}, val={val.shape}, test={test.shape}")
        
        return train, val, test