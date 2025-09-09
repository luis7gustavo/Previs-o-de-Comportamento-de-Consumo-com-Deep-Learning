import pandas as pd
import numpy as np
from typing import List, Union
from datetime import datetime
import holidays

from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """Class for engineering features for time series forecasting"""
    
    def __init__(self):
        """Initialize feature engineering class"""
        self.br_holidays = holidays.Brazil()
        
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features from datetime index
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime index
            
        Returns:
            pd.DataFrame: Dataframe with added time features
        """
        df = df.copy()
        
        # Extract datetime components
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['is_weekend'] = df.index.dayofweek >= 5
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Create cyclical features for day of week and month
        df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek/7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek/7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)
        
        logger.info("Added time-based features")
        return df
        
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add holiday indicators as features
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime index
            
        Returns:
            pd.DataFrame: Dataframe with added holiday features
        """
        df = df.copy()
        
        # Check if date is holiday
        df['is_holiday'] = df.index.map(lambda x: x in self.br_holidays).astype(int)
        
        # Check day before and after holiday
        dates = df.index.to_pydatetime()
        df['is_day_before_holiday'] = [
            (date + pd.Timedelta(days=1)).date() in self.br_holidays
            for date in dates
        ]
        df['is_day_after_holiday'] = [
            (date - pd.Timedelta(days=1)).date() in self.br_holidays
            for date in dates
        ]
        
        logger.info("Added holiday features")
        return df
        
    def add_lag_features(self, df: pd.DataFrame, 
                        target_col: str, 
                        lag_periods: List[int]) -> pd.DataFrame:
        """
        Add lagged values of target as features
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            lag_periods (List[int]): List of lag periods to add
            
        Returns:
            pd.DataFrame: Dataframe with added lag features
        """
        df = df.copy()
        
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        logger.info(f"Added lag features: {lag_periods}")
        return df
        
    def add_rolling_features(self, df: pd.DataFrame, 
                           target_col: str,
                           windows: List[int]) -> pd.DataFrame:
        """
        Add rolling statistics as features
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            windows (List[int]): List of window sizes for rolling calculations
            
        Returns:
            pd.DataFrame: Dataframe with added rolling features
        """
        df = df.copy()
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            
        logger.info(f"Added rolling features with windows: {windows}")
        return df
        
    def add_all_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Apply all feature engineering transformations
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime index
            target_col (str): Target column name
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        df = self.add_time_features(df)
        df = self.add_holiday_features(df)
        df = self.add_lag_features(df, target_col, [1, 7, 14, 28])
        df = self.add_rolling_features(df, target_col, [7, 14, 30])
        
        # Drop rows with NaN values resulting from lag/rolling features
        df_filled = df.fillna(method='bfill')
        
        logger.info(f"Final dataframe shape after feature engineering: {df_filled.shape}")
        return df_filled