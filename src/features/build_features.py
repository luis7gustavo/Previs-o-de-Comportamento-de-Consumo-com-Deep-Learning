"""
Feature engineering for time series forecasting.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import List, Optional

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def add_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Add time-based features from date column.
    
    Args:
        df: Input DataFrame
        date_column: Name of date column
        
    Returns:
        pd.DataFrame: DataFrame with additional time features
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract time components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    logger.info("Added time features based on date column")
    return df


def add_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    """
    Add lag features for a specific column.
    
    Args:
        df: Input DataFrame
        column: Target column to create lags for
        lags: List of lag periods
        
    Returns:
        pd.DataFrame: DataFrame with additional lag features
    """
    df = df.copy()
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
    logger.info(f"Added lag features for column '{column}' with lags {lags}")
    return df


def add_rolling_features(df: pd.DataFrame, column: str, 
                         windows: List[int]) -> pd.DataFrame:
    """
    Add rolling window features (mean, std, min, max) for a column.
    
    Args:
        df: Input DataFrame
        column: Target column for rolling features
        windows: List of window sizes
        
    Returns:
        pd.DataFrame: DataFrame with additional rolling features
    """
    df = df.copy()
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
        df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
        df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
    
    logger.info(f"Added rolling features for column '{column}' with windows {windows}")
    return df


def add_holiday_features(df: pd.DataFrame, date_column: str,
                         country_code: str = 'BR') -> pd.DataFrame:
    """
    Add holiday indicator features.
    
    Args:
        df: Input DataFrame
        date_column: Name of date column
        country_code: Country code for holidays (default: Brazil)
        
    Returns:
        pd.DataFrame: DataFrame with holiday features
    """
    try:
        from holidays import country_holidays
        
        df = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
            
        # Get holidays for the date range
        start_year = df[date_column].min().year
        end_year = df[date_column].max().year
        holidays_dict = country_holidays(country_code, years=range(start_year, end_year + 1))
        
        # Create holiday feature
        df['is_holiday'] = df[date_column].isin(holidays_dict).astype(int)
        
        logger.info(f"Added holiday features for {country_code}")
        return df
    except ImportError:
        logger.warning("holidays package not installed. Skipping holiday features.")
        df['is_holiday'] = 0
        return df


def normalize_features(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize numerical features to range [0, 1].
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from normalization
        
    Returns:
        pd.DataFrame: DataFrame with normalized features
    """
    df = df.copy()
    exclude_cols = exclude_cols or []
    
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Check if min != max to avoid division by zero
            if min_val != max_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    
    logger.info("Normalized numerical features")
    return df


def main():
    """Run feature engineering pipeline."""
    # Load processed data
    train_path = os.path.join('data', 'processed', 'train.parquet')
    test_path = os.path.join('data', 'processed', 'test.parquet')
    
    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        logger.info(f"Loaded training data with shape {train_df.shape}")
        logger.info(f"Loaded test data with shape {test_df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Apply feature engineering
    date_col = 'date'  # Adjust based on your dataset
    target_col = 'sales'  # Adjust based on your dataset
    
    # Time features
    train_df = add_time_features(train_df, date_col)
    test_df = add_time_features(test_df, date_col)
    
    # Lag features
    train_df = add_lag_features(train_df, target_col, lags=[1, 7, 14, 30])
    test_df = add_lag_features(test_df, target_col, lags=[1, 7, 14, 30])
    
    # Rolling features
    train_df = add_rolling_features(train_df, target_col, windows=[7, 14, 30])
    test_df = add_rolling_features(test_df, target_col, windows=[7, 14, 30])
    
    # Holiday features
    train_df = add_holiday_features(train_df, date_col)
    test_df = add_holiday_features(test_df, date_col)
    
    # Normalize features
    exclude_from_norm = [date_col, target_col, 'is_holiday', 'is_weekend']
    train_df = normalize_features(train_df, exclude_cols=exclude_from_norm)
    test_df = normalize_features(test_df, exclude_cols=exclude_from_norm)
    
    # Save feature-engineered data
    feature_dir = os.path.join('data', 'features')
    os.makedirs(feature_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(feature_dir, 'train_features.parquet'))
    test_df.to_parquet(os.path.join(feature_dir, 'test_features.parquet'))
    logger.info("Feature engineering completed successfully")


if __name__ == '__main__':
    main()