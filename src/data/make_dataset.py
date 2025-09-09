"""
Module for data acquisition and dataset creation.
"""
import os
import pandas as pd
import logging
from typing import Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def download_data(source_url: str, target_path: str) -> bool:
    """
    Download data from source URL and save to target path.
    
    Args:
        source_url: URL to download data from
        target_path: Local path to save data to
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading data from {source_url}")
        # Implementation of download logic
        return True
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from local file path.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Optional[pd.DataFrame]: Loaded data as DataFrame or None if error
    """
    try:
        logger.info(f"Loading data from {file_path}")
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def split_data(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    For time series data, this uses a chronological split rather than random.
    
    Args:
        df: Input DataFrame
        test_ratio: Proportion of data to use for testing
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    logger.info(f"Splitting data with test ratio {test_ratio}")
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    return train_df, test_df


def main():
    """Run the data processing pipeline."""
    # Example usage
    raw_data_path = os.path.join('data', 'raw', 'sales_data.csv')
    processed_data_dir = os.path.join('data', 'processed')
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Load data
    df = load_data(raw_data_path)
    if df is None:
        return
    
    # Split data
    train_df, test_df = split_data(df)
    
    # Save processed data
    train_df.to_parquet(os.path.join(processed_data_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(processed_data_dir, 'test.parquet'))
    logger.info("Data processing completed successfully")


if __name__ == '__main__':
    main()