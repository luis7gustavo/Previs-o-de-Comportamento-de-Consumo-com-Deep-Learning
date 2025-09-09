import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesPreprocessor:
    """Class for preprocessing time series data for deep learning models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration"""
        from config.model_config import load_config, get_preprocessing_params
        self.config = load_config(config_path)
        self.params = get_preprocessing_params()
        self.scaler = None
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform standard preprocessing on sales data
        
        Args:
            df (pd.DataFrame): Raw sales data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        df = df.copy()
        date_col = self.params.get("date_column", "data")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Set date as index
        df.set_index(date_col, inplace=True)
        
        # Fill missing values if any
        if df.isnull().sum().any():
            logger.info("Filling missing values")
            df = df.interpolate(method="time")
        
        # Save preprocessed data
        output_path = self.params.get("processed_data", "data/processed/vendas_preprocessed.csv")
        df.to_csv(output_path)
        logger.info(f"Saved preprocessed data to {output_path}")
        
        return df
    
    def create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for time series model
        
        Args:
            data (np.ndarray): Time series data
            seq_length (int): Sequence length for input
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def scale_data(self, train_data: np.ndarray, val_data: np.ndarray = None, 
                  test_data: np.ndarray = None) -> Tuple:
        """
        Scale time series data using StandardScaler
        
        Args:
            train_data (np.ndarray): Training data
            val_data (np.ndarray, optional): Validation data
            test_data (np.ndarray, optional): Test data
            
        Returns:
            Tuple: Scaled train, validation, and test data
        """
        # Initialize and fit scaler on training data only
        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        
        # Transform validation and test data if provided
        val_scaled = None
        test_scaled = None
        
        if val_data is not None:
            val_scaled = self.scaler.transform(val_data.reshape(-1, 1)).flatten()
        
        if test_data is not None:
            test_scaled = self.scaler.transform(test_data.reshape(-1, 1)).flatten()
        
        # Save the scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        logger.info("Saved scaler to models/scaler.pkl")
        
        return train_scaled, val_scaled, test_scaled
    
    def prepare_data_for_training(self, df: pd.DataFrame) -> Dict:
        """
        Prepare data for model training
        
        Args:
            df (pd.DataFrame): Preprocessed data
            
        Returns:
            Dict: Dictionary containing train, val, test sequences
        """
        from src.data.data_loader import DataLoader
        
        # Extract params
        target_col = self.params.get("target_column", "vendas")
        seq_length = self.params.get("sequence_length", 30)
        
        # Split data
        loader = DataLoader()
        train_df, val_df, test_df = loader.split_data(df)
        
        # Extract target series
        train_series = train_df[target_col].values
        val_series = val_df[target_col].values
        test_series = test_df[target_col].values
        
        # Scale data
        train_scaled, val_scaled, test_scaled = self.scale_data(
            train_series, val_series, test_series
        )
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, seq_length)
        X_val, y_val = self.create_sequences(val_scaled, seq_length)
        X_test, y_test = self.create_sequences(test_scaled, seq_length)
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        logger.info(f"Prepared sequences: X_train={X_train.shape}, y_train={y_train.shape}")
        
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "scaler": self.scaler
        }