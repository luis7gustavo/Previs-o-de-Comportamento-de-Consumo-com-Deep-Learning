import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Union
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)

class SalesPredictor:
    """Class for making predictions using the trained model"""
    
    def __init__(self, model_path: str = "models/sales_forecaster.h5", 
                 scaler_path: str = "models/scaler.pkl"):
        """
        Initialize predictor with trained model
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
        """
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        from config.model_config import get_preprocessing_params
        self.params = get_preprocessing_params()
        self.seq_length = self.params.get("sequence_length", 30)
        
        logger.info(f"Loaded model from {model_path} and scaler from {scaler_path}")
        
    def prepare_single_sequence(self, input_data: np.ndarray) -> np.ndarray:
        """
        Prepare a single sequence for prediction
        
        Args:
            input_data (np.ndarray): Input time series data
            
        Returns:
            np.ndarray: Processed input sequence
        """
        # Ensure we have the right amount of data
        if len(input_data) < self.seq_length:
            raise ValueError(f"Input data must contain at least {self.seq_length} points")
            
        # Use the most recent data points
        recent_data = input_data[-self.seq_length:]
        
        # Scale the data
        scaled_data = self.scaler.transform(recent_data.reshape(-1, 1)).flatten()
        
        # Reshape for model input [1, seq_length, 1]
        model_input = scaled_data.reshape(1, self.seq_length, 1)
        
        return model_input
        
    def predict(self, input_data: np.ndarray) -> float:
        """
        Generate a single prediction using the model
        
        Args:
            input_data (np.ndarray): Input time series data
            
        Returns:
            float: Predicted value
        """
        # Prepare input sequence
        model_input = self.prepare_single_sequence(input_data)
        
        # Make prediction
        scaled_prediction = self.model.predict(model_input)[0][0]
        
        # Inverse transform to get original scale
        prediction = self.scaler.inverse_transform(
            np.array([[scaled_prediction]])
        )[0][0]
        
        logger.info(f"Generated prediction: {prediction}")
        return prediction
        
    def predict_future(self, historical_data: np.ndarray, steps: int = 30) -> np.ndarray:
        """
        Generate multiple future predictions
        
        Args:
            historical_data (np.ndarray): Historical time series data
            steps (int): Number of future steps to predict
            
        Returns:
            np.ndarray: Array of predictions
        """
        # Start with historical data
        working_data = historical_data.copy()
        predictions = []
        
        # Generate predictions one by one
        for _ in range(steps):
            # Prepare sequence
            model_input = self.prepare_single_sequence(working_data)
            
            # Make prediction
            scaled_pred = self.model.predict(model_input)[0][0]
            
            # Inverse transform
            pred = self.scaler.inverse_transform(
                np.array([[scaled_pred]])
            )[0][0]
            
            # Store prediction
            predictions.append(pred)
            
            # Update working data by appending the new prediction
            working_data = np.append(working_data, pred)
            
        logger.info(f"Generated {steps} future predictions")
        return np.array(predictions)
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Generate predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_true = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape
        }
        
        logger.info(f"Model evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        return metrics