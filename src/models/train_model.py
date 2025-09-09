import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import joblib
from typing import Dict, Any, Tuple

from src.data.data_loader import DataLoader
from src.data.preprocessing import TimeSeriesPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SalesForecaster:
    """Class for building and training time series forecasting models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model with configuration"""
        from config.model_config import load_config, get_model_params
        self.config = load_config(config_path)
        self.model_params = get_model_params()
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape (Tuple[int, int]): Input shape (timesteps, features)
            
        Returns:
            Sequential: Compiled Keras model
        """
        # Extract parameters
        lstm_units = self.model_params.get("lstm_units", 128)
        dense_units = self.model_params.get("dense_units", 64)
        dropout_rate = self.model_params.get("dropout_rate", 0.2)
        learning_rate = self.model_params.get("learning_rate", 0.001)
        
        # Build model
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=lstm_units, 
                       return_sequences=True, 
                       input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=lstm_units//2, 
                       return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(units=dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1))  # Output layer
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"Built model with {model.count_params()} parameters")
        model.summary(print_fn=logger.info)
        
        self.model = model
        return model
    
    def train(self, training_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Train the forecasting model
        
        Args:
            training_data (Dict): Dictionary with X_train, y_train, X_val, y_val
            
        Returns:
            Dict: Training history and metrics
        """
        # Extract data
        X_train = training_data["X_train"]
        y_train = training_data["y_train"]
        X_val = training_data["X_val"] 
        y_val = training_data["y_val"]
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.model_params.get("patience", 15),
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        logger.info("Starting model training")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.model_params.get("epochs", 100),
            batch_size=self.model_params.get("batch_size", 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save history
        self.history = history.history
        
        # Evaluate on validation set
        val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation loss: {val_loss}, MAE: {val_mae}")
        
        # Save model and history
        self.save_model()
        
        return {
            "history": self.history,
            "val_loss": val_loss,
            "val_mae": val_mae
        }
    
    def save_model(self, path: str = "models") -> None:
        """
        Save the trained model and training history
        
        Args:
            path (str): Directory to save model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(path, "sales_forecaster.h5")
        self.model.save(model_path)
        
        # Save history
        if self.history:
            history_path = os.path.join(path, "training_history.pkl")
            joblib.dump(self.history, history_path)
            
        logger.info(f"Model and history saved to {path}")
        
    def load_model(self, model_path: str = "models/sales_forecaster.h5") -> Sequential:
        """
        Load a trained model
        
        Args:
            model_path (str): Path to saved model
            
        Returns:
            Sequential: Loaded Keras model
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return self.model