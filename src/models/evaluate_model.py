"""
Module for evaluating time series forecasting models.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
import tensorflow as tf
from ..utils.utils import calculate_metrics

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate forecast accuracy using multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    return calculate_metrics(y_true, y_pred)


def plot_forecast_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                           title: str = "Forecast vs Actual",
                           save_path: Optional[str] = None) -> None:
    """
    Plot forecast values against actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure to (if None, figure is displayed)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Forecast', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()


def load_model_and_predict(model_path: str, X: np.ndarray) -> np.ndarray:
    """
    Load a model and make predictions.
    
    Args:
        model_path: Path to the saved model
        X: Input features for prediction
        
    Returns:
        np.ndarray: Model predictions
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Make predictions
        y_pred = model.predict(X)
        return y_pred
    except Exception as e:
        logger.error(f"Error loading model or making predictions: {e}")
        return np.array([])


def main():
    """Run model evaluation pipeline."""
    # Load test data
    test_features_path = os.path.join('data', 'features', 'test_features.parquet')
    try:
        test_df = pd.read_parquet(test_features_path)
        logger.info(f"Loaded test data with shape {test_df.shape}")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # Load the model
    model_path = os.path.join('models', 'final_model.h5')
    
    # Extract features and target
    target_col = 'sales'  # Adjust based on your dataset
    y_true = test_df[target_col].values
    
    # Prepare test data for prediction
    # This should match the same preprocessing used during training
    # Assuming same TimeSeriesGenerator logic from train_model.py
    lookback = 30
    X_test = []
    for i in range(len(y_true) - lookback):
        X_test.append(y_true[i:i+lookback])
    
    X_test = np.array(X_test).reshape(-1, lookback, 1)
    
    # Make predictions
    y_pred = load_model_and_predict(model_path, X_test)
    
    # If model predicts multiple steps, take first step for comparison
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred[:, 0]
    
    # Ensure y_true matches y_pred in length
    y_true = y_true[lookback:lookback+len(y_pred)]
    
    # Evaluate predictions
    metrics = evaluate_forecast(y_true, y_pred)
    logger.info("Forecast evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Plot results
    reports_dir = os.path.join('reports', 'figures')
    os.makedirs(reports_dir, exist_ok=True)
    
    plot_forecast_vs_actual(
        y_true,
        y_pred,
        title="Forecast vs Actual Sales",
        save_path=os.path.join(reports_dir, 'forecast_vs_actual.png')
    )


if __name__ == '__main__':
    main()