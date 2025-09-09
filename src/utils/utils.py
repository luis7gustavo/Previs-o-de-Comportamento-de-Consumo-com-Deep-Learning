"""
Utility functions for the project.
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def save_model(model: Any, model_path: str) -> bool:
    """
    Save model to disk.
    
    Args:
        model: Model object to save
        model_path: Path to save model to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(model_path: str) -> Optional[Any]:
    """
    Load model from disk.
    
    Args:
        model_path: Path to load model from
        
    Returns:
        Optional[Any]: Loaded model or None if error
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str,
                     title: str = "Time Series Plot", figsize: Tuple[int, int] = (12, 6),
                     save_path: Optional[str] = None) -> None:
    """
    Plot time series data.
    
    Args:
        df: DataFrame containing time series data
        date_col: Column name for dates
        value_col: Column name for values
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Path to save figure to (if None, figure is displayed)
    """
    plt.figure(figsize=figsize)
    sns.lineplot(data=df, x=date_col, y=value_col)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()


def calculate_metrics(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> Dict[str, float]:
    """
    Calculate regression metrics between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Coefficient of Determination (R²)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }
    
    return metrics


def setup_directories() -> None:
    """Create standard project directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'data/features',
        'models',
        'notebooks/exploratory',
        'notebooks/results',
        'reports/figures',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Project directories created")