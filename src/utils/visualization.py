import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import os

def setup_plot_style() -> None:
    """Set up matplotlib style for consistent visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    sns.set_palette("deep")

def plot_sales_history(df: pd.DataFrame, 
                      target_col: str = 'vendas', 
                      date_col: str = None,
                      title: str = 'Histórico de Vendas Online',
                      save_path: Optional[str] = None) -> None:
    """
    Plot historical sales data
    
    Args:
        df (pd.DataFrame): Sales dataframe
        target_col (str): Column containing sales values
        date_col (str, optional): Column containing dates
        title (str): Plot title
        save_path (str, optional): Path to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Check if date is index or column
    if date_col is not None:
        x = df[date_col]
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            x = df.index
        else:
            x = range(len(df))
    
    ax.plot(x, df[target_col], linewidth=2, color='#1f77b4')
    
    # Add rolling average
    if len(df) > 30:
        rolling_avg = df[target_col].rolling(window=30).mean()
        ax.plot(x, rolling_avg, linewidth=2, color='#ff7f0e', 
                label='Média Móvel (30 dias)')
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Vendas', fontsize=14)
    ax.set_xlabel('Data', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_prediction_vs_actual(actual: np.ndarray, 
                             predicted: np.ndarray,
                             dates: Optional[pd.DatetimeIndex] = None,
                             title: str = 'Previsão vs. Real',
                             save_path: Optional[str] = None) -> None:
    """
    Plot predicted vs actual values
    
    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        dates (pd.DatetimeIndex, optional): Dates for x-axis
        title (str): Plot title
        save_path (str, optional): Path to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = dates if dates is not None else range(len(actual))
    
    ax.plot(x, actual, 'b-', linewidth=2, label='Valores Reais')
    ax.plot(x, predicted, 'r--', linewidth=2, label='Valores Previstos')
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Vendas', fontsize=14)
    ax.set_xlabel('Data' if dates is not None else 'Período', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add metrics
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    metrics_text = f'MAE: {mae:.2f}\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_future_forecast(historical: np.ndarray, 
                        forecast: np.ndarray,
                        historical_dates: Optional[pd.DatetimeIndex] = None,
                        forecast_dates: Optional[pd.DatetimeIndex] = None,
                        title: str = 'Previsão de Vendas Futuras',
                        save_path: Optional[str] = None) -> None:
    """
    Plot historical data with future forecast
    
    Args:
        historical (np.ndarray): Historical values
        forecast (np.ndarray): Forecasted values
        historical_dates (pd.DatetimeIndex, optional): Dates for historical data
        forecast_dates (pd.DatetimeIndex, optional): Dates for forecast
        title (str): Plot title
        save_path (str, optional): Path to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define x values
    if historical_dates is not None and forecast_dates is not None:
        x_hist = historical_dates
        x_fore = forecast_dates
    else:
        x_hist = range(len(historical))
        x_fore = range(len(historical), len(historical) + len(forecast))
    
    # Plot historical data
    ax.plot(x_hist, historical, 'b-', linewidth=2, label='Dados Históricos')
    
    # Plot forecast
    ax.plot(x_fore, forecast, 'r-', linewidth=2, label='Previsão')
    
    # Add confidence intervals (using a simple method)
    forecast_std = np.std(historical[-len(forecast):]) if len(forecast) <= len(historical) else np.std(historical)
    ax.fill_between(x_fore, 
                   forecast - 1.96 * forecast_std,
                   forecast + 1.96 * forecast_std,
                   color='r', alpha=0.2, label='Intervalo de Confiança (95%)')
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Vendas', fontsize=14)
    ax.set_xlabel('Data' if historical_dates is not None else 'Período', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add vertical line to separate historical and forecast
    if historical_dates is not None and forecast_dates is not None:
        ax.axvline(x=historical_dates[-1], color='k', linestyle='--', alpha=0.3)
    else:
        ax.axvline(x=len(historical)-0.5, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_training_history(history: Dict[str, List[float]],
                         title: str = 'Histórico de Treinamento',
                         save_path: Optional[str] = None) -> None:
    """
    Plot model training history
    
    Args:
        history (Dict[str, List[float]]): Training history dictionary
        title (str): Plot title
        save_path (str, optional): Path to save plot
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot loss
    ax1.plot(history['loss'], label='Treino', linewidth=2)
    ax1.plot(history['val_loss'], label='Validação', linewidth=2)
    ax1.set_title(f'{title} - Perda', fontsize=16)
    ax1.set_ylabel('Erro Quadrático Médio (MSE)', fontsize=14)
    ax1.set_xlabel('Épocas', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE if available
    if 'mae' in history and 'val_mae' in history:
        ax2.plot(history['mae'], label='Treino', linewidth=2)
        ax2.plot(history['val_mae'], label='Validação', linewidth=2)
        ax2.set_title(f'{title} - Erro Absoluto Médio', fontsize=16)
        ax2.set_ylabel('MAE', fontsize=14)
        ax2.set_xlabel('Épocas', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()