import os
import argparse
import logging

from src.data.data_loader import DataLoader
from src.data.preprocessing import TimeSeriesPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import SalesForecaster
from src.models.predict_model import SalesPredictor
from src.utils.logger import get_logger
from src.utils.visualization import (
    plot_sales_history,
    plot_prediction_vs_actual,
    plot_future_forecast,
    plot_training_history
)

logger = get_logger("main")

def train_pipeline(data_path=None):
    """Run the complete training pipeline"""
    logger.info("Starting training pipeline")
    
    # Load data
    loader = DataLoader()
    df = loader.load_raw_data() if data_path is None else loader.load_raw_data(data_path)
    
    # Validate data
    if not loader.validate_data(df):
        logger.error("Data validation failed. Please check your input data.")
        return
        
    # Preprocess data
    preprocessor = TimeSeriesPreprocessor()
    preprocessed_df = preprocessor.preprocess(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    from config.model_config import get_preprocessing_params
    params = get_preprocessing_params()
    target_col = params.get("target_column", "vendas")
    
    featured_df = engineer.add_all_features(preprocessed_df, target_col)
    
    # Prepare data for training
    training_data = preprocessor.prepare_data_for_training(featured_df)
    
    # Build and train model
    forecaster = SalesForecaster()
    train_results = forecaster.train(training_data)
    
    # Plot training history
    plot_training_history(
        train_results["history"],
        save_path="reports/figures/training_history.png"
    )
    
    # Evaluate on test data
    predictor = SalesPredictor()
    test_metrics = predictor.evaluate(
        training_data["X_test"],
        training_data["y_test"]
    )
    
    # Generate predictions on test data
    y_pred = predictor.model.predict(training_data["X_test"])
    y_pred = predictor.scaler.inverse_transform(y_pred).flatten()
    y_true = predictor.scaler.inverse_transform(
        training_data["y_test"].reshape(-1, 1)
    ).flatten()
    
    # Plot predictions vs actual
    plot_prediction_vs_actual(
        y_true, y_pred,
        title="Previsão vs. Real (Conjunto de Teste)",
        save_path="reports/figures/test_predictions.png"
    )
    
    # Generate future forecast
    historical = featured_df[target_col].values
    forecast = predictor.predict_future(historical, steps=30)
    
    # Plot future forecast
    plot_future_forecast(
        historical, forecast,
        title="Previsão de Vendas para os Próximos 30 Dias",
        save_path="reports/figures/future_forecast.png"
    )
    
    logger.info("Training pipeline completed successfully")

def predict_pipeline(data_path=None, steps=30):
    """Run prediction pipeline on new data"""
    logger.info("Starting prediction pipeline")
    
    # Load data
    loader = DataLoader()
    df = loader.load_raw_data() if data_path is None else loader.load_raw_data(data_path)
    
    # Load predictor
    predictor = SalesPredictor()
    
    from config.model_config import get_preprocessing_params
    params = get_preprocessing_params()
    target_col = params.get("target_column", "vendas")
    
    # Generate forecast
    historical = df[target_col].values
    forecast = predictor.predict_future(historical, steps=steps)
    
    # Plot forecast
    plot_future_forecast(
        historical, forecast,
        title=f"Previsão de Vendas para os Próximos {steps} Dias",
        save_path="reports/figures/new_forecast.png"
    )
    
    logger.info("Prediction pipeline completed successfully")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Sales Forecasting Pipeline")
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                      help='Pipeline mode: train a new model or predict with existing model')
    parser.add_argument('--data', type=str, help='Path to input data file')
    parser.add_argument('--steps', type=int, default=30, help='Number of steps to forecast')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_pipeline(args.data)
    else:
        predict_pipeline(args.data, args.steps)