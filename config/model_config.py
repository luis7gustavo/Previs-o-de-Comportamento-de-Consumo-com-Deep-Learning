import os
import yaml

def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
    
def get_model_params():
    """Get model parameters from config file"""
    config = load_config()
    return config.get("model", {})
    
def get_preprocessing_params():
    """Get preprocessing parameters from config file"""
    config = load_config()
    return config.get("preprocessing", {})
    
def get_paths():
    """Get data paths from config file"""
    config = load_config()
    return config.get("paths", {})