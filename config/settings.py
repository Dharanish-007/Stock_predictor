"""
Configuration settings for the Stock Predictor project
Environment: Production
"""

from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Model hyperparameters"""
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    test_size: float = 0.2
    
@dataclass
class FeatureConfig:
    """Features used for training"""
    technical_indicators: List[str] = None
    
    def __post_init__(self):
        self.technical_indicators = [
            'MA5',
            'MA20', 
            'RSI',
            'Volatility',
            'BB_Upper',
            'BB_Lower'
        ]

@dataclass
class DataConfig:
    """Data collection settings"""
    default_period: str = "5y"
    default_interval: str = "1d"
    cache_data: bool = True
    
class ProductionConfig:
    """Production environment configuration"""
    MODEL_CONFIG = ModelConfig()
    FEATURE_CONFIG = FeatureConfig()
    DATA_CONFIG = DataConfig()
    DEBUG = False
    API_TIMEOUT = 30

class DevelopmentConfig:
    """Development environment configuration"""
    MODEL_CONFIG = ModelConfig()
    FEATURE_CONFIG = FeatureConfig()
    DATA_CONFIG = DataConfig()
    DEBUG = True
    API_TIMEOUT = 60

# Select environment
Config = ProductionConfig()  # Change to DevelopmentConfig() for dev