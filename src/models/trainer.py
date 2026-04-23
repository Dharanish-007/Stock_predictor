"""
Model Training Module
Professional ML model training with cross-validation and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Professional model trainer with multiple algorithms and hyperparameter tuning
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize model trainer
        
        Args:
            model_type: One of ['logistic', 'random_forest', 'xgboost', 'gradient_boosting']
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.feature_columns = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    def _initialize_model(self):
        """Initialize the selected model with default parameters"""
        models = {
            'logistic': LogisticRegression(
                random_state=Config.MODEL_CONFIG.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=Config.MODEL_CONFIG.n_estimators,
                max_depth=Config.MODEL_CONFIG.max_depth,
                random_state=Config.MODEL_CONFIG.random_state,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=Config.MODEL_CONFIG.n_estimators,
                max_depth=Config.MODEL_CONFIG.max_depth,
                random_state=Config.MODEL_CONFIG.random_state,
                verbosity=0
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=Config.MODEL_CONFIG.n_estimators,
                max_depth=Config.MODEL_CONFIG.max_depth,
                random_state=Config.MODEL_CONFIG.random_state
            )
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
        return models[self.model_type]
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: list) -> tuple:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            
        Returns:
            X, y, feature_columns
        """
        self.feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[self.feature_columns]
        y = df['Target']
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def time_series_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Perform time series split (no random shuffling)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - Config.MODEL_CONFIG.test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Train period: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0)
        }
        
        logger.info(f"Evaluation results: {metrics}")
        
        # Print detailed report
        print("\n" + "="*50)
        print(f"📊 {self.model_type.upper()} MODEL EVALUATION")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall:    {metrics['recall']:.2%}")
        print(f"F1 Score:  {metrics['f1_score']:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['DOWN', 'UP']))
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            print("\n📈 Feature Importance:")
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.2%}")
        
        return metrics
    
    def save_model(self, filename: str = None) -> str:
        """
        Save trained model to disk
        
        Returns:
            Path to saved model
        """
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        filepath = self.models_dir / filename
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def load_model(filepath: str):
        """Load a saved model"""
        data = joblib.load(filepath)
        return data['model'], data['feature_columns']

# Example usage
if __name__ == "__main__":
    # Test the trainer
    import yfinance as yf
    from src.features.technical_indicators import TechnicalIndicators
    
    # Load and prepare data
    df = yf.download("RELIANCE.NS", period="3y", progress=False)
    df = TechnicalIndicators.engineer_features(df)
    
    # Train model
    trainer = ModelTrainer(model_type="random_forest")
    feature_cols = ['MA5', 'MA20', 'RSI', 'Volatility', 'BB_Width']
    X, y = trainer.prepare_data(df, feature_cols)
    X_train, X_test, y_train, y_test = trainer.time_series_split(X, y)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.save_model()