import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Automated model selection logic based on dataset characteristics
    """
    
    @staticmethod
    def select_model(df: pd.DataFrame) -> str:
        """
        Select the best model type based on the dataset size and characteristics
        
        Args:
            df: The processed dataframe with features
            
        Returns:
            model_type: String identifier for the recommended model
        """
        data_size = len(df)
        
        # Simple heuristic for model selection
        if data_size < 500:
            # Small dataset: use a simpler model to avoid overfitting
            model_type = "logistic"
            reason = "Small dataset size (< 500 samples). Logistic Regression is used for stability."
        elif data_size < 1500:
            # Medium dataset: Random Forest handles non-linear relationships well
            model_type = "random_forest"
            reason = "Medium dataset size (500-1500 samples). Random Forest is used for robust ensemble learning."
        else:
            # Large dataset: XGBoost thrives with more data
            model_type = "xgboost"
            reason = "Large dataset size (> 1500 samples). XGBoost is used for high-performance gradient boosting."
            
        logger.info(f"Automated Model Selection: Chosen {model_type} because: {reason}")
        
        return model_type, reason
