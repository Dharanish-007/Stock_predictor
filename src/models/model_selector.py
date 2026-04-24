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
        
        # Heuristic for automated model selection
        if data_size < 300:
            model_type = "logistic"
            reason = f"Small dataset ({data_size} samples). Logistic Regression used for stability."
        elif data_size < 800:
            model_type = "svm"
            reason = f"Small-medium dataset ({data_size} samples). SVM used for high-dimensional boundary mapping."
        elif data_size < 1500:
            model_type = "random_forest"
            reason = f"Medium dataset ({data_size} samples). Random Forest used for robust ensemble learning."
        elif data_size < 3000:
            model_type = "neural_network"
            reason = f"Medium-large dataset ({data_size} samples). Neural Network used for complex pattern recognition."
        else:
            model_type = "xgboost"
            reason = f"Large dataset ({data_size} samples). XGBoost used for high-performance gradient boosting."
            
        logger.info(f"Automated Model Selection: Chosen {model_type} because: {reason}")
        
        return model_type, reason
