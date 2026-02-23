import joblib
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStorage:
    """Handles saving and loading ML models"""
    
    def __init__(self, model_dir="saved_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, model_name, metadata=None):
        """Save model with timestamp and optional metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = self.model_dir / filename
        
        # Save model
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata if provided
        if metadata:
            meta_file = self.model_dir / f"{model_name}_{timestamp}_metadata.txt"
            with open(meta_file, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
        
        return str(filepath)
    
    def load_latest_model(self, model_name):
        """Load the most recent model for a given name"""
        model_files = list(self.model_dir.glob(f"{model_name}_*.pkl"))
        if not model_files:
            logger.warning(f"No model found for {model_name}")
            return None
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"Loading model from {latest_model}")
        return joblib.load(latest_model)
    
    def list_models(self, model_name=None):
        """List all saved models, optionally filtered by name"""
        if model_name:
            models = list(self.model_dir.glob(f"{model_name}_*.pkl"))
        else:
            models = list(self.model_dir.glob("*.pkl"))
        
        return [str(m) for m in models]