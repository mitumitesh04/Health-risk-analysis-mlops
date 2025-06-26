import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "RandomForestClassifier"
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    test_size: float = 0.2

@dataclass
class DataConfig:
    """Data configuration"""
    n_samples: int = 1000
    random_state: int = 42
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['heart_rate', 'steps_daily', 'sleep_hours', 'age']

@dataclass
class MLOpsConfig:
    """MLOps configuration"""
    experiment_name: str = "health_risk_v2"
    model_registry: str = "models"
    tracking_uri: str = "sqlite:///mlflow.db"
    enable_model_validation: bool = True

# Global config
CONFIG = {
    'model': ModelConfig(),
    'data': DataConfig(),
    'mlops': MLOpsConfig()
}