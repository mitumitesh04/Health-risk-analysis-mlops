import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class HealthModelTrainer:
    """Enhanced ML trainer with comprehensive evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train model with comprehensive tracking"""
        
        with mlflow.start_run():
            logger.info("🤖 Training Random Forest model...")
            
            # Initialize model
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1  # Use all CPU cores
            )
            
            # Train model
            start_time = datetime.now()
            self.model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Predictions
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            metrics['training_time'] = training_time
            
            # Log everything to MLflow
            self._log_to_mlflow(metrics, X_train.columns.tolist())
            
            # Save model and artifacts
            self._save_model_artifacts(metrics)
            
            # Display results
            self._display_results(metrics, y_test, y_pred)
            
            return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)"""
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _log_to_mlflow(self, metrics, feature_names):
        """Log everything to MLflow"""
        # Log parameters
        mlflow.log_params({
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'test_size': self.config.test_size,
            'random_state': self.config.random_state,
            'n_features': len(feature_names)
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Log model
        mlflow.sklearn.log_model(
            self.model, 
            "model",
            registered_model_name="health_risk_classifier"
        )
    
    def _save_model_artifacts(self, metrics):
        """Save model and metadata locally"""
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, 'models/health_model.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'features': self.config.features,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'random_state': self.config.random_state
            }
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("💾 Model and metadata saved")
    
    def _display_results(self, metrics, y_test, y_pred):
        """Display training results"""
        logger.info("✅ Model training completed!")
        logger.info("\n📊 Performance Metrics:")
        for metric, value in metrics.items():
            if metric != 'training_time':
                logger.info(f"   {metric.capitalize()}: {value:.3f}")
        
        logger.info(f"   Training time: {metrics['training_time']:.2f}s")
        
        # Feature importance
        feature_importance = sorted(
            zip(self.config.features, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        
        logger.info("\n🎯 Feature Importance:")
        for feature, importance in feature_importance:
            logger.info(f"   {feature}: {importance:.3f}")