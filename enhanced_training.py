import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    logger.info("📁 Created necessary directories")

def generate_health_data():
    """Generate synthetic health data"""
    logger.info("📊 Generating synthetic health data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic health data
    ages = np.random.randint(18, 80, n_samples)
    
    # Heart rate depends on age and fitness
    base_hr = 220 - ages  # Max HR formula
    resting_hr = np.random.normal(70, 12, n_samples) + (ages - 40) * 0.2
    heart_rate = np.clip(resting_hr, 45, 120)
    
    # Steps depend on age and lifestyle
    base_steps = np.random.normal(8000, 3000, n_samples) - (ages - 30) * 20
    steps_daily = np.clip(base_steps, 1000, 25000)
    
    # Sleep patterns
    sleep_hours = np.random.normal(7.5, 1.2, n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 11)
    
    # Create realistic risk patterns
    hr_risk = np.where(heart_rate > 100, 1.0, np.where(heart_rate < 60, 0.7, 0.0))
    activity_risk = np.where(steps_daily < 5000, 0.8, np.where(steps_daily < 3000, 1.0, 0.0))
    sleep_risk = np.where(sleep_hours < 6, 0.7, np.where(sleep_hours < 5, 1.0, 0.0))
    age_risk = np.where(ages > 65, 0.6, np.where(ages > 75, 0.8, 0.0))
    
    # Weighted risk score
    weights = {'hr': 0.35, 'activity': 0.25, 'sleep': 0.25, 'age': 0.15}
    risk_score = (
        hr_risk * weights['hr'] + 
        activity_risk * weights['activity'] + 
        sleep_risk * weights['sleep'] + 
        age_risk * weights['age']
    )
    
    # Add some randomness but keep it realistic
    noise = np.random.normal(0, 0.1, len(risk_score))
    risk_score = np.clip(risk_score + noise, 0, 1)
    
    # Binary classification with threshold
    high_risk = (risk_score > 0.4).astype(int)
    
    df = pd.DataFrame({
        'heart_rate': heart_rate.round(0),
        'steps_daily': steps_daily.round(0),
        'sleep_hours': sleep_hours.round(1),
        'age': ages,
        'risk_score': risk_score,
        'high_risk': high_risk
    })
    
    logger.info(f"✅ Generated {len(df)} health records")
    logger.info(f"   High risk cases: {df['high_risk'].sum()} ({df['high_risk'].mean():.1%})")
    
    return df

def validate_data(df):
    """Validate data quality"""
    logger.info("🔍 Validating data quality...")
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        logger.warning("⚠️ Missing values detected")
    
    # Check for outliers
    outliers = {
        'heart_rate': (df['heart_rate'] < 40) | (df['heart_rate'] > 150),
        'steps_daily': (df['steps_daily'] < 0) | (df['steps_daily'] > 30000),
        'sleep_hours': (df['sleep_hours'] < 3) | (df['sleep_hours'] > 12)
    }
    
    for feature, mask in outliers.items():
        if mask.sum() > 0:
            logger.warning(f"⚠️ {mask.sum()} outliers in {feature}")
    
    # Check class balance
    class_ratio = df['high_risk'].mean()
    if class_ratio < 0.1 or class_ratio > 0.9:
        logger.warning(f"⚠️ Imbalanced classes: {class_ratio:.1%} high risk")
    
    logger.info("✅ Data validation completed")

def prepare_features(df):
    """Prepare features for ML"""
    logger.info("🔧 Feature engineering...")
    
    feature_cols = ['heart_rate', 'steps_daily', 'sleep_hours', 'age']
    X = df[feature_cols].copy()
    y = df['high_risk'].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("💾 Scaler saved to models/scaler.pkl")
    
    return X_scaled, y, feature_cols

def train_model_with_mlflow(X, y, feature_names):
    """Train model with MLflow tracking"""
    logger.info("🤖 Training Random Forest model with MLflow...")
    
    # Set up MLflow
    mlflow.set_experiment("health_risk_enhanced")
    
    with mlflow.start_run():
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"   Training set: {len(X_train)} samples")
        logger.info(f"   Test set: {len(X_test)} samples")
        
        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'specificity': calculate_specificity(y_test, y_pred),
            'training_time': training_time
        }
        
        # Log parameters to MLflow
        mlflow.log_params({
            'n_estimators': 100,
            'max_depth': 10,
            'test_size': 0.2,
            'random_state': 42,
            'n_features': len(feature_names)
        })
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="health_risk_classifier"
        )
        
        # Save model locally
        joblib.dump(model, 'models/health_model.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'features': feature_names,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Display results
        logger.info("✅ Model training completed!")
        logger.info("\n📊 Performance Metrics:")
        for metric, value in metrics.items():
            if metric != 'training_time':
                logger.info(f"   {metric.capitalize()}: {value:.3f}")
        
        logger.info(f"   Training time: {metrics['training_time']:.2f}s")
        
        # Feature importance
        feature_importance = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        
        logger.info("\n🎯 Feature Importance:")
        for feature, importance in feature_importance:
            logger.info(f"   {feature}: {importance:.3f}")
        
        return model, metrics

def calculate_specificity(y_true, y_pred):
    """Calculate specificity (true negative rate)"""
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting Enhanced Health Risk MLOps Pipeline")
    
    try:
        # Step 1: Create directories
        create_directories()
        
        # Step 2: Generate and validate data
        logger.info("📊 Step 1: Data Generation & Validation")
        df = generate_health_data()
        validate_data(df)
        
        # Step 3: Feature engineering
        logger.info("🔧 Step 2: Feature Engineering")
        X, y, feature_names = prepare_features(df)
        
        # Step 4: Model training with MLflow
        logger.info("🤖 Step 3: Model Training & Evaluation")
        model, metrics = train_model_with_mlflow(X, y, feature_names)
        
        # Step 5: Pipeline completion
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"   Final accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"   Model saved to: models/health_model.pkl")
        
        print("\n" + "="*60)
        print("🎉 ENHANCED HEALTH MLOPS PIPELINE COMPLETED!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. 🚀 Start API Server:")
        print("   python enhanced_api.py")
        print("\n2. 🌐 Launch Dashboard:")
        print("   streamlit run enhanced_dashboard.py")
        print("\n3. 📊 View MLflow Experiments:")
        print("   mlflow ui")
        print("\n4. 🔍 Test API:")
        print("   curl -X POST http://localhost:8000/predict \\")
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"heart_rate":75,"steps_daily":8000,"sleep_hours":7.5,"age":35}\'')
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Training completed successfully! Ready for deployment.")
    else:
        print("\n❌ Training failed. Check logs for details.")