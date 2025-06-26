import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthDataPipeline:
    """Enhanced data pipeline with validation and monitoring"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self):
        """Generate synthetic health data with realistic patterns"""
        logger.info("🔄 Generating synthetic health data...")
        
        np.random.seed(self.config.random_state)
        n = self.config.n_samples
        
        # Generate realistic health data
        ages = np.random.randint(18, 80, n)
        
        # Heart rate depends on age and fitness
        base_hr = 220 - ages  # Max HR formula
        resting_hr = np.random.normal(70, 12, n) + (ages - 40) * 0.2
        heart_rate = np.clip(resting_hr, 45, 120)
        
        # Steps depend on age and lifestyle
        base_steps = np.random.normal(8000, 3000, n) - (ages - 30) * 20
        steps_daily = np.clip(base_steps, 1000, 25000)
        
        # Sleep patterns
        sleep_hours = np.random.normal(7.5, 1.2, n)
        sleep_hours = np.clip(sleep_hours, 4, 11)
        
        # Create realistic risk patterns
        risk_factors = self._calculate_risk_factors(heart_rate, steps_daily, sleep_hours, ages)
        
        df = pd.DataFrame({
            'heart_rate': heart_rate.round(0),
            'steps_daily': steps_daily.round(0),
            'sleep_hours': sleep_hours.round(1),
            'age': ages,
            'risk_score': risk_factors['risk_score'],
            'high_risk': risk_factors['high_risk']
        })
        
        # Data validation
        self._validate_data(df)
        
        logger.info(f"✅ Generated {len(df)} records")
        logger.info(f"   High risk: {df['high_risk'].sum()} ({df['high_risk'].mean():.1%})")
        
        return df
    
    def _calculate_risk_factors(self, hr, steps, sleep, age):
        """Calculate health risk with medical knowledge"""
        
        # Normalized risk factors (0-1)
        hr_risk = np.where(hr > 100, 1.0, np.where(hr < 60, 0.7, 0.0))
        activity_risk = np.where(steps < 5000, 0.8, np.where(steps < 3000, 1.0, 0.0))
        sleep_risk = np.where(sleep < 6, 0.7, np.where(sleep < 5, 1.0, 0.0))
        age_risk = np.where(age > 65, 0.6, np.where(age > 75, 0.8, 0.0))
        
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
        
        return {'risk_score': risk_score, 'high_risk': high_risk}
    
    def _validate_data(self, df):
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
    
    def prepare_features(self, df, fit_scaler=True):
        """Prepare features for ML"""
        feature_cols = self.config.features
        X = df[feature_cols].copy()
        y = df['high_risk'].copy()
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            # Save scaler
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.scaler, 'models/scaler.pkl')
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        return X_scaled, y
    
    def split_data(self, X, y):
        """Split data with stratification"""
        return train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
