"""
Device Predictor - Machine Learning Model for Device State Forecasting

This module trains Random Forest models to predict device states.

Why Random Forest?
- Handles non-linear patterns well
- Works great with time-series features
- Fast training and prediction
- Built-in feature importance
- Resistant to overfitting

The model learns patterns like:
- "This light turns on between 6-8pm on weekdays"
- "Temperature sensor reads higher in afternoon"
- "Motion sensor active mornings and evenings"
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from utils.feature_engineering import create_feature_vector, get_feature_columns

logger = logging.getLogger(__name__)


class DevicePredictor:
    """
    Predicts future device states using Random Forest classification.
    
    Workflow:
    1. Load historical data
    2. Engineer features
    3. Train Random Forest model
    4. Cache model to disk
    5. Generate predictions for future timepoints
    """
    
    def __init__(self, model_cache_dir: str = "/app/models"):
        """
        Initialize predictor.
        
        Args:
            model_cache_dir: Where to save trained models
        """
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}  # In-memory model cache
        self.scalers = {}  # Feature scalers
        self.metadata = {}  # Model training metadata
        
        logger.info(f"DevicePredictor initialized, cache dir: {self.model_cache_dir}")
    
    def _get_model_path(self, entity_id: str) -> Path:
        """Get file path for cached model"""
        safe_name = entity_id.replace('.', '_').replace(':', '_')
        return self.model_cache_dir / f"{safe_name}.joblib"
    
    def _prepare_training_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with engineered features
        
        Returns:
            X: Feature matrix
            y: Target labels
            feature_names: Column names
        """
        feature_cols = get_feature_columns()
        
        # Ensure all feature columns exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        X = df[feature_cols].values
        y = df['state_numeric'].values  # Target: current state
        
        return X, y, feature_cols
    
    def train(
        self, 
        entity_id: str, 
        historical_data: pd.DataFrame,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train a prediction model for a device.
        
        Args:
            entity_id: Device to train for
            historical_data: DataFrame with 'timestamp' and 'state' columns
            force_retrain: Retrain even if cached model exists
        
        Returns:
            Dict with training results (accuracy, metadata, etc.)
        """
        model_path = self._get_model_path(entity_id)
        
        # Check if model exists and is recent
        if not force_retrain and model_path.exists():
            model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            if model_age < timedelta(days=1):  # Models are valid for 1 day
                logger.info(f"Using cached model for {entity_id} (age: {model_age})")
                self._load_model(entity_id)
                return {
                    'status': 'cached',
                    'model_age_hours': model_age.total_seconds() / 3600,
                    'entity_id': entity_id
                }
        
        logger.info(f"Training model for {entity_id}...")
        
        # Validate input data
        if historical_data.empty:
            return {
                'status': 'error',
                'reason': 'no_data',
                'entity_id': entity_id
            }
        
        if len(historical_data) < 100:
            return {
                'status': 'error',
                'reason': 'insufficient_data',
                'records': len(historical_data),
                'minimum_required': 100,
                'entity_id': entity_id
            }
        
        # Feature engineering
        try:
            df = create_feature_vector(historical_data)
        except Exception as e:
            logger.error(f"Feature engineering failed for {entity_id}: {e}")
            return {
                'status': 'error',
                'reason': 'feature_engineering_failed',
                'error': str(e),
                'entity_id': entity_id
            }
        
        if len(df) < 50:  # After dropping NaN, need minimum records
            return {
                'status': 'error',
                'reason': 'insufficient_data_after_processing',
                'records_after_processing': len(df),
                'entity_id': entity_id
            }
        
        # Prepare training data
        try:
            X, y, feature_names = self._prepare_training_data(df)
        except Exception as e:
            logger.error(f"Data preparation failed for {entity_id}: {e}")
            return {
                'status': 'error',
                'reason': 'data_preparation_failed',
                'error': str(e),
                'entity_id': entity_id
            }
        
        # Split train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features (important for some algorithms, less so for Random Forest)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        # n_estimators: number of trees (more = better, but slower)
        # max_depth: prevent overfitting (limit tree depth)
        # min_samples_split: require minimum samples before splitting
        model = RandomForestClassifier(
            n_estimators=100,  # 100 trees is a good balance
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=10,  # Need 10+ samples to split
            random_state=42,  # Reproducible results
            n_jobs=-1  # Use all CPU cores
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        # Get feature importances (which features matter most?)
        feature_importance = dict(zip(
            feature_names,
            model.feature_importances_
        ))
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        logger.info(f"Model trained for {entity_id}: "
                   f"train_acc={train_accuracy:.3f}, test_acc={test_accuracy:.3f}")
        
        # Store in memory
        self.models[entity_id] = model
        self.scalers[entity_id] = scaler
        self.metadata[entity_id] = {
            'trained_at': datetime.now().isoformat(),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'training_records': len(X),
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'features': feature_names
        }
        
        # Save to disk
        self._save_model(entity_id)
        
        return {
            'status': 'success',
            'entity_id': entity_id,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'training_records': len(X),
            'top_features': list(feature_importance.keys())
        }
    
    def _save_model(self, entity_id: str):
        """Save model, scaler, and metadata to disk"""
        model_path = self._get_model_path(entity_id)
        
        joblib.dump({
            'model': self.models[entity_id],
            'scaler': self.scalers[entity_id],
            'metadata': self.metadata[entity_id]
        }, model_path)
        
        logger.info(f"Model saved for {entity_id}: {model_path}")
    
    def _load_model(self, entity_id: str) -> bool:
        """Load model from disk"""
        model_path = self._get_model_path(entity_id)
        
        if not model_path.exists():
            logger.warning(f"No cached model for {entity_id}")
            return False
        
        try:
            data = joblib.load(model_path)
            self.models[entity_id] = data['model']
            self.scalers[entity_id] = data['scaler']
            self.metadata[entity_id] = data['metadata']
            logger.info(f"Model loaded for {entity_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model for {entity_id}: {e}")
            return False
    
    def predict(
        self,
        entity_id: str,
        hours_ahead: int = 24,
        current_state: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for next N hours.
        
        Args:
            entity_id: Device to predict
            hours_ahead: How many hours to forecast
            current_state: Current device state (optional, used for lag features)
        
        Returns:
            List of predictions, one per hour:
            [
                {
                    'timestamp': '2026-01-09T15:00:00',
                    'hour': 15,
                    'predicted_state': 1,  # 0 or 1
                    'predicted_state_label': 'on',  # or 'off'
                    'probability_on': 0.85,
                    'probability_off': 0.15,
                    'confidence': 0.85  # max probability
                },
                ...
            ]
        """
        # Ensure model is loaded
        if entity_id not in self.models:
            if not self._load_model(entity_id):
                logger.error(f"No model available for {entity_id}")
                return []
        
        model = self.models[entity_id]
        scaler = self.scalers[entity_id]
        
        # Generate future timestamps (hourly predictions)
        now = datetime.now()
        future_timestamps = [now + timedelta(hours=h) for h in range(hours_ahead)]
        
        predictions = []
        
        # Initialize lag features
        last_state = current_state if current_state is not None else 0.0
        last_state_2 = 0.0
        rolling_mean = 0.5  # Start with neutral value
        rolling_std = 0.0
        rolling_sum = 0.0
        state_duration = 0
        
        for ts in future_timestamps:
            # Build feature vector for this timestamp
            features = self._create_future_feature_vector(
                ts, 
                last_state, 
                last_state_2,
                rolling_mean,
                rolling_std,
                rolling_sum,
                state_duration
            )
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict
            predicted_state = int(model.predict(features_scaled)[0])
            probabilities = model.predict_proba(features_scaled)[0]
            
            # probabilities is [prob_off, prob_on] for binary classification
            prob_off = float(probabilities[0]) if len(probabilities) > 1 else float(1 - predicted_state)
            prob_on = float(probabilities[1]) if len(probabilities) > 1 else float(predicted_state)
            
            confidence = max(prob_on, prob_off)
            
            predictions.append({
                'timestamp': ts.isoformat(),
                'hour': ts.hour,
                'predicted_state': predicted_state,
                'predicted_state_label': 'on' if predicted_state == 1 else 'off',
                'probability_on': prob_on,
                'probability_off': prob_off,
                'confidence': confidence
            })
            
            # Update lag features for next iteration
            last_state_2 = last_state
            last_state = float(predicted_state)
            
            # Update state duration
            if predicted_state == last_state_2:
                state_duration += 1
            else:
                state_duration = 0
            
            # Simple rolling updates (simplified for forecasting)
            rolling_mean = (rolling_mean * 0.9) + (predicted_state * 0.1)
        
        return predictions
    
    def _create_future_feature_vector(
        self,
        timestamp: datetime,
        prev_state: float,
        prev_state_2: float,
        rolling_mean: float,
        rolling_std: float,
        rolling_sum: float,
        state_duration: int
    ) -> List[float]:
        """
        Create feature vector for a future timestamp.
        
        This matches the feature columns from get_feature_columns()
        """
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = int(day_of_week in [5, 6])
        month = timestamp.month
        
        # Time of day buckets
        is_morning = int(6 <= hour <= 10)
        is_afternoon = int(11 <= hour <= 17)
        is_evening = int(18 <= hour <= 22)
        is_night = int(hour >= 22 or hour < 6)
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Lag features
        state_changed = int(prev_state != prev_state_2)
        
        # Build feature vector in EXACT order of get_feature_columns()
        return [
            hour,
            day_of_week,
            is_weekend,
            month,
            is_morning,
            is_afternoon,
            is_evening,
            is_night,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            prev_state,
            prev_state_2,
            state_changed,
            state_duration,
            rolling_mean,
            rolling_std,
            rolling_sum,
        ]
    
    def get_model_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a trained model.
        
        Returns None if model doesn't exist.
        """
        if entity_id not in self.metadata:
            if not self._load_model(entity_id):
                return None
        
        return self.metadata.get(entity_id)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    from datetime import datetime, timedelta
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        periods=500,
        freq='30min'
    )
    
    # Simulate light with evening pattern
    states = []
    for dt in dates:
        hour = dt.hour
        # Light on between 18:00-23:00, plus random noise
        if 18 <= hour <= 23:
            states.append('on' if np.random.random() > 0.1 else 'off')
        else:
            states.append('off' if np.random.random() > 0.1 else 'on')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'state': states
    })
    
    print(f"Sample data: {len(df)} records")
    print(df.head())
    
    # Train model
    predictor = DevicePredictor(model_cache_dir="/tmp/test_models")
    result = predictor.train('light.test', df)
    
    print(f"\nTraining result: {result}")
    
    if result['status'] == 'success':
        # Generate predictions
        predictions = predictor.predict('light.test', hours_ahead=24)
        
        print(f"\n24-hour forecast:")
        for pred in predictions[:10]:  # Show first 10 hours
            print(f"  {pred['hour']:02d}:00 - {pred['predicted_state_label']} "
                  f"(confidence: {pred['confidence']:.2%})")