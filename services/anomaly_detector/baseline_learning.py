"""
Baseline Learning - Learns normal behavior patterns from historical data

This module analyzes historical device states to establish baselines for:
- Typical state durations (how long devices stay on/off)
- Change frequencies (how often devices change state)
- Active hours (when devices are typically used)
- Value ranges (for numeric sensors like temperature)

These baselines are used to detect anomalies.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)


class BaselineLearner:
    """
    Learns normal behavior patterns for entities.
    """
    
    def __init__(self):
        self.baselines = {}  # {entity_id: baseline_data}
    
    def learn_baseline(self, entity_id: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn baseline behavior from historical data.
        
        Args:
            entity_id: Entity to learn baseline for
            historical_data: DataFrame with columns: timestamp, state
        
        Returns:
            Baseline data:
            {
                "entity_id": "light.kitchen",
                "data_points": 1000,
                "state_type": "binary",  # or "numeric"
                "typical_state_duration_minutes": 45.5,
                "typical_changes_per_day": 4.2,
                "typical_on_hours": [6, 7, 18, 19, 20],
                "value_mean": 22.5,  # for numeric sensors
                "value_std": 2.3,
                "value_min": 18.0,
                "value_max": 28.0,
                "learned_at": "2026-01-09T13:00:00"
            }
        """
        if historical_data.empty:
            logger.warning(f"No data to learn baseline for {entity_id}")
            return {}
        
        logger.info(f"Learning baseline for {entity_id} from {len(historical_data)} records")
        
        baseline = {
            "entity_id": entity_id,
            "data_points": len(historical_data),
            "learned_at": datetime.now().isoformat()
        }
        
        # Determine state type (binary on/off or numeric)
        state_type = self._detect_state_type(historical_data)
        baseline["state_type"] = state_type
        
        # Learn type-specific baselines
        if state_type == "binary":
            baseline.update(self._learn_binary_baseline(historical_data))
        elif state_type == "numeric":
            baseline.update(self._learn_numeric_baseline(historical_data))
        
        # Store baseline
        self.baselines[entity_id] = baseline
        
        logger.info(f"Baseline learned for {entity_id}: {state_type} type")
        return baseline
    
    def _detect_state_type(self, df: pd.DataFrame) -> str:
        """
        Detect if entity has binary states (on/off) or numeric values.
        """
        unique_states = df['state'].unique()
        
        # Binary states
        binary_states = {'on', 'off', 'true', 'false', 'open', 'closed', 
                        'home', 'away', 'active', 'inactive'}
        
        if len(unique_states) <= 10:
            # Check if states are binary-like
            states_lower = {str(s).lower() for s in unique_states}
            if states_lower.intersection(binary_states):
                return "binary"
        
        # Try to parse as numeric
        try:
            pd.to_numeric(df['state'], errors='raise')
            return "numeric"
        except:
            return "categorical"
    
    def _learn_binary_baseline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn baseline for binary states (on/off, etc.)
        """
        baseline = {}
        
        # Convert states to binary (1 = on/active, 0 = off/inactive)
        df['state_binary'] = df['state'].apply(self._to_binary)
        
        # Calculate state durations
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60  # minutes
        
        # Average time in each state
        on_durations = df[df['state_binary'] == 1]['time_diff'].dropna()
        off_durations = df[df['state_binary'] == 0]['time_diff'].dropna()
        
        if len(on_durations) > 0:
            baseline['typical_on_duration_minutes'] = float(on_durations.median())
        if len(off_durations) > 0:
            baseline['typical_off_duration_minutes'] = float(off_durations.median())
        
        # Average state duration overall
        all_durations = df['time_diff'].dropna()
        if len(all_durations) > 0:
            baseline['typical_state_duration_minutes'] = float(all_durations.median())
        
        # Changes per day
        days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        state_changes = (df['state_binary'] != df['state_binary'].shift()).sum()
        baseline['typical_changes_per_day'] = float(state_changes / days)
        
        # Active hours (hours when typically ON)
        df['hour'] = df['timestamp'].dt.hour
        on_hours = df[df['state_binary'] == 1]['hour'].value_counts()
        
        # Hours with >10% of total ON time
        if len(on_hours) > 0:
            threshold = on_hours.sum() * 0.1
            typical_hours = on_hours[on_hours > threshold].index.tolist()
            baseline['typical_on_hours'] = sorted(typical_hours)
        else:
            baseline['typical_on_hours'] = []
        
        # ON percentage
        baseline['typical_on_percentage'] = float(df['state_binary'].mean() * 100)
        
        return baseline
    
    def _learn_numeric_baseline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn baseline for numeric sensors (temperature, humidity, etc.)
        """
        baseline = {}
        
        # Convert to numeric
        df['value'] = pd.to_numeric(df['state'], errors='coerce')
        df = df.dropna(subset=['value'])
        
        if len(df) == 0:
            return baseline
        
        values = df['value']
        
        # Basic statistics
        baseline['value_mean'] = float(values.mean())
        baseline['value_std'] = float(values.std())
        baseline['value_median'] = float(values.median())
        baseline['value_min'] = float(values.min())
        baseline['value_max'] = float(values.max())
        
        # Percentiles (for outlier detection)
        baseline['value_p5'] = float(values.quantile(0.05))
        baseline['value_p95'] = float(values.quantile(0.95))
        
        # IQR for outlier detection
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        baseline['value_iqr'] = float(iqr)
        baseline['value_q1'] = float(q1)
        baseline['value_q3'] = float(q3)
        
        # Change rate (how fast values typically change)
        df['value_diff'] = df['value'].diff().abs()
        baseline['typical_change_rate'] = float(df['value_diff'].median())
        baseline['max_change_rate'] = float(df['value_diff'].quantile(0.95))
        
        # Hour-based patterns (temperature varies by time of day)
        df['hour'] = df['timestamp'].dt.hour
        hourly_means = df.groupby('hour')['value'].mean()
        baseline['hourly_pattern'] = hourly_means.to_dict()
        
        return baseline
    
    def _to_binary(self, state: str) -> int:
        """Convert state to binary (1=on/active, 0=off/inactive)"""
        state_str = str(state).lower().strip()
        
        on_states = {'on', 'true', '1', 'open', 'home', 'active', 'detected', 'occupied'}
        
        return 1 if state_str in on_states else 0
    
    def get_baseline(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get learned baseline for entity"""
        return self.baselines.get(entity_id)
    
    def has_baseline(self, entity_id: str) -> bool:
        """Check if baseline exists for entity"""
        return entity_id in self.baselines
    
    def get_all_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Get all learned baselines"""
        return self.baselines.copy()
    
    def save_baseline(self, entity_id: str, baseline: Dict[str, Any]):
        """Manually save a baseline"""
        self.baselines[entity_id] = baseline
        logger.info(f"Saved baseline for {entity_id}")
    
    def clear_baseline(self, entity_id: str):
        """Remove baseline for entity"""
        if entity_id in self.baselines:
            del self.baselines[entity_id]
            logger.info(f"Cleared baseline for {entity_id}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='30min'
    )
    
    # Simulate light with evening pattern
    states = []
    for dt in dates:
        hour = dt.hour
        if 18 <= hour <= 23:
            states.append('on')
        else:
            states.append('off')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'state': states
    })
    
    print(f"Sample data: {len(df)} records")
    
    # Learn baseline
    learner = BaselineLearner()
    baseline = learner.learn_baseline('light.test', df)
    
    print(f"\nLearned baseline:")
    for key, value in baseline.items():
        if key != 'hourly_pattern':  # Skip hourly pattern (too verbose)
            print(f"  {key}: {value}")