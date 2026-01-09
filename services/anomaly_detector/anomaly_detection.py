"""
Anomaly Detection - Detects unusual behavior using statistical methods

This module provides multiple anomaly detection methods:
1. Z-Score Method - Detects values far from mean
2. IQR Method - Detects outliers using interquartile range
3. Pattern-Based - Detects unusual state changes or timing
4. Prediction-Based - Compares reality vs ML predictions

Each method returns an anomaly score and description.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies using multiple statistical methods.
    """
    
    def __init__(self, baseline_learner):
        """
        Initialize with baseline learner.
        
        Args:
            baseline_learner: BaselineLearner instance with learned baselines
        """
        self.baseline_learner = baseline_learner
    
    def detect_value_anomaly(
        self, 
        entity_id: str, 
        current_value: float,
        method: str = "zscore"
    ) -> Tuple[bool, float, str]:
        """
        Detect if a numeric value is anomalous.
        
        Args:
            entity_id: Entity to check
            current_value: Current sensor value
            method: "zscore" or "iqr"
        
        Returns:
            (is_anomaly, score, description)
            
        Example:
            is_anomaly, score, desc = detector.detect_value_anomaly(
                "sensor.temperature", 
                45.0  # Current temp
            )
            # Returns: (True, 5.2, "Value 5.2σ above mean")
        """
        baseline = self.baseline_learner.get_baseline(entity_id)
        
        if not baseline or baseline.get('state_type') != 'numeric':
            return (False, 0.0, "No numeric baseline available")
        
        if method == "zscore":
            return self._detect_zscore_anomaly(baseline, current_value)
        elif method == "iqr":
            return self._detect_iqr_anomaly(baseline, current_value)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_zscore_anomaly(
        self, 
        baseline: Dict[str, Any], 
        value: float,
        threshold: float = 3.0
    ) -> Tuple[bool, float, str]:
        """
        Detect anomaly using Z-score method.
        
        Z-score = (value - mean) / std
        Anomaly if |Z-score| > threshold (default: 3.0)
        """
        mean = baseline.get('value_mean', 0)
        std = baseline.get('value_std', 1)
        
        if std == 0:
            return (False, 0.0, "No variance in baseline")
        
        zscore = abs((value - mean) / std)
        is_anomaly = zscore > threshold
        
        if is_anomaly:
            direction = "above" if value > mean else "below"
            description = f"Value {zscore:.1f}σ {direction} mean (expected: {mean:.1f}±{std:.1f})"
        else:
            description = f"Value within normal range ({zscore:.1f}σ from mean)"
        
        return (is_anomaly, float(zscore), description)
    
    def _detect_iqr_anomaly(
        self, 
        baseline: Dict[str, Any], 
        value: float
    ) -> Tuple[bool, float, str]:
        """
        Detect anomaly using IQR (Interquartile Range) method.
        
        Outlier if: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
        """
        q1 = baseline.get('value_q1', 0)
        q3 = baseline.get('value_q3', 0)
        iqr = baseline.get('value_iqr', 0)
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        is_anomaly = value < lower_bound or value > upper_bound
        
        # Calculate how many IQRs away from bounds
        if value < lower_bound:
            score = (lower_bound - value) / iqr if iqr > 0 else 0
            description = f"Value {score:.1f} IQRs below lower bound (expected: {lower_bound:.1f}-{upper_bound:.1f})"
        elif value > upper_bound:
            score = (value - upper_bound) / iqr if iqr > 0 else 0
            description = f"Value {score:.1f} IQRs above upper bound (expected: {lower_bound:.1f}-{upper_bound:.1f})"
        else:
            score = 0.0
            description = f"Value within IQR bounds ({lower_bound:.1f}-{upper_bound:.1f})"
        
        return (is_anomaly, float(score), description)
    
    def detect_pattern_anomaly(
        self,
        entity_id: str,
        current_state: str,
        current_hour: int,
        state_duration_minutes: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Detect pattern-based anomalies for binary states.
        
        Checks:
        - Is device active at unusual hour?
        - Has device been in current state too long?
        - Is state change frequency unusual?
        
        Args:
            entity_id: Entity to check
            current_state: Current state ("on" or "off")
            current_hour: Current hour (0-23)
            state_duration_minutes: How long in current state
        
        Returns:
            (is_anomaly, score, description)
        """
        baseline = self.baseline_learner.get_baseline(entity_id)
        
        if not baseline or baseline.get('state_type') != 'binary':
            return (False, 0.0, "No binary baseline available")
        
        anomalies = []
        max_score = 0.0
        
        # Check 1: Unusual hour for ON state
        if current_state.lower() in ['on', 'active', 'home', 'open']:
            typical_on_hours = baseline.get('typical_on_hours', [])
            if typical_on_hours and current_hour not in typical_on_hours:
                score = 2.0  # Moderate anomaly
                anomalies.append((score, f"Active at unusual hour {current_hour}:00 (typical: {typical_on_hours})"))
                max_score = max(max_score, score)
        
        # Check 2: Excessive state duration
        if state_duration_minutes:
            typical_duration = baseline.get('typical_state_duration_minutes', 60)
            if state_duration_minutes > typical_duration * 3:  # 3x longer than normal
                score = state_duration_minutes / typical_duration
                anomalies.append((score, f"State duration {state_duration_minutes:.0f}min exceeds typical {typical_duration:.0f}min by {score:.1f}x"))
                max_score = max(max_score, score)
        
        # Return most significant anomaly
        if anomalies:
            # Sort by score descending
            anomalies.sort(key=lambda x: x[0], reverse=True)
            is_anomaly = max_score >= 2.0  # Threshold for reporting
            return (is_anomaly, max_score, anomalies[0][1])
        
        return (False, 0.0, "Pattern within normal range")
    
    def detect_prediction_mismatch(
        self,
        entity_id: str,
        actual_state: str,
        predicted_state: str,
        prediction_confidence: float,
        confidence_threshold: float = 0.75
    ) -> Tuple[bool, float, str]:
        """
        Detect when reality differs from ML prediction.
        
        This is a powerful anomaly detection method because:
        - ML models learn typical patterns
        - When reality differs significantly, something unusual happened
        - High confidence mismatches are strong anomaly indicators
        
        Args:
            entity_id: Entity to check
            actual_state: Current actual state
            predicted_state: What ML predicted
            prediction_confidence: ML's confidence (0-1)
            confidence_threshold: Minimum confidence to consider (default: 0.75)
        
        Returns:
            (is_anomaly, score, description)
        
        Example:
            # ML predicted light ON with 89% confidence, but it's OFF
            is_anomaly, score, desc = detector.detect_prediction_mismatch(
                "light.kitchen",
                actual_state="off",
                predicted_state="on",
                prediction_confidence=0.89
            )
            # Returns: (True, 0.89, "Predicted ON (89% conf) but actual state is OFF")
        """
        # Normalize states
        actual = actual_state.lower().strip()
        predicted = predicted_state.lower().strip()
        
        # Check if states match
        if actual == predicted:
            return (False, 0.0, f"State matches prediction ({actual})")
        
        # States differ - check if prediction was confident
        if prediction_confidence < confidence_threshold:
            # Low confidence prediction, mismatch not surprising
            return (False, prediction_confidence, 
                   f"Low confidence prediction ({prediction_confidence:.0%}), mismatch expected")
        
        # High confidence mismatch = anomaly!
        score = prediction_confidence
        description = (f"Predicted {predicted.upper()} ({prediction_confidence:.0%} conf) "
                      f"but actual state is {actual.upper()}")
        
        # Higher confidence mismatches are stronger anomalies
        is_anomaly = prediction_confidence >= confidence_threshold
        
        return (is_anomaly, score, description)
    
    def detect_change_rate_anomaly(
        self,
        entity_id: str,
        recent_changes: int,
        time_window_hours: float = 1.0
    ) -> Tuple[bool, float, str]:
        """
        Detect if entity is changing state too frequently.
        
        Examples of anomalies:
        - Light flickering (20 changes/minute)
        - HVAC short-cycling (30 changes/hour)
        - Sensor glitching (constant state changes)
        
        Args:
            entity_id: Entity to check
            recent_changes: Number of state changes in time window
            time_window_hours: Time window for counting changes
        
        Returns:
            (is_anomaly, score, description)
        """
        baseline = self.baseline_learner.get_baseline(entity_id)
        
        if not baseline:
            return (False, 0.0, "No baseline available")
        
        # Calculate changes per day for comparison
        changes_per_day = recent_changes / time_window_hours * 24
        
        typical_changes_per_day = baseline.get('typical_changes_per_day', 10)
        
        if typical_changes_per_day == 0:
            return (False, 0.0, "No typical change rate in baseline")
        
        # Calculate ratio
        ratio = changes_per_day / typical_changes_per_day
        
        # Anomaly if >3x typical rate
        is_anomaly = ratio > 3.0
        
        if is_anomaly:
            description = (f"Change rate {changes_per_day:.1f}/day is {ratio:.1f}x typical "
                          f"({typical_changes_per_day:.1f}/day)")
        else:
            description = f"Change rate within normal range ({ratio:.1f}x typical)"
        
        return (is_anomaly, float(ratio), description)
    
    def calculate_anomaly_severity(self, score: float, anomaly_type: str) -> str:
        """
        Calculate severity level based on score and type.
        
        Returns: "low", "medium", "high", or "critical"
        """
        if anomaly_type == "value":
            # Z-score based
            if score >= 5.0:
                return "critical"
            elif score >= 4.0:
                return "high"
            elif score >= 3.0:
                return "medium"
            else:
                return "low"
        
        elif anomaly_type == "prediction_mismatch":
            # Confidence based
            if score >= 0.95:
                return "high"
            elif score >= 0.85:
                return "medium"
            else:
                return "low"
        
        elif anomaly_type == "change_rate":
            # Ratio based
            if score >= 10.0:
                return "critical"
            elif score >= 5.0:
                return "high"
            elif score >= 3.0:
                return "medium"
            else:
                return "low"
        
        else:
            # Pattern or unknown
            if score >= 5.0:
                return "high"
            elif score >= 3.0:
                return "medium"
            else:
                return "low"


# Example usage
if __name__ == "__main__":
    from baseline_learning import BaselineLearner
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    
    # Create baseline
    dates = pd.date_range(start='2026-01-01', periods=100, freq='1h')
    temps = [20 + np.random.normal(0, 2) for _ in range(100)]  # Normal: 20±2°C
    
    df = pd.DataFrame({
        'timestamp': dates,
        'state': temps
    })
    
    learner = BaselineLearner()
    baseline = learner.learn_baseline('sensor.temperature', df)
    
    print(f"Baseline: mean={baseline['value_mean']:.1f}, std={baseline['value_std']:.1f}")
    
    # Test anomaly detection
    detector = AnomalyDetector(learner)
    
    # Test 1: Normal value
    is_anom, score, desc = detector.detect_value_anomaly('sensor.temperature', 21.0)
    print(f"\nTest 1 (21.0°C): {is_anom}, score={score:.2f}, {desc}")
    
    # Test 2: Anomalous value
    is_anom, score, desc = detector.detect_value_anomaly('sensor.temperature', 35.0)
    print(f"Test 2 (35.0°C): {is_anom}, score={score:.2f}, {desc}")
    
    # Test 3: Prediction mismatch
    is_anom, score, desc = detector.detect_prediction_mismatch(
        'light.kitchen',
        actual_state='off',
        predicted_state='on',
        prediction_confidence=0.89
    )
    print(f"\nTest 3 (Prediction mismatch): {is_anom}, score={score:.2f}, {desc}")