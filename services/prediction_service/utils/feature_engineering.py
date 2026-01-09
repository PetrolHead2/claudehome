"""
Feature Engineering - Convert raw device states into ML-ready features

This module transforms Home Assistant state data into features that
machine learning models can use to predict future behavior.

Key Insight:
Device patterns are heavily time-dependent:
- Lights turn on at sunset
- Heating adjusts based on time of day
- Activity follows weekly rhythms (weekday vs weekend)

So we extract TIME-BASED features from timestamps.
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def state_to_numeric(state: str) -> float:
    """
    Convert Home Assistant state strings to numeric values.
    
    Binary states (on/off, home/away, etc.) → 0 or 1
    Numeric states (temperature, brightness) → float value
    
    Args:
        state: State string from Home Assistant
    
    Returns:
        Numeric representation
    
    Examples:
        'on' → 1.0
        'off' → 0.0
        '23.5' → 23.5
        'home' → 1.0
        'unavailable' → 0.0
    """
    if pd.isna(state):
        return 0.0
    
    state_lower = str(state).lower().strip()
    
    # Binary positive states
    if state_lower in ['on', 'true', 'home', 'open', 'active', 'occupied']:
        return 1.0
    
    # Binary negative states
    if state_lower in ['off', 'false', 'away', 'closed', 'inactive', 'not_home', 'unavailable']:
        return 0.0
    
    # Try to parse as number
    try:
        return float(state)
    except (ValueError, TypeError):
        # Unknown state, default to 0
        logger.debug(f"Unknown state '{state}', defaulting to 0.0")
        return 0.0


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamps.
    
    These features capture WHEN things happen, which is crucial for
    predicting device behavior.
    
    Features extracted:
    - hour: 0-23 (when during the day)
    - day_of_week: 0-6 (Monday=0, Sunday=6)
    - is_weekend: 0/1 (Saturday/Sunday)
    - month: 1-12 (seasonal patterns)
    - day_of_month: 1-31 (monthly patterns)
    - is_morning: 1 if 6-10am
    - is_evening: 1 if 18-22pm
    - is_night: 1 if 22pm-6am
    
    Args:
        df: DataFrame with 'timestamp' column
    
    Returns:
        DataFrame with time features added
    """
    if df.empty or 'timestamp' not in df.columns:
        return df
    
    df = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month
    df['day_of_month'] = df['timestamp'].dt.day
    
    # Time-of-day buckets (useful for pattern recognition)
    df['is_morning'] = df['hour'].between(6, 10).astype(int)
    df['is_afternoon'] = df['hour'].between(11, 17).astype(int)
    df['is_evening'] = df['hour'].between(18, 22).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    
    # Cyclical encoding for hour (important!)
    # Hour 23 and hour 0 are actually close, but 23 vs 0 looks far apart
    # Sin/cos encoding captures this cyclical nature
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def extract_lag_features(df: pd.DataFrame, state_column: str = 'state_numeric') -> pd.DataFrame:
    """
    Extract lag features (previous states).
    
    The previous state often predicts the next state:
    - If light was on, it's likely to stay on (inertia)
    - If light was off, turning on might indicate evening arrival
    
    Args:
        df: DataFrame with state data
        state_column: Column containing numeric state
    
    Returns:
        DataFrame with lag features
    """
    if df.empty or state_column not in df.columns:
        return df
    
    df = df.copy()
    
    # Previous state (lag-1)
    df['prev_state'] = df[state_column].shift(1)
    
    # State 2 steps ago (lag-2) - helps capture patterns
    df['prev_state_2'] = df[state_column].shift(2)
    
    # Did state change from previous?
    df['state_changed'] = (df[state_column] != df['prev_state']).astype(int)
    
    # How long in current state (duration counter)
    # Resets every time state changes
    df['state_duration'] = df.groupby(
        (df['state_changed'] == 1).cumsum()
    ).cumcount()
    
    return df


def extract_rolling_features(df: pd.DataFrame, state_column: str = 'state_numeric', window_hours: int = 3) -> pd.DataFrame:
    """
    Extract rolling statistics (patterns over time windows).
    
    Rolling features capture recent behavior trends:
    - Rolling mean: How often was light on in last 3 hours?
    - Rolling std: How stable has the state been?
    
    Args:
        df: DataFrame with state data
        state_column: Column with numeric state
        window_hours: Window size in hours (default: 3)
    
    Returns:
        DataFrame with rolling features
    """
    if df.empty or state_column not in df.columns:
        return df
    
    df = df.copy()
    
    # Estimate window size in records
    # Assuming ~5-10 minute intervals on average
    # 3 hours = 180 minutes = 18-36 records
    window_size = max(10, window_hours * 12)  # Conservative estimate
    
    # Rolling mean (what % of time was device on?)
    df[f'rolling_mean_{window_hours}h'] = df[state_column].rolling(
        window=window_size, 
        min_periods=1
    ).mean()
    
    # Rolling std (how variable is the state?)
    df[f'rolling_std_{window_hours}h'] = df[state_column].rolling(
        window=window_size,
        min_periods=1
    ).std().fillna(0)
    
    # Rolling sum (total activity count)
    df[f'rolling_sum_{window_hours}h'] = df[state_column].rolling(
        window=window_size,
        min_periods=1
    ).sum()
    
    return df


def create_feature_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: Create complete feature set from raw state data.
    
    This is the main entry point for feature engineering.
    Takes raw Home Assistant data and produces ML-ready features.
    
    Pipeline:
    1. Convert states to numeric
    2. Extract time features
    3. Extract lag features
    4. Extract rolling statistics
    5. Drop NaN values
    
    Args:
        df: Raw DataFrame with 'timestamp' and 'state' columns
    
    Returns:
        DataFrame with full feature set, ready for ML
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to feature engineering")
        return df
    
    logger.info(f"Starting feature engineering on {len(df)} records")
    
    # Step 1: Convert state to numeric
    df['state_numeric'] = df['state'].apply(state_to_numeric)
    
    # Step 2: Time-based features
    df = extract_time_features(df)
    
    # Step 3: Lag features (previous states)
    df = extract_lag_features(df, 'state_numeric')
    
    # Step 4: Rolling statistics
    df = extract_rolling_features(df, 'state_numeric', window_hours=3)
    
    # Step 5: Handle NaN values intelligently
    # Don't drop all NaN - fill them with sensible defaults
    initial_count = len(df)
    
    # Fill NaN in lag features with 0 (assume off/inactive initially)
    lag_cols = ['prev_state', 'prev_state_2', 'state_changed', 'state_duration']
    for col in lag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Fill NaN in rolling features with current state_numeric
    # (first few records don't have enough history for rolling window)
    rolling_cols = [col for col in df.columns if col.startswith('rolling_')]
    for col in rolling_cols:
        if col in df.columns:
            if 'mean' in col or 'sum' in col:
                df[col] = df[col].fillna(df['state_numeric'])
            elif 'std' in col:
                df[col] = df[col].fillna(0)
    
    # Now only drop rows where critical columns are still NaN
    df = df.dropna(subset=['state_numeric', 'hour', 'day_of_week'])
    
    dropped = initial_count - len(df)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing critical data")
    
    logger.info(f"Feature engineering complete: {len(df)} records with {len(df.columns)} features")
    
    return df


def get_feature_columns() -> list:
    """
    Return the list of feature column names to use for ML training.
    
    These are the columns we'll pass to the Random Forest model.
    
    Returns:
        List of feature column names
    """
    return [
        # Time features
        'hour',
        'day_of_week',
        'is_weekend',
        'month',
        'is_morning',
        'is_afternoon',
        'is_evening',
        'is_night',
        'hour_sin',
        'hour_cos',
        'day_sin',
        'day_cos',
        
        # Lag features
        'prev_state',
        'prev_state_2',
        'state_changed',
        'state_duration',
        
        # Rolling features
        'rolling_mean_3h',
        'rolling_std_3h',
        'rolling_sum_3h',
    ]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    from datetime import datetime, timedelta
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='10min'  # Every 10 minutes
    )
    
    # Simulate light that turns on in evening, off at night
    states = []
    for dt in dates:
        hour = dt.hour
        if 18 <= hour <= 23:  # Evening
            states.append('on')
        else:
            states.append('off')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'state': states
    })
    
    print("Original data:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    
    # Apply feature engineering
    df_features = create_feature_vector(df)
    
    print("\n\nFeature-engineered data:")
    print(df_features.head(10))
    print(f"\nShape: {df_features.shape}")
    print(f"\nFeature columns: {get_feature_columns()}")
    print(f"\nTotal features: {len(get_feature_columns())}")