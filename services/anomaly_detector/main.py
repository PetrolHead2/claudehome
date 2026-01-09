"""
Anomaly Detector - Phase 3.2 Enhanced Statistical and ML-Based Detection

This service detects anomalies using multiple methods:
1. Baseline Learning - Learns normal behavior from historical data
2. Statistical Detection - Z-score and IQR methods for numeric sensors
3. Pattern Detection - Unusual state changes, timing, or durations
4. Prediction Comparison - Compares reality vs ML predictions
5. Change Rate Detection - Detects excessive state changes (flickering, cycling)

The detector learns baselines automatically and triggers alerts when anomalies detected.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import redis.asyncio as redis
import aiohttp
import pandas as pd
import logging

# Import our Phase 3.2 modules
from baseline_learning import BaselineLearner
from anomaly_detection import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClaudeHome Anomaly Detector - Phase 3.2")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
PREDICTION_SERVICE_URL = os.getenv("PREDICTION_SERVICE_URL", "http://prediction_service:8000")
MARIADB_HOST = os.getenv("MARIADB_HOST")
MARIADB_PORT = int(os.getenv("MARIADB_PORT", "3306"))
MARIADB_USER = os.getenv("MARIADB_USER")
MARIADB_PASSWORD = os.getenv("MARIADB_PASSWORD")
MARIADB_DATABASE = os.getenv("MARIADB_DATABASE")

# Global instances
redis_client: Optional[redis.Redis] = None
baseline_learner: Optional[BaselineLearner] = None
anomaly_detector: Optional[AnomalyDetector] = None


# Pydantic Models
class Event(BaseModel):
    entity_id: str
    state: str
    old_state: Optional[str] = None
    timestamp: str
    attributes: Dict[str, Any] = {}


class Anomaly(BaseModel):
    anomaly_id: str
    entity_id: str
    anomaly_type: str  # value, pattern, prediction_mismatch, change_rate
    severity: str  # low, medium, high, critical
    description: str
    current_value: Any
    expected_value: Optional[Any] = None
    score: float
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = {}


class BaselineRequest(BaseModel):
    entity_id: str
    lookback_days: int = 7


@app.on_event("startup")
async def startup():
    """Initialize services"""
    global redis_client, baseline_learner, anomaly_detector
    
    logger.info("Starting Anomaly Detector Phase 3.2...")
    
    # Initialize Redis
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connected")
    
    # Initialize baseline learner
    baseline_learner = BaselineLearner()
    logger.info("BaselineLearner initialized")
    
    # Initialize anomaly detector
    anomaly_detector = AnomalyDetector(baseline_learner)
    logger.info("AnomalyDetector initialized")
    
    # Load any saved baselines from Redis
    await load_baselines_from_redis()
    
    logger.info("Anomaly Detector Phase 3.2 started successfully")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup"""
    if redis_client:
        # Save baselines to Redis before shutdown
        await save_baselines_to_redis()
        await redis_client.close()
        logger.info("Shut down complete")


async def load_baselines_from_redis():
    """Load saved baselines from Redis"""
    try:
        keys = await redis_client.keys("claudehome:baseline:*")
        
        for key in keys:
            data = await redis_client.get(key)
            if data:
                baseline = json.loads(data)
                entity_id = baseline.get('entity_id')
                if entity_id:
                    baseline_learner.save_baseline(entity_id, baseline)
                    logger.info(f"Loaded baseline for {entity_id}")
        
        logger.info(f"Loaded {len(keys)} baselines from Redis")
        
    except Exception as e:
        logger.error(f"Failed to load baselines: {e}")


async def save_baselines_to_redis():
    """Save baselines to Redis"""
    try:
        baselines = baseline_learner.get_all_baselines()
        
        for entity_id, baseline in baselines.items():
            key = f"claudehome:baseline:{entity_id}"
            value = json.dumps(baseline)
            # Save for 30 days
            await redis_client.setex(key, 86400 * 30, value)
        
        logger.info(f"Saved {len(baselines)} baselines to Redis")
        
    except Exception as e:
        logger.error(f"Failed to save baselines: {e}")


# Health Check
@app.get("/health")
async def health():
    """Service health check"""
    return {
        "status": "healthy",
        "service": "anomaly_detector",
        "version": "3.2",
        "baselines_loaded": len(baseline_learner.get_all_baselines()),
        "timestamp": datetime.now().isoformat()
    }


# Baseline Management
@app.post("/baseline/learn")
async def learn_baseline(request: BaselineRequest):
    """
    Learn baseline for an entity from historical data.
    
    This fetches historical data from MariaDB and learns normal patterns.
    """
    try:
        from utils.db_client import DatabaseClient
        
        logger.info(f"Learning baseline for {request.entity_id}")
        
        # Fetch historical data
        db_client = DatabaseClient()
        historical_data = db_client.load_historical_states(
            request.entity_id, 
            days=request.lookback_days
        )
        
        if historical_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for {request.entity_id}"
            )
        
        # Learn baseline
        baseline = baseline_learner.learn_baseline(request.entity_id, historical_data)
        
        # Save to Redis
        await save_baseline_to_redis(request.entity_id, baseline)
        
        return {
            "status": "success",
            "entity_id": request.entity_id,
            "baseline": baseline
        }
        
    except Exception as e:
        logger.error(f"Baseline learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def save_baseline_to_redis(entity_id: str, baseline: Dict[str, Any]):
    """Save single baseline to Redis"""
    key = f"claudehome:baseline:{entity_id}"
    value = json.dumps(baseline)
    await redis_client.setex(key, 86400 * 30, value)


@app.get("/baselines")
async def list_baselines():
    """List all learned baselines"""
    baselines = baseline_learner.get_all_baselines()
    
    return {
        "count": len(baselines),
        "baselines": list(baselines.keys()),
        "details": baselines
    }


@app.get("/baseline/{entity_id}")
async def get_baseline(entity_id: str):
    """Get baseline for specific entity"""
    baseline = baseline_learner.get_baseline(entity_id)
    
    if not baseline:
        raise HTTPException(status_code=404, detail=f"No baseline for {entity_id}")
    
    return baseline


@app.delete("/baseline/{entity_id}")
async def delete_baseline(entity_id: str):
    """Delete baseline for entity"""
    baseline_learner.clear_baseline(entity_id)
    
    # Delete from Redis
    key = f"claudehome:baseline:{entity_id}"
    await redis_client.delete(key)
    
    return {"status": "deleted", "entity_id": entity_id}


# Anomaly Detection
@app.post("/detect")
async def detect_anomalies(event: Event):
    """
    Detect anomalies in current event.
    
    This is the main endpoint called by other services to check for anomalies.
    """
    try:
        anomalies = []
        entity_id = event.entity_id
        
        # Check if baseline exists
        if not baseline_learner.has_baseline(entity_id):
            return {
                "has_baseline": False,
                "message": f"No baseline for {entity_id}. Learn baseline first.",
                "anomalies": []
            }
        
        baseline = baseline_learner.get_baseline(entity_id)
        state_type = baseline.get('state_type')
        
        # Detect based on state type
        if state_type == 'numeric':
            # Try to parse state as numeric
            try:
                value = float(event.state)
                
                # Value anomaly detection (Z-score)
                is_anom, score, desc = anomaly_detector.detect_value_anomaly(
                    entity_id, value, method="zscore"
                )
                
                if is_anom:
                    anomaly = await create_anomaly(
                        entity_id=entity_id,
                        anomaly_type="value",
                        description=desc,
                        current_value=value,
                        expected_value=baseline.get('value_mean'),
                        score=score
                    )
                    anomalies.append(anomaly)
                    
            except ValueError:
                logger.warning(f"Could not parse {event.state} as numeric")
        
        elif state_type == 'binary':
            # Pattern anomaly detection
            current_hour = datetime.fromisoformat(event.timestamp).hour
            
            is_anom, score, desc = anomaly_detector.detect_pattern_anomaly(
                entity_id=entity_id,
                current_state=event.state,
                current_hour=current_hour
            )
            
            if is_anom:
                anomaly = await create_anomaly(
                    entity_id=entity_id,
                    anomaly_type="pattern",
                    description=desc,
                    current_value=event.state,
                    score=score
                )
                anomalies.append(anomaly)
        
        # Prediction-based anomaly detection (if model exists)
        prediction_anomaly = await detect_prediction_anomaly(entity_id, event.state)
        if prediction_anomaly:
            anomalies.append(prediction_anomaly)
        
        # Store anomalies
        for anomaly in anomalies:
            await store_anomaly(anomaly)
        
        return {
            "has_baseline": True,
            "entity_id": entity_id,
            "anomalies_detected": len(anomalies),
            "anomalies": [a.dict() for a in anomalies]
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def detect_prediction_anomaly(entity_id: str, actual_state: str) -> Optional[Anomaly]:
    """
    Check if current state differs from ML prediction.
    """
    try:
        # Get prediction from prediction service
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{PREDICTION_SERVICE_URL}/predict/{entity_id}",
                params={"hours_ahead": 1},
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                predictions = data.get('predictions', [])
                
                if not predictions:
                    return None
                
                # Get current hour prediction
                current_prediction = predictions[0]
                predicted_state = current_prediction['predicted_state_label']
                confidence = current_prediction['confidence']
                
                # Detect mismatch
                is_anom, score, desc = anomaly_detector.detect_prediction_mismatch(
                    entity_id=entity_id,
                    actual_state=actual_state,
                    predicted_state=predicted_state,
                    prediction_confidence=confidence
                )
                
                if is_anom:
                    return await create_anomaly(
                        entity_id=entity_id,
                        anomaly_type="prediction_mismatch",
                        description=desc,
                        current_value=actual_state,
                        expected_value=predicted_state,
                        score=score,
                        metadata={"prediction_confidence": confidence}
                    )
                
    except Exception as e:
        logger.debug(f"Prediction anomaly check failed for {entity_id}: {e}")
    
    return None


async def create_anomaly(
    entity_id: str,
    anomaly_type: str,
    description: str,
    current_value: Any,
    expected_value: Optional[Any] = None,
    score: float = 0.0,
    metadata: Dict[str, Any] = {}
) -> Anomaly:
    """Create anomaly object"""
    
    # Calculate severity
    severity = anomaly_detector.calculate_anomaly_severity(score, anomaly_type)
    
    # Generate ID
    anomaly_id = f"anom_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    return Anomaly(
        anomaly_id=anomaly_id,
        entity_id=entity_id,
        anomaly_type=anomaly_type,
        severity=severity,
        description=description,
        current_value=current_value,
        expected_value=expected_value,
        score=score,
        confidence=min(score / 5.0, 1.0) if anomaly_type == "value" else score,
        timestamp=datetime.now().isoformat(),
        metadata=metadata
    )


async def store_anomaly(anomaly: Anomaly):
    """Store anomaly in Redis"""
    try:
        # Store anomaly
        key = f"claudehome:anomaly:{anomaly.anomaly_id}"
        value = json.dumps(anomaly.dict())
        await redis_client.setex(key, 86400 * 7, value)  # 7 days TTL
        
        # Add to entity anomaly list
        entity_key = f"claudehome:anomalies:{anomaly.entity_id}"
        await redis_client.lpush(entity_key, anomaly.anomaly_id)
        await redis_client.ltrim(entity_key, 0, 99)  # Keep last 100
        
        # Add to severity list
        severity_key = f"claudehome:anomalies:severity:{anomaly.severity}"
        await redis_client.lpush(severity_key, anomaly.anomaly_id)
        await redis_client.ltrim(severity_key, 0, 99)
        
        # Publish to event bus
        await redis_client.publish(
            "claudehome:anomaly:detected",
            json.dumps({
                "anomaly_id": anomaly.anomaly_id,
                "entity_id": anomaly.entity_id,
                "severity": anomaly.severity,
                "type": anomaly.anomaly_type
            })
        )
        
        logger.info(f"Anomaly detected: {anomaly.entity_id} - {anomaly.description}")
        
    except Exception as e:
        logger.error(f"Failed to store anomaly: {e}")


@app.get("/anomalies")
async def list_anomalies(severity: Optional[str] = None, limit: int = 50):
    """
    List recent anomalies.
    
    Args:
        severity: Filter by severity (low, medium, high, critical)
        limit: Max number to return
    """
    try:
        if severity:
            key = f"claudehome:anomalies:severity:{severity}"
        else:
            # Get all anomalies (merge from all severity levels)
            all_ids = []
            for sev in ['critical', 'high', 'medium', 'low']:
                key = f"claudehome:anomalies:severity:{sev}"
                ids = await redis_client.lrange(key, 0, limit - 1)
                all_ids.extend(ids)
            
            # Fetch anomaly details
            anomalies = []
            for anom_id in all_ids[:limit]:
                anom_key = f"claudehome:anomaly:{anom_id}"
                data = await redis_client.get(anom_key)
                if data:
                    anomalies.append(json.loads(data))
            
            return {
                "count": len(anomalies),
                "anomalies": anomalies
            }
        
        # Severity-specific
        anomaly_ids = await redis_client.lrange(key, 0, limit - 1)
        
        anomalies = []
        for anom_id in anomaly_ids:
            anom_key = f"claudehome:anomaly:{anom_id}"
            data = await redis_client.get(anom_key)
            if data:
                anomalies.append(json.loads(data))
        
        return {
            "severity": severity,
            "count": len(anomalies),
            "anomalies": anomalies
        }
        
    except Exception as e:
        logger.error(f"Failed to list anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies/{entity_id}")
async def get_entity_anomalies(entity_id: str, limit: int = 20):
    """Get anomalies for specific entity"""
    try:
        key = f"claudehome:anomalies:{entity_id}"
        anomaly_ids = await redis_client.lrange(key, 0, limit - 1)
        
        anomalies = []
        for anom_id in anomaly_ids:
            anom_key = f"claudehome:anomaly:{anom_id}"
            data = await redis_client.get(anom_key)
            if data:
                anomalies.append(json.loads(data))
        
        return {
            "entity_id": entity_id,
            "count": len(anomalies),
            "anomalies": anomalies
        }
        
    except Exception as e:
        logger.error(f"Failed to get entity anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get anomaly detection statistics"""
    try:
        # Count by severity
        stats = {}
        for sev in ['critical', 'high', 'medium', 'low']:
            key = f"claudehome:anomalies:severity:{sev}"
            count = await redis_client.llen(key)
            stats[f"{sev}_count"] = count
        
        stats["total_baselines"] = len(baseline_learner.get_all_baselines())
        stats["total_anomalies"] = sum(stats[f"{sev}_count"] for sev in ['critical', 'high', 'medium', 'low'])
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)