"""
Prediction Service - REST API for Device State Forecasting

This service provides HTTP endpoints for:
- Training ML models for devices
- Getting predictions (24-48 hour forecasts)
- Model management and health checks

Architecture:
- FastAPI for REST API
- Redis for caching predictions
- MariaDB for historical data (read-only)
- Disk storage for trained models

Endpoints:
- POST /train/{entity_id} - Train model for device
- GET /predict/{entity_id} - Get forecast
- GET /models - List all trained models
- GET /health - Service health check
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import redis.asyncio as redis

from models.device_predictor import DevicePredictor
from utils.db_client import DatabaseClient
from utils.feature_engineering import get_feature_columns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ClaudeHome Prediction Service",
    description="ML-powered device state forecasting",
    version="3.1.0"
)

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
PREDICTION_CACHE_TTL = int(os.getenv("PREDICTION_CACHE_TTL", "3600"))  # 1 hour

# Global instances
redis_client: Optional[redis.Redis] = None
db_client: Optional[DatabaseClient] = None
predictor: Optional[DevicePredictor] = None


# Pydantic models for request/response
class TrainRequest(BaseModel):
    """Request to train a model"""
    lookback_days: int = Field(30, description="Days of history to use", ge=7, le=90)
    force_retrain: bool = Field(False, description="Force retraining even if cached")


class TrainResponse(BaseModel):
    """Response from training"""
    status: str
    entity_id: str
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    training_records: Optional[int] = None
    top_features: Optional[List[str]] = None
    message: Optional[str] = None


class Prediction(BaseModel):
    """Single prediction point"""
    timestamp: str
    hour: int
    predicted_state: int
    predicted_state_label: str
    probability_on: float
    probability_off: float
    confidence: float


class PredictResponse(BaseModel):
    """Response with predictions"""
    entity_id: str
    generated_at: str
    predictions: List[Prediction]
    source: str  # 'cache' or 'generated'


class ModelInfo(BaseModel):
    """Model metadata"""
    entity_id: str
    trained_at: str
    train_accuracy: float
    test_accuracy: float
    training_records: int
    top_features: Dict[str, float]


# Startup/Shutdown
@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global redis_client, db_client, predictor
    
    logger.info("Starting Prediction Service...")
    
    # Initialize Redis
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connected")
    
    # Initialize database client
    db_client = DatabaseClient()
    health = db_client.health_check()
    if health['status'] == 'healthy':
        logger.info(f"Database connected: {health['database']} on {health['host']}")
    else:
        logger.error(f"Database connection failed: {health.get('error')}")
    
    # Initialize predictor
    predictor = DevicePredictor(model_cache_dir=MODEL_CACHE_DIR)
    logger.info("DevicePredictor initialized")
    
    logger.info("Prediction Service started successfully")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


# Health Check
@app.get("/health")
async def health_check():
    """
    Service health check.
    
    Returns status of all components.
    """
    health = {
        "service": "prediction_service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check Redis
    try:
        await redis_client.ping()
        health["components"]["redis"] = "healthy"
    except Exception as e:
        health["components"]["redis"] = f"unhealthy: {e}"
        health["status"] = "degraded"
    
    # Check Database
    db_health = db_client.health_check()
    health["components"]["database"] = db_health["status"]
    if db_health["status"] != "healthy":
        health["status"] = "degraded"
    
    # Check model cache
    model_count = len(list(predictor.model_cache_dir.glob("*.joblib")))
    health["components"]["model_cache"] = {
        "status": "healthy",
        "cached_models": model_count
    }
    
    return health


# Training Endpoints
@app.post("/train/{entity_id}", response_model=TrainResponse)
async def train_model(
    entity_id: str, 
    lookback_days: int = 30,
    force_retrain: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Train a prediction model for a device.
    
    This endpoint:
    1. Fetches historical data from MariaDB
    2. Engineers features
    3. Trains Random Forest model
    4. Caches model to disk
    5. Returns training metrics
    
    Args:
        entity_id: Device to train (e.g., 'light.living_room')
        lookback_days: Number of days of history to use (default: 30)
        force_retrain: Force retraining even if cached model exists (default: False)
    
    Returns:
        Training results including accuracy metrics
    
    Example:
        POST /train/light.living_room?lookback_days=30&force_retrain=false
    """
    try:
        logger.info(f"Training request for {entity_id} (lookback: {lookback_days} days)")
        
        # Load historical data
        historical_data = db_client.load_historical_states(entity_id, days=lookback_days)
        
        if historical_data.empty:
            return TrainResponse(
                status="error",
                entity_id=entity_id,
                message=f"No historical data found for {entity_id}"
            )
        
        # Train model
        result = predictor.train(
            entity_id=entity_id,
            historical_data=historical_data,
            force_retrain=force_retrain
        )
        
        # Invalidate cached predictions after training
        cache_key = f"prediction:{entity_id}"
        await redis_client.delete(cache_key)
        
        # Return response
        return TrainResponse(
            status=result['status'],
            entity_id=entity_id,
            train_accuracy=result.get('train_accuracy'),
            test_accuracy=result.get('test_accuracy'),
            training_records=result.get('training_records'),
            top_features=result.get('top_features'),
            message=result.get('reason')
        )
        
    except Exception as e:
        logger.error(f"Training failed for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/batch")
async def train_batch(background_tasks: BackgroundTasks):
    """
    Train models for all devices with sufficient data.
    
    This is a batch training endpoint that:
    1. Discovers devices with enough historical data
    2. Trains models for each device
    3. Runs in background to avoid timeout
    
    Use this for initial setup or periodic retraining.
    
    Returns:
        Status indicating batch training started
    """
    try:
        # Get entities with sufficient data
        entities = db_client.get_available_entities(limit=50)
        
        if not entities:
            return {
                "status": "no_entities",
                "message": "No entities found with sufficient data"
            }
        
        # Start batch training in background
        background_tasks.add_task(batch_train_models, entities)
        
        return {
            "status": "started",
            "entities_to_train": len(entities),
            "entities": entities,
            "message": "Batch training started in background"
        }
        
    except Exception as e:
        logger.error(f"Batch training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def batch_train_models(entities: List[str]):
    """Background task for batch training"""
    logger.info(f"Starting batch training for {len(entities)} entities")
    
    results = {
        "success": 0,
        "failed": 0,
        "cached": 0
    }
    
    for entity_id in entities:
        try:
            # Check if enough data
            count = db_client.get_entity_state_count(entity_id, days=30)
            if count < 100:
                logger.info(f"Skipping {entity_id}: only {count} records")
                continue
            
            # Load and train
            historical_data = db_client.load_historical_states(entity_id, days=30)
            result = predictor.train(entity_id, historical_data, force_retrain=False)
            
            if result['status'] == 'success':
                results['success'] += 1
            elif result['status'] == 'cached':
                results['cached'] += 1
            else:
                results['failed'] += 1
            
            # Small delay to avoid overwhelming system
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to train {entity_id}: {e}")
            results['failed'] += 1
    
    logger.info(f"Batch training complete: {results}")
    
    # Store results in Redis
    await redis_client.setex(
        "batch_training:last_run",
        86400,  # 24 hours
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "results": results
        })
    )


# Prediction Endpoints
@app.get("/predict/{entity_id}", response_model=PredictResponse)
async def get_prediction(entity_id: str, hours_ahead: int = 24):
    """
    Get prediction for a device.
    
    Returns cached prediction if available (< 1 hour old),
    otherwise generates new prediction.
    
    Args:
        entity_id: Device to predict
        hours_ahead: Forecast horizon (default: 24 hours)
    
    Returns:
        List of hourly predictions
    
    Example:
        GET /predict/light.living_room?hours_ahead=24
    """
    try:
        # Check cache first
        cache_key = f"prediction:{entity_id}"
        cached = await redis_client.get(cache_key)
        
        if cached:
            logger.info(f"Returning cached prediction for {entity_id}")
            data = json.loads(cached)
            return PredictResponse(
                entity_id=entity_id,
                generated_at=data['generated_at'],
                predictions=data['predictions'],
                source='cache'
            )
        
        # Generate new prediction
        logger.info(f"Generating prediction for {entity_id}")
        
        predictions = predictor.predict(entity_id, hours_ahead=hours_ahead)
        
        if not predictions:
            raise HTTPException(
                status_code=404,
                detail=f"No model available for {entity_id}. Train it first with POST /train/{entity_id}"
            )
        
        # Prepare response
        response = PredictResponse(
            entity_id=entity_id,
            generated_at=datetime.now().isoformat(),
            predictions=[Prediction(**p) for p in predictions],
            source='generated'
        )
        
        # Cache the prediction
        cache_data = {
            "entity_id": entity_id,
            "generated_at": response.generated_at,
            "predictions": [p.dict() for p in response.predictions]
        }
        await redis_client.setex(
            cache_key,
            PREDICTION_CACHE_TTL,
            json.dumps(cache_data)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{entity_id}/summary")
async def get_prediction_summary(entity_id: str):
    """
    Get human-readable prediction summary.
    
    Returns:
        Summary with key insights (peak hours, next state change, etc.)
    
    Example:
        GET /predict/light.living_room/summary
        
        Returns:
        {
            "entity_id": "light.living_room",
            "next_24h": "Expected ON during 18:00-23:00",
            "high_probability_hours": [18, 19, 20, 21, 22],
            "low_probability_hours": [1, 2, 3, 4, 5],
            "next_state_change": {
                "time": "18:00",
                "to": "on",
                "confidence": 0.89
            }
        }
    """
    try:
        # Get full prediction
        prediction_response = await get_prediction(entity_id, hours_ahead=24)
        predictions = prediction_response.predictions
        
        # Analyze predictions
        high_prob_hours = [p.hour for p in predictions if p.probability_on > 0.8]
        low_prob_hours = [p.hour for p in predictions if p.probability_on < 0.2]
        
        # Find next significant state change
        current_state = predictions[0].predicted_state
        next_change = None
        
        for p in predictions[1:]:
            if p.predicted_state != current_state and p.confidence > 0.7:
                next_change = {
                    "time": f"{p.hour:02d}:00",
                    "to": p.predicted_state_label,
                    "confidence": p.confidence
                }
                break
        
        return {
            "entity_id": entity_id,
            "forecast_period": "24 hours",
            "high_probability_hours": high_prob_hours,
            "low_probability_hours": low_prob_hours,
            "next_state_change": next_change,
            "overall_confidence": sum(p.confidence for p in predictions) / len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Summary generation failed for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Management
@app.get("/models")
async def list_models():
    """
    List all trained models.
    
    Returns metadata for each trained model including accuracy.
    """
    try:
        model_files = list(predictor.model_cache_dir.glob("*.joblib"))
        
        models = []
        for model_file in model_files:
            entity_id = model_file.stem.replace('_', '.')
            
            # Get model info
            info = predictor.get_model_info(entity_id)
            if info:
                models.append({
                    "entity_id": entity_id,
                    "trained_at": info['trained_at'],
                    "test_accuracy": info['test_accuracy'],
                    "training_records": info['training_records']
                })
        
        return {
            "total_models": len(models),
            "models": models
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{entity_id}")
async def get_model_info_endpoint(entity_id: str):
    """
    Get detailed information about a specific model.
    
    Returns:
        Model metadata including feature importance
    """
    try:
        info = predictor.get_model_info(entity_id)
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for {entity_id}"
            )
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{entity_id}")
async def delete_model(entity_id: str):
    """
    Delete a trained model.
    
    Removes model from disk and clears cache.
    """
    try:
        model_path = predictor._get_model_path(entity_id)
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No model found for {entity_id}"
            )
        
        # Delete from disk
        model_path.unlink()
        
        # Clear from memory
        if entity_id in predictor.models:
            del predictor.models[entity_id]
        if entity_id in predictor.scalers:
            del predictor.scalers[entity_id]
        if entity_id in predictor.metadata:
            del predictor.metadata[entity_id]
        
        # Clear cached predictions
        await redis_client.delete(f"prediction:{entity_id}")
        
        logger.info(f"Model deleted for {entity_id}")
        
        return {
            "status": "deleted",
            "entity_id": entity_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Discovery Endpoint
@app.get("/discover")
async def discover_entities(min_records: int = 100):
    """
    Discover entities with sufficient data for training.
    
    WARNING: With very large databases (6M+ records/week), this can be slow.
    Consider using /discover/pattern instead for faster results.
    
    Args:
        min_records: Minimum state changes required
    
    Returns:
        List of entities with data counts
    """
    try:
        entities = db_client.get_available_entities(limit=100)
        
        entity_info = []
        for entity_id in entities:
            count = db_client.get_entity_state_count(entity_id, days=30)
            if count >= min_records:
                entity_info.append({
                    "entity_id": entity_id,
                    "state_count_30d": count,
                    "trainable": True
                })
        
        return {
            "total_found": len(entity_info),
            "entities": entity_info
        }
        
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/discover/pattern/{pattern}")
async def discover_by_pattern(pattern: str, limit: int = 50):
    """
    Discover entities by pattern (FAST - no history scan).
    
    This bypasses the slow history scan and just returns entities
    matching a pattern. Use this for large databases!
    
    Args:
        pattern: Entity pattern (e.g., 'light', 'switch', 'binary_sensor')
        limit: Max entities to return
    
    Returns:
        List of entity IDs
    
    Examples:
        GET /discover/pattern/light - All lights
        GET /discover/pattern/switch - All switches
        GET /discover/pattern/binary_sensor - All binary sensors
    """
    try:
        # Convert simple pattern to SQL LIKE pattern
        if not '%' in pattern and not '_' in pattern:
            sql_pattern = f"{pattern}.%"
        else:
            sql_pattern = pattern
        
        entities = db_client.get_entities_by_pattern(sql_pattern, limit=limit)
        
        return {
            "pattern": sql_pattern,
            "total_found": len(entities),
            "entities": entities,
            "note": "Use POST /train/{entity_id} to train these entities"
        }
        
    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)