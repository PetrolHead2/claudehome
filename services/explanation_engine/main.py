"""
Explanation Engine - Phase 4.1
Makes system decisions transparent and understandable

Primary Focus: Anomaly Explanations
Storage: Hybrid (cache recent, generate old)
Detail Level: Configurable (simple/detailed)
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import redis.asyncio as redis
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClaudeHome Explanation Engine")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ANOMALY_DETECTOR_URL = os.getenv("ANOMALY_DETECTOR_URL", "http://anomaly_detector:8000")
PREDICTION_SERVICE_URL = os.getenv("PREDICTION_SERVICE_URL", "http://prediction_service:8000")

# Global instances
redis_client: Optional[redis.Redis] = None

# Constants
EXPLANATION_CACHE_TTL = 86400 * 7  # 7 days


# Pydantic Models
class ExplanationRequest(BaseModel):
    target_id: str  # anomaly_id, entity_id, etc.
    target_type: str  # 'anomaly', 'prediction', 'automation'
    detail_level: str = "simple"  # 'simple' or 'detailed'


class CascadeEffect(BaseModel):
    effect: str
    probability: float
    severity: str  # 'low', 'medium', 'high'
    description: str


class WhatIfScenario(BaseModel):
    scenario_description: str
    changes: Dict[str, Any]
    detail_level: str = "simple"


class Explanation(BaseModel):
    explanation_id: str
    target_id: str
    target_type: str
    detail_level: str
    summary: str
    decision_chain: List[Dict[str, Any]] = []
    probable_causes: List[Dict[str, Any]] = []
    cascade_effects: List[CascadeEffect] = []
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = {}


@app.on_event("startup")
async def startup():
    """Initialize services"""
    global redis_client
    
    logger.info("Starting Explanation Engine...")
    
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connected")
    
    logger.info("Explanation Engine started successfully")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup"""
    if redis_client:
        await redis_client.close()
        logger.info("Shut down complete")


@app.get("/health")
async def health():
    """Service health check"""
    return {
        "status": "healthy",
        "service": "explanation_engine",
        "version": "4.1",
        "focus": "anomaly_explanations",
        "timestamp": datetime.now().isoformat()
    }


# ========== ANOMALY EXPLANATIONS ==========

@app.get("/explain/anomaly/{anomaly_id}")
async def explain_anomaly(
    anomaly_id: str,
    detail_level: str = Query("simple", regex="^(simple|detailed)$")
):
    """
    Explain why an anomaly was detected.
    
    Args:
        anomaly_id: Anomaly to explain
        detail_level: 'simple' or 'detailed'
    
    Returns:
        Comprehensive explanation of the anomaly
    """
    try:
        # Check cache first (hybrid storage)
        cache_key = f"claudehome:explanation:anomaly:{anomaly_id}:{detail_level}"
        cached = await redis_client.get(cache_key)
        
        if cached:
            logger.info(f"Returning cached explanation for {anomaly_id}")
            return json.loads(cached)
        
        # Generate new explanation
        logger.info(f"Generating explanation for anomaly {anomaly_id}")
        
        # Fetch anomaly details
        anomaly = await fetch_anomaly_details(anomaly_id)
        
        if not anomaly:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        # Generate explanation
        explanation = await generate_anomaly_explanation(anomaly, detail_level)
        
        # Cache explanation
        await redis_client.setex(
            cache_key,
            EXPLANATION_CACHE_TTL,
            json.dumps(explanation.dict())
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_anomaly_details(anomaly_id: str) -> Optional[Dict[str, Any]]:
    """Fetch anomaly from anomaly detector"""
    try:
        async with aiohttp.ClientSession() as session:
            # Try to get from Redis directly
            key = f"claudehome:anomaly:{anomaly_id}"
            data = await redis_client.get(key)
            
            if data:
                return json.loads(data)
            
            # Could also query anomaly detector API if needed
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch anomaly: {e}")
        return None


async def generate_anomaly_explanation(
    anomaly: Dict[str, Any],
    detail_level: str
) -> Explanation:
    """
    Generate comprehensive anomaly explanation.
    """
    entity_id = anomaly['entity_id']
    anomaly_type = anomaly['anomaly_type']
    
    # Fetch baseline for comparison
    baseline = await fetch_baseline(entity_id)
    
    # Generate summary
    if detail_level == "simple":
        summary = generate_simple_anomaly_summary(anomaly, baseline)
        decision_chain = []
        probable_causes = generate_probable_causes_simple(anomaly, baseline)
    else:
        summary = generate_detailed_anomaly_summary(anomaly, baseline)
        decision_chain = generate_decision_chain(anomaly, baseline)
        probable_causes = generate_probable_causes_detailed(anomaly, baseline)
    
    # Generate cascade effects
    cascade_effects = await generate_cascade_effects(anomaly, baseline)
    
    # Create explanation
    explanation_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    explanation = Explanation(
        explanation_id=explanation_id,
        target_id=anomaly['anomaly_id'],
        target_type='anomaly',
        detail_level=detail_level,
        summary=summary,
        decision_chain=decision_chain,
        probable_causes=probable_causes,
        cascade_effects=cascade_effects,
        confidence=anomaly.get('confidence', 0.0),
        timestamp=datetime.now().isoformat(),
        metadata={
            "entity_id": entity_id,
            "anomaly_type": anomaly_type,
            "severity": anomaly.get('severity', 'unknown')
        }
    )
    
    return explanation


async def fetch_baseline(entity_id: str) -> Optional[Dict[str, Any]]:
    """Fetch baseline from anomaly detector"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ANOMALY_DETECTOR_URL}/baseline/{entity_id}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
    except Exception as e:
        logger.debug(f"Failed to fetch baseline: {e}")
        return None


def generate_simple_anomaly_summary(
    anomaly: Dict[str, Any],
    baseline: Optional[Dict[str, Any]]
) -> str:
    """Generate simple 1-2 sentence summary"""
    entity_id = anomaly['entity_id']
    description = anomaly['description']
    severity = anomaly['severity']
    
    if baseline:
        return (
            f"{entity_id} is showing unusual behavior: {description}. "
            f"This is a {severity} severity anomaly based on learned patterns."
        )
    else:
        return (
            f"{entity_id} triggered an anomaly: {description}. "
            f"Severity: {severity}."
        )


def generate_detailed_anomaly_summary(
    anomaly: Dict[str, Any],
    baseline: Optional[Dict[str, Any]]
) -> str:
    """Generate detailed multi-paragraph summary"""
    entity_id = anomaly['entity_id']
    anomaly_type = anomaly['anomaly_type']
    description = anomaly['description']
    severity = anomaly['severity']
    score = anomaly.get('score', 0)
    
    summary_parts = []
    
    # Main anomaly description
    summary_parts.append(
        f"An anomaly was detected for {entity_id} at {anomaly['timestamp'][:19]}. "
        f"Type: {anomaly_type.replace('_', ' ').title()}. "
        f"Severity: {severity.upper()}."
    )
    
    # Statistical details
    summary_parts.append(
        f"Statistical Analysis: {description} "
        f"(Anomaly score: {score:.2f})"
    )
    
    # Baseline comparison
    if baseline:
        state_type = baseline.get('state_type', 'unknown')
        
        if state_type == 'binary':
            typical_hours = baseline.get('typical_on_hours', [])
            summary_parts.append(
                f"Baseline: This device is typically active during hours {typical_hours}. "
                f"It shows activity {baseline.get('typical_on_percentage', 0):.1f}% of the time. "
                f"The baseline was learned from {baseline.get('data_points', 0)} observations."
            )
        elif state_type == 'numeric':
            mean = baseline.get('value_mean', 0)
            std = baseline.get('value_std', 0)
            summary_parts.append(
                f"Baseline: Normal range is {mean:.1f} ± {std:.1f}. "
                f"Min: {baseline.get('value_min', 0):.1f}, Max: {baseline.get('value_max', 0):.1f}. "
                f"The baseline was learned from {baseline.get('data_points', 0)} observations."
            )
    
    return " ".join(summary_parts)


def generate_decision_chain(
    anomaly: Dict[str, Any],
    baseline: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate step-by-step decision chain"""
    chain = []
    
    # Step 1: Observation
    chain.append({
        "step": 1,
        "factor": "Current Observation",
        "value": f"{anomaly['entity_id']} state: {anomaly.get('current_value')}",
        "weight": 0.3,
        "confidence": 1.0
    })
    
    # Step 2: Baseline comparison
    if baseline:
        chain.append({
            "step": 2,
            "factor": "Baseline Comparison",
            "value": f"Learned pattern from {baseline.get('data_points', 0)} observations",
            "weight": 0.3,
            "confidence": 0.85
        })
    
    # Step 3: Statistical analysis
    chain.append({
        "step": 3,
        "factor": "Statistical Analysis",
        "value": f"{anomaly['description']}",
        "weight": 0.25,
        "confidence": anomaly.get('confidence', 0.8)
    })
    
    # Step 4: Severity assessment
    chain.append({
        "step": 4,
        "factor": "Severity Assessment",
        "value": f"Classified as {anomaly['severity']} severity",
        "weight": 0.15,
        "confidence": 0.9
    })
    
    return chain


def generate_probable_causes_simple(
    anomaly: Dict[str, Any],
    baseline: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate simple probable causes"""
    anomaly_type = anomaly['anomaly_type']
    
    causes = []
    
    if anomaly_type == 'pattern':
        causes.append({
            "cause": "Unusual user activity",
            "probability": 0.6
        })
        causes.append({
            "cause": "Automation error or timing issue",
            "probability": 0.3
        })
        causes.append({
            "cause": "Manual override",
            "probability": 0.1
        })
    
    elif anomaly_type == 'value':
        causes.append({
            "cause": "Sensor malfunction",
            "probability": 0.7
        })
        causes.append({
            "cause": "Actual extreme condition",
            "probability": 0.2
        })
        causes.append({
            "cause": "Electrical interference",
            "probability": 0.1
        })
    
    elif anomaly_type == 'prediction_mismatch':
        causes.append({
            "cause": "Behavior changed from learned pattern",
            "probability": 0.5
        })
        causes.append({
            "cause": "Automation triggered unexpectedly",
            "probability": 0.3
        })
        causes.append({
            "cause": "Guest or unusual circumstances",
            "probability": 0.2
        })
    
    elif anomaly_type == 'change_rate':
        causes.append({
            "cause": "Device malfunction (flickering/cycling)",
            "probability": 0.6
        })
        causes.append({
            "cause": "Automation conflict or loop",
            "probability": 0.3
        })
        causes.append({
            "cause": "External trigger (power fluctuation)",
            "probability": 0.1
        })
    
    return causes


def generate_probable_causes_detailed(
    anomaly: Dict[str, Any],
    baseline: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate detailed probable causes with evidence"""
    simple_causes = generate_probable_causes_simple(anomaly, baseline)
    
    # Enhance with evidence and details
    for cause in simple_causes:
        if cause['cause'] == "Unusual user activity":
            cause['evidence'] = "No regular pattern matches this behavior"
            cause['recommendation'] = "Monitor for recurrence; may become new normal"
        
        elif cause['cause'] == "Sensor malfunction":
            cause['evidence'] = "Reading falls outside physically possible range"
            cause['recommendation'] = "Check sensor, recalibrate or replace"
        
        elif cause['cause'] == "Device malfunction (flickering/cycling)":
            cause['evidence'] = f"Change rate {anomaly.get('score', 0):.1f}x normal"
            cause['recommendation'] = "Inspect device, check electrical connections"
        
        elif cause['cause'] == "Behavior changed from learned pattern":
            cause['evidence'] = f"ML prediction confidence was {anomaly.get('score', 0):.0%}"
            cause['recommendation'] = "Pattern may be evolving; continue monitoring"
    
    return simple_causes


async def generate_cascade_effects(
    anomaly: Dict[str, Any],
    baseline: Optional[Dict[str, Any]]
) -> List[CascadeEffect]:
    """
    Generate cascade effects - what could this anomaly lead to?
    """
    effects = []
    
    anomaly_type = anomaly['anomaly_type']
    severity = anomaly['severity']
    entity_id = anomaly['entity_id']
    
    # Pattern anomalies
    if anomaly_type == 'pattern':
        effects.append(CascadeEffect(
            effect="Energy waste",
            probability=0.7,
            severity="low" if severity == "low" else "medium",
            description=f"If {entity_id} remains in unusual state, may waste energy"
        ))
        
        effects.append(CascadeEffect(
            effect="User annoyance",
            probability=0.5,
            severity="low",
            description="Unexpected device behavior may frustrate user"
        ))
        
        if severity in ['high', 'critical']:
            effects.append(CascadeEffect(
                effect="Security concern",
                probability=0.3,
                severity="high",
                description="Unusual activity at odd hours may indicate security issue"
            ))
    
    # Value anomalies
    elif anomaly_type == 'value':
        effects.append(CascadeEffect(
            effect="Incorrect automation triggers",
            probability=0.8,
            severity="medium",
            description="Faulty sensor readings may cause incorrect automations"
        ))
        
        effects.append(CascadeEffect(
            effect="Safety risk",
            probability=0.4 if severity in ['high', 'critical'] else 0.1,
            severity="high" if severity in ['high', 'critical'] else "low",
            description="Extreme readings may indicate hazardous conditions"
        ))
        
        effects.append(CascadeEffect(
            effect="Data corruption",
            probability=0.6,
            severity="low",
            description="Bad sensor data pollutes historical records and ML models"
        ))
    
    # Prediction mismatches
    elif anomaly_type == 'prediction_mismatch':
        effects.append(CascadeEffect(
            effect="ML model degradation",
            probability=0.5,
            severity="medium",
            description="If pattern truly changed, predictions will become less accurate"
        ))
        
        effects.append(CascadeEffect(
            effect="Automation timing issues",
            probability=0.6,
            severity="low",
            description="Predictive automations may trigger at wrong times"
        ))
    
    # Change rate anomalies
    elif anomaly_type == 'change_rate':
        effects.append(CascadeEffect(
            effect="Device failure imminent",
            probability=0.7,
            severity="high",
            description="Rapid cycling often precedes complete device failure"
        ))
        
        effects.append(CascadeEffect(
            effect="Increased wear",
            probability=0.9,
            severity="medium",
            description="Excessive switching reduces device lifespan"
        ))
        
        effects.append(CascadeEffect(
            effect="Energy spike",
            probability=0.5,
            severity="low",
            description="Frequent state changes consume more power"
        ))
    
    return effects


# ========== PREDICTION EXPLANATIONS ==========

@app.get("/explain/prediction/{entity_id}")
async def explain_prediction(
    entity_id: str,
    detail_level: str = Query("simple", regex="^(simple|detailed)$")
):
    """
    Explain why a prediction was made.
    
    Args:
        entity_id: Entity to explain prediction for
        detail_level: 'simple' or 'detailed'
    
    Returns:
        Explanation of prediction reasoning
    """
    try:
        # Check cache
        cache_key = f"claudehome:explanation:prediction:{entity_id}:{detail_level}"
        cached = await redis_client.get(cache_key)
        
        if cached:
            logger.info(f"Returning cached prediction explanation for {entity_id}")
            return json.loads(cached)
        
        # Generate explanation
        logger.info(f"Generating prediction explanation for {entity_id}")
        
        # Fetch prediction and model details
        prediction = await fetch_prediction(entity_id)
        model_info = await fetch_model_info(entity_id)
        
        if not prediction or not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction or model available for {entity_id}"
            )
        
        # Generate explanation
        explanation = generate_prediction_explanation(
            entity_id,
            prediction,
            model_info,
            detail_level
        )
        
        # Cache (shorter TTL for predictions - they change)
        await redis_client.setex(
            cache_key,
            3600,  # 1 hour
            json.dumps(explanation)
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Prediction explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_prediction(entity_id: str) -> Optional[Dict[str, Any]]:
    """Fetch prediction from prediction service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{PREDICTION_SERVICE_URL}/predict/{entity_id}",
                params={"hours_ahead": 24},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
    except Exception as e:
        logger.debug(f"Failed to fetch prediction: {e}")
        return None


async def fetch_model_info(entity_id: str) -> Optional[Dict[str, Any]]:
    """Fetch model details from prediction service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{PREDICTION_SERVICE_URL}/models/{entity_id}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
    except Exception as e:
        logger.debug(f"Failed to fetch model info: {e}")
        return None


def generate_prediction_explanation(
    entity_id: str,
    prediction: Dict[str, Any],
    model_info: Dict[str, Any],
    detail_level: str
) -> Dict[str, Any]:
    """Generate prediction explanation"""
    
    if detail_level == "simple":
        summary = (
            f"Prediction for {entity_id}: Based on {model_info.get('training_records', 0)} "
            f"historical records with {model_info.get('test_accuracy', 0)*100:.1f}% accuracy. "
            f"Next 24 hours forecast available."
        )
        factors = []
    else:
        summary = (
            f"ML Model for {entity_id}: Random Forest with {model_info.get('training_records', 0)} "
            f"training samples. Test accuracy: {model_info.get('test_accuracy', 0)*100:.1f}%. "
            f"Top features: {', '.join(model_info.get('top_features', [])[:3])}. "
            f"Model trained on {model_info.get('trained_at', 'unknown')[:10]}."
        )
        
        # Feature importance
        factors = []
        for idx, feature in enumerate(model_info.get('top_features', [])[:5]):
            factors.append({
                "rank": idx + 1,
                "feature": feature,
                "importance": "High" if idx < 2 else "Medium"
            })
    
    return {
        "entity_id": entity_id,
        "summary": summary,
        "model_accuracy": model_info.get('test_accuracy', 0),
        "training_records": model_info.get('training_records', 0),
        "factors": factors,
        "detail_level": detail_level
    }


# ========== UTILITY ENDPOINTS ==========

@app.get("/explanations/recent")
async def get_recent_explanations(limit: int = 10):
    """Get recently generated explanations"""
    try:
        # Get from Redis (recently cached explanations)
        keys = await redis_client.keys("claudehome:explanation:*")
        
        explanations = []
        for key in keys[:limit]:
            data = await redis_client.get(key)
            if data:
                explanations.append(json.loads(data))
        
        return {
            "count": len(explanations),
            "explanations": explanations
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent explanations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/explanations/cache")
async def clear_explanation_cache():
    """Clear explanation cache"""
    try:
        keys = await redis_client.keys("claudehome:explanation:*")
        if keys:
            await redis_client.delete(*keys)
        
        return {
            "status": "cache_cleared",
            "count": len(keys)
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)