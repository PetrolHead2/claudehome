"""
Meta-Learner - Phase 4.3
Learns how to learn - improves learning strategies themselves

Capabilities:
- Analyzes which prediction models perform best
- Learns optimal confidence thresholds
- Discovers meta-patterns (patterns about patterns)
- Experiments with strategy variations
- Evolves learning approaches based on outcomes
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import redis.asyncio as redis
import aiohttp
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClaudeHome Meta-Learner")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
OUTCOME_TRACKER_URL = os.getenv("OUTCOME_TRACKER_URL", "http://outcome_tracker:8000")
PREDICTION_SERVICE_URL = os.getenv("PREDICTION_SERVICE_URL", "http://prediction_service:8000")
ANOMALY_DETECTOR_URL = os.getenv("ANOMALY_DETECTOR_URL", "http://anomaly_detector:8000")
PROACTIVE_ENGINE_URL = os.getenv("PROACTIVE_ENGINE_URL", "http://proactive_engine:8000")

# Global instances
redis_client: Optional[redis.Redis] = None


# Pydantic Models
class LearningStrategy(BaseModel):
    strategy_id: str
    strategy_type: str  # 'prediction_threshold', 'anomaly_sensitivity', 'pattern_confidence'
    parameters: Dict[str, Any]
    performance_score: float = 0.0
    trials: int = 0
    successes: int = 0
    failures: int = 0
    last_updated: str
    metadata: Dict[str, Any] = {}


class MetaPattern(BaseModel):
    pattern_id: str
    pattern_type: str
    description: str
    evidence: List[Dict[str, Any]]
    confidence: float
    discovered_at: str
    impact: str  # 'high', 'medium', 'low'
    actionable_insight: str
    metadata: Dict[str, Any] = {}


class StrategyRecommendation(BaseModel):
    recommendation_id: str
    target_service: str  # 'prediction_service', 'anomaly_detector', etc.
    current_strategy: Dict[str, Any]
    recommended_strategy: Dict[str, Any]
    expected_improvement: float
    confidence: float
    reasoning: str
    created_at: str


class LearningInsight(BaseModel):
    insight_id: str
    insight_type: str
    title: str
    description: str
    evidence: List[str]
    confidence: float
    timestamp: str
    actionable: bool
    action_taken: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Initialize services"""
    global redis_client
    
    logger.info("Starting Meta-Learner...")
    
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connected")
    
    # Start background learning tasks
    asyncio.create_task(analyze_outcomes_periodically())
    asyncio.create_task(discover_meta_patterns_periodically())
    
    logger.info("Meta-Learner started successfully")


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
        "service": "meta_learner",
        "version": "4.3",
        "learning": "active",
        "timestamp": datetime.now().isoformat()
    }


# ========== STRATEGY ANALYSIS ==========

@app.get("/analyze/strategies")
async def analyze_learning_strategies():
    """
    Analyze which learning strategies are performing best.
    
    Compares:
    - Different prediction confidence thresholds
    - Anomaly detection sensitivity levels
    - Pattern recognition parameters
    """
    try:
        insights = []
        
        # Analyze prediction strategies
        pred_insights = await analyze_prediction_strategies()
        insights.extend(pred_insights)
        
        # Analyze anomaly detection strategies
        anom_insights = await analyze_anomaly_strategies()
        insights.extend(anom_insights)
        
        # Analyze pattern recognition
        pattern_insights = await analyze_pattern_strategies()
        insights.extend(pattern_insights)
        
        return {
            "analyzed_at": datetime.now().isoformat(),
            "total_insights": len(insights),
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Strategy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def analyze_prediction_strategies() -> List[LearningInsight]:
    """Analyze prediction model performance"""
    insights = []
    
    try:
        # Get all prediction models
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PREDICTION_SERVICE_URL}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get('models', [])
                else:
                    return insights
        
        # Get outcome data
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OUTCOME_TRACKER_URL}/analytics/summary") as resp:
                if resp.status == 200:
                    outcomes = await resp.json()
                else:
                    return insights
        
        # Analyze model accuracy vs real-world success
        for model in models:
            entity_id = model['entity_id']
            test_accuracy = model.get('test_accuracy', 0)
            
            # Compare to actual automation success for this entity
            # High test accuracy but low automation success = model overfitting or wrong features
            
            insight_id = f"insight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            if test_accuracy > 0.9:
                insights.append(LearningInsight(
                    insight_id=insight_id,
                    insight_type="model_performance",
                    title=f"High accuracy model for {entity_id}",
                    description=f"Model achieves {test_accuracy*100:.1f}% test accuracy. Monitor real-world performance.",
                    evidence=[f"Test accuracy: {test_accuracy:.2%}", f"Training records: {model.get('training_records', 0)}"],
                    confidence=0.9,
                    timestamp=datetime.now().isoformat(),
                    actionable=True
                ))
            elif test_accuracy < 0.7:
                insights.append(LearningInsight(
                    insight_id=insight_id,
                    insight_type="model_needs_improvement",
                    title=f"Low accuracy model for {entity_id}",
                    description=f"Model only achieves {test_accuracy*100:.1f}% accuracy. Consider retraining with more data or different features.",
                    evidence=[f"Test accuracy: {test_accuracy:.2%}"],
                    confidence=0.85,
                    timestamp=datetime.now().isoformat(),
                    actionable=True
                ))
        
        # Meta-insight: Which feature engineering works best?
        if len(models) >= 3:
            avg_accuracy = np.mean([m.get('test_accuracy', 0) for m in models])
            
            insights.append(LearningInsight(
                insight_id=f"insight_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="meta_learning",
                title="Overall prediction performance",
                description=f"Average model accuracy across {len(models)} models: {avg_accuracy*100:.1f}%",
                evidence=[f"{len(models)} models analyzed", f"Average accuracy: {avg_accuracy:.2%}"],
                confidence=0.95,
                timestamp=datetime.now().isoformat(),
                actionable=True if avg_accuracy < 0.75 else False
            ))
        
        return insights
        
    except Exception as e:
        logger.error(f"Prediction strategy analysis failed: {e}")
        return insights


async def analyze_anomaly_strategies() -> List[LearningInsight]:
    """Analyze anomaly detection effectiveness"""
    insights = []
    
    try:
        # Get anomaly stats
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ANOMALY_DETECTOR_URL}/stats") as resp:
                if resp.status == 200:
                    stats = await resp.json()
                else:
                    return insights
        
        total_anomalies = stats.get('total_anomalies', 0)
        critical_count = stats.get('critical_count', 0)
        high_count = stats.get('high_count', 0)
        medium_count = stats.get('medium_count', 0)
        low_count = stats.get('low_count', 0)
        
        # Analyze severity distribution
        if total_anomalies > 0:
            critical_rate = critical_count / total_anomalies
            high_rate = high_count / total_anomalies
            
            insight_id = f"insight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Too many criticals = too sensitive
            if critical_rate > 0.3:
                insights.append(LearningInsight(
                    insight_id=insight_id,
                    insight_type="anomaly_sensitivity",
                    title="High critical anomaly rate",
                    description=f"{critical_rate*100:.1f}% of anomalies are critical. May be too sensitive.",
                    evidence=[f"Critical: {critical_count}", f"Total: {total_anomalies}"],
                    confidence=0.75,
                    timestamp=datetime.now().isoformat(),
                    actionable=True
                ))
            
            # Too few alerts = too lenient
            elif critical_rate < 0.05 and high_rate < 0.1:
                insights.append(LearningInsight(
                    insight_id=insight_id,
                    insight_type="anomaly_sensitivity",
                    title="Low severity anomaly rate",
                    description=f"Only {(critical_rate+high_rate)*100:.1f}% are high severity. Detection may be too lenient.",
                    evidence=[f"Critical+High: {critical_count+high_count}", f"Total: {total_anomalies}"],
                    confidence=0.7,
                    timestamp=datetime.now().isoformat(),
                    actionable=True
                ))
            
            # Good balance
            else:
                insights.append(LearningInsight(
                    insight_id=insight_id,
                    insight_type="anomaly_balance",
                    title="Anomaly detection well-balanced",
                    description=f"Good severity distribution: {critical_count} critical, {high_count} high, {medium_count} medium.",
                    evidence=[f"Total: {total_anomalies}", "Balanced distribution"],
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    actionable=False
                ))
        
        return insights
        
    except Exception as e:
        logger.error(f"Anomaly strategy analysis failed: {e}")
        return insights


async def analyze_pattern_strategies() -> List[LearningInsight]:
    """Analyze pattern recognition effectiveness"""
    insights = []
    
    try:
        # Get proactive engine suggestions
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROACTIVE_ENGINE_URL}/suggestions/pending") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    suggestions = data.get('suggestions', [])
                else:
                    return insights
        
        # Analyze suggestion quality
        if len(suggestions) > 0:
            avg_confidence = np.mean([s.get('confidence', 0) for s in suggestions])
            
            insight_id = f"insight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            insights.append(LearningInsight(
                insight_id=insight_id,
                insight_type="pattern_recognition",
                title=f"Pattern recognition generating {len(suggestions)} suggestions",
                description=f"Average confidence: {avg_confidence*100:.1f}%",
                evidence=[f"{len(suggestions)} pending suggestions", f"Avg confidence: {avg_confidence:.2%}"],
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                actionable=True if avg_confidence < 0.6 else False
            ))
        
        return insights
        
    except Exception as e:
        logger.error(f"Pattern strategy analysis failed: {e}")
        return insights


# ========== META-PATTERN DISCOVERY ==========

@app.get("/discover/meta-patterns")
async def discover_meta_patterns():
    """
    Discover patterns about patterns.
    
    Examples:
    - "Models with >200 training samples consistently perform better"
    - "Anomalies at night are more likely false positives"
    - "Prediction-enhanced suggestions have higher acceptance rate"
    """
    try:
        patterns = []
        
        # Pattern 1: Training data size vs accuracy
        pattern = await discover_training_size_pattern()
        if pattern:
            patterns.append(pattern)
        
        # Pattern 2: Temporal anomaly patterns
        pattern = await discover_temporal_anomaly_pattern()
        if pattern:
            patterns.append(pattern)
        
        # Pattern 3: Prediction-enhanced suggestion effectiveness
        pattern = await discover_prediction_enhancement_pattern()
        if pattern:
            patterns.append(pattern)
        
        # Store discovered patterns
        for pattern in patterns:
            await store_meta_pattern(pattern)
        
        return {
            "discovered_at": datetime.now().isoformat(),
            "patterns_found": len(patterns),
            "patterns": [p.dict() for p in patterns]
        }
        
    except Exception as e:
        logger.error(f"Meta-pattern discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def discover_training_size_pattern() -> Optional[MetaPattern]:
    """Discover relationship between training data size and model performance"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PREDICTION_SERVICE_URL}/models") as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                models = data.get('models', [])
        
        if len(models) < 3:
            return None
        
        # Group by training size
        small_models = [m for m in models if m.get('training_records', 0) < 150]
        large_models = [m for m in models if m.get('training_records', 0) >= 150]
        
        if len(small_models) == 0 or len(large_models) == 0:
            return None
        
        small_avg_acc = np.mean([m.get('test_accuracy', 0) for m in small_models])
        large_avg_acc = np.mean([m.get('test_accuracy', 0) for m in large_models])
        
        # Significant difference?
        if large_avg_acc - small_avg_acc > 0.05:  # 5% improvement
            pattern_id = f"metapat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return MetaPattern(
                pattern_id=pattern_id,
                pattern_type="training_data_correlation",
                description=f"Models with ≥150 training records perform {(large_avg_acc-small_avg_acc)*100:.1f}% better",
                evidence=[
                    {"type": "small_models", "count": len(small_models), "avg_accuracy": small_avg_acc},
                    {"type": "large_models", "count": len(large_models), "avg_accuracy": large_avg_acc}
                ],
                confidence=0.85,
                discovered_at=datetime.now().isoformat(),
                impact="high",
                actionable_insight="Prioritize training models with at least 150 historical records"
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Training size pattern discovery failed: {e}")
        return None


async def discover_temporal_anomaly_pattern() -> Optional[MetaPattern]:
    """Discover if anomalies cluster at certain times"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ANOMALY_DETECTOR_URL}/anomalies?limit=100") as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                anomalies = data.get('anomalies', [])
        
        if len(anomalies) < 20:
            return None
        
        # Analyze by hour
        hours = defaultdict(int)
        for anom in anomalies:
            timestamp = anom.get('timestamp', '')
            if timestamp:
                hour = datetime.fromisoformat(timestamp[:19]).hour
                hours[hour] += 1
        
        # Find peak hours
        if hours:
            max_hour = max(hours, key=hours.get)
            max_count = hours[max_hour]
            total = sum(hours.values())
            
            # If >20% occur in single hour, that's significant
            if max_count / total > 0.2:
                pattern_id = f"metapat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                return MetaPattern(
                    pattern_id=pattern_id,
                    pattern_type="temporal_clustering",
                    description=f"{max_count/total*100:.1f}% of anomalies occur around {max_hour}:00",
                    evidence=[
                        {"hour": max_hour, "count": max_count, "percentage": max_count/total},
                        {"total_anomalies": total}
                    ],
                    confidence=0.75,
                    discovered_at=datetime.now().isoformat(),
                    impact="medium",
                    actionable_insight=f"Investigate why anomalies cluster at {max_hour}:00. May be normal activity pattern."
                )
        
        return None
        
    except Exception as e:
        logger.error(f"Temporal pattern discovery failed: {e}")
        return None


async def discover_prediction_enhancement_pattern() -> Optional[MetaPattern]:
    """Discover if prediction-enhanced suggestions perform better"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROACTIVE_ENGINE_URL}/suggestions/pending") as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                suggestions = data.get('suggestions', [])
        
        if len(suggestions) < 5:
            return None
        
        # Check which have prediction data
        with_predictions = [s for s in suggestions if s.get('pattern', {}).get('predictions')]
        without_predictions = [s for s in suggestions if not s.get('pattern', {}).get('predictions')]
        
        if len(with_predictions) > 0 and len(without_predictions) > 0:
            with_avg_conf = np.mean([s.get('confidence', 0) for s in with_predictions])
            without_avg_conf = np.mean([s.get('confidence', 0) for s in without_predictions])
            
            if with_avg_conf - without_avg_conf > 0.1:  # 10% higher confidence
                pattern_id = f"metapat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                return MetaPattern(
                    pattern_id=pattern_id,
                    pattern_type="prediction_enhancement_effectiveness",
                    description=f"Prediction-enhanced suggestions have {(with_avg_conf-without_avg_conf)*100:.1f}% higher confidence",
                    evidence=[
                        {"type": "with_predictions", "count": len(with_predictions), "avg_confidence": with_avg_conf},
                        {"type": "without_predictions", "count": len(without_predictions), "avg_confidence": without_avg_conf}
                    ],
                    confidence=0.8,
                    discovered_at=datetime.now().isoformat(),
                    impact="high",
                    actionable_insight="Prioritize training prediction models for frequently used entities"
                )
        
        return None
        
    except Exception as e:
        logger.error(f"Prediction enhancement pattern discovery failed: {e}")
        return None


async def store_meta_pattern(pattern: MetaPattern):
    """Store discovered meta-pattern"""
    try:
        key = f"claudehome:metalearner:pattern:{pattern.pattern_id}"
        value = json.dumps(pattern.dict())
        await redis_client.setex(key, 86400 * 30, value)  # 30 days
        
        # Add to list
        list_key = "claudehome:metalearner:patterns"
        await redis_client.lpush(list_key, pattern.pattern_id)
        await redis_client.ltrim(list_key, 0, 99)  # Keep last 100
        
        logger.info(f"Stored meta-pattern: {pattern.description}")
        
    except Exception as e:
        logger.error(f"Failed to store meta-pattern: {e}")


# ========== STRATEGY RECOMMENDATIONS ==========

@app.get("/recommendations")
async def get_strategy_recommendations():
    """
    Generate recommendations for improving learning strategies.
    
    Based on meta-patterns and performance analysis.
    """
    try:
        recommendations = []
        
        # Get recent insights
        insights = await analyze_learning_strategies()
        
        for insight in insights.get('insights', []):
            if insight.get('actionable'):
                # Generate recommendation from insight
                rec = await generate_recommendation_from_insight(insight)
                if rec:
                    recommendations.append(rec)
        
        # Get meta-patterns
        pattern_keys = await redis_client.lrange("claudehome:metalearner:patterns", 0, 9)
        
        for pattern_id in pattern_keys:
            key = f"claudehome:metalearner:pattern:{pattern_id}"
            data = await redis_client.get(key)
            
            if data:
                pattern = MetaPattern(**json.loads(data))
                rec = await generate_recommendation_from_pattern(pattern)
                if rec:
                    recommendations.append(rec)
        
        return {
            "generated_at": datetime.now().isoformat(),
            "recommendation_count": len(recommendations),
            "recommendations": [r.dict() for r in recommendations]
        }
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_recommendation_from_insight(insight: Dict[str, Any]) -> Optional[StrategyRecommendation]:
    """Generate actionable recommendation from insight"""
    try:
        insight_type = insight.get('insight_type')
        
        if insight_type == "model_needs_improvement":
            rec_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            return StrategyRecommendation(
                recommendation_id=rec_id,
                target_service="prediction_service",
                current_strategy={"training_records": "insufficient"},
                recommended_strategy={"action": "collect_more_data", "target": "200+ records"},
                expected_improvement=0.15,
                confidence=0.8,
                reasoning=insight.get('description', ''),
                created_at=datetime.now().isoformat()
            )
        
        elif insight_type == "anomaly_sensitivity":
            rec_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            if "too sensitive" in insight.get('description', '').lower():
                return StrategyRecommendation(
                    recommendation_id=rec_id,
                    target_service="anomaly_detector",
                    current_strategy={"threshold": "current"},
                    recommended_strategy={"action": "increase_threshold", "z_score": 3.5},
                    expected_improvement=0.2,
                    confidence=0.75,
                    reasoning="Too many critical anomalies suggests over-sensitivity",
                    created_at=datetime.now().isoformat()
                )
        
        return None
        
    except Exception as e:
        logger.error(f"Recommendation generation from insight failed: {e}")
        return None


async def generate_recommendation_from_pattern(pattern: MetaPattern) -> Optional[StrategyRecommendation]:
    """Generate recommendation from meta-pattern"""
    try:
        if pattern.pattern_type == "training_data_correlation":
            rec_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            return StrategyRecommendation(
                recommendation_id=rec_id,
                target_service="prediction_service",
                current_strategy={"min_training_records": "any"},
                recommended_strategy={"min_training_records": 150},
                expected_improvement=0.05,
                confidence=pattern.confidence,
                reasoning=pattern.actionable_insight,
                created_at=datetime.now().isoformat()
            )
        
        elif pattern.pattern_type == "prediction_enhancement_effectiveness":
            rec_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            return StrategyRecommendation(
                recommendation_id=rec_id,
                target_service="proactive_engine",
                current_strategy={"prediction_usage": "opportunistic"},
                recommended_strategy={"prediction_usage": "prioritize_trained_entities"},
                expected_improvement=0.1,
                confidence=pattern.confidence,
                reasoning=pattern.actionable_insight,
                created_at=datetime.now().isoformat()
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Recommendation generation from pattern failed: {e}")
        return None


# ========== BACKGROUND TASKS ==========

async def analyze_outcomes_periodically():
    """Analyze outcomes every hour"""
    logger.info("Starting periodic outcome analysis...")
    
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            
            logger.info("Running periodic strategy analysis...")
            await analyze_learning_strategies()
            
        except Exception as e:
            logger.error(f"Periodic analysis error: {e}")
            await asyncio.sleep(3600)


async def discover_meta_patterns_periodically():
    """Discover meta-patterns every 6 hours"""
    logger.info("Starting periodic meta-pattern discovery...")
    
    while True:
        try:
            await asyncio.sleep(21600)  # Every 6 hours
            
            logger.info("Running periodic meta-pattern discovery...")
            await discover_meta_patterns()
            
        except Exception as e:
            logger.error(f"Periodic discovery error: {e}")
            await asyncio.sleep(21600)


@app.get("/insights/recent")
async def get_recent_insights(limit: int = 20):
    """Get recent learning insights"""
    try:
        # This would typically query a database
        # For now, return from recent analysis
        insights_data = await analyze_learning_strategies()
        
        return {
            "count": len(insights_data.get('insights', [])),
            "insights": insights_data.get('insights', [])[:limit]
        }
        
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns/meta")
async def get_meta_patterns(limit: int = 20):
    """Get discovered meta-patterns"""
    try:
        pattern_ids = await redis_client.lrange("claudehome:metalearner:patterns", 0, limit-1)
        
        patterns = []
        for pattern_id in pattern_ids:
            key = f"claudehome:metalearner:pattern:{pattern_id}"
            data = await redis_client.get(key)
            
            if data:
                patterns.append(json.loads(data))
        
        return {
            "count": len(patterns),
            "patterns": patterns
        }
        
    except Exception as e:
        logger.error(f"Failed to get meta-patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)