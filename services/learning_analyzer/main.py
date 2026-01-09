"""
Learning Analyzer - System Intelligence Engine

Analyzes outcomes, overrides, and patterns to continuously improve:
- Tracks success rates by pattern type
- Measures prediction accuracy
- Identifies improvement opportunities
- Generates learning digests
- Updates confidence thresholds
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learning_analyzer")

app = FastAPI(title="ClaudeHome Learning Analyzer")

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000")
ANALYSIS_INTERVAL_HOURS = int(os.getenv("ANALYSIS_INTERVAL_HOURS", "24"))

redis_client: Optional[redis.Redis] = None


class PatternTypeMetrics(BaseModel):
    """Metrics for a specific pattern type"""
    pattern_type: str
    total_suggestions: int = 0
    deployed: int = 0
    successful: int = 0
    failed: int = 0
    overridden: int = 0
    success_rate: float = 0.0
    override_rate: float = 0.0
    avg_confidence: float = 0.0


class LearningInsight(BaseModel):
    """A learned insight"""
    insight_id: str
    insight_type: str  # "success_pattern", "failure_pattern", "improvement"
    title: str
    description: str
    supporting_data: Dict[str, Any]
    confidence: float
    actionable: bool
    action: Optional[str] = None
    created_at: str


class LearningDigest(BaseModel):
    """Daily learning summary"""
    digest_id: str
    date: str
    period_start: str
    period_end: str
    total_suggestions: int
    total_deployments: int
    total_overrides: int
    pattern_metrics: List[PatternTypeMetrics]
    insights: List[LearningInsight]
    recommendations: List[str]
    overall_success_rate: float


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Learning Analyzer started")
    
    # Start background analysis
    asyncio.create_task(periodic_analysis())
    asyncio.create_task(listen_for_events())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "learning_analyzer"
    }


@app.get("/metrics/pattern_types")
async def get_pattern_metrics():
    """Get metrics by pattern type"""
    try:
        metrics = await analyze_pattern_types()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights")
async def get_insights(limit: int = 20):
    """Get recent insights"""
    try:
        keys = []
        async for key in redis_client.scan_iter("claudehome:insights:*"):
            keys.append(key)
        
        insights = []
        for key in sorted(keys, reverse=True)[:limit]:
            data = await redis_client.get(key)
            if data:
                insights.append(json.loads(data))
        
        return {"count": len(insights), "insights": insights}
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/digest/latest")
async def get_latest_digest():
    """Get latest learning digest"""
    try:
        # Get most recent digest
        keys = []
        async for key in redis_client.scan_iter("claudehome:digests:*"):
            keys.append(key)
        
        if not keys:
            return {"message": "No digests available yet"}
        
        latest_key = sorted(keys, reverse=True)[0]
        data = await redis_client.get(latest_key)
        
        if data:
            return json.loads(data)
        else:
            return {"message": "No digests available yet"}
            
    except Exception as e:
        logger.error(f"Error getting digest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/now")
async def trigger_analysis():
    """Manually trigger analysis"""
    try:
        digest = await generate_learning_digest()
        return {"status": "analysis_complete", "digest": digest.dict()}
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def periodic_analysis():
    """Run analysis periodically"""
    while True:
        try:
            await asyncio.sleep(ANALYSIS_INTERVAL_HOURS * 3600)
            
            logger.info("Starting periodic analysis...")
            digest = await generate_learning_digest()
            
            # Publish digest
            await redis_client.publish(
                "claudehome:learning_digest",
                json.dumps(digest.dict())
            )
            
            logger.info(f"Analysis complete: {digest.total_deployments} deployments, {digest.overall_success_rate:.1%} success rate")
            
        except Exception as e:
            logger.error(f"Periodic analysis error: {e}")


async def listen_for_events():
    """Listen for outcome and override events"""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(
        "claudehome:proactive:outcome_verified",
        "claudehome:proactive:override_detected",
        "claudehome:proactive:rule_generated"
    )
    
    logger.info("Listening for learning events")
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                event = json.loads(message["data"])
                await process_learning_event(message["channel"], event)
            except Exception as e:
                logger.error(f"Error processing event: {e}")


async def process_learning_event(channel: str, event: Dict[str, Any]):
    """Process a learning-relevant event"""
    
    if "outcome_verified" in channel:
        # Outcome verified - update success metrics
        await update_outcome_metrics(event)
        
    elif "override_detected" in channel:
        # Override detected - update override metrics
        await update_override_metrics(event)
        
    elif "rule_generated" in channel:
        # New rule generated - create insight
        await create_rule_insight(event)


async def update_outcome_metrics(event: Dict[str, Any]):
    """Update metrics when outcome is verified"""
    suggestion_id = event.get("suggestion_id")
    status = event.get("status")  # success, failure, unclear
    
    if not suggestion_id:
        return
    
    # Get original suggestion to determine pattern type
    suggestion_key = f"claudehome:suggestions:pending:{suggestion_id}"
    suggestion_data = await redis_client.get(suggestion_key)
    
    if not suggestion_data:
        # Try approved
        suggestion_key = f"claudehome:suggestions:approved:{suggestion_id}"
        suggestion_data = await redis_client.get(suggestion_key)
    
    if suggestion_data:
        suggestion = json.loads(suggestion_data)
        pattern_type = suggestion.get("pattern", {}).get("pattern_type", "unknown")
        
        # Update pattern type metrics
        metric_key = f"claudehome:metrics:pattern:{pattern_type}"
        
        if status == "success":
            await redis_client.hincrby(metric_key, "successful", 1)
        elif status == "failure":
            await redis_client.hincrby(metric_key, "failed", 1)
        else:
            await redis_client.hincrby(metric_key, "unclear", 1)
        
        logger.info(f"Updated metrics for {pattern_type}: {status}")


async def update_override_metrics(event: Dict[str, Any]):
    """Update metrics when override is detected"""
    automation_id = event.get("automation_id")
    entity_id = event.get("entity_id")
    
    # Increment override count
    metric_key = f"claudehome:metrics:overrides"
    await redis_client.hincrby(metric_key, automation_id, 1)
    
    logger.info(f"Recorded override for {automation_id}")


async def create_rule_insight(event: Dict[str, Any]):
    """Create insight when rule is generated"""
    insight_id = f"insight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    insight = LearningInsight(
        insight_id=insight_id,
        insight_type="success_pattern",
        title="User Preference Learned",
        description=f"Generated blocking rule: {event.get('reason')}",
        supporting_data=event,
        confidence=event.get("confidence", 0.8),
        actionable=True,
        action="apply_blocking_rule",
        created_at=datetime.utcnow().isoformat()
    )
    
    # Store insight
    key = f"claudehome:insights:{insight_id}"
    await redis_client.setex(key, 86400 * 30, json.dumps(insight.dict()))
    
    logger.info(f"💡 Created insight: {insight.title}")


async def analyze_pattern_types() -> List[Dict[str, Any]]:
    """Analyze metrics by pattern type"""
    
    pattern_types = ["manual_action", "time_based", "sequence", "system_optimization"]
    metrics = []
    
    for pattern_type in pattern_types:
        metric_key = f"claudehome:metrics:pattern:{pattern_type}"
        data = await redis_client.hgetall(metric_key)
        
        if data:
            successful = int(data.get("successful", 0))
            failed = int(data.get("failed", 0))
            unclear = int(data.get("unclear", 0))
            total = successful + failed + unclear
            
            success_rate = successful / total if total > 0 else 0.0
            
            metrics.append({
                "pattern_type": pattern_type,
                "total": total,
                "successful": successful,
                "failed": failed,
                "unclear": unclear,
                "success_rate": success_rate
            })
    
    return metrics


async def generate_learning_digest() -> LearningDigest:
    """Generate comprehensive learning digest"""
    
    now = datetime.utcnow()
    period_start = now - timedelta(hours=ANALYSIS_INTERVAL_HOURS)
    
    digest_id = f"digest_{now.strftime('%Y%m%d_%H%M%S')}"
    
    # Analyze pattern type metrics
    pattern_metrics_data = await analyze_pattern_types()
    pattern_metrics = [PatternTypeMetrics(**m) for m in pattern_metrics_data]
    
    # Calculate overall success rate
    total_outcomes = sum(m.total for m in pattern_metrics_data)
    total_successful = sum(m.get("successful", 0) for m in pattern_metrics_data)
    overall_success_rate = total_successful / total_outcomes if total_outcomes > 0 else 0.0
    
    # Get override count
    override_keys = await redis_client.keys("claudehome:overrides:*")
    total_overrides = len(override_keys)
    
    # Get deployment count
    deployment_keys = await redis_client.keys("claudehome:deployments:*")
    total_deployments = len(deployment_keys)
    
    # Get suggestion count
    suggestion_keys = await redis_client.keys("claudehome:suggestions:*")
    total_suggestions = len(suggestion_keys)
    
    # Generate insights
    insights = await generate_insights(pattern_metrics_data, total_overrides)
    
    # Generate recommendations
    recommendations = await generate_recommendations(pattern_metrics_data, overall_success_rate)
    
    digest = LearningDigest(
        digest_id=digest_id,
        date=now.strftime("%Y-%m-%d"),
        period_start=period_start.isoformat(),
        period_end=now.isoformat(),
        total_suggestions=total_suggestions,
        total_deployments=total_deployments,
        total_overrides=total_overrides,
        pattern_metrics=pattern_metrics,
        insights=insights,
        recommendations=recommendations,
        overall_success_rate=overall_success_rate
    )
    
    # Store digest
    key = f"claudehome:digests:{digest_id}"
    await redis_client.setex(key, 86400 * 90, json.dumps(digest.dict()))
    
    logger.info(f"📊 Generated learning digest: {overall_success_rate:.1%} success rate")
    
    return digest


async def generate_insights(pattern_metrics: List[Dict], total_overrides: int) -> List[LearningInsight]:
    """Generate insights from metrics"""
    insights = []
    
    # Find best performing pattern
    if pattern_metrics:
        best_pattern = max(pattern_metrics, key=lambda x: x.get("success_rate", 0))
        if best_pattern["total"] > 0:
            insights.append(LearningInsight(
                insight_id=f"insight_best_{datetime.utcnow().strftime('%Y%m%d')}",
                insight_type="success_pattern",
                title=f"Best Pattern: {best_pattern['pattern_type']}",
                description=f"{best_pattern['pattern_type']} patterns have {best_pattern['success_rate']:.1%} success rate ({best_pattern['successful']}/{best_pattern['total']})",
                supporting_data=best_pattern,
                confidence=0.9,
                actionable=True,
                action="increase_confidence_threshold",
                created_at=datetime.utcnow().isoformat()
            ))
        
        # Find worst performing pattern
        worst_pattern = min(pattern_metrics, key=lambda x: x.get("success_rate", 0))
        if worst_pattern["total"] > 0 and worst_pattern["success_rate"] < 0.7:
            insights.append(LearningInsight(
                insight_id=f"insight_worst_{datetime.utcnow().strftime('%Y%m%d')}",
                insight_type="failure_pattern",
                title=f"Needs Improvement: {worst_pattern['pattern_type']}",
                description=f"{worst_pattern['pattern_type']} patterns only have {worst_pattern['success_rate']:.1%} success rate ({worst_pattern['successful']}/{worst_pattern['total']})",
                supporting_data=worst_pattern,
                confidence=0.8,
                actionable=True,
                action="add_validation_rules",
                created_at=datetime.utcnow().isoformat()
            ))
    
    # Override insight
    if total_overrides > 0:
        insights.append(LearningInsight(
            insight_id=f"insight_overrides_{datetime.utcnow().strftime('%Y%m%d')}",
            insight_type="improvement",
            title="User Overrides Detected",
            description=f"Users manually reversed {total_overrides} automation actions. Blocking rules have been generated.",
            supporting_data={"override_count": total_overrides},
            confidence=1.0,
            actionable=False,
            created_at=datetime.utcnow().isoformat()
        ))
    
    return insights


async def generate_recommendations(pattern_metrics: List[Dict], overall_success_rate: float) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if overall_success_rate < 0.7:
        recommendations.append("Overall success rate is low. Consider increasing confidence threshold for suggestions.")
    
    for metric in pattern_metrics:
        if metric["total"] > 5 and metric["success_rate"] < 0.6:
            recommendations.append(f"Review {metric['pattern_type']} pattern detection - low success rate.")
        
        if metric["total"] > 10 and metric["success_rate"] > 0.9:
            recommendations.append(f"Consider auto-deploying {metric['pattern_type']} patterns (high success rate).")
    
    if not recommendations:
        recommendations.append("System performing well. Continue monitoring patterns.")
    
    return recommendations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)