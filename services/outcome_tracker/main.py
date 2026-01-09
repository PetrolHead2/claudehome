"""
Outcome Tracker - Phase 4.2
Monitors automation outcomes to learn what works

Tracks:
- Automation executions
- User manual overrides
- Success/failure patterns
- User satisfaction signals
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
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClaudeHome Outcome Tracker")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
HOME_ASSISTANT_URL = os.getenv("HOME_ASSISTANT_URL")
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

# Global instances
redis_client: Optional[redis.Redis] = None

# Constants
OUTCOME_RETENTION_DAYS = 90  # Keep 3 months of data


# Pydantic Models
class AutomationExecution(BaseModel):
    execution_id: str
    automation_id: str
    entity_id: str
    action: str
    triggered_at: str
    trigger_reason: str
    metadata: Dict[str, Any] = {}


class UserOverride(BaseModel):
    override_id: str
    automation_execution_id: Optional[str] = None
    entity_id: str
    original_state: str
    override_state: str
    override_at: str
    time_since_automation: Optional[float] = None  # Seconds
    metadata: Dict[str, Any] = {}


class Outcome(BaseModel):
    outcome_id: str
    automation_execution_id: str
    outcome_type: str  # 'success', 'override', 'ignored', 'complaint'
    score: float  # -1.0 to 1.0 (-1=bad, 0=neutral, 1=good)
    timestamp: str
    evidence: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class PerformanceMetrics(BaseModel):
    automation_id: str
    total_executions: int
    successful: int
    overridden: int
    ignored: int
    success_rate: float
    override_rate: float
    average_score: float
    last_executed: str
    metadata: Dict[str, Any] = {}


@app.on_event("startup")
async def startup():
    """Initialize services"""
    global redis_client
    
    logger.info("Starting Outcome Tracker...")
    
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connected")
    
    # Start background monitoring
    asyncio.create_task(monitor_state_changes())
    
    logger.info("Outcome Tracker started successfully")


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
        "service": "outcome_tracker",
        "version": "4.2",
        "monitoring": "active",
        "timestamp": datetime.now().isoformat()
    }


# ========== AUTOMATION EXECUTION TRACKING ==========

@app.post("/track/execution")
async def track_execution(execution: AutomationExecution):
    """
    Record an automation execution.
    
    Called by other services when automation triggers.
    """
    try:
        # Store execution
        key = f"claudehome:outcome:execution:{execution.execution_id}"
        value = json.dumps(execution.dict())
        await redis_client.setex(key, 86400 * OUTCOME_RETENTION_DAYS, value)
        
        # Add to entity execution list
        entity_key = f"claudehome:outcome:executions:{execution.entity_id}"
        await redis_client.lpush(entity_key, execution.execution_id)
        await redis_client.ltrim(entity_key, 0, 999)  # Keep last 1000
        
        # Add to automation execution list
        auto_key = f"claudehome:outcome:auto_executions:{execution.automation_id}"
        await redis_client.lpush(auto_key, execution.execution_id)
        await redis_client.ltrim(auto_key, 0, 999)
        
        # Schedule outcome evaluation (check for overrides in 5 minutes)
        await schedule_outcome_evaluation(execution.execution_id, delay_seconds=300)
        
        logger.info(f"Tracked execution: {execution.execution_id} for {execution.entity_id}")
        
        return {"status": "tracked", "execution_id": execution.execution_id}
        
    except Exception as e:
        logger.error(f"Failed to track execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def schedule_outcome_evaluation(execution_id: str, delay_seconds: int):
    """
    Schedule outcome evaluation after delay.
    
    Gives time for user to override if they disagree with automation.
    """
    try:
        # Store in pending evaluations
        eval_time = datetime.now() + timedelta(seconds=delay_seconds)
        key = "claudehome:outcome:pending_evaluations"
        value = json.dumps({
            "execution_id": execution_id,
            "evaluate_at": eval_time.isoformat()
        })
        await redis_client.zadd(key, {value: eval_time.timestamp()})
        
        logger.debug(f"Scheduled evaluation for {execution_id} in {delay_seconds}s")
        
    except Exception as e:
        logger.error(f"Failed to schedule evaluation: {e}")


# ========== USER OVERRIDE TRACKING ==========

@app.post("/track/override")
async def track_override(override: UserOverride):
    """
    Record a user manual override.
    
    Called when user manually changes state after automation.
    """
    try:
        # Store override
        key = f"claudehome:outcome:override:{override.override_id}"
        value = json.dumps(override.dict())
        await redis_client.setex(key, 86400 * OUTCOME_RETENTION_DAYS, value)
        
        # Add to entity override list
        entity_key = f"claudehome:outcome:overrides:{override.entity_id}"
        await redis_client.lpush(entity_key, override.override_id)
        await redis_client.ltrim(entity_key, 0, 999)
        
        # If linked to automation execution, create negative outcome
        if override.automation_execution_id:
            await create_outcome(
                execution_id=override.automation_execution_id,
                outcome_type="override",
                score=-0.7,  # Override = bad (user disagreed)
                evidence={
                    "override_id": override.override_id,
                    "time_since_automation": override.time_since_automation,
                    "original": override.original_state,
                    "override": override.override_state
                }
            )
        
        logger.info(f"Tracked override: {override.override_id} for {override.entity_id}")
        
        return {"status": "tracked", "override_id": override.override_id}
        
    except Exception as e:
        logger.error(f"Failed to track override: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== OUTCOME EVALUATION ==========

async def create_outcome(
    execution_id: str,
    outcome_type: str,
    score: float,
    evidence: Dict[str, Any] = {}
):
    """Create outcome record for an execution"""
    try:
        outcome_id = f"outcome_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        outcome = Outcome(
            outcome_id=outcome_id,
            automation_execution_id=execution_id,
            outcome_type=outcome_type,
            score=score,
            timestamp=datetime.now().isoformat(),
            evidence=evidence
        )
        
        # Store outcome
        key = f"claudehome:outcome:result:{outcome_id}"
        value = json.dumps(outcome.dict())
        await redis_client.setex(key, 86400 * OUTCOME_RETENTION_DAYS, value)
        
        # Link to execution
        exec_key = f"claudehome:outcome:execution_outcome:{execution_id}"
        await redis_client.set(exec_key, outcome_id, ex=86400 * OUTCOME_RETENTION_DAYS)
        
        # Update metrics
        await update_metrics(execution_id, outcome)
        
        logger.info(f"Created outcome: {outcome_type} (score: {score}) for {execution_id}")
        
        return outcome
        
    except Exception as e:
        logger.error(f"Failed to create outcome: {e}")


async def evaluate_execution_outcome(execution_id: str):
    """
    Evaluate outcome of an automation execution.
    
    Checks:
    - Was it overridden by user?
    - Was state maintained?
    - Any complaints/issues?
    """
    try:
        # Get execution details
        exec_key = f"claudehome:outcome:execution:{execution_id}"
        exec_data = await redis_client.get(exec_key)
        
        if not exec_data:
            logger.warning(f"Execution {execution_id} not found")
            return
        
        execution = json.loads(exec_data)
        entity_id = execution['entity_id']
        
        # Check if already evaluated
        outcome_key = f"claudehome:outcome:execution_outcome:{execution_id}"
        existing_outcome = await redis_client.get(outcome_key)
        
        if existing_outcome:
            logger.debug(f"Execution {execution_id} already evaluated")
            return
        
        # Check for override
        override_found = await check_for_override(execution_id, entity_id)
        
        if override_found:
            # Already handled by track_override
            return
        
        # No override detected - assume success
        await create_outcome(
            execution_id=execution_id,
            outcome_type="success",
            score=0.5,  # Modest success (no override = user didn't disagree)
            evidence={"reason": "No override detected within 5 minutes"}
        )
        
    except Exception as e:
        logger.error(f"Failed to evaluate execution {execution_id}: {e}")


async def check_for_override(execution_id: str, entity_id: str) -> bool:
    """Check if execution was overridden by user"""
    try:
        # Get recent overrides for entity
        override_key = f"claudehome:outcome:overrides:{entity_id}"
        override_ids = await redis_client.lrange(override_key, 0, 9)  # Last 10
        
        for override_id in override_ids:
            key = f"claudehome:outcome:override:{override_id}"
            data = await redis_client.get(key)
            
            if data:
                override = json.loads(data)
                if override.get('automation_execution_id') == execution_id:
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to check for override: {e}")
        return False


# ========== METRICS & ANALYTICS ==========

async def update_metrics(execution_id: str, outcome: Outcome):
    """Update performance metrics for automation"""
    try:
        # Get execution
        exec_key = f"claudehome:outcome:execution:{execution_id}"
        exec_data = await redis_client.get(exec_key)
        
        if not exec_data:
            return
        
        execution = json.loads(exec_data)
        automation_id = execution['automation_id']
        
        # Get current metrics
        metrics = await get_metrics(automation_id)
        
        # Update counts
        metrics.total_executions += 1
        
        if outcome.outcome_type == "success":
            metrics.successful += 1
        elif outcome.outcome_type == "override":
            metrics.overridden += 1
        elif outcome.outcome_type == "ignored":
            metrics.ignored += 1
        
        # Calculate rates
        metrics.success_rate = metrics.successful / metrics.total_executions
        metrics.override_rate = metrics.overridden / metrics.total_executions
        
        # Update average score (running average)
        old_avg = metrics.average_score
        old_count = metrics.total_executions - 1
        metrics.average_score = (old_avg * old_count + outcome.score) / metrics.total_executions
        
        metrics.last_executed = execution['triggered_at']
        
        # Save metrics
        metrics_key = f"claudehome:outcome:metrics:{automation_id}"
        await redis_client.set(metrics_key, json.dumps(metrics.dict()))
        
        logger.debug(f"Updated metrics for {automation_id}: {metrics.success_rate:.2%} success")
        
    except Exception as e:
        logger.error(f"Failed to update metrics: {e}")


async def get_metrics(automation_id: str) -> PerformanceMetrics:
    """Get metrics for automation (or create new)"""
    try:
        metrics_key = f"claudehome:outcome:metrics:{automation_id}"
        data = await redis_client.get(metrics_key)
        
        if data:
            return PerformanceMetrics(**json.loads(data))
        else:
            # Create new metrics
            return PerformanceMetrics(
                automation_id=automation_id,
                total_executions=0,
                successful=0,
                overridden=0,
                ignored=0,
                success_rate=0.0,
                override_rate=0.0,
                average_score=0.0,
                last_executed=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return PerformanceMetrics(
            automation_id=automation_id,
            total_executions=0,
            successful=0,
            overridden=0,
            ignored=0,
            success_rate=0.0,
            override_rate=0.0,
            average_score=0.0,
            last_executed=datetime.now().isoformat()
        )


@app.get("/metrics/{automation_id}")
async def get_automation_metrics(automation_id: str):
    """Get performance metrics for specific automation"""
    try:
        metrics = await get_metrics(automation_id)
        return metrics.dict()
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_all_metrics(limit: int = 20):
    """Get metrics for all automations"""
    try:
        # Get all metric keys
        keys = await redis_client.keys("claudehome:outcome:metrics:*")
        
        metrics_list = []
        for key in keys[:limit]:
            data = await redis_client.get(key)
            if data:
                metrics_list.append(json.loads(data))
        
        # Sort by success rate
        metrics_list.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return {
            "count": len(metrics_list),
            "metrics": metrics_list
        }
        
    except Exception as e:
        logger.error(f"Failed to get all metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outcomes/recent")
async def get_recent_outcomes(limit: int = 50):
    """Get recent automation outcomes"""
    try:
        # Get outcome keys (sorted by timestamp)
        keys = await redis_client.keys("claudehome:outcome:result:*")
        
        outcomes = []
        for key in keys[:limit]:
            data = await redis_client.get(key)
            if data:
                outcomes.append(json.loads(data))
        
        # Sort by timestamp (newest first)
        outcomes.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            "count": len(outcomes),
            "outcomes": outcomes[:limit]
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent outcomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get overall system performance summary"""
    try:
        # Get all metrics
        keys = await redis_client.keys("claudehome:outcome:metrics:*")
        
        total_executions = 0
        total_successful = 0
        total_overridden = 0
        total_score = 0.0
        automation_count = len(keys)
        
        for key in keys:
            data = await redis_client.get(key)
            if data:
                metrics = json.loads(data)
                total_executions += metrics['total_executions']
                total_successful += metrics['successful']
                total_overridden += metrics['overridden']
                total_score += metrics['average_score'] * metrics['total_executions']
        
        overall_success_rate = total_successful / total_executions if total_executions > 0 else 0
        overall_override_rate = total_overridden / total_executions if total_executions > 0 else 0
        overall_score = total_score / total_executions if total_executions > 0 else 0
        
        return {
            "automations_tracked": automation_count,
            "total_executions": total_executions,
            "overall_success_rate": overall_success_rate,
            "overall_override_rate": overall_override_rate,
            "overall_score": overall_score,
            "health_status": "excellent" if overall_success_rate > 0.8 else "good" if overall_success_rate > 0.6 else "needs_improvement"
        }
        
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== BACKGROUND MONITORING ==========

async def monitor_state_changes():
    """
    Background task to monitor HA state changes for override detection.
    
    Watches for manual state changes that contradict recent automations.
    """
    logger.info("Starting state change monitoring...")
    
    # Subscribe to Redis channel for HA events
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("homeassistant:state_changed")
    
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    event = json.loads(message['data'])
                    await process_state_change(event)
                except Exception as e:
                    logger.error(f"Error processing state change: {e}")
    
    except Exception as e:
        logger.error(f"State monitoring error: {e}")
    finally:
        await pubsub.unsubscribe("homeassistant:state_changed")


async def process_state_change(event: Dict[str, Any]):
    """Process a state change event to detect potential overrides"""
    try:
        entity_id = event.get('entity_id')
        new_state = event.get('new_state')
        old_state = event.get('old_state')
        context = event.get('context', {})
        
        # Check if this was a manual change (no automation context)
        if context.get('user_id') or context.get('parent_id') is None:
            # Potentially a manual override - check recent automations
            await check_potential_override(entity_id, old_state, new_state, event['timestamp'])
        
    except Exception as e:
        logger.error(f"Error processing state change: {e}")


async def check_potential_override(
    entity_id: str,
    old_state: str,
    new_state: str,
    timestamp: str
):
    """Check if state change is an override of recent automation"""
    try:
        # Get recent executions for this entity
        exec_key = f"claudehome:outcome:executions:{entity_id}"
        exec_ids = await redis_client.lrange(exec_key, 0, 4)  # Last 5
        
        change_time = datetime.fromisoformat(timestamp)
        
        for exec_id in exec_ids:
            exec_data_key = f"claudehome:outcome:execution:{exec_id}"
            exec_data = await redis_client.get(exec_data_key)
            
            if not exec_data:
                continue
            
            execution = json.loads(exec_data)
            exec_time = datetime.fromisoformat(execution['triggered_at'])
            
            # Check if state change happened within 10 minutes of automation
            time_diff = (change_time - exec_time).total_seconds()
            
            if 0 < time_diff < 600:  # Within 10 minutes
                # Check if state contradicts automation action
                if is_contradictory_action(execution['action'], old_state, new_state):
                    # This looks like an override!
                    override_id = f"override_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    
                    override = UserOverride(
                        override_id=override_id,
                        automation_execution_id=exec_id,
                        entity_id=entity_id,
                        original_state=old_state,
                        override_state=new_state,
                        override_at=timestamp,
                        time_since_automation=time_diff
                    )
                    
                    await track_override(override)
                    break
        
    except Exception as e:
        logger.error(f"Error checking potential override: {e}")


def is_contradictory_action(automation_action: str, old_state: str, new_state: str) -> bool:
    """
    Check if manual action contradicts automation.
    
    Example: Automation turned light ON, user turned it OFF shortly after
    """
    # Automation turned something ON, user turned OFF
    if "turn_on" in automation_action.lower() and new_state == "off":
        return True
    
    # Automation turned something OFF, user turned ON
    if "turn_off" in automation_action.lower() and new_state == "on":
        return True
    
    # Add more sophisticated logic as needed
    
    return False


# ========== BACKGROUND TASK: EVALUATE PENDING ==========

@app.on_event("startup")
async def start_evaluation_task():
    """Start background task to evaluate pending outcomes"""
    asyncio.create_task(evaluate_pending_outcomes())


async def evaluate_pending_outcomes():
    """Background task that evaluates outcomes when their time comes"""
    logger.info("Starting pending outcome evaluation task...")
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            # Get pending evaluations that are due
            key = "claudehome:outcome:pending_evaluations"
            now = datetime.now().timestamp()
            
            # Get evaluations due now
            due_evaluations = await redis_client.zrangebyscore(key, 0, now)
            
            for eval_data in due_evaluations:
                try:
                    evaluation = json.loads(eval_data)
                    execution_id = evaluation['execution_id']
                    
                    # Evaluate
                    await evaluate_execution_outcome(execution_id)
                    
                    # Remove from pending
                    await redis_client.zrem(key, eval_data)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {eval_data}: {e}")
            
        except Exception as e:
            logger.error(f"Evaluation task error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)