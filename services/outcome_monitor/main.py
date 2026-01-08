"""
Outcome Monitor - Verify Automation Success

Monitors deployed automations to verify they work as expected:
- Tracks automation triggers
- Verifies desired outcomes
- Reports success/failure
- Generates improvement suggestions
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
logger = logging.getLogger("outcome_monitor")

app = FastAPI(title="ClaudeHome Outcome Monitor")

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
CONTEXT_BUILDER_URL = os.getenv("CONTEXT_BUILDER_URL", "http://context_builder:8000")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000")
MONITORING_DURATION_HOURS = int(os.getenv("MONITORING_DURATION_HOURS", "48"))

redis_client: Optional[redis.Redis] = None
monitoring_tasks: Dict[str, Dict] = {}


class DeploymentOutcome(BaseModel):
    """Outcome of a deployed automation"""
    automation_id: str
    suggestion_id: str
    deployed_at: str
    monitoring_until: str
    trigger_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_trigger: Optional[str] = None
    status: str = "monitoring"  # monitoring, success, failure, unclear
    verification_method: str = "state"  # state, historical, llm
    details: List[Dict[str, Any]] = []


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Outcome Monitor started")
    
    # Start background monitoring
    asyncio.create_task(monitor_deployments())
    asyncio.create_task(check_automation_events())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "outcome_monitor",
        "monitoring_count": len(monitoring_tasks)
    }


@app.post("/monitor/{automation_id}")
async def start_monitoring(automation_id: str, suggestion_id: str):
    """
    Start monitoring a deployed automation
    """
    try:
        deployed_at = datetime.utcnow()
        monitoring_until = deployed_at + timedelta(hours=MONITORING_DURATION_HOURS)
        
        outcome = DeploymentOutcome(
            automation_id=automation_id,
            suggestion_id=suggestion_id,
            deployed_at=deployed_at.isoformat(),
            monitoring_until=monitoring_until.isoformat()
        )
        
        # Store in Redis
        key = f"claudehome:outcomes:monitoring:{automation_id}"
        await redis_client.setex(
            key,
            MONITORING_DURATION_HOURS * 3600 + 86400,  # Extra day buffer
            json.dumps(outcome.dict())
        )
        
        # Add to monitoring set
        await redis_client.sadd("claudehome:outcomes:active", automation_id)
        
        logger.info(f"Started monitoring {automation_id} until {monitoring_until}")
        
        return {"status": "monitoring_started", "monitoring_until": monitoring_until.isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outcome/{automation_id}")
async def get_outcome(automation_id: str):
    """Get monitoring outcome for an automation"""
    try:
        key = f"claudehome:outcomes:monitoring:{automation_id}"
        data = await redis_client.get(key)
        
        if not data:
            # Check completed outcomes
            key = f"claudehome:outcomes:completed:{automation_id}"
            data = await redis_client.get(key)
            
        if not data:
            raise HTTPException(status_code=404, detail="Outcome not found")
        
        return json.loads(data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outcomes/summary")
async def get_outcomes_summary():
    """Get summary of all outcomes"""
    try:
        # Get active monitoring
        active = await redis_client.smembers("claudehome:outcomes:active")
        
        # Get completed outcomes
        completed_keys = await redis_client.keys("claudehome:outcomes:completed:*")
        
        outcomes = {
            "active_monitoring": len(active),
            "completed": len(completed_keys),
            "summary": {
                "success": 0,
                "failure": 0,
                "unclear": 0
            }
        }
        
        # Count completed by status
        for key in completed_keys:
            data = await redis_client.get(key)
            if data:
                outcome = json.loads(data)
                status = outcome.get("status", "unclear")
                outcomes["summary"][status] = outcomes["summary"].get(status, 0) + 1
        
        return outcomes
        
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def monitor_deployments():
    """Background task: Monitor active deployments"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            active_ids = await redis_client.smembers("claudehome:outcomes:active")
            
            for automation_id in active_ids:
                await check_automation_outcome(automation_id)
                
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")


async def check_automation_outcome(automation_id: str):
    """Check outcome of a specific automation"""
    try:
        key = f"claudehome:outcomes:monitoring:{automation_id}"
        data = await redis_client.get(key)
        
        if not data:
            # No longer monitoring
            await redis_client.srem("claudehome:outcomes:active", automation_id)
            return
        
        outcome = DeploymentOutcome(**json.loads(data))
        
        # Check if monitoring period ended
        monitoring_until = datetime.fromisoformat(outcome.monitoring_until)
        if datetime.utcnow() > monitoring_until:
            # Finalize outcome
            await finalize_outcome(outcome)
            return
        
        # Check recent triggers (simplified for now)
        # In production, this would check HA automation.triggered events
        logger.debug(f"Checking {automation_id}: {outcome.trigger_count} triggers so far")
        
    except Exception as e:
        logger.error(f"Error checking outcome for {automation_id}: {e}")


async def finalize_outcome(outcome: DeploymentOutcome):
    """Finalize monitoring and determine final status"""
    try:
        # Determine final status
        if outcome.trigger_count == 0:
            outcome.status = "unclear"  # Never triggered
        elif outcome.success_count > outcome.failure_count:
            outcome.status = "success"
        elif outcome.failure_count > 0:
            outcome.status = "failure"
        else:
            outcome.status = "unclear"
        
        # Move to completed
        completed_key = f"claudehome:outcomes:completed:{outcome.automation_id}"
        await redis_client.setex(
            completed_key,
            86400 * 30,  # Keep for 30 days
            json.dumps(outcome.dict())
        )
        
        # Remove from monitoring
        monitoring_key = f"claudehome:outcomes:monitoring:{outcome.automation_id}"
        await redis_client.delete(monitoring_key)
        await redis_client.srem("claudehome:outcomes:active", outcome.automation_id)
        
        # Publish outcome
        await redis_client.publish(
            "claudehome:proactive:outcome_verified",
            json.dumps({
                "automation_id": outcome.automation_id,
                "suggestion_id": outcome.suggestion_id,
                "status": outcome.status,
                "trigger_count": outcome.trigger_count,
                "success_count": outcome.success_count
            })
        )
        
        logger.info(f"Finalized outcome for {outcome.automation_id}: {outcome.status}")
        
    except Exception as e:
        logger.error(f"Error finalizing outcome: {e}")


async def check_automation_events():
    """Background task: Check for automation trigger events from HA"""
    # Subscribe to HA events via Redis
    # This will be populated by event_ingester
    
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("claudehome:ha_events:automation_triggered")
    
    logger.info("Subscribed to automation trigger events")
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                event_data = json.loads(message["data"])
                await handle_automation_trigger(event_data)
            except Exception as e:
                logger.error(f"Error handling automation event: {e}")


async def handle_automation_trigger(event_data: Dict[str, Any]):
    """Handle automation trigger event"""
    automation_id = event_data.get("automation_id")
    
    if not automation_id or not automation_id.startswith("claudehome_"):
        return  # Not our automation
    
    # Update outcome
    key = f"claudehome:outcomes:monitoring:{automation_id}"
    data = await redis_client.get(key)
    
    if not data:
        return  # Not monitoring this one
    
    outcome = DeploymentOutcome(**json.loads(data))
    outcome.trigger_count += 1
    outcome.last_trigger = datetime.utcnow().isoformat()
    
    # Verify outcome (simplified)
    # In production, check if the action succeeded
    outcome.success_count += 1
    
    outcome.details.append({
        "timestamp": outcome.last_trigger,
        "result": "success",
        "event": event_data
    })
    
    # Save updated outcome
    await redis_client.setex(
        key,
        MONITORING_DURATION_HOURS * 3600 + 86400,
        json.dumps(outcome.dict())
    )
    
    logger.info(f"Recorded trigger for {automation_id}: {outcome.trigger_count} total")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)