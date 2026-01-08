"""
Override Detector - Learn from User Behavior

Detects when users manually reverse automation actions:
- Light turned on by automation → User turns off within 60s
- Temperature set by automation → User changes it
- Generates blocking rules to prevent unwanted automations
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
logger = logging.getLogger("override_detector")

app = FastAPI(title="ClaudeHome Override Detector")

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000")
OVERRIDE_WINDOW_SECONDS = int(os.getenv("OVERRIDE_WINDOW_SECONDS", "60"))
MIN_OVERRIDES_FOR_RULE = int(os.getenv("MIN_OVERRIDES_FOR_RULE", "3"))

redis_client: Optional[redis.Redis] = None

# Track recent automation actions
recent_automation_actions: Dict[str, Dict] = {}


class Override(BaseModel):
    """Detected override event"""
    override_id: str
    automation_id: str
    entity_id: str
    automation_action: str  # "turn_on", "set_brightness", etc.
    user_action: str        # "turn_off", "set_brightness", etc.
    automation_value: Optional[Any] = None
    user_value: Optional[Any] = None
    time_delta_seconds: float
    timestamp: str
    context: Dict[str, Any] = {}


class BlockingRule(BaseModel):
    """Generated blocking rule"""
    rule_id: str
    entity_id: str
    automation_id: str
    rule_type: str  # "time_block", "state_block", "value_preference"
    condition: Dict[str, Any]
    reason: str
    override_count: int
    created_at: str
    confidence: float


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Override Detector started")
    
    # Start background tasks
    asyncio.create_task(monitor_automation_actions())
    asyncio.create_task(monitor_user_actions())
    asyncio.create_task(analyze_override_patterns())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "override_detector",
        "recent_actions_tracked": len(recent_automation_actions)
    }


@app.get("/overrides")
async def get_overrides(limit: int = 50):
    """Get recent overrides"""
    try:
        keys = []
        async for key in redis_client.scan_iter("claudehome:overrides:*"):
            keys.append(key)
        
        overrides = []
        for key in sorted(keys, reverse=True)[:limit]:
            data = await redis_client.get(key)
            if data:
                overrides.append(json.loads(data))
        
        return {"count": len(overrides), "overrides": overrides}
        
    except Exception as e:
        logger.error(f"Error getting overrides: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rules")
async def get_blocking_rules():
    """Get active blocking rules"""
    try:
        keys = []
        async for key in redis_client.scan_iter("claudehome:blocking_rules:*"):
            keys.append(key)
        
        rules = []
        for key in keys:
            data = await redis_client.get(key)
            if data:
                rules.append(json.loads(data))
        
        return {"count": len(rules), "rules": rules}
        
    except Exception as e:
        logger.error(f"Error getting rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete a blocking rule"""
    try:
        key = f"claudehome:blocking_rules:{rule_id}"
        deleted = await redis_client.delete(key)
        
        if deleted:
            logger.info(f"Deleted blocking rule: {rule_id}")
            return {"status": "deleted", "rule_id": rule_id}
        else:
            raise HTTPException(status_code=404, detail="Rule not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def monitor_automation_actions():
    """Monitor automation actions via Redis pub/sub"""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("claudehome:automation_triggered")
    
    logger.info("Monitoring automation actions")
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                event = json.loads(message["data"])
                await handle_automation_action(event)
            except Exception as e:
                logger.error(f"Error handling automation action: {e}")


async def handle_automation_action(event: Dict[str, Any]):
    """Record automation action for override detection"""
    automation_id = event.get("automation_id")
    entity_id = event.get("entity_id")
    action = event.get("action")  # turn_on, turn_off, set_brightness, etc.
    value = event.get("value")
    
    if not automation_id or not entity_id:
        return
    
    # Store in memory for quick lookup
    key = f"{automation_id}:{entity_id}"
    recent_automation_actions[key] = {
        "automation_id": automation_id,
        "entity_id": entity_id,
        "action": action,
        "value": value,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.debug(f"Recorded automation action: {automation_id} → {action} on {entity_id}")
    
    # Clean old actions (older than 5 minutes)
    cleanup_threshold = datetime.utcnow() - timedelta(minutes=5)
    to_remove = []
    for k, v in recent_automation_actions.items():
        action_time = datetime.fromisoformat(v["timestamp"])
        if action_time < cleanup_threshold:
            to_remove.append(k)
    
    for k in to_remove:
        del recent_automation_actions[k]


async def monitor_user_actions():
    """Monitor user actions for potential overrides"""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("claudehome:user_action")
    
    logger.info("Monitoring user actions")
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                event = json.loads(message["data"])
                await detect_override(event)
            except Exception as e:
                logger.error(f"Error detecting override: {e}")


async def detect_override(user_event: Dict[str, Any]):
    """Detect if user action is overriding a recent automation"""
    entity_id = user_event.get("entity_id")
    user_action = user_event.get("action")
    user_value = user_event.get("value")
    user_time = datetime.fromisoformat(user_event.get("timestamp", datetime.utcnow().isoformat()))
    
    if not entity_id:
        return
    
    # Check if there's a recent automation action on this entity
    matched_automation = None
    
    for key, auto_action in recent_automation_actions.items():
        if auto_action["entity_id"] == entity_id:
            auto_time = datetime.fromisoformat(auto_action["timestamp"])
            time_delta = (user_time - auto_time).total_seconds()
            
            # Within override window?
            if 0 < time_delta <= OVERRIDE_WINDOW_SECONDS:
                # Check if it's actually an override
                if is_override(auto_action["action"], user_action, auto_action.get("value"), user_value):
                    matched_automation = auto_action
                    matched_automation["time_delta"] = time_delta
                    break
    
    if matched_automation:
        await record_override(matched_automation, user_event)


def is_override(auto_action: str, user_action: str, auto_value: Any, user_value: Any) -> bool:
    """Determine if user action overrides automation action"""
    
    # Direct reversals
    if auto_action == "turn_on" and user_action == "turn_off":
        return True
    if auto_action == "turn_off" and user_action == "turn_on":
        return True
    
    # Value changes
    if auto_action == "set_brightness" and user_action == "set_brightness":
        if auto_value and user_value and abs(auto_value - user_value) > 10:
            return True
    
    if auto_action == "set_temperature" and user_action == "set_temperature":
        if auto_value and user_value and abs(auto_value - user_value) > 1:
            return True
    
    return False


async def record_override(auto_action: Dict, user_event: Dict):
    """Record an override event"""
    override_id = f"override_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    override = Override(
        override_id=override_id,
        automation_id=auto_action["automation_id"],
        entity_id=auto_action["entity_id"],
        automation_action=auto_action["action"],
        user_action=user_event["action"],
        automation_value=auto_action.get("value"),
        user_value=user_event.get("value"),
        time_delta_seconds=auto_action["time_delta"],
        timestamp=datetime.utcnow().isoformat(),
        context={
            "hour": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().strftime("%A"),
            "automation_timestamp": auto_action["timestamp"],
            "user_timestamp": user_event["timestamp"]
        }
    )
    
    # Store override
    key = f"claudehome:overrides:{override_id}"
    await redis_client.setex(
        key,
        86400 * 30,  # Keep for 30 days
        json.dumps(override.dict())
    )
    
    # Track override count for this automation+entity
    count_key = f"claudehome:override_count:{auto_action['automation_id']}:{auto_action['entity_id']}"
    count = await redis_client.incr(count_key)
    await redis_client.expire(count_key, 86400 * 7)  # Weekly count
    
    logger.info(f"🚫 Override detected: {auto_action['automation_id']} → {auto_action['action']} overridden after {auto_action['time_delta']:.1f}s (count: {count})")
    
    # Check if we should generate a blocking rule
    if count >= MIN_OVERRIDES_FOR_RULE:
        await generate_blocking_rule(override, count)
    
    # Publish override event
    await redis_client.publish(
        "claudehome:proactive:override_detected",
        json.dumps(override.dict())
    )


async def generate_blocking_rule(override: Override, override_count: int):
    """Generate a blocking rule from override pattern"""
    
    rule_id = f"rule_{override.automation_id}_{override.entity_id}_{datetime.utcnow().strftime('%Y%m%d')}"
    
    # Determine rule type and condition
    hour = override.context.get("hour")
    
    if override.automation_action == "turn_on" and override.user_action == "turn_off":
        rule_type = "time_block"
        condition = {
            "entity_id": override.entity_id,
            "action": "turn_on",
            "time_range": f"{hour:02d}:00-{(hour+1)%24:02d}:00",
            "block": True
        }
        reason = f"User repeatedly turns off {override.entity_id} around {hour:02d}:00 ({override_count} times)"
    
    elif "brightness" in override.automation_action:
        rule_type = "value_preference"
        condition = {
            "entity_id": override.entity_id,
            "action": "set_brightness",
            "preferred_value": override.user_value,
            "avoid_value": override.automation_value
        }
        reason = f"User prefers brightness {override.user_value}% not {override.automation_value}% ({override_count} times)"
    
    else:
        rule_type = "state_block"
        condition = {
            "entity_id": override.entity_id,
            "action": override.automation_action,
            "block": True
        }
        reason = f"User repeatedly reverses this automation ({override_count} times)"
    
    rule = BlockingRule(
        rule_id=rule_id,
        entity_id=override.entity_id,
        automation_id=override.automation_id,
        rule_type=rule_type,
        condition=condition,
        reason=reason,
        override_count=override_count,
        created_at=datetime.utcnow().isoformat(),
        confidence=min(override_count / 5.0, 1.0)  # Confidence increases with overrides
    )
    
    # Store rule
    key = f"claudehome:blocking_rules:{rule_id}"
    await redis_client.set(key, json.dumps(rule.dict()))
    
    logger.info(f"📋 Generated blocking rule: {rule_id} - {reason}")
    
    # Publish rule generation event
    await redis_client.publish(
        "claudehome:proactive:rule_generated",
        json.dumps(rule.dict())
    )


async def analyze_override_patterns():
    """Periodic analysis of override patterns"""
    while True:
        try:
            await asyncio.sleep(3600)  # Analyze every hour
            
            # Get all overrides from past week
            # TODO: Analyze patterns across time/entities
            
            logger.info("Analyzed override patterns")
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)