"""
Triage Engine - Event Classification and Routing

Receives anonymized events and uses Claude to classify them:
- PROACTIVE_OPP: Repeated patterns → automation suggestions
- ANOMALY_DRIFT: Unusual behavior → alerts
- CAUSAL_INFERENCE: User overrides → rule learning
- ROUTINE: Normal operation, log only
"""

import os
import json
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
from anthropic import AsyncAnthropic
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="ClaudeHome Triage Engine")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Initialize clients
redis_client = None
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# Event classification cache (5 min TTL)
classification_cache = {}


class Event(BaseModel):
    entity_id: str
    state: str
    old_state: str = None
    timestamp: str
    context: Dict[str, Any] = {}


class Classification(BaseModel):
    category: str  # PROACTIVE_OPP, ANOMALY_DRIFT, CAUSAL_INFERENCE, ROUTINE
    confidence: float
    reasoning: str
    priority: int  # 1=high, 2=medium, 3=low


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Triage Engine started - Connected to Redis")


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "triage_engine"}


@app.post("/classify", response_model=Classification)
async def classify_event(event: Event):
    """
    Classify a single event using Claude API with prompt caching
    """
    try:
        # Check cache first
        cache_key = f"{event.entity_id}:{event.state}:{event.old_state}"
        if cache_key in classification_cache:
            logger.info(f"Cache hit for {event.entity_id}")
            return classification_cache[cache_key]
        
        # Build context from Redis
        context = await build_context(event)
        
        # Classify with Claude
        classification = await classify_with_claude(event, context)
        
        # Cache result (5 min)
        classification_cache[cache_key] = classification
        asyncio.create_task(expire_cache(cache_key, 300))
        
        # Store classification in Redis
        await store_classification(event, classification)
        
        # Publish to internal event bus
        await publish_classification(event, classification)
        
        return classification
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def build_context(event: Event) -> Dict[str, Any]:
    """
    Build context from Redis for better classification
    """
    context = {
        "recent_events": [],
        "entity_history": [],
        "active_rules": []
    }
    
    try:
        # Get recent events from same entity (last 1 hour)
        history_key = f"claudehome:history:{event.entity_id}"
        history = await redis_client.lrange(history_key, 0, 10)
        context["entity_history"] = [json.loads(h) for h in history]
        
        # Get recent events from all entities (last 5 min)
        recent_key = "claudehome:events:recent"
        recent = await redis_client.lrange(recent_key, 0, 20)
        context["recent_events"] = [json.loads(r) for r in recent]
        
        # Get active house rules
        rules_key = "claudehome:rules:active"
        rules = await redis_client.smembers(rules_key)
        context["active_rules"] = list(rules)
        
    except Exception as e:
        logger.warning(f"Context building error: {e}")
    
    return context


async def classify_with_claude(event: Event, context: Dict[str, Any]) -> Classification:
    """
    Use Claude API to classify the event with prompt caching
    """
    
    # System prompt (cached for cost efficiency)
    system_prompt = """You are the Triage Engine for ClaudeHome, an autonomous smart home AI.

Your job: Classify events into categories to route them to the right handler.

CATEGORIES:

1. PROACTIVE_OPP (Proactive Opportunity)
   - Repeated manual actions (user did this 3+ times recently)
   - Predictable patterns (happens same time daily)
   - Obvious automation candidates
   Example: "User turned on kitchen lights at 6 AM (3rd time this week)"

2. ANOMALY_DRIFT (Anomaly or System Drift)
   - Unusual sensor readings (temp spike, unexpected values)
   - Device unavailable/unresponsive
   - System performance degradation
   Example: "Temperature jumped from 20°C to 35°C in 5 minutes"

3. CAUSAL_INFERENCE (User Override or Causality Learning)
   - User reversed an automation within 60 seconds
   - Automation triggered unexpected consequences
   - User preference signal (they want different behavior)
   Example: "User turned off light that automation just turned on"

4. ROUTINE (Normal Operation)
   - Expected state changes
   - Regular sensor updates
   - Normal automation behavior
   Example: "Motion sensor detected movement, light turned on as expected"

Respond ONLY with JSON:
{
  "category": "PROACTIVE_OPP|ANOMALY_DRIFT|CAUSAL_INFERENCE|ROUTINE",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation (1-2 sentences)",
  "priority": 1-3
}

Priority: 1=urgent/actionable, 2=notable, 3=log-only
"""
    
    # User prompt with event details
    user_prompt = f"""Event to classify:

Entity: {event.entity_id}
State: {event.old_state} → {event.state}
Time: {event.timestamp}

Recent History (this entity):
{json.dumps(context.get('entity_history', [])[:5], indent=2)}

Recent Events (all entities, last 5 min):
{json.dumps(context.get('recent_events', [])[:10], indent=2)}

Active Rules:
{json.dumps(context.get('active_rules', []), indent=2)}

Classify this event."""
    
    try:
        response = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Cache system prompt
                }
            ],
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse response
        result_text = response.content[0].text
        result = json.loads(result_text)
        
        logger.info(f"Classified {event.entity_id} as {result['category']} (confidence: {result['confidence']})")
        
        return Classification(**result)
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        # Fallback to ROUTINE if classification fails
        return Classification(
            category="ROUTINE",
            confidence=0.5,
            reasoning=f"Classification failed: {str(e)}",
            priority=3
        )


async def store_classification(event: Event, classification: Classification):
    """
    Store classification result in Redis for analysis
    """
    try:
        key = f"claudehome:classifications:{classification.category}"
        value = json.dumps({
            "entity_id": event.entity_id,
            "state": event.state,
            "timestamp": event.timestamp,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning
        })
        await redis_client.lpush(key, value)
        await redis_client.ltrim(key, 0, 99)  # Keep last 100
    except Exception as e:
        logger.warning(f"Failed to store classification: {e}")


async def publish_classification(event: Event, classification: Classification):
    """
    Publish to internal event bus for other services to consume
    """
    try:
        channel = f"claudehome:triage:{classification.category}"
        message = json.dumps({
            "event": event.dict(),
            "classification": classification.dict()
        })
        await redis_client.publish(channel, message)
        logger.info(f"Published {classification.category} to event bus")
    except Exception as e:
        logger.warning(f"Failed to publish: {e}")


async def expire_cache(key: str, ttl: int):
    """
    Remove key from cache after TTL
    """
    await asyncio.sleep(ttl)
    if key in classification_cache:
        del classification_cache[key]


@app.post("/batch_classify")
async def batch_classify(events: List[Event]):
    """
    Classify multiple events in batch
    """
    results = []
    for event in events:
        classification = await classify_event(event)
        results.append({
            "event": event.dict(),
            "classification": classification.dict()
        })
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)