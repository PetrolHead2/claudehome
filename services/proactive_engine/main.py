"""
Proactive Engine - Automation Suggestion Generator

Analyzes patterns and generates automation suggestions:
- Detects repeated manual actions
- Identifies time-based patterns
- Suggests system optimizations
- Validates suggestions before presenting
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
from anthropic import AsyncAnthropic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClaudeHome Proactive Engine")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
HOME_ASSISTANT_URL = os.getenv("HOME_ASSISTANT_URL")
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

# Initialize clients
redis_client = None
anthropic_client = None


class Pattern(BaseModel):
    pattern_type: str  # manual_action, time_based, system_optimization
    entities: List[str]
    frequency: int
    confidence: float
    description: str
    context: Dict[str, Any] = {}


class Suggestion(BaseModel):
    suggestion_id: str
    title: str
    description: str
    automation_yaml: str
    confidence: float
    benefits: List[str]
    pattern: Pattern


@app.on_event("startup")
async def startup():
    global redis_client, anthropic_client
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    logger.info("Proactive Engine started")


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "proactive_engine"}


@app.post("/analyze_patterns")
async def analyze_patterns():
    """
    Analyze recent events for patterns and generate suggestions
    """
    try:
        # Detect patterns
        patterns = await detect_patterns()
        
        if not patterns:
            return {"message": "No actionable patterns detected", "patterns": []}
        
        # Generate suggestions for high-confidence patterns
        suggestions = []
        for pattern in patterns:
            if pattern.confidence >= 0.7:  # Only high-confidence patterns
                suggestion = await generate_suggestion(pattern)
                if suggestion:
                    suggestions.append(suggestion)
        
        # Store suggestions for user review
        for suggestion in suggestions:
            await store_suggestion(suggestion)
        
        # Publish to event bus
        for suggestion in suggestions:
            await publish_suggestion(suggestion)
        
        return {
            "patterns_detected": len(patterns),
            "suggestions_generated": len(suggestions),
            "suggestions": [s.dict() for s in suggestions]
        }
        
    except Exception as e:
        logger.error(f"Pattern analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def detect_patterns() -> List[Pattern]:
    """
    Detect patterns from recent PROACTIVE_OPP classifications
    """
    patterns = []
    
    try:
        # Get recent proactive opportunities
        key = "claudehome:classifications:PROACTIVE_OPP"
        events = await redis_client.lrange(key, 0, 99)
        
        if len(events) < 3:
            logger.info("Not enough events for pattern detection")
            return patterns
        
        # Parse events
        parsed_events = [json.loads(e) for e in events]
        
        # Analyze with Claude
        patterns = await analyze_events_with_claude(parsed_events)
        
        logger.info(f"Detected {len(patterns)} patterns")
        
    except Exception as e:
        logger.error(f"Pattern detection error: {e}")
    
    return patterns


async def analyze_events_with_claude(events: List[Dict]) -> List[Pattern]:
    """
    Use Claude to analyze events and detect patterns
    """
    
    system_prompt = """You are the Pattern Detector for ClaudeHome.

Your job: Analyze recent events and identify ACTIONABLE patterns that could become automations.

Look for:
1. REPEATED MANUAL ACTIONS - User did the same thing 3+ times
   Example: "User turned on kitchen lights at 6 AM (5 times this week)"

2. TIME-BASED PATTERNS - Actions happen at predictable times
   Example: "Coffee maker turns on, then lights, every weekday morning"

3. SEQUENCE PATTERNS - Event A always followed by Event B
   Example: "Motion detected → lights on (happens 90% of the time)"

4. SYSTEM OPTIMIZATIONS - Configuration inefficiencies
   Example: "Database growing 600MB/month, exclude high-frequency sensors"

Respond ONLY with JSON array:
[
  {
    "pattern_type": "manual_action|time_based|sequence|system_optimization",
    "entities": ["entity_id_1", "entity_id_2"],
    "frequency": 5,
    "confidence": 0.0-1.0,
    "description": "Brief description",
    "context": {
      "times": ["06:00", "06:05"],
      "days": ["Mon", "Tue", "Wed"],
      "trigger_entity": "sensor.motion",
      "action_entity": "light.kitchen"
    }
  }
]

Confidence guidelines:
- 0.9+: Pattern happens >90% of the time
- 0.7-0.9: Pattern is consistent but has exceptions
- 0.5-0.7: Pattern exists but is irregular
- <0.5: Too inconsistent to suggest

Only return patterns with confidence >= 0.7
"""
    
    user_prompt = f"""Recent events classified as PROACTIVE_OPP:

{json.dumps(events[:50], indent=2)}

Analyze these events and identify actionable patterns."""
    
    try:
        response = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result_text = response.content[0].text
        patterns_data = json.loads(result_text)
        
        patterns = [Pattern(**p) for p in patterns_data]
        logger.info(f"Claude identified {len(patterns)} patterns")
        
        return patterns
        
    except Exception as e:
        logger.error(f"Claude pattern analysis error: {e}")
        return []


async def generate_suggestion(pattern: Pattern) -> Optional[Suggestion]:
    """
    Generate automation suggestion from pattern using Claude
    """
    
    system_prompt = """You are the Automation Generator for ClaudeHome.

Your job: Convert patterns into Home Assistant automation YAML.

Requirements:
1. Generate valid YAML syntax
2. Include clear alias/description
3. Add safety checks (time limits, conditions)
4. Use appropriate triggers and actions
5. Include comments explaining logic

Example output:
```yaml
- id: claudehome_coffee_lights_001
  alias: "ClaudeHome: Coffee Maker → Kitchen Lights"
  description: "Turn on kitchen lights when coffee maker starts (weekday mornings)"
  trigger:
    - platform: state
      entity_id: sensor.coffee_maker_power
      to: "on"
  condition:
    - condition: time
      after: "05:00:00"
      before: "09:00:00"
      weekday:
        - mon
        - tue
        - wed
        - thu
        - fri
  action:
    - service: light.turn_on
      target:
        entity_id: light.kitchen
      data:
        brightness_pct: 70
        transition: 2
```

Respond with JSON:
{
  "title": "Brief title",
  "description": "What this automation does",
  "automation_yaml": "Valid YAML string",
  "benefits": ["Benefit 1", "Benefit 2"],
  "confidence": 0.0-1.0
}
"""
    
    user_prompt = f"""Pattern to convert into automation:

Type: {pattern.pattern_type}
Description: {pattern.description}
Entities: {', '.join(pattern.entities)}
Frequency: {pattern.frequency} times
Context: {json.dumps(pattern.context, indent=2)}

Generate a Home Assistant automation for this pattern."""
    
    try:
        response = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result_text = response.content[0].text
        suggestion_data = json.loads(result_text)
        
        # Create suggestion ID
        suggestion_id = f"suggestion_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        suggestion = Suggestion(
            suggestion_id=suggestion_id,
            pattern=pattern,
            **suggestion_data
        )
        
        logger.info(f"Generated suggestion: {suggestion.title}")
        
        return suggestion
        
    except Exception as e:
        logger.error(f"Suggestion generation error: {e}")
        return None


async def store_suggestion(suggestion: Suggestion):
    """
    Store suggestion in Redis for user review
    """
    try:
        key = f"claudehome:suggestions:pending:{suggestion.suggestion_id}"
        value = json.dumps(suggestion.dict())
        await redis_client.setex(key, 86400 * 7, value)  # 7 days TTL
        
        # Add to pending list
        await redis_client.sadd("claudehome:suggestions:pending", suggestion.suggestion_id)
        
        logger.info(f"Stored suggestion: {suggestion.suggestion_id}")
        
    except Exception as e:
        logger.error(f"Failed to store suggestion: {e}")


async def publish_suggestion(suggestion: Suggestion):
    """
    Publish suggestion to event bus
    """
    try:
        channel = "claudehome:proactive:suggestion_ready"
        message = json.dumps({
            "suggestion_id": suggestion.suggestion_id,
            "title": suggestion.title,
            "confidence": suggestion.confidence
        })
        await redis_client.publish(channel, message)
        logger.info(f"Published suggestion: {suggestion.title}")
        
    except Exception as e:
        logger.error(f"Failed to publish suggestion: {e}")


@app.get("/suggestions/pending")
async def get_pending_suggestions():
    """
    Get all pending suggestions awaiting user review
    """
    try:
        suggestion_ids = await redis_client.smembers("claudehome:suggestions:pending")
        
        suggestions = []
        for sid in suggestion_ids:
            key = f"claudehome:suggestions:pending:{sid}"
            data = await redis_client.get(key)
            if data:
                suggestions.append(json.loads(data))
        
        return {"count": len(suggestions), "suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"Error fetching suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suggestions/{suggestion_id}/approve")
async def approve_suggestion(suggestion_id: str):
    """
    Mark suggestion as approved (will be deployed by orchestrator)
    """
    try:
        # Move from pending to approved
        key = f"claudehome:suggestions:pending:{suggestion_id}"
        data = await redis_client.get(key)
        
        if not data:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        # Store in approved
        approved_key = f"claudehome:suggestions:approved:{suggestion_id}"
        await redis_client.setex(approved_key, 86400 * 30, data)  # 30 days
        
        # Remove from pending
        await redis_client.delete(key)
        await redis_client.srem("claudehome:suggestions:pending", suggestion_id)
        
        # Publish approval event
        await redis_client.publish(
            "claudehome:proactive:suggestion_approved",
            json.dumps({"suggestion_id": suggestion_id})
        )
        
        logger.info(f"Approved suggestion: {suggestion_id}")
        
        return {"status": "approved", "suggestion_id": suggestion_id}
        
    except Exception as e:
        logger.error(f"Error approving suggestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suggestions/{suggestion_id}/reject")
async def reject_suggestion(suggestion_id: str, reason: str = "User rejected"):
    """
    Reject a suggestion
    """
    try:
        key = f"claudehome:suggestions:pending:{suggestion_id}"
        await redis_client.delete(key)
        await redis_client.srem("claudehome:suggestions:pending", suggestion_id)
        
        # Store rejection for learning
        rejection_key = f"claudehome:suggestions:rejected:{suggestion_id}"
        await redis_client.setex(rejection_key, 86400 * 30, json.dumps({"reason": reason}))
        
        logger.info(f"Rejected suggestion: {suggestion_id} - {reason}")
        
        return {"status": "rejected", "suggestion_id": suggestion_id}
        
    except Exception as e:
        logger.error(f"Error rejecting suggestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)