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
import aiohttp
import logging

# Import prediction client
from prediction_client import enhance_pattern_with_predictions, get_prediction_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClaudeHome Proactive Engine")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000")
HOME_ASSISTANT_URL = os.getenv("HOME_ASSISTANT_URL")
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

# Initialize clients
redis_client = None
# anthropic_client = None


async def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
    """
    Call LLM service (supports multiple providers)
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LLM_SERVICE_URL}/generate",
                json={
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"LLM service error: {error_text}")
                
                data = await resp.json()
                logger.info(f"LLM response from {data.get('provider')}/{data.get('model_used')}")
                return data["text"]
                
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that might contain markdown or extra content
    """
    import re
    
    # Remove markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
    
    # Remove common prefixes
    text = text.strip()
    lines = text.split('\n')
    
    # Skip lines until we find JSON start
    json_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('[') or stripped.startswith('{'):
            json_start = i
            break
    
    if json_start >= 0:
        text = '\n'.join(lines[json_start:])
    
    # Find matching braces/brackets
    if text.strip().startswith('['):
        # Array
        bracket_count = 0
        for i, char in enumerate(text):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return text[:i+1]
    elif text.strip().startswith('{'):
        # Object
        brace_count = 0
        for i, char in enumerate(text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[:i+1]
    
    return text.strip()
    
    
class Pattern(BaseModel):
    pattern_type: str  # manual_action, time_based, system_optimization
    entities: List[str]
    frequency: int
    confidence: float
    description: str
    context: Dict[str, Any] = {}
    # Phase 3: ML predictions
    predictions: Dict[str, Any] = {}  # {entity_id: {next_on_time, confidence, ...}}
    suggested_timing: Optional[Dict[str, Any]] = None  # {trigger_time, reason, confidence}


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
    # anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
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
    PHASE 3: Enhanced with ML predictions for smarter timing
    """
    try:
        # Detect patterns
        patterns = await detect_patterns()
        
        if not patterns:
            return {"message": "No actionable patterns detected", "patterns": []}
        
        # PHASE 3: Enhance patterns with predictions
        enhanced_patterns = []
        prediction_enhancements = 0
        
        for pattern in patterns:
            try:
                # Add predictive context to pattern
                enhanced = await enhance_pattern_with_predictions(pattern.dict())
                
                # Convert back to Pattern object but preserve predictions
                pattern.predictions = enhanced.get('predictions', {})
                pattern.suggested_timing = enhanced.get('suggested_timing')
                
                enhanced_patterns.append(pattern)
                
                if pattern.predictions:
                    prediction_enhancements += 1
                    logger.info(f"Enhanced '{pattern.description}' with predictions for {len(pattern.predictions)} entities")
                    
            except Exception as e:
                logger.warning(f"Could not enhance pattern with predictions: {e}")
                enhanced_patterns.append(pattern)
        
        # Generate suggestions for high-confidence patterns
        suggestions = []
        for pattern in enhanced_patterns:
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
            "patterns_enhanced_with_predictions": prediction_enhancements,
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


async def get_context_summary() -> Dict[str, Any]:
    """
    Get context summary from Context Builder
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://context_builder:8000/stats",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        return {}
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        return {}


async def validate_entities(entities: List[str]) -> List[str]:
    """
    Validate that entities actually exist in HA
    """
    valid_entities = []
    
    try:
        async with aiohttp.ClientSession() as session:
            for entity_id in entities:
                async with session.get(
                    f"http://context_builder:8000/entity/{entity_id}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        valid_entities.append(entity_id)
                    else:
                        logger.warning(f"Entity {entity_id} not found in HA")
    except Exception as e:
        logger.error(f"Entity validation error: {e}")
        # Return all if validation fails
        return entities
    
    return valid_entities


async def check_duplicate_automation(pattern_description: str) -> bool:
    """
    Check if similar automation already exists
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://context_builder:8000/automations",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    automations = data.get("automations", [])
                    
                    # Simple similarity check
                    for auto in automations:
                        alias = auto.get("alias", "").lower()
                        description = auto.get("description", "").lower()
                        
                        # Check if pattern description is very similar
                        pattern_lower = pattern_description.lower()
                        if (pattern_lower in alias or 
                            pattern_lower in description or
                            alias in pattern_lower):
                            logger.info(f"Found similar automation: {alias}")
                            return True
                    
        return False
        
    except Exception as e:
        logger.error(f"Duplicate check error: {e}")
        return False


async def analyze_events_with_claude(events: List[Dict]) -> List[Pattern]:
    """
    Use Claude to analyze events and detect patterns (ENHANCED with context)
    """
    
    # Get context from Context Builder
    context = await get_context_summary()
    
    # Build context summary for Claude
    context_summary = ""
    if context:
        context_summary = f"""
SYSTEM CONTEXT (Your Home Assistant Installation):
- Total Entities: {context.get('entities_total', 0)}
- Domains: {', '.join(list(context.get('entities_by_domain', {}).keys())[:15])}...
- Areas: {context.get('areas_total', 0)} rooms
- Existing Automations: {context.get('automations_total', 0)} already configured
- Integrations: {context.get('integrations_total', 0)} installed

IMPORTANT: Only suggest automations using entities that exist in this installation.
Do not suggest generic entity names - use actual domain counts above to validate.
"""
    
    system_prompt = f"""You are the Pattern Detector for ClaudeHome.

{context_summary}

Your job: Analyze recent events and identify ACTIONABLE patterns that could become automations.

CRITICAL RULES:
1. Only suggest patterns using entity domains that exist in the system
2. Be specific about entity types (use "light" if lights exist, "switch" if switches exist)
3. Consider that {context.get('automations_total', 0)} automations already exist - avoid duplicates
4. Use realistic entity naming conventions

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
  {{
    "pattern_type": "manual_action|time_based|sequence|system_optimization",
    "entities": ["entity_id_1", "entity_id_2"],
    "frequency": 5,
    "confidence": 0.0-1.0,
    "description": "Brief description",
    "context": {{
      "times": ["06:00", "06:05"],
      "days": ["Mon", "Tue", "Wed"],
      "trigger_entity": "sensor.motion",
      "action_entity": "light.kitchen"
    }}
  }}
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

Analyze these events and identify actionable patterns using ONLY entities that exist in this installation."""
    
    try:
        result_text = await call_llm(system_prompt, user_prompt, max_tokens=2000)
        
        logger.info(f"Raw LLM response: {result_text[:200]}...")  # Log first 200 chars
        
        # Extract JSON from text
        json_text = extract_json_from_text(result_text)
        
        logger.info(f"Extracted JSON: {json_text[:200]}...")
        
        patterns_data = json.loads(json_text)

        patterns = []
        for p in patterns_data:
            # Validate entities exist
            entities = p.get("entities", [])
            valid_entities = await validate_entities(entities)
            
            if not valid_entities:
                logger.warning(f"Pattern skipped - no valid entities: {p['description']}")
                continue
            
            # Check for duplicates
            is_duplicate = await check_duplicate_automation(p['description'])
            if is_duplicate:
                logger.info(f"Pattern skipped - duplicate detected: {p['description']}")
                continue
            
            # Use validated entities
            p["entities"] = valid_entities
            patterns.append(Pattern(**p))
        
        logger.info(f"Claude identified {len(patterns)} valid patterns")
        
        return patterns
        
    except Exception as e:
        logger.error(f"Claude pattern analysis error: {e}")
        return []


async def generate_suggestion(pattern: Pattern) -> Optional[Suggestion]:
    """
    Generate automation suggestion from pattern using Claude (ENHANCED with context)
    """
    
    # Get entity details for entities in pattern
    entity_details = {}
    try:
        async with aiohttp.ClientSession() as session:
            for entity_id in pattern.entities:
                async with session.get(
                    f"http://context_builder:8000/entity/{entity_id}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        entity_details[entity_id] = await resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch entity details: {e}")
    
    # Build entity context
    entity_context = ""
    if entity_details:
        entity_context = "\n\nENTITY DETAILS:\n"
        for eid, details in entity_details.items():
            entity_context += f"- {eid}:\n"
            entity_context += f"  Platform: {details.get('platform')}\n"
            entity_context += f"  Device Class: {details.get('device_class')}\n"
            if details.get('capabilities'):
                entity_context += f"  Capabilities: {details.get('capabilities')}\n"
    
    system_prompt = f"""You are the Automation Generator for ClaudeHome.

{entity_context}

CRITICAL: You MUST respond with ONLY a JSON object. No other text, no markdown, no YAML.

Your job: Convert patterns into Home Assistant automation YAML.

Requirements:
1. Generate valid YAML syntax for the automation
2. Use ONLY the entities provided (they are validated to exist)
3. Include clear alias/description
4. Add safety checks (time limits, conditions)
5. Use appropriate triggers and actions based on entity capabilities
6. DO NOT include 'id:' field - it will be added automatically

Response format (JSON ONLY - no markdown):
{{
  "title": "Brief title (max 50 chars)",
  "description": "What this automation does (1-2 sentences)",
  "automation_yaml": "alias: 'Automation Name'\\ndescription: 'Description'\\ntrigger:\\n  - platform: time\\n    at: '06:00:00'\\naction:\\n  - service: light.turn_on\\n    target:\\n      entity_id: light.kitchen_roof_1\\n    data:\\n      brightness_pct: 70",
  "benefits": ["Benefit 1", "Benefit 2"],
  "confidence": 0.9
}}

IMPORTANT: 
- The automation_yaml field should contain the YAML as a single string with \\n for newlines
- Do NOT wrap response in markdown code blocks
- Do NOT include any text before or after the JSON
- Respond with PURE JSON ONLY
"""
    
    user_prompt = f"""Pattern to convert into automation:

Type: {pattern.pattern_type}
Description: {pattern.description}
Entities: {', '.join(pattern.entities)}
Frequency: {pattern.frequency} times
Confidence: {pattern.confidence}
Context: {json.dumps(pattern.context, indent=2)}

Generate a Home Assistant automation for this pattern using the provided entities.

Remember: Respond with JSON ONLY, no markdown, no extra text."""
    
    try:
        result_text = await call_llm(system_prompt, user_prompt, max_tokens=1500)
        
        logger.info(f"Raw suggestion response (first 300 chars): {result_text[:300]}")
        
        # Extract JSON from text
        json_text = extract_json_from_text(result_text)
        
        logger.info(f"Extracted JSON (first 300 chars): {json_text[:300]}")
        
        # Parse JSON
        try:
            suggestion_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Failed JSON text: {json_text}")
            return None
        
        # Validate structure
        if not isinstance(suggestion_data, dict):
            logger.error(f"Expected dict, got {type(suggestion_data)}: {suggestion_data}")
            return None
        
        # Check required fields
        required_fields = ['title', 'description', 'automation_yaml', 'benefits', 'confidence']
        missing_fields = [f for f in required_fields if f not in suggestion_data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            logger.error(f"Received data: {suggestion_data}")
            return None
        
        # Create suggestion ID
        suggestion_id = f"suggestion_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create suggestion
        suggestion = Suggestion(
            suggestion_id=suggestion_id,
            pattern=pattern,
            **suggestion_data
        )
        
        logger.info(f"Generated context-aware suggestion: {suggestion.title}")
        
        return suggestion
        
    except Exception as e:
        logger.error(f"Suggestion generation error: {e}", exc_info=True)
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