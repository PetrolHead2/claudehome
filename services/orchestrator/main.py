#!/usr/bin/env python3
"""
ClaudeHome Orchestrator Service

Main coordination service that connects all ClaudeHome components and
implements the event processing pipeline.

Pipeline: Redis Queue → Privacy Filter → Triage → Claude API → Actions
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from contextlib import asynccontextmanager

import anthropic
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orchestrator")

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
PRIVACY_FILTER_URL = os.environ.get("PRIVACY_FILTER_URL", "http://privacy_filter:8000")
HEALTH_MONITOR_URL = os.environ.get("HEALTH_MONITOR_URL", "http://health_monitor:8003")
EVENT_INGESTER_URL = os.environ.get("EVENT_INGESTER_URL", "http://event_ingester:8002")

# Processing configuration
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
BATCH_WAIT_SECONDS = float(os.environ.get("BATCH_WAIT_SECONDS", "5.0"))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Global state
redis_client: Optional[redis.Redis] = None
anthropic_client: Optional[anthropic.Anthropic] = None
http_client: Optional[httpx.AsyncClient] = None
running: bool = False


class EventCategory(str, Enum):
    """Event classification categories."""
    ROUTINE = "routine"          # Normal state changes, ignore
    ANOMALY = "anomaly"          # Unusual pattern, investigate
    PROACTIVE = "proactive"      # Opportunity for automation
    CAUSAL = "causal"            # User override detected
    CRITICAL = "critical"        # Requires immediate attention


class TriageResult(BaseModel):
    """Result of event triage."""
    event_id: str
    category: EventCategory
    confidence: float
    reasoning: str
    suggested_action: Optional[str] = None
    priority: int = 5  # 1-10, 1 highest


class OrchestratorStats(BaseModel):
    """Orchestrator statistics."""
    events_processed: int = 0
    events_by_category: dict = {}
    claude_calls: int = 0
    claude_tokens_used: int = 0
    errors: int = 0
    uptime_seconds: float = 0
    last_event_time: Optional[str] = None
    services_healthy: dict = {}


stats = OrchestratorStats()
start_time = time.time()

# System prompt for event triage
TRIAGE_SYSTEM_PROMPT = """You are the ClaudeHome triage engine. Your job is to classify Home Assistant events into categories.

Categories:
- ROUTINE: Normal expected behavior (light turned on during day, temperature fluctuation within normal range)
- ANOMALY: Unusual pattern that warrants attention (device offline, unexpected state change, unusual timing)
- PROACTIVE: Opportunity to suggest an automation or optimization (repeated manual action, energy waste)
- CAUSAL: User override of automation detected (manual light toggle after automation triggered)
- CRITICAL: Requires immediate attention (security event, device failure, safety issue)

For each event, respond with JSON:
{
  "category": "ROUTINE|ANOMALY|PROACTIVE|CAUSAL|CRITICAL",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "suggested_action": "optional action to take",
  "priority": 1-10
}

Consider:
- Time of day and typical patterns
- Entity type and expected behavior
- Frequency and recency of similar events
- Potential safety implications
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global redis_client, anthropic_client, http_client, running

    # Startup
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_URL}")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

    if ANTHROPIC_API_KEY:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized")
    else:
        logger.warning("ANTHROPIC_API_KEY not set - Claude features disabled")

    http_client = httpx.AsyncClient(timeout=30.0)

    # Start background processing
    running = True
    asyncio.create_task(process_event_queue())

    yield

    # Shutdown
    running = False
    if redis_client:
        await redis_client.close()
    if http_client:
        await http_client.aclose()


app = FastAPI(
    title="ClaudeHome Orchestrator",
    description="Main coordination service for ClaudeHome autonomous smart home system",
    version="1.0.0",
    lifespan=lifespan
)


async def check_service_health(name: str, url: str) -> bool:
    """Check if a service is healthy."""
    try:
        response = await http_client.get(f"{url}/health")
        return response.status_code == 200
    except Exception:
        return False


async def anonymize_event(event: dict) -> dict:
    """Send event through privacy filter for anonymization."""
    try:
        response = await http_client.post(
            f"{PRIVACY_FILTER_URL}/anonymize",
            json={"data": event, "context": "event_triage"}
        )
        if response.status_code == 200:
            return response.json()["data"]
    except Exception as e:
        logger.error(f"Privacy filter error: {e}")
    return event  # Return original if anonymization fails


async def deanonymize_response(data: dict) -> dict:
    """De-anonymize Claude response for execution."""
    try:
        response = await http_client.post(
            f"{PRIVACY_FILTER_URL}/deanonymize",
            json={"data": data}
        )
        if response.status_code == 200:
            return response.json()["data"]
    except Exception as e:
        logger.error(f"Privacy filter de-anonymize error: {e}")
    return data


async def triage_event(event: dict) -> Optional[TriageResult]:
    """Use Claude to triage an event."""
    global stats

    if not anthropic_client:
        logger.warning("Claude not available for triage")
        return None

    try:
        # Anonymize event first
        anon_event = await anonymize_event(event)

        message = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=TRIAGE_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Triage this Home Assistant event:\n{json.dumps(anon_event, indent=2)}"
            }]
        )

        stats.claude_calls += 1
        stats.claude_tokens_used += message.usage.input_tokens + message.usage.output_tokens

        # Parse response
        response_text = message.content[0].text
        # Try to extract JSON from response
        try:
            # Handle potential markdown code blocks
            if "```" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                response_text = response_text[json_start:json_end]

            result_data = json.loads(response_text)

            return TriageResult(
                event_id=event.get("state_id", "unknown"),
                category=EventCategory(result_data.get("category", "routine").lower()),
                confidence=float(result_data.get("confidence", 0.5)),
                reasoning=result_data.get("reasoning", ""),
                suggested_action=result_data.get("suggested_action"),
                priority=int(result_data.get("priority", 5))
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse triage response: {e}")
            return None

    except Exception as e:
        logger.error(f"Triage error: {e}")
        stats.errors += 1
        return None


async def process_event_queue():
    """Process events from Redis queue."""
    global stats, running

    logger.info("Starting event queue processor")

    while running:
        try:
            if not redis_client:
                await asyncio.sleep(5)
                continue

            # Batch collect events
            events = []
            for _ in range(BATCH_SIZE):
                event_json = await redis_client.rpop("claudehome:events")
                if event_json:
                    try:
                        events.append(json.loads(event_json))
                    except json.JSONDecodeError:
                        pass
                else:
                    break

            if not events:
                await asyncio.sleep(BATCH_WAIT_SECONDS)
                continue

            logger.info(f"Processing batch of {len(events)} events")

            for event in events:
                await process_single_event(event)

        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            stats.errors += 1
            await asyncio.sleep(5)


async def process_single_event(event: dict):
    """Process a single event through the pipeline."""
    global stats

    entity_id = event.get("entity_id", "unknown")
    stats.events_processed += 1
    stats.last_event_time = datetime.now().isoformat()

    # Quick filter: skip routine state updates for certain domains
    if should_skip_event(event):
        stats.events_by_category["skipped"] = stats.events_by_category.get("skipped", 0) + 1
        return

    # Triage the event
    result = await triage_event(event)

    if result:
        category = result.category.value
        stats.events_by_category[category] = stats.events_by_category.get(category, 0) + 1

        # Log non-routine events
        if result.category != EventCategory.ROUTINE:
            logger.info(f"[{result.category.value.upper()}] {entity_id}: {result.reasoning}")

            # Store for later analysis
            await store_triage_result(event, result)

            # Handle critical events immediately
            if result.category == EventCategory.CRITICAL:
                await handle_critical_event(event, result)


def should_skip_event(event: dict) -> bool:
    """Quick filter to skip obviously routine events."""
    entity_id = event.get("entity_id", "")
    state = event.get("state", "")

    # Skip unavailable -> unavailable
    if state == "unavailable":
        return True

    # Skip high-frequency sensor updates (these are typically noise)
    skip_patterns = [
        "_rssi", "_linkquality", "_uptime", "_signal",
        "_power",  # Power sensors update frequently
        "_voltage", "_current", "_amps"
    ]

    for pattern in skip_patterns:
        if pattern in entity_id.lower():
            return True

    return False


async def store_triage_result(event: dict, result: TriageResult):
    """Store triage result for analysis."""
    if not redis_client:
        return

    try:
        storage_data = {
            "event": event,
            "result": result.model_dump(),
            "timestamp": datetime.now().isoformat()
        }

        # Store in category-specific list
        key = f"claudehome:triage:{result.category.value}"
        await redis_client.lpush(key, json.dumps(storage_data))
        await redis_client.ltrim(key, 0, 999)  # Keep last 1000 per category

    except Exception as e:
        logger.error(f"Failed to store triage result: {e}")


async def handle_critical_event(event: dict, result: TriageResult):
    """Handle critical events that need immediate attention."""
    logger.warning(f"CRITICAL EVENT: {event.get('entity_id')} - {result.reasoning}")

    # Could trigger:
    # - Discord notification
    # - Push notification
    # - Automatic remediation

    # For now, just store in critical queue
    if redis_client:
        await redis_client.lpush(
            "claudehome:critical",
            json.dumps({
                "event": event,
                "result": result.model_dump(),
                "timestamp": datetime.now().isoformat()
            })
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    services = {
        "redis": False,
        "privacy_filter": False,
        "health_monitor": False,
        "event_ingester": False,
        "claude": bool(anthropic_client)
    }

    if redis_client:
        try:
            await redis_client.ping()
            services["redis"] = True
        except Exception:
            pass

    services["privacy_filter"] = await check_service_health("privacy_filter", PRIVACY_FILTER_URL)
    services["health_monitor"] = await check_service_health("health_monitor", HEALTH_MONITOR_URL)
    services["event_ingester"] = await check_service_health("event_ingester", EVENT_INGESTER_URL)

    stats.services_healthy = services

    all_healthy = all(services.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services,
        "processing": running
    }


@app.get("/stats", response_model=OrchestratorStats)
async def get_stats():
    """Get orchestrator statistics."""
    stats.uptime_seconds = time.time() - start_time
    return stats


@app.get("/triage/recent")
async def get_recent_triage(category: Optional[str] = None, limit: int = 20):
    """Get recent triage results."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    results = []

    if category:
        categories = [category]
    else:
        categories = [c.value for c in EventCategory]

    for cat in categories:
        key = f"claudehome:triage:{cat}"
        items = await redis_client.lrange(key, 0, limit - 1)
        for item in items:
            try:
                results.append(json.loads(item))
            except json.JSONDecodeError:
                pass

    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return {"count": len(results), "results": results[:limit]}


@app.get("/critical")
async def get_critical_events(limit: int = 10):
    """Get critical events queue."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    items = await redis_client.lrange("claudehome:critical", 0, limit - 1)
    events = []
    for item in items:
        try:
            events.append(json.loads(item))
        except json.JSONDecodeError:
            pass

    return {"count": len(events), "events": events}


@app.post("/triage/manual")
async def manual_triage(event: dict):
    """Manually submit an event for triage."""
    result = await triage_event(event)
    if result:
        return result.model_dump()
    raise HTTPException(status_code=500, detail="Triage failed")


@app.post("/reset/stats")
async def reset_stats():
    """Reset statistics."""
    global stats
    stats = OrchestratorStats()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
