#!/usr/bin/env python3
"""
Event Ingester Service for ClaudeHome

Monitors Home Assistant for state changes and pushes events to Redis queue
for processing by other ClaudeHome services.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("event_ingester")

# Configuration
HA_CONFIG_DIR = os.environ.get("HA_CONFIG_DIR", "/media/pi/NextCloud/homeassistant")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1.0"))
DB_PATH = Path(HA_CONFIG_DIR) / "home-assistant_v2.db"

# Global state
redis_client: Optional[redis.Redis] = None
last_state_id: int = 0
running: bool = False


class EventStats(BaseModel):
    """Event ingestion statistics."""
    events_processed: int = 0
    last_event_time: Optional[str] = None
    last_state_id: int = 0
    uptime_seconds: float = 0
    redis_connected: bool = False


stats = EventStats()
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global redis_client, running

    # Startup
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_URL}")
        stats.redis_connected = True
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        stats.redis_connected = False

    # Start background polling
    running = True
    asyncio.create_task(poll_state_changes())

    yield

    # Shutdown
    running = False
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="ClaudeHome Event Ingester",
    description="Monitors Home Assistant state changes and queues events",
    version="1.0.0",
    lifespan=lifespan
)


def get_db_connection() -> Optional[sqlite3.Connection]:
    """Get connection to Home Assistant database."""
    if not DB_PATH.exists():
        logger.warning(f"Database not found at {DB_PATH}")
        return None

    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None


async def poll_state_changes():
    """Poll database for state changes."""
    global last_state_id

    logger.info("Starting state change polling")

    while running:
        try:
            conn = get_db_connection()
            if not conn:
                await asyncio.sleep(POLL_INTERVAL * 5)
                continue

            cursor = conn.cursor()

            # Get the latest state_id if we don't have one
            if last_state_id == 0:
                cursor.execute("SELECT MAX(state_id) FROM states")
                result = cursor.fetchone()
                if result and result[0]:
                    last_state_id = result[0]
                    stats.last_state_id = last_state_id
                    logger.info(f"Starting from state_id: {last_state_id}")

            # Query for new states
            cursor.execute("""
                SELECT
                    s.state_id,
                    s.entity_id,
                    s.state,
                    s.attributes,
                    s.last_changed_ts,
                    s.last_updated_ts
                FROM states s
                WHERE s.state_id > ?
                ORDER BY s.state_id ASC
                LIMIT 100
            """, (last_state_id,))

            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                await process_state_change(dict(row))
                last_state_id = row["state_id"]
                stats.last_state_id = last_state_id

        except Exception as e:
            logger.error(f"Polling error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


async def process_state_change(state: dict):
    """Process a single state change event."""
    global stats

    entity_id = state.get("entity_id", "unknown")

    # Create event payload
    event = {
        "type": "state_changed",
        "entity_id": entity_id,
        "state": state.get("state"),
        "timestamp": datetime.now().isoformat(),
        "state_id": state.get("state_id")
    }

    # Parse attributes if present
    if state.get("attributes"):
        try:
            event["attributes"] = json.loads(state["attributes"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Push to Redis queue if connected
    if redis_client and stats.redis_connected:
        try:
            await redis_client.lpush("claudehome:events", json.dumps(event))
            await redis_client.ltrim("claudehome:events", 0, 9999)  # Keep last 10k events
            logger.debug(f"Queued event for {entity_id}")
        except Exception as e:
            logger.error(f"Redis push error: {e}")
            stats.redis_connected = False

    stats.events_processed += 1
    stats.last_event_time = event["timestamp"]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_available = DB_PATH.exists()

    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
            stats.redis_connected = True
        except Exception:
            stats.redis_connected = False

    return {
        "status": "healthy" if db_available else "degraded",
        "database": str(DB_PATH),
        "database_available": db_available,
        "redis_connected": redis_ok,
        "polling": running
    }


@app.get("/stats", response_model=EventStats)
async def get_stats():
    """Get ingestion statistics."""
    stats.uptime_seconds = time.time() - start_time
    return stats


@app.get("/queue/length")
async def queue_length():
    """Get current queue length."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not connected")

    try:
        length = await redis_client.llen("claudehome:events")
        return {"queue": "claudehome:events", "length": length}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/peek")
async def peek_queue(count: int = 10):
    """Peek at events in the queue without removing them."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not connected")

    try:
        events = await redis_client.lrange("claudehome:events", 0, count - 1)
        return {
            "count": len(events),
            "events": [json.loads(e) for e in events]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/clear")
async def clear_queue():
    """Clear the event queue."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not connected")

    try:
        await redis_client.delete("claudehome:events")
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_position():
    """Reset the state position to start from latest."""
    global last_state_id

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database not available")

    cursor = conn.cursor()
    cursor.execute("SELECT MAX(state_id) FROM states")
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        last_state_id = result[0]
        stats.last_state_id = last_state_id
        return {"status": "reset", "new_position": last_state_id}

    raise HTTPException(status_code=500, detail="Could not get max state_id")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
