#!/usr/bin/env python3
"""
Event Ingester Service for ClaudeHome

Monitors Home Assistant via MQTT (real-time) and MariaDB (backup)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Set
from contextlib import asynccontextmanager

import redis.asyncio as redis
import aiomysql
import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("event_ingester")

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
DB_POLL_INTERVAL = float(os.environ.get("DB_POLL_INTERVAL", "30.0"))  # Reduced to 30s

# MariaDB configuration
MARIADB_HOST = os.environ.get("MARIADB_HOST")
MARIADB_PORT = int(os.environ.get("MARIADB_PORT", "3306"))
MARIADB_USER = os.environ.get("MARIADB_USER")
MARIADB_PASSWORD = os.environ.get("MARIADB_PASSWORD")
MARIADB_DATABASE = os.environ.get("MARIADB_DATABASE", "homeassistant")

# MQTT configuration
MQTT_BROKER = os.environ.get("MQTT_BROKER")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "")

# Global state
redis_client: Optional[redis.Redis] = None
db_pool: Optional[aiomysql.Pool] = None
mqtt_client: Optional[mqtt.Client] = None
last_state_id: int = 0
running: bool = False
processed_state_ids: Set[int] = set()  # Deduplication
main_loop: Optional[asyncio.AbstractEventLoop] = None  # ADD THIS LINE


class EventStats(BaseModel):
    """Event ingestion statistics."""
    events_processed: int = 0
    mqtt_events: int = 0
    db_events: int = 0
    last_event_time: Optional[str] = None
    last_state_id: int = 0
    uptime_seconds: float = 0
    redis_connected: bool = False
    mariadb_connected: bool = False
    mqtt_connected: bool = False


stats = EventStats()
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global redis_client, db_pool, mqtt_client, running, main_loop
    
    main_loop = asyncio.get_running_loop()

    # Connect to Redis
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_URL}")
        stats.redis_connected = True
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        stats.redis_connected = False

    # Connect to MariaDB
    if MARIADB_HOST:
        try:
            db_pool = await aiomysql.create_pool(
                host=MARIADB_HOST,
                port=MARIADB_PORT,
                user=MARIADB_USER,
                password=MARIADB_PASSWORD,
                db=MARIADB_DATABASE,
                maxsize=5,
                autocommit=True
            )
            logger.info(f"Connected to MariaDB at {MARIADB_HOST}:{MARIADB_PORT}")
            stats.mariadb_connected = True
        except Exception as e:
            logger.error(f"Failed to connect to MariaDB: {e}")
            stats.mariadb_connected = False

    # Setup MQTT
    if MQTT_BROKER:
        setup_mqtt()

    # Start background polling (reduced frequency - MQTT is primary)
    running = True
    asyncio.create_task(poll_state_changes())

    yield

    # Shutdown
    running = False
    if redis_client:
        await redis_client.close()
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


app = FastAPI(
    title="ClaudeHome Event Ingester",
    description="Monitors Home Assistant state changes via MQTT and MariaDB",
    version="2.0.0",
    lifespan=lifespan
)


def setup_mqtt():
    """Setup MQTT client for real-time events."""
    global mqtt_client

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
            stats.mqtt_connected = True
            # Subscribe to all state changes
            client.subscribe("homeassistant/+/+/state")
            client.subscribe("homeassistant/+/+/+/state")
            logger.info("Subscribed to Home Assistant state topics")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            stats.mqtt_connected = False

    def on_disconnect(client, userdata, rc):
        logger.warning(f"MQTT disconnected with code {rc}")
        stats.mqtt_connected = False

    def on_message(client, userdata, msg):
        try:
            # Parse MQTT message
            payload = msg.payload.decode()
            
            # Extract entity_id from topic
            # Topic format: homeassistant/sensor/bedroom_temp/state
            parts = msg.topic.split('/')
            if len(parts) >= 4:
                domain = parts[1]
                object_id = parts[2]
                entity_id = f"{domain}.{object_id}"
            else:
                logger.warning(f"Unexpected topic format: {msg.topic}")
                return

            # Create event
            event = {
                "type": "state_changed",
                "entity_id": entity_id,
                "state": payload,
                "timestamp": datetime.now().isoformat(),
                "source": "mqtt"
            }

            # Schedule in the main event loop
            if main_loop:
                asyncio.run_coroutine_threadsafe(process_mqtt_event(event), main_loop)
            # Get the running event loop and schedule the coroutine
            # loop = asyncio.get_event_loop()
            # asyncio.run_coroutine_threadsafe(process_mqtt_event(event), loop)

        except Exception as e:
            logger.error(f"MQTT message processing error: {e}")

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message = on_message

    if MQTT_USER:
        mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        logger.info("MQTT client started")
    except Exception as e:
        logger.error(f"Failed to connect to MQTT: {e}")
        stats.mqtt_connected = False


async def process_mqtt_event(event: dict):
    """Process event from MQTT (real-time)."""
    global stats

    entity_id = event.get("entity_id")

    # Push to Redis
    if redis_client and stats.redis_connected:
        try:
            await redis_client.lpush("claudehome:events", json.dumps(event))
            await redis_client.ltrim("claudehome:events", 0, 9999)
            logger.debug(f"MQTT event queued: {entity_id}")
        except Exception as e:
            logger.error(f"Redis push error: {e}")
            stats.redis_connected = False

    stats.events_processed += 1
    stats.mqtt_events += 1
    stats.last_event_time = event["timestamp"]


async def poll_state_changes():
    """Poll MariaDB as backup (catches anything MQTT missed)."""
    global last_state_id, processed_state_ids

    logger.info(f"Starting database backup polling (every {DB_POLL_INTERVAL}s)")

    if not db_pool:
        logger.error("Database pool not available")
        return

    while running:
        try:
            async with db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # Initialize position
                    if last_state_id == 0:
                        await cursor.execute("SELECT MAX(state_id) FROM states")
                        result = await cursor.fetchone()
                        if result and result.get('MAX(state_id)'):
                            last_state_id = result['MAX(state_id)']
                            stats.last_state_id = last_state_id
                            logger.info(f"Starting from state_id: {last_state_id}")

                    # Query for new states
                    query = """
                        SELECT
                            s.state_id,
                            sm.entity_id,
                            s.state,
                            s.attributes,
                            s.last_updated_ts
                        FROM states s
                        LEFT JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                        WHERE s.state_id > %s
                        ORDER BY s.state_id ASC
                        LIMIT 100
                    """
                    
                    await cursor.execute(query, (last_state_id,))
                    rows = await cursor.fetchall()

                    new_events = 0
                    for row in rows:
                        state_id = row["state_id"]
                        
                        # Skip if already processed via MQTT
                        if state_id in processed_state_ids:
                            last_state_id = state_id
                            continue

                        await process_db_event(row)
                        last_state_id = state_id
                        stats.last_state_id = last_state_id
                        new_events += 1

                        # Add to dedup set (limit size to 10000)
                        processed_state_ids.add(state_id)
                        if len(processed_state_ids) > 10000:
                            # Remove oldest half
                            to_remove = sorted(processed_state_ids)[:5000]
                            processed_state_ids.difference_update(to_remove)

                    if new_events > 0:
                        logger.info(f"DB backup caught {new_events} missed events")

        except Exception as e:
            logger.error(f"DB polling error: {e}")
            stats.mariadb_connected = False
            await asyncio.sleep(DB_POLL_INTERVAL * 2)
            continue

        await asyncio.sleep(DB_POLL_INTERVAL)


async def process_db_event(state: dict):
    """Process event from database (backup)."""
    global stats

    entity_id = state.get("entity_id", "unknown")

    event = {
        "type": "state_changed",
        "entity_id": entity_id,
        "state": state.get("state"),
        "timestamp": datetime.now().isoformat(),
        "state_id": state.get("state_id"),
        "source": "database"
    }

    # Parse attributes
    if state.get("attributes"):
        try:
            event["attributes"] = json.loads(state["attributes"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Push to Redis
    if redis_client and stats.redis_connected:
        try:
            await redis_client.lpush("claudehome:events", json.dumps(event))
            await redis_client.ltrim("claudehome:events", 0, 9999)
            logger.debug(f"DB event queued: {entity_id}")
        except Exception as e:
            logger.error(f"Redis push error: {e}")
            stats.redis_connected = False

    stats.events_processed += 1
    stats.db_events += 1
    stats.last_event_time = event["timestamp"]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
            stats.redis_connected = True
        except Exception:
            stats.redis_connected = False

    return {
        "status": "healthy",
        "mariadb_connected": stats.mariadb_connected,
        "mqtt_connected": stats.mqtt_connected,
        "redis_connected": redis_ok,
        "polling": running,
        "primary_source": "mqtt" if stats.mqtt_connected else "database"
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
    """Peek at events in the queue."""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)