#!/usr/bin/env python3
"""
Health Monitor Service for ClaudeHome

Monitors Home Assistant logs for errors, tracks patterns, and sends alerts.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("health_monitor")

# Configuration
HA_CONFIG_DIR = os.environ.get("HA_CONFIG_DIR", "/media/pi/NextCloud/homeassistant")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
LOG_PATH = Path(HA_CONFIG_DIR) / "home-assistant.log"
CHECK_INTERVAL = float(os.environ.get("CHECK_INTERVAL", "10.0"))

# Global state
running: bool = False
last_position: int = 0
error_buffer: deque = deque(maxlen=100)
pattern_counts: dict = {}


class ErrorEntry(BaseModel):
    """A single error entry."""
    timestamp: str
    level: str
    component: str
    message: str
    count: int = 1


class HealthStats(BaseModel):
    """Health monitoring statistics."""
    errors_last_hour: int = 0
    errors_last_24h: int = 0
    top_error_patterns: list = []
    last_check: Optional[str] = None
    log_file_size: int = 0
    uptime_seconds: float = 0


stats = HealthStats()
start_time = time.time()

# Error pattern definitions
ERROR_PATTERNS = {
    "connection_refused": r"Connection refused|ConnectionRefusedError",
    "timeout": r"TimeoutError|timed out|Timeout",
    "authentication": r"AuthenticationError|401|403|Unauthorized",
    "integration_setup": r"Error setting up|Setup failed|Unable to set up",
    "entity_unavailable": r"Entity not available|unavailable",
    "database": r"database is locked|OperationalError|sqlite",
    "memory": r"MemoryError|Out of memory",
    "template": r"TemplateError|UndefinedError",
    "yaml": r"YAML|yaml\.scanner|Invalid config",
    "api": r"API error|APIError|api_call",
}

# Known fixable patterns
AUTO_FIX_SUGGESTIONS = {
    "database is locked": "Consider reducing recorder commit_interval or adding exclusions",
    "Entity not available": "Check if device is powered on and connected",
    "Connection refused": "Verify the service is running and port is correct",
    "timed out": "Increase timeout or check network connectivity",
    "Setup failed": "Check integration configuration and credentials",
    "TemplateError": "Review template syntax in Developer Tools > Template",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global running

    running = True
    asyncio.create_task(monitor_logs())

    yield

    running = False


app = FastAPI(
    title="ClaudeHome Health Monitor",
    description="Monitors Home Assistant health and sends alerts",
    version="1.0.0",
    lifespan=lifespan
)


def parse_log_line(line: str) -> Optional[dict]:
    """Parse a Home Assistant log line."""
    # HA log format: YYYY-MM-DD HH:MM:SS.mmm LEVEL (component) [context] message
    pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) (ERROR|WARNING|CRITICAL) \(([^)]+)\)(?: \[[^\]]+\])? (.+)$"

    match = re.match(pattern, line)
    if not match:
        return None

    return {
        "timestamp": match.group(1),
        "level": match.group(2),
        "component": match.group(3),
        "message": match.group(4).strip()
    }


def classify_error(message: str) -> str:
    """Classify an error message into a pattern category."""
    for pattern_name, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, message, re.IGNORECASE):
            return pattern_name
    return "unknown"


def get_fix_suggestion(message: str) -> Optional[str]:
    """Get an auto-fix suggestion for an error."""
    for pattern, suggestion in AUTO_FIX_SUGGESTIONS.items():
        if pattern.lower() in message.lower():
            return suggestion
    return None


async def send_discord_alert(error: dict, pattern: str, suggestion: Optional[str]):
    """Send alert to Discord webhook."""
    if not DISCORD_WEBHOOK:
        return

    embed = {
        "title": f"🚨 Home Assistant {error['level']}",
        "color": 15158332 if error["level"] == "ERROR" else 16776960,
        "fields": [
            {"name": "Component", "value": error["component"], "inline": True},
            {"name": "Pattern", "value": pattern, "inline": True},
            {"name": "Message", "value": error["message"][:500], "inline": False},
        ],
        "timestamp": datetime.now().isoformat()
    }

    if suggestion:
        embed["fields"].append({
            "name": "💡 Suggestion",
            "value": suggestion,
            "inline": False
        })

    payload = {"embeds": [embed]}

    try:
        async with httpx.AsyncClient() as client:
            await client.post(DISCORD_WEBHOOK, json=payload, timeout=10.0)
            logger.info(f"Sent Discord alert for {pattern}")
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")


async def monitor_logs():
    """Monitor log file for errors."""
    global last_position, running

    logger.info(f"Starting log monitoring: {LOG_PATH}")

    while running:
        try:
            if not LOG_PATH.exists():
                logger.warning(f"Log file not found: {LOG_PATH}")
                await asyncio.sleep(CHECK_INTERVAL * 5)
                continue

            file_size = LOG_PATH.stat().st_size
            stats.log_file_size = file_size

            # If file was truncated/rotated, start from beginning
            if file_size < last_position:
                last_position = 0
                logger.info("Log file rotated, starting from beginning")

            with open(LOG_PATH, "r", errors="ignore") as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()

            for line in new_lines:
                parsed = parse_log_line(line.strip())
                if parsed and parsed["level"] in ("ERROR", "CRITICAL"):
                    await process_error(parsed)

            stats.last_check = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Log monitoring error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)


async def process_error(error: dict):
    """Process a detected error."""
    global error_buffer, pattern_counts

    pattern = classify_error(error["message"])
    suggestion = get_fix_suggestion(error["message"])

    # Add to buffer
    entry = ErrorEntry(
        timestamp=error["timestamp"],
        level=error["level"],
        component=error["component"],
        message=error["message"][:500]
    )
    error_buffer.append(entry.model_dump())

    # Track pattern counts
    if pattern not in pattern_counts:
        pattern_counts[pattern] = {"count": 0, "last_seen": None}
    pattern_counts[pattern]["count"] += 1
    pattern_counts[pattern]["last_seen"] = error["timestamp"]

    logger.warning(f"Error detected [{pattern}]: {error['component']} - {error['message'][:100]}")

    # Send Discord alert for critical errors or high-frequency patterns
    if error["level"] == "CRITICAL" or pattern_counts[pattern]["count"] > 5:
        await send_discord_alert(error, pattern, suggestion)


def calculate_stats() -> HealthStats:
    """Calculate current health statistics."""
    now = datetime.now()
    hour_ago = now - timedelta(hours=1)
    day_ago = now - timedelta(hours=24)

    errors_1h = 0
    errors_24h = 0

    for entry in error_buffer:
        try:
            ts = datetime.fromisoformat(entry["timestamp"].replace(" ", "T"))
            if ts > hour_ago:
                errors_1h += 1
            if ts > day_ago:
                errors_24h += 1
        except (ValueError, KeyError):
            pass

    # Get top patterns
    top_patterns = sorted(
        [{"pattern": k, **v} for k, v in pattern_counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:5]

    return HealthStats(
        errors_last_hour=errors_1h,
        errors_last_24h=errors_24h,
        top_error_patterns=top_patterns,
        last_check=stats.last_check,
        log_file_size=stats.log_file_size,
        uptime_seconds=time.time() - start_time
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "log_file": str(LOG_PATH),
        "log_exists": LOG_PATH.exists(),
        "monitoring": running,
        "discord_configured": bool(DISCORD_WEBHOOK)
    }


@app.get("/stats", response_model=HealthStats)
async def get_stats():
    """Get health monitoring statistics."""
    return calculate_stats()


@app.get("/errors/recent")
async def recent_errors(limit: int = 20):
    """Get recent errors."""
    errors = list(error_buffer)[-limit:]
    errors.reverse()
    return {"count": len(errors), "errors": errors}


@app.get("/patterns")
async def get_patterns():
    """Get error pattern statistics."""
    return {
        "patterns": pattern_counts,
        "definitions": list(ERROR_PATTERNS.keys())
    }


@app.post("/patterns/reset")
async def reset_patterns():
    """Reset pattern counts."""
    global pattern_counts
    pattern_counts = {}
    return {"status": "reset"}


@app.post("/test/alert")
async def test_alert():
    """Send a test alert to Discord."""
    if not DISCORD_WEBHOOK:
        raise HTTPException(status_code=400, detail="Discord webhook not configured")

    test_error = {
        "timestamp": datetime.now().isoformat(),
        "level": "WARNING",
        "component": "health_monitor",
        "message": "This is a test alert from ClaudeHome Health Monitor"
    }

    await send_discord_alert(test_error, "test", "This is a test suggestion")
    return {"status": "sent"}


@app.get("/suggestions")
async def get_suggestions():
    """Get all available auto-fix suggestions."""
    return {"suggestions": AUTO_FIX_SUGGESTIONS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
