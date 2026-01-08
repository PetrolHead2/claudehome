"""
Anomaly Detector - Statistical and Semantic Drift Detection
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
from anthropic import AsyncAnthropic
import logging
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClaudeHome Anomaly Detector")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

redis_client = None
anthropic_client = None

Z_SCORE_THRESHOLD = 2.0
MIN_SAMPLES = 10


class Event(BaseModel):
    entity_id: str
    state: str
    old_state: Optional[str] = None
    timestamp: str
    attributes: Dict[str, Any] = {}


class Anomaly(BaseModel):
    anomaly_id: str
    entity_id: str
    anomaly_type: str
    severity: str
    description: str
    current_value: Any
    expected_range: Optional[Dict[str, Any]] = None
    confidence: float
    timestamp: str


@app.on_event("startup")
async def startup():
    global redis_client, anthropic_client
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    logger.info("Anomaly Detector started")


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "anomaly_detector"}


@app.get("/system/health")
async def check_system_health():
    """Check system health metrics"""
    try:
        info = await redis_client.info('memory')
        used_memory_mb = info.get('used_memory', 0) / 1024 / 1024
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "HEALTHY",
            "redis_memory_mb": round(used_memory_mb, 2)
        }
    except Exception as e:
        logger.error(f"System health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)