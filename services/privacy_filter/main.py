#!/usr/bin/env python3
"""
Privacy Filter Service for ClaudeHome

FastAPI service that anonymizes and de-anonymizes Home Assistant entity data
before sending to Claude API and after receiving responses.
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging to never log real entity IDs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("privacy_filter")

app = FastAPI(
    title="ClaudeHome Privacy Filter",
    description="Anonymizes sensitive Home Assistant data before external processing",
    version="1.0.0"
)

# Path to entity mappings
MAPPINGS_PATH = Path("/opt/claudehome/.claude/skills/entity_mappings.json")


class AnonymizeRequest(BaseModel):
    """Request to anonymize data."""
    data: Any
    context: Optional[str] = None


class AnonymizeResponse(BaseModel):
    """Response with anonymized data."""
    data: Any
    mappings_used: int


class DeAnonymizeRequest(BaseModel):
    """Request to de-anonymize data."""
    data: Any


class DeAnonymizeResponse(BaseModel):
    """Response with de-anonymized data."""
    data: Any


@lru_cache(maxsize=1)
def load_mappings() -> dict:
    """Load entity mappings from JSON file."""
    if not MAPPINGS_PATH.exists():
        logger.warning("Mappings file not found, using empty mappings")
        return {"mappings": {}, "sensitive_patterns": [], "never_log": []}

    with open(MAPPINGS_PATH) as f:
        return json.load(f)


def get_flat_mappings() -> dict[str, str]:
    """Get flattened forward mappings (real -> anonymous)."""
    mappings = load_mappings()
    flat = {}

    for category, mapping in mappings.get("mappings", {}).items():
        if isinstance(mapping, dict):
            flat.update(mapping)

    return flat


def get_reverse_mappings() -> dict[str, str]:
    """Get reverse mappings (anonymous -> real)."""
    forward = get_flat_mappings()
    return {v: k for k, v in forward.items()}


def anonymize_string(text: str) -> tuple[str, int]:
    """
    Anonymize a string by replacing known entities and patterns.
    Returns (anonymized_text, count_of_replacements).
    """
    if not isinstance(text, str):
        return text, 0

    mappings = get_flat_mappings()
    config = load_mappings()
    count = 0

    result = text

    # Replace known entity mappings
    for real, anonymous in mappings.items():
        if real in result:
            result = result.replace(real, anonymous)
            count += 1

    # Replace location strings
    location_strings = config.get("mappings", {}).get("location_strings", {})
    for real, anonymous in location_strings.items():
        if real in result:
            result = result.replace(real, anonymous)
            count += 1

    # Redact sensitive patterns
    for pattern in config.get("sensitive_patterns", []):
        matches = re.findall(pattern, result, re.IGNORECASE)
        for match in matches:
            result = result.replace(match, "[REDACTED]")
            count += 1

    # Never log these values
    for never in config.get("never_log", []):
        if never in result:
            result = result.replace(never, "[SENSITIVE]")
            count += 1

    return result, count


def anonymize_value(value: Any) -> tuple[Any, int]:
    """Recursively anonymize a value (dict, list, or string)."""
    total_count = 0

    if isinstance(value, str):
        return anonymize_string(value)

    elif isinstance(value, dict):
        result = {}
        for k, v in value.items():
            anon_key, key_count = anonymize_string(k)
            anon_val, val_count = anonymize_value(v)
            result[anon_key] = anon_val
            total_count += key_count + val_count
        return result, total_count

    elif isinstance(value, list):
        result = []
        for item in value:
            anon_item, item_count = anonymize_value(item)
            result.append(anon_item)
            total_count += item_count
        return result, total_count

    else:
        return value, 0


def deanonymize_string(text: str) -> str:
    """De-anonymize a string by reversing known mappings."""
    if not isinstance(text, str):
        return text

    mappings = get_reverse_mappings()
    result = text

    for anonymous, real in mappings.items():
        if anonymous in result:
            result = result.replace(anonymous, real)

    return result


def deanonymize_value(value: Any) -> Any:
    """Recursively de-anonymize a value."""
    if isinstance(value, str):
        return deanonymize_string(value)

    elif isinstance(value, dict):
        result = {}
        for k, v in value.items():
            result[deanonymize_string(k)] = deanonymize_value(v)
        return result

    elif isinstance(value, list):
        return [deanonymize_value(item) for item in value]

    else:
        return value


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    mappings = load_mappings()
    mapping_count = sum(
        len(m) for m in mappings.get("mappings", {}).values()
        if isinstance(m, dict)
    )
    return {
        "status": "healthy",
        "mappings_loaded": mapping_count,
        "version": "1.0.0"
    }


@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize(request: AnonymizeRequest):
    """
    Anonymize data before sending to Claude API.

    Replaces real entity IDs, names, and locations with anonymous equivalents.
    """
    try:
        anonymized, count = anonymize_value(request.data)
        logger.info(f"Anonymized {count} items (context: {request.context or 'none'})")
        return AnonymizeResponse(data=anonymized, mappings_used=count)
    except Exception as e:
        logger.error(f"Anonymization error: {e}")
        raise HTTPException(status_code=500, detail="Anonymization failed")


@app.post("/deanonymize", response_model=DeAnonymizeResponse)
async def deanonymize(request: DeAnonymizeRequest):
    """
    De-anonymize data received from Claude API.

    Converts anonymous entity IDs back to real ones for execution.
    """
    try:
        deanonymized = deanonymize_value(request.data)
        logger.info("De-anonymized response data")
        return DeAnonymizeResponse(data=deanonymized)
    except Exception as e:
        logger.error(f"De-anonymization error: {e}")
        raise HTTPException(status_code=500, detail="De-anonymization failed")


@app.post("/reload")
async def reload_mappings():
    """Reload mappings from disk."""
    load_mappings.cache_clear()
    mappings = load_mappings()
    mapping_count = sum(
        len(m) for m in mappings.get("mappings", {}).values()
        if isinstance(m, dict)
    )
    logger.info(f"Reloaded {mapping_count} mappings")
    return {"status": "reloaded", "mappings_count": mapping_count}


@app.get("/mappings/stats")
async def mapping_stats():
    """Get statistics about loaded mappings."""
    mappings = load_mappings()
    stats = {}

    for category, mapping in mappings.get("mappings", {}).items():
        if isinstance(mapping, dict):
            stats[category] = len(mapping)

    return {
        "categories": stats,
        "total": sum(stats.values()),
        "sensitive_patterns": len(mappings.get("sensitive_patterns", [])),
        "never_log_items": len(mappings.get("never_log", []))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
