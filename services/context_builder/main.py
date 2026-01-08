#!/usr/bin/env python3
"""
Context Builder Service for ClaudeHome

Discovers and maintains complete knowledge of Home Assistant installation:
- Entity registry (all devices, sensors, etc.)
- Existing automations
- Installed integrations
- Configuration details
- Device/Area mappings

Auto-refreshes every 15 minutes to detect new devices.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from contextlib import asynccontextmanager

import redis.asyncio as redis
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("context_builder")

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
HEALTH_MONITOR_URL = os.environ.get("HEALTH_MONITOR_URL", "http://health_monitor:8003")
HA_CONFIG_DIR = os.environ.get("HA_CONFIG_DIR", "/media/pi/NextCloud/homeassistant")
REFRESH_INTERVAL = int(os.environ.get("CONTEXT_REFRESH_INTERVAL", "900"))  # 15 minutes

# Global state
redis_client: Optional[redis.Redis] = None
running: bool = False
entity_cache: Dict[str, Dict] = {}
automation_cache: List[Dict] = []
integration_cache: List[str] = []
area_cache: Dict[str, str] = {}
device_cache: Dict[str, Dict] = {}
last_refresh: Optional[datetime] = None
bootstrap_complete: bool = False


class ContextStats(BaseModel):
    """Context builder statistics"""
    entities_total: int = 0
    entities_by_domain: Dict[str, int] = {}
    automations_total: int = 0
    integrations_total: int = 0
    areas_total: int = 0
    devices_total: int = 0
    last_refresh: Optional[str] = None
    bootstrap_complete: bool = False
    auto_refresh_enabled: bool = True


async def ssh_execute(command: str) -> Dict[str, Any]:
    """Execute command via health_monitor SSH"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HEALTH_MONITOR_URL}/ssh/execute",
                json={"command": command},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"SSH execute failed: {resp.status}")
                return await resp.json()
    except Exception as e:
        logger.error(f"SSH execute error: {e}")
        raise


async def ssh_read_file(path: str) -> str:
    """Read file via health_monitor SSH"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HEALTH_MONITOR_URL}/ssh/file/read",
                json={"path": path},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"SSH read failed: {resp.status}")
                result = await resp.json()
                return result.get("content", "")
    except Exception as e:
        logger.error(f"SSH read error: {e}")
        raise


async def load_entity_registry():
    """Load and parse entity registry"""
    global entity_cache, last_refresh
    
    logger.info("Loading entity registry...")
    
    try:
        # Read entity registry
        content = await ssh_read_file(f"{HA_CONFIG_DIR}/.storage/core.entity_registry")
        registry = json.loads(content)
        
        entities_data = registry.get("data", {}).get("entities", [])
        old_count = len(entity_cache)
        new_cache = {}
        
        for entity in entities_data:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue
            
            # Build entity metadata
            metadata = {
                "entity_id": entity_id,
                "platform": entity.get("platform"),
                "device_id": entity.get("device_id"),
                "area_id": entity.get("area_id"),
                "device_class": entity.get("device_class"),
                "original_name": entity.get("original_name"),
                "capabilities": entity.get("capabilities", {}),
                "disabled_by": entity.get("disabled_by"),
                "hidden_by": entity.get("hidden_by"),
                "unique_id": entity.get("unique_id")
            }
            
            new_cache[entity_id] = metadata
            
            # Store in Redis with 24h TTL
            await redis_client.setex(
                f"claudehome:entities:{entity_id}",
                86400,
                json.dumps(metadata)
            )
        
        # Detect changes
        if old_count > 0:
            old_entities = set(entity_cache.keys())
            new_entities = set(new_cache.keys())
            
            added = new_entities - old_entities
            removed = old_entities - new_entities
            
            if added:
                logger.info(f"🆕 New entities detected: {len(added)}")
                for eid in list(added)[:5]:
                    logger.info(f"  + {eid}")
                if len(added) > 5:
                    logger.info(f"  ... and {len(added)-5} more")
            
            if removed:
                logger.info(f"🗑️ Removed entities: {len(removed)}")
        
        entity_cache = new_cache
        
        # Store domain counts
        domain_counts = {}
        for entity_id in entity_cache.keys():
            domain = entity_id.split(".")[0]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        await redis_client.setex(
            "claudehome:context:domain_counts",
            86400,
            json.dumps(domain_counts)
        )
        
        logger.info(f"✅ Loaded {len(entity_cache)} entities across {len(domain_counts)} domains")
        last_refresh = datetime.utcnow()
        
        return len(entity_cache)
        
    except Exception as e:
        logger.error(f"Failed to load entity registry: {e}")
        return 0


async def load_area_registry():
    """Load and parse area registry"""
    global area_cache
    
    logger.info("Loading area registry...")
    
    try:
        content = await ssh_read_file(f"{HA_CONFIG_DIR}/.storage/core.area_registry")
        registry = json.loads(content)
        
        areas_data = registry.get("data", {}).get("areas", [])
        area_cache = {}
        
        for area in areas_data:
            area_id = area.get("id")
            area_name = area.get("name")
            if area_id and area_name:
                area_cache[area_id] = area_name
                
                await redis_client.setex(
                    f"claudehome:areas:{area_id}",
                    86400,
                    json.dumps({"id": area_id, "name": area_name})
                )
        
        logger.info(f"✅ Loaded {len(area_cache)} areas")
        return len(area_cache)
        
    except Exception as e:
        logger.error(f"Failed to load area registry: {e}")
        return 0


async def load_device_registry():
    """Load and parse device registry"""
    global device_cache
    
    logger.info("Loading device registry...")
    
    try:
        content = await ssh_read_file(f"{HA_CONFIG_DIR}/.storage/core.device_registry")
        registry = json.loads(content)
        
        devices_data = registry.get("data", {}).get("devices", [])
        device_cache = {}
        
        for device in devices_data:
            device_id = device.get("id")
            if not device_id:
                continue
            
            device_info = {
                "id": device_id,
                "name": device.get("name"),
                "manufacturer": device.get("manufacturer"),
                "model": device.get("model"),
                "area_id": device.get("area_id"),
                "config_entries": device.get("config_entries", []),
                "identifiers": device.get("identifiers", [])
            }
            
            device_cache[device_id] = device_info
            
            await redis_client.setex(
                f"claudehome:devices:{device_id}",
                86400,
                json.dumps(device_info)
            )
        
        logger.info(f"✅ Loaded {len(device_cache)} devices")
        return len(device_cache)
        
    except Exception as e:
        logger.error(f"Failed to load device registry: {e}")
        return 0


async def load_automations():
    """Load existing automations"""
    global automation_cache
    
    logger.info("Loading automations...")
    
    try:
        # Read main automations.yaml
        content = await ssh_read_file(f"{HA_CONFIG_DIR}/automations.yaml")
        
        import yaml
        automations = yaml.safe_load(content) or []
        
        if not isinstance(automations, list):
            automations = [automations]
        
        automation_cache = []
        for auto in automations:
            if isinstance(auto, dict):
                automation_cache.append({
                    "id": auto.get("id", "unknown"),
                    "alias": auto.get("alias", "Unknown"),
                    "description": auto.get("description", ""),
                    "trigger": str(auto.get("trigger", [])),
                    "action": str(auto.get("action", []))
                })
        
        # Store in Redis
        await redis_client.setex(
            "claudehome:context:automations",
            86400,
            json.dumps(automation_cache)
        )
        
        logger.info(f"✅ Loaded {len(automation_cache)} automations")
        return len(automation_cache)
        
    except Exception as e:
        logger.error(f"Failed to load automations: {e}")
        return 0


async def load_integrations():
    """Load installed integrations"""
    global integration_cache
    
    logger.info("Loading integrations...")
    
    try:
        content = await ssh_read_file(f"{HA_CONFIG_DIR}/.storage/core.config_entries")
        registry = json.loads(content)
        
        entries = registry.get("data", {}).get("entries", [])
        integration_cache = []
        
        for entry in entries:
            domain = entry.get("domain")
            if domain:
                integration_cache.append({
                    "domain": domain,
                    "title": entry.get("title", domain),
                    "entry_id": entry.get("entry_id")
                })
        
        # Store in Redis
        await redis_client.setex(
            "claudehome:context:integrations",
            86400,
            json.dumps(integration_cache)
        )
        
        logger.info(f"✅ Loaded {len(integration_cache)} integrations")
        return len(integration_cache)
        
    except Exception as e:
        logger.error(f"Failed to load integrations: {e}")
        return 0


async def bootstrap():
    """Initial bootstrap - load all context"""
    global bootstrap_complete
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Context Builder Bootstrap")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load all registries
        entities_count = await load_entity_registry()
        areas_count = await load_area_registry()
        devices_count = await load_device_registry()
        automations_count = await load_automations()
        integrations_count = await load_integrations()
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("✅ Bootstrap Complete!")
        logger.info(f"   Entities: {entities_count}")
        logger.info(f"   Areas: {areas_count}")
        logger.info(f"   Devices: {devices_count}")
        logger.info(f"   Automations: {automations_count}")
        logger.info(f"   Integrations: {integrations_count}")
        logger.info(f"   Time: {elapsed:.1f}s")
        logger.info("=" * 60)
        
        bootstrap_complete = True
        
        # Store bootstrap status
        await redis_client.setex(
            "claudehome:context:bootstrap_complete",
            86400,
            json.dumps({
                "complete": True,
                "timestamp": datetime.utcnow().isoformat(),
                "entities": entities_count,
                "areas": areas_count,
                "devices": devices_count,
                "automations": automations_count,
                "integrations": integrations_count
            })
        )
        
    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        bootstrap_complete = False


async def auto_refresh():
    """Auto-refresh context periodically"""
    global running
    
    logger.info(f"Auto-refresh enabled (every {REFRESH_INTERVAL}s)")
    
    while running:
        await asyncio.sleep(REFRESH_INTERVAL)
        
        logger.info("🔄 Auto-refreshing context...")
        
        try:
            await load_entity_registry()
            await load_area_registry()
            await load_device_registry()
            await load_automations()
            await load_integrations()
            
            logger.info("✅ Auto-refresh complete")
            
        except Exception as e:
            logger.error(f"Auto-refresh failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global redis_client, running
    
    # Connect to Redis
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    logger.info(f"Connected to Redis at {REDIS_URL}")
    
    # Bootstrap
    await bootstrap()
    
    # Start auto-refresh
    running = True
    asyncio.create_task(auto_refresh())
    
    yield
    
    # Shutdown
    running = False
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="ClaudeHome Context Builder",
    description="Discovers and maintains knowledge of Home Assistant installation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "context_builder",
        "bootstrap_complete": bootstrap_complete,
        "last_refresh": last_refresh.isoformat() if last_refresh else None
    }


@app.get("/stats")
async def get_stats() -> ContextStats:
    """Get context statistics"""
    domain_counts = {}
    for entity_id in entity_cache.keys():
        domain = entity_id.split(".")[0]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return ContextStats(
        entities_total=len(entity_cache),
        entities_by_domain=domain_counts,
        automations_total=len(automation_cache),
        integrations_total=len(integration_cache),
        areas_total=len(area_cache),
        devices_total=len(device_cache),
        last_refresh=last_refresh.isoformat() if last_refresh else None,
        bootstrap_complete=bootstrap_complete
    )


@app.get("/entity/{entity_id}")
async def get_entity(entity_id: str):
    """Get entity metadata"""
    if entity_id not in entity_cache:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return entity_cache[entity_id]


@app.get("/entities/domain/{domain}")
async def get_entities_by_domain(domain: str):
    """Get all entities in a domain"""
    entities = [
        entity_cache[eid] for eid in entity_cache.keys()
        if eid.startswith(f"{domain}.")
    ]
    return {"domain": domain, "count": len(entities), "entities": entities}


@app.get("/entities/area/{area_id}")
async def get_entities_by_area(area_id: str):
    """Get all entities in an area"""
    entities = [
        entity_cache[eid] for eid, data in entity_cache.items()
        if data.get("area_id") == area_id
    ]
    return {"area_id": area_id, "count": len(entities), "entities": entities}


@app.get("/areas")
async def get_areas():
    """Get all areas"""
    return {
        "count": len(area_cache),
        "areas": [{"id": aid, "name": name} for aid, name in area_cache.items()]
    }


@app.get("/devices")
async def get_devices():
    """Get all devices"""
    return {"count": len(device_cache), "devices": list(device_cache.values())}


@app.get("/automations")
async def get_automations():
    """Get existing automations"""
    return {"count": len(automation_cache), "automations": automation_cache}


@app.get("/integrations")
async def get_integrations():
    """Get installed integrations"""
    return {"count": len(integration_cache), "integrations": integration_cache}


@app.post("/refresh")
async def trigger_refresh():
    """Manually trigger context refresh"""
    logger.info("Manual refresh triggered")
    
    try:
        await load_entity_registry()
        await load_area_registry()
        await load_device_registry()
        await load_automations()
        await load_integrations()
        
        return {"status": "refreshed", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)