#!/usr/bin/env python3
"""
Automation Deployer Service for ClaudeHome

Deploys approved automation suggestions to Home Assistant via SSH.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional
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
logger = logging.getLogger("automation_deployer")

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
HEALTH_MONITOR_URL = os.environ.get("HEALTH_MONITOR_URL", "http://health_monitor:8003")
HOME_ASSISTANT_URL = os.environ.get("HOME_ASSISTANT_URL")
HOME_ASSISTANT_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN")
HA_CONFIG_DIR = os.environ.get("HA_CONFIG_DIR", "/media/pi/NextCloud/homeassistant")
AUTOMATIONS_DIR = f"{HA_CONFIG_DIR}/automations/claudehome"

# Global state
redis_client: Optional[redis.Redis] = None
running: bool = False
deployments_count: int = 0


class DeploymentRequest(BaseModel):
    """Request to deploy an automation."""
    suggestion_id: str
    automation_yaml: str
    title: str
    description: str


class DeploymentResult(BaseModel):
    """Result of a deployment."""
    suggestion_id: str
    success: bool
    automation_id: str = None
    file_path: str = None
    error: str = None
    git_committed: bool = False
    ha_reloaded: bool = False


async def ssh_execute(command: str) -> Dict[str, Any]:
    """Execute command via health_monitor SSH."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HEALTH_MONITOR_URL}/ssh/execute",
                json={"command": command},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"SSH execute failed: {resp.status}")
                return await resp.json()
    except Exception as e:
        logger.error(f"SSH execute error: {e}")
        raise


async def ssh_write_file(path: str, content: str) -> bool:
    """Write file via health_monitor SSH."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HEALTH_MONITOR_URL}/ssh/file/write",
                json={"path": path, "content": content},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"SSH write failed: {resp.status}")
                result = await resp.json()
                return result.get("success", False)
    except Exception as e:
        logger.error(f"SSH write error: {e}")
        raise


async def git_commit(message: str) -> bool:
    """Commit changes via health_monitor."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HEALTH_MONITOR_URL}/git/commit",
                params={"message": message},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Git commit failed: {resp.status}")
                result = await resp.json()
                return result.get("success", False)
    except Exception as e:
        logger.error(f"Git commit error: {e}")
        return False


async def reload_ha_automations() -> bool:
    """Reload Home Assistant automations via API."""
    if not HOME_ASSISTANT_URL or not HOME_ASSISTANT_TOKEN:
        logger.warning("HA URL/Token not configured, skipping reload")
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HOME_ASSISTANT_URL}/api/services/automation/reload",
                headers={"Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                return resp.status == 200
    except Exception as e:
        logger.error(f"HA reload error: {e}")
        return False


async def ensure_automations_directory() -> bool:
    """Ensure the claudehome automations directory exists."""
    try:
        # Check if directory exists
        result = await ssh_execute(f"test -d {AUTOMATIONS_DIR} && echo 'exists' || echo 'not_found'")
        
        if "not_found" in result.get("output", ""):
            # Create directory
            logger.info(f"Creating automations directory: {AUTOMATIONS_DIR}")
            create_result = await ssh_execute(f"mkdir -p {AUTOMATIONS_DIR}")
            if not create_result.get("success"):
                raise Exception("Failed to create automations directory")
        
        return True
    except Exception as e:
        logger.error(f"Failed to ensure directory: {e}")
        return False


def generate_automation_id(title: str) -> str:
    """Generate a unique automation ID."""
    # Clean title for use in ID
    clean_title = title.lower().replace(" ", "_").replace("-", "_")
    # Remove special characters
    clean_title = "".join(c for c in clean_title if c.isalnum() or c == "_")
    # Add timestamp for uniqueness
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"claudehome_{clean_title}_{timestamp}"


def generate_filename(automation_id: str) -> str:
    """Generate filename for automation."""
    return f"{automation_id}.yaml"


async def deploy_automation(request: DeploymentRequest) -> DeploymentResult:
    """Deploy an automation to Home Assistant."""
    global deployments_count
    
    result = DeploymentResult(
        suggestion_id=request.suggestion_id,
        success=False
    )
    
    try:
        # Ensure directory exists
        if not await ensure_automations_directory():
            result.error = "Failed to create automations directory"
            return result
        
        # Generate automation ID and filename
        automation_id = generate_automation_id(request.title)
        filename = generate_filename(automation_id)
        file_path = f"{AUTOMATIONS_DIR}/{filename}"
        
        result.automation_id = automation_id
        result.file_path = file_path
        
        # Parse and modify YAML to ensure it has an ID
        yaml_content = request.automation_yaml.strip()
        
        # If YAML is a list (starts with -), convert to single automation
        if yaml_content.startswith("-"):
            yaml_content = yaml_content[1:].strip()
        
        # Ensure it has an id field
        if "id:" not in yaml_content:
            # Add id at the beginning
            yaml_content = f"id: {automation_id}\n{yaml_content}"
        
        # Add alias if not present
        if "alias:" not in yaml_content:
            yaml_content = f"alias: '{request.title}'\n{yaml_content}"
        
        # Add description if not present
        if "description:" not in yaml_content:
            yaml_content = f"description: '{request.description}'\n{yaml_content}"
        
        # Wrap in list format for Home Assistant
        full_yaml = f"# ClaudeHome Generated Automation\n# Deployed: {datetime.utcnow().isoformat()}\n# Suggestion ID: {request.suggestion_id}\n\n- {yaml_content}\n"
        
        logger.info(f"Deploying automation to: {file_path}")
        
        # Write file via SSH
        write_success = await ssh_write_file(file_path, full_yaml)
        if not write_success:
            result.error = "Failed to write automation file"
            return result
        
        logger.info(f"Automation file written successfully")
        
        # Commit to git
        git_message = f"ClaudeHome: Deploy automation '{request.title}' (ID: {automation_id})"
        git_success = await git_commit(git_message)
        result.git_committed = git_success
        
        if git_success:
            logger.info("Changes committed to git")
        else:
            logger.warning("Git commit failed (non-fatal)")
        
        # Reload Home Assistant automations
        reload_success = await reload_ha_automations()
        result.ha_reloaded = reload_success
        
        if reload_success:
            logger.info("Home Assistant automations reloaded")
        else:
            logger.warning("HA reload failed (non-fatal)")
        
        # Mark as successful
        result.success = True
        deployments_count += 1
        
        logger.info(f"✅ Automation deployed successfully: {automation_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        result.error = str(e)
        return result


async def listen_for_approvals():
    """Listen for approved suggestions on Redis pub/sub."""
    global running
    
    logger.info("Starting approval listener...")
    
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("claudehome:proactive:suggestion_approved")
    
    logger.info("Subscribed to suggestion approvals")
    
    while running:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            
            if message and message["type"] == "message":
                data = json.loads(message["data"])
                suggestion_id = data.get("suggestion_id")
                
                logger.info(f"Received approval for suggestion: {suggestion_id}")
                
                # Fetch suggestion details from Redis
                suggestion_key = f"claudehome:suggestions:approved:{suggestion_id}"
                suggestion_data = await redis_client.get(suggestion_key)
                
                if not suggestion_data:
                    logger.error(f"Suggestion not found: {suggestion_id}")
                    continue
                
                suggestion = json.loads(suggestion_data)
                
                # Create deployment request
                deploy_request = DeploymentRequest(
                    suggestion_id=suggestion_id,
                    automation_yaml=suggestion["automation_yaml"],
                    title=suggestion["title"],
                    description=suggestion["description"]
                )
                
                # Deploy the automation
                result = await deploy_automation(deploy_request)
                
                # Store deployment result
                result_key = f"claudehome:deployments:{suggestion_id}"
                await redis_client.setex(result_key, 86400 * 30, json.dumps(result.dict()))
                
                # Publish result
                if result.success:
                    await redis_client.publish(
                        "claudehome:deployment:success",
                        json.dumps({"suggestion_id": suggestion_id, "automation_id": result.automation_id})
                    )
                else:
                    await redis_client.publish(
                        "claudehome:deployment:failure",
                        json.dumps({"suggestion_id": suggestion_id, "error": result.error})
                    )
            
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Approval listener error: {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global redis_client, running
    
    # Connect to Redis
    try:
        redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_URL}")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise
    
    # Start background listener
    running = True
    asyncio.create_task(listen_for_approvals())
    
    yield
    
    # Shutdown
    running = False
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="ClaudeHome Automation Deployer",
    description="Deploys approved automation suggestions to Home Assistant",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "service": "automation_deployer",
        "deployments_total": deployments_count,
        "ha_config_dir": HA_CONFIG_DIR,
        "automations_dir": AUTOMATIONS_DIR
    }


@app.post("/deploy", response_model=DeploymentResult)
async def deploy_automation_endpoint(request: DeploymentRequest):
    """Manually deploy an automation (for testing)."""
    return await deploy_automation(request)


@app.get("/deployments/{suggestion_id}")
async def get_deployment_result(suggestion_id: str):
    """Get deployment result for a suggestion."""
    result_key = f"claudehome:deployments:{suggestion_id}"
    data = await redis_client.get(result_key)
    
    if not data:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    return json.loads(data)


@app.get("/deployments")
async def list_deployments():
    """List recent deployments."""
    keys = []
    async for key in redis_client.scan_iter("claudehome:deployments:*"):
        keys.append(key)
    
    deployments = []
    for key in keys[:20]:  # Last 20
        data = await redis_client.get(key)
        if data:
            deployments.append(json.loads(data))
    
    return {"count": len(deployments), "deployments": deployments}


@app.get("/stats")
async def get_stats():
    """Get deployment statistics."""
    return {
        "deployments_total": deployments_count,
        "uptime_seconds": time.time() - start_time
    }


start_time = time.time()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)