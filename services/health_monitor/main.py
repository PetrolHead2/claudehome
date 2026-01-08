#!/usr/bin/env python3
"""
Health Monitor Service for ClaudeHome

Monitors Home Assistant logs for errors AND provides SSH operations for file management.
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
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import httpx
import paramiko
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("health_monitor")

# Configuration
HA_CONFIG_DIR = os.environ.get("HA_CONFIG_DIR", "/config")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
LOG_PATH = Path(HA_CONFIG_DIR) / "home-assistant.log"
CHECK_INTERVAL = float(os.environ.get("CHECK_INTERVAL", "10.0"))

# SSH Configuration
SSH_HOST = os.environ.get("SSH_HOST")
SSH_PORT = int(os.environ.get("SSH_PORT", "22"))
SSH_USER = os.environ.get("SSH_USER", "root")
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "/app/.ssh/ha_key")

# Global state
running: bool = False
last_position: int = 0
error_buffer: deque = deque(maxlen=100)
pattern_counts: dict = {}
ssh_client: Optional[paramiko.SSHClient] = None


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
    ssh_connected: bool = False


class SSHCommand(BaseModel):
    """SSH command execution request."""
    command: str
    timeout: int = 30


class FileContent(BaseModel):
    """File content for reading/writing."""
    path: str
    content: Optional[str] = None


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

AUTO_FIX_SUGGESTIONS = {
    "database is locked": "Consider reducing recorder commit_interval or adding exclusions",
    "Entity not available": "Check if device is powered on and connected",
    "Connection refused": "Verify the service is running and port is correct",
    "timed out": "Increase timeout or check network connectivity",
    "Setup failed": "Check integration configuration and credentials",
    "TemplateError": "Review template syntax in Developer Tools > Template",
}


async def connect_ssh():
    """Connect to Home Assistant via SSH."""
    global ssh_client
    
    if not SSH_HOST:
        logger.warning("SSH_HOST not configured, SSH operations disabled")
        return False
    
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Try key-based auth first
        if os.path.exists(SSH_KEY_PATH):
            ssh_client.connect(
                hostname=SSH_HOST,
                port=SSH_PORT,
                username=SSH_USER,
                key_filename=SSH_KEY_PATH,
                timeout=10
            )
            logger.info(f"Connected to {SSH_HOST} via SSH (key auth)")
        else:
            logger.warning(f"SSH key not found at {SSH_KEY_PATH}")
            return False
        
        stats.ssh_connected = True
        return True
        
    except Exception as e:
        logger.error(f"SSH connection failed: {e}")
        stats.ssh_connected = False
        return False


def ensure_ssh_connected():
    """Ensure SSH connection is active."""
    if not ssh_client or not ssh_client.get_transport() or not ssh_client.get_transport().is_active():
        logger.info("SSH connection lost, reconnecting...")
        return asyncio.create_task(connect_ssh())
    return True


async def ssh_execute(command: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute command via SSH."""
    if not ensure_ssh_connected():
        raise HTTPException(status_code=503, detail="SSH not connected")
    
    try:
        stdin, stdout, stderr = ssh_client.exec_command(command, timeout=timeout)
        
        exit_code = stdout.channel.recv_exit_status()
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        return {
            "exit_code": exit_code,
            "output": output,
            "error": error,
            "success": exit_code == 0
        }
        
    except Exception as e:
        logger.error(f"SSH execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def ssh_read_file(path: str) -> str:
    """Read file content via SSH."""
    result = await ssh_execute(f"cat {path}")
    if not result["success"]:
        raise HTTPException(status_code=404, detail=f"File not found or cannot read: {path}")
    return result["output"]


async def ssh_write_file(path: str, content: str) -> bool:
    """Write file content via SSH."""
    try:
        # Use heredoc to write file
        command = f"""cat > {path} << 'EOF'
{content}
EOF"""
        result = await ssh_execute(command)
        return result["success"]
    except Exception as e:
        logger.error(f"Failed to write file {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def ssh_append_file(path: str, content: str) -> bool:
    """Append content to file via SSH."""
    try:
        command = f"""cat >> {path} << 'EOF'
{content}
EOF"""
        result = await ssh_execute(command)
        return result["success"]
    except Exception as e:
        logger.error(f"Failed to append to file {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def ssh_file_exists(path: str) -> bool:
    """Check if file exists via SSH."""
    result = await ssh_execute(f"test -f {path} && echo 'exists' || echo 'not_found'")
    return "exists" in result["output"]


async def ssh_list_directory(path: str) -> List[str]:
    """List directory contents via SSH."""
    result = await ssh_execute(f"ls -1 {path}")
    if not result["success"]:
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
    return [line.strip() for line in result["output"].split('\n') if line.strip()]


async def git_commit(message: str, files: List[str] = None) -> bool:
    """Commit changes to git."""
    try:
        if files:
            # Add specific files
            for file in files:
                await ssh_execute(f"cd {HA_CONFIG_DIR} && git add {file}")
        else:
            # Add all changes
            await ssh_execute(f"cd {HA_CONFIG_DIR} && git add .")
        
        # Commit
        result = await ssh_execute(f"cd {HA_CONFIG_DIR} && git commit -m '{message}'")
        return result["success"]
        
    except Exception as e:
        logger.error(f"Git commit failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global running

    # Connect SSH
    await connect_ssh()
    
    # Start log monitoring
    running = True
    asyncio.create_task(monitor_logs())

    yield

    # Cleanup
    running = False
    if ssh_client:
        ssh_client.close()


app = FastAPI(
    title="ClaudeHome Health Monitor",
    description="Monitors Home Assistant health and provides SSH operations",
    version="2.0.0",
    lifespan=lifespan
)


# ===== EXISTING LOG MONITORING FUNCTIONS =====

def parse_log_line(line: str) -> Optional[dict]:
    """Parse a Home Assistant log line."""
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
    """Classify an error message."""
    for pattern_name, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, message, re.IGNORECASE):
            return pattern_name
    return "unknown"


def get_fix_suggestion(message: str) -> Optional[str]:
    """Get auto-fix suggestion."""
    for pattern, suggestion in AUTO_FIX_SUGGESTIONS.items():
        if pattern.lower() in message.lower():
            return suggestion
    return None


async def send_discord_alert(error: dict, pattern: str, suggestion: Optional[str]):
    """Send alert to Discord."""
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
        embed["fields"].append({"name": "💡 Suggestion", "value": suggestion, "inline": False})
    payload = {"embeds": [embed]}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(DISCORD_WEBHOOK, json=payload, timeout=10.0)
    except Exception as e:
        logger.error(f"Discord alert failed: {e}")


async def monitor_logs():
    """Monitor log file."""
    global last_position, running
    logger.info(f"Starting log monitoring: {LOG_PATH}")
    while running:
        try:
            if not LOG_PATH.exists():
                await asyncio.sleep(CHECK_INTERVAL * 5)
                continue
            file_size = LOG_PATH.stat().st_size
            stats.log_file_size = file_size
            if file_size < last_position:
                last_position = 0
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
    """Process detected error."""
    global error_buffer, pattern_counts
    pattern = classify_error(error["message"])
    suggestion = get_fix_suggestion(error["message"])
    entry = ErrorEntry(
        timestamp=error["timestamp"],
        level=error["level"],
        component=error["component"],
        message=error["message"][:500]
    )
    error_buffer.append(entry.model_dump())
    if pattern not in pattern_counts:
        pattern_counts[pattern] = {"count": 0, "last_seen": None}
    pattern_counts[pattern]["count"] += 1
    pattern_counts[pattern]["last_seen"] = error["timestamp"]
    if error["level"] == "CRITICAL" or pattern_counts[pattern]["count"] > 5:
        await send_discord_alert(error, pattern, suggestion)


def calculate_stats() -> HealthStats:
    """Calculate statistics."""
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
        uptime_seconds=time.time() - start_time,
        ssh_connected=stats.ssh_connected
    )


# ===== API ENDPOINTS =====

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "log_file": str(LOG_PATH),
        "log_exists": LOG_PATH.exists(),
        "monitoring": running,
        "ssh_connected": stats.ssh_connected,
        "discord_configured": bool(DISCORD_WEBHOOK)
    }


@app.get("/stats", response_model=HealthStats)
async def get_stats():
    """Get statistics."""
    return calculate_stats()


@app.get("/errors/recent")
async def recent_errors(limit: int = 20):
    """Get recent errors."""
    errors = list(error_buffer)[-limit:]
    errors.reverse()
    return {"count": len(errors), "errors": errors}


@app.get("/patterns")
async def get_patterns():
    """Get error patterns."""
    return {"patterns": pattern_counts, "definitions": list(ERROR_PATTERNS.keys())}


@app.post("/patterns/reset")
async def reset_patterns():
    """Reset patterns."""
    global pattern_counts
    pattern_counts = {}
    return {"status": "reset"}


# ===== NEW SSH ENDPOINTS =====

@app.post("/ssh/execute")
async def execute_ssh_command(cmd: SSHCommand):
    """Execute SSH command."""
    result = await ssh_execute(cmd.command, cmd.timeout)
    return result


@app.post("/ssh/file/read")
async def read_file(file: FileContent):
    """Read file via SSH."""
    content = await ssh_read_file(file.path)
    return {"path": file.path, "content": content}


@app.post("/ssh/file/write")
async def write_file(file: FileContent):
    """Write file via SSH."""
    if not file.content:
        raise HTTPException(status_code=400, detail="Content required")
    success = await ssh_write_file(file.path, file.content)
    return {"path": file.path, "success": success}


@app.post("/ssh/file/append")
async def append_file(file: FileContent):
    """Append to file via SSH."""
    if not file.content:
        raise HTTPException(status_code=400, detail="Content required")
    success = await ssh_append_file(file.path, file.content)
    return {"path": file.path, "success": success}


@app.get("/ssh/file/exists/{path:path}")
async def check_file_exists(path: str):
    """Check if file exists."""
    exists = await ssh_file_exists(path)
    return {"path": path, "exists": exists}


@app.get("/ssh/directory/{path:path}")
async def list_dir(path: str):
    """List directory."""
    files = await ssh_list_directory(path)
    return {"path": path, "files": files}


@app.post("/git/commit")
async def commit_changes(message: str):
    """Commit changes to git."""
    try:
        # Escape single quotes in the message for shell safety
        safe_message = message.replace("'", "'\\''")
        
        # Build the full git command
        command = f"cd {HA_CONFIG_DIR} && git add . && git commit -m '{safe_message}'"
        
        logger.info(f"Executing git commit with message: {message}")
        result = await ssh_execute(command)
        
        logger.info(f"Git result: exit_code={result['exit_code']}, success={result['success']}")
        if result.get('output'):
            logger.info(f"Git output: {result['output']}")
        if result.get('error'):
            logger.error(f"Git error: {result['error']}")
        
        # Git commit returns exit code 0 on success
        success = result.get("exit_code") == 0 and result.get("success", False)
        
        return {
            "success": success,
            "message": message,
            "output": result.get("output", ""),
            "error": result.get("error", "")
        }
        
    except Exception as e:
        logger.error(f"Git commit exception: {e}", exc_info=True)
        return {
            "success": False,
            "message": message,
            "error": str(e),
            "output": ""
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)