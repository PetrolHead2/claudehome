"""
Shared Home Assistant API Client
Handles HTTPS with self-signed certificates
"""

import ssl
import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def create_ha_ssl_context():
    """Create SSL context that accepts self-signed certificates"""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


async def call_ha_service(
    base_url: str,
    token: str,
    domain: str,
    service: str,
    service_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Call Home Assistant service
    
    Args:
        base_url: HA URL (e.g., https://192.168.50.250:8123)
        token: Long-lived access token
        domain: Service domain (e.g., 'automation')
        service: Service name (e.g., 'reload')
        service_data: Optional service data
    
    Returns:
        True if successful
    """
    try:
        ssl_context = create_ha_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            url = f"{base_url}/api/services/{domain}/{service}"
            
            async with session.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json=service_data or {},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Called {domain}.{service} successfully")
                    return True
                else:
                    error_text = await resp.text()
                    logger.error(f"HA service call failed: {resp.status} - {error_text}")
                    return False
                    
    except Exception as e:
        logger.error(f"HA service call error: {e}")
        return False


async def get_ha_states(base_url: str, token: str, entity_id: Optional[str] = None) -> Optional[Any]:
    """
    Get Home Assistant states
    
    Args:
        base_url: HA URL
        token: Access token
        entity_id: Optional specific entity (otherwise returns all)
    
    Returns:
        State data or None
    """
    try:
        ssl_context = create_ha_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            if entity_id:
                url = f"{base_url}/api/states/{entity_id}"
            else:
                url = f"{base_url}/api/states"
            
            async with session.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Get states failed: {resp.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Get states error: {e}")
        return None