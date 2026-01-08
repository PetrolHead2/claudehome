"""
LLM Service - Unified interface for multiple LLM providers
Supports: LiteLLM, Anthropic, OpenAI, Ollama direct
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_service")

app = FastAPI(title="ClaudeHome LLM Service")

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "litellm")  # litellm, anthropic, openai, ollama
LITELLM_URL = os.getenv("LITELLM_URL", "http://192.168.50.158:4100")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "local-chat")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.50.158:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")


class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    model: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7


class LLMResponse(BaseModel):
    text: str
    model_used: str
    provider: str
    tokens_used: Optional[int] = None


class LLMProvider:
    """Base class for LLM providers"""
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class LiteLLMProvider(LLMProvider):
    """LiteLLM unified interface (recommended)"""
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or LITELLM_MODEL
        
        try:
            headers = {"Content-Type": "application/json"}
            if LITELLM_API_KEY:
                headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{LITELLM_URL}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": request.system_prompt},
                            {"role": "user", "content": request.user_prompt}
                        ],
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, 
                                      detail=f"LiteLLM error: {response.text}")
                
                data = response.json()
                
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    model_used=model,
                    provider="litellm",
                    tokens_used=data.get("usage", {}).get("total_tokens")
                )
                
        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


class OllamaProvider(LLMProvider):
    """Direct Ollama access (fallback)"""
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or OLLAMA_MODEL
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": f"{request.system_prompt}\n\n{request.user_prompt}",
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens
                        }
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code,
                                      detail=f"Ollama error: {response.text}")
                
                data = response.json()
                
                return LLMResponse(
                    text=data["response"],
                    model_used=model,
                    provider="ollama"
                )
                
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API"""
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="Anthropic API key not set")
        
        model = request.model or "claude-sonnet-4-20250514"
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": model,
                        "max_tokens": request.max_tokens,
                        "system": request.system_prompt,
                        "messages": [
                            {"role": "user", "content": request.user_prompt}
                        ]
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code,
                                      detail=f"Anthropic error: {response.text}")
                
                data = response.json()
                
                return LLMResponse(
                    text=data["content"][0]["text"],
                    model_used=model,
                    provider="anthropic",
                    tokens_used=data.get("usage", {}).get("input_tokens", 0) + 
                               data.get("usage", {}).get("output_tokens", 0)
                )
                
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Provider registry
PROVIDERS = {
    "litellm": LiteLLMProvider(),
    "ollama": OllamaProvider(),
    "anthropic": AnthropicProvider()
}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "provider": LLM_PROVIDER,
        "model": LITELLM_MODEL if LLM_PROVIDER == "litellm" else OLLAMA_MODEL
    }


@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    """
    Generate LLM response using configured provider
    """
    provider = PROVIDERS.get(LLM_PROVIDER)
    
    if not provider:
        raise HTTPException(status_code=500, 
                          detail=f"Unknown provider: {LLM_PROVIDER}")
    
    logger.info(f"Generating with {LLM_PROVIDER}")
    
    return await provider.generate(request)


@app.get("/providers")
async def list_providers():
    """List available providers"""
    return {
        "current": LLM_PROVIDER,
        "available": list(PROVIDERS.keys()),
        "config": {
            "litellm_url": LITELLM_URL,
            "litellm_model": LITELLM_MODEL,
            "ollama_url": OLLAMA_URL,
            "ollama_model": OLLAMA_MODEL
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)