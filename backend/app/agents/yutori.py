"""
Yutori Agent - Web scouting and research using Yutori API.

Yutori provides:
- Research API: One-time deep web research
- Scouting API: Continuous monitoring for news/updates

API: https://api.yutori.com/v1
Docs: https://docs.yutori.com/
"""

import httpx
import logging
from dataclasses import dataclass
from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class YutoriResearchResult:
    """Result from a Yutori research task."""
    task_id: str
    status: str
    view_url: str | None = None
    content: str | None = None
    sources: list[dict] = None
    error: str | None = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class YutoriAgent:
    """Agent for web research using Yutori API."""
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.yutori_api_key
        self.base_url = settings.yutori_base_url
        
        logger.info(f"🔑 Yutori API Key configured: {'Yes' if self.api_key else 'No'}")
        logger.info(f"🌐 Yutori Base URL: {self.base_url}")
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
    
    async def research(self, query: str, webhook_url: str | None = None) -> YutoriResearchResult:
        """Launch a one-time deep research task."""
        payload = {"query": query}
        if webhook_url:
            payload["webhook_url"] = webhook_url
        
        logger.info(f"🔍 [YUTORI] Starting research: {query[:50]}...")
        
        try:
            response = await self.client.post("/research/tasks", json=payload)
            
            logger.info(f"🔍 [YUTORI] Response status: {response.status_code}")
            logger.debug(f"🔍 [YUTORI] Response body: {response.text[:500]}")
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✅ [YUTORI] Task created: {data.get('task_id')}")
            
            return YutoriResearchResult(
                task_id=data.get("task_id", ""),
                status=data.get("status", "queued"),
                view_url=data.get("view_url"),
            )
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"❌ [YUTORI] API Error: {error_msg}")
            return YutoriResearchResult(
                task_id="",
                status="error",
                error=error_msg,
            )
        except httpx.HTTPError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(f"❌ [YUTORI] {error_msg}")
            return YutoriResearchResult(
                task_id="",
                status="error",
                error=error_msg,
            )
    
    async def get_research_status(self, task_id: str) -> YutoriResearchResult:
        """Check the status of a research task."""
        logger.debug(f"🔄 [YUTORI] Checking status for task: {task_id}")
        
        try:
            response = await self.client.get(f"/research/tasks/{task_id}")
            
            logger.debug(f"🔄 [YUTORI] Status response: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            status = data.get("status", "unknown")
            logger.info(f"📊 [YUTORI] Task {task_id} status: {status}")
            
            return YutoriResearchResult(
                task_id=task_id,
                status=status,
                view_url=data.get("view_url"),
                content=data.get("content"),
                sources=data.get("sources", []),
            )
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"❌ [YUTORI] Status check error: {error_msg}")
            return YutoriResearchResult(
                task_id=task_id,
                status="error",
                error=error_msg,
            )
        except httpx.HTTPError as e:
            error_msg = str(e)
            logger.error(f"❌ [YUTORI] Status check failed: {error_msg}")
            return YutoriResearchResult(
                task_id=task_id,
                status="error",
                error=error_msg,
            )
    
    async def close(self):
        await self.client.aclose()


# Singleton instance
yutori_agent = YutoriAgent()
