"""
Yutori Agent - Web scouting and research using Yutori API.

Yutori provides:
- Research API: One-time deep web research
- Scouting API: Continuous monitoring for news/updates
"""

import httpx
from typing import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from app.core.config import get_settings


@dataclass
class YutoriResearchResult:
    """Result from a Yutori research task."""
    task_id: str
    status: str
    view_url: str | None = None
    content: str | None = None
    sources: list[dict] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class YutoriAgent:
    """
    Agent for web research and scouting using Yutori API.
    
    API Reference: https://docs.yutori.com/
    """
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.yutori_api_key
        self.base_url = settings.yutori_base_url
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
    
    async def research(self, query: str, webhook_url: str | None = None) -> YutoriResearchResult:
        """
        Launch a one-time deep research task.
        
        Args:
            query: The research query
            webhook_url: Optional webhook for completion notification
            
        Returns:
            YutoriResearchResult with task details
        """
        payload = {"query": query}
        if webhook_url:
            payload["webhook_url"] = webhook_url
        
        try:
            response = await self.client.post("/research/tasks", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return YutoriResearchResult(
                task_id=data.get("task_id", ""),
                status=data.get("status", "queued"),
                view_url=data.get("view_url"),
            )
        except httpx.HTTPError as e:
            return YutoriResearchResult(
                task_id="",
                status="error",
                content=f"Research API error: {str(e)}",
            )
    
    async def get_research_status(self, task_id: str) -> YutoriResearchResult:
        """Check the status of a research task."""
        try:
            response = await self.client.get(f"/research/tasks/{task_id}")
            response.raise_for_status()
            data = response.json()
            
            return YutoriResearchResult(
                task_id=task_id,
                status=data.get("status", "unknown"),
                view_url=data.get("view_url"),
                content=data.get("content"),
                sources=data.get("sources", []),
            )
        except httpx.HTTPError as e:
            return YutoriResearchResult(
                task_id=task_id,
                status="error",
                content=f"Status check error: {str(e)}",
            )
    
    async def create_scout(
        self, 
        query: str, 
        display_name: str | None = None,
        webhook_url: str | None = None,
    ) -> dict:
        """
        Create a scouting task for continuous monitoring.
        
        Args:
            query: What to scout for (e.g., "latest AI news")
            display_name: Human-readable name for the scout
            webhook_url: Webhook for notifications
            
        Returns:
            Scout task details
        """
        payload = {"query": query}
        if display_name:
            payload["display_name"] = display_name
        if webhook_url:
            payload["webhook_url"] = webhook_url
        
        try:
            response = await self.client.post("/scouting/tasks", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e), "status": "error"}
    
    async def get_scout_outputs(self, scout_id: str) -> list[dict]:
        """Get outputs from a scouting task."""
        try:
            response = await self.client.get(f"/scouting/tasks/{scout_id}/outputs")
            response.raise_for_status()
            return response.json().get("outputs", [])
        except httpx.HTTPError as e:
            return [{"error": str(e)}]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton instance
yutori_agent = YutoriAgent()
