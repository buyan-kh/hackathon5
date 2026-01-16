"""
Tonic Fabricate Agent - Synthetic data generation.

Tonic Fabricate API:
- Endpoint: https://fabricate.tonic.ai/api/v1
- Auth: Bearer token in Authorization header
- Docs: https://docs.tonic.ai/fabricate
"""

import httpx
import logging
from dataclasses import dataclass, field
from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FabricateResult:
    """Result from Tonic Fabricate."""
    status: str = "pending"
    data: dict = field(default_factory=dict)
    error: str | None = None


class FabricateAgent:
    """Agent for synthetic data generation using Tonic Fabricate API."""
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.tonic_api_key
        self.base_url = settings.tonic_base_url
        
        logger.info(f"🔑 Tonic Fabricate API Key configured: {'Yes' if self.api_key else 'No'}")
        logger.info(f"🌐 Tonic Base URL: {self.base_url}")
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
    
    async def list_workspaces(self) -> dict:
        """List available workspaces."""
        logger.info("📂 [FABRICATE] Listing workspaces...")
        
        try:
            response = await self.client.get("/workspaces")
            
            logger.info(f"📂 [FABRICATE] Response status: {response.status_code}")
            logger.debug(f"📂 [FABRICATE] Response: {response.text[:500]}")
            
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"❌ [FABRICATE] Workspaces error: {error_msg}")
            return {"status": "error", "error": error_msg}
        except httpx.HTTPError as e:
            error_msg = str(e)
            logger.error(f"❌ [FABRICATE] Connection error: {error_msg}")
            return {"status": "error", "error": error_msg}
    
    async def list_databases(self, workspace: str) -> dict:
        """List databases in a workspace."""
        logger.info(f"📊 [FABRICATE] Listing databases in workspace: {workspace}")
        
        try:
            response = await self.client.get(f"/workspaces/{workspace}/databases")
            
            logger.info(f"📊 [FABRICATE] Response status: {response.status_code}")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"❌ [FABRICATE] Databases error: {error_msg}")
            return {"status": "error", "error": error_msg}
        except httpx.HTTPError as e:
            error_msg = str(e)
            logger.error(f"❌ [FABRICATE] Connection error: {error_msg}")
            return {"status": "error", "error": error_msg}
    
    async def generate_data(
        self,
        workspace: str,
        database: str,
        format: str = "jsonl",
    ) -> FabricateResult:
        """Generate synthetic data from a Fabricate database."""
        logger.info(f"🏭 [FABRICATE] Generating data: {workspace}/{database}")
        
        payload = {
            "workspace": workspace,
            "database": database,
            "format": format,
        }
        
        try:
            response = await self.client.post("/generate", json=payload)
            
            logger.info(f"🏭 [FABRICATE] Generate response: {response.status_code}")
            logger.debug(f"🏭 [FABRICATE] Response: {response.text[:500]}")
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✅ [FABRICATE] Data generated successfully")
            
            return FabricateResult(
                status="completed",
                data=data,
            )
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"❌ [FABRICATE] Generate error: {error_msg}")
            return FabricateResult(
                status="error",
                error=error_msg,
            )
        except httpx.HTTPError as e:
            error_msg = str(e)
            logger.error(f"❌ [FABRICATE] Connection error: {error_msg}")
            return FabricateResult(
                status="error",
                error=error_msg,
            )
    
    async def test_connection(self) -> dict:
        """Test the API connection."""
        logger.info("🧪 [FABRICATE] Testing API connection...")
        return await self.list_workspaces()
    
    async def generate_headline(self, topic: str, context: str = "general") -> str:
        """
        Generate a synthetic headline using Fabricate's generative engine.
        In a real scenario, this would hit an LLM endpoint.
        Here we simulate the network latency and return a generated response.
        """
        import asyncio
        import random
        
        logger.info(f"🏭 [FABRICATE] Generating headline for topic: {topic} ({context})")
        
        # Simulate API network latency (0.5 - 1.5s)
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Advanced template engine to simulate LLM output
        templates = [
            f"Global Markets Rattle as {topic} Scenario Intensifies",
            f"Exclusive: Inside the {topic} Crisis - What Traders Need to Know",
            f"Breaking: {topic} Sparks Volatility Across Major Indices",
            f"Analysis: How the {topic} Situation Could Reshape the Economy",
            f"Live Updates: diplomatic tensions rise over {topic}",
            f"Projected: {topic} Impact on Supply Chains 'Severe'",
            f"Urgent: Central Banks Monitor {topic} Developments Closely",
            f"Opinion: Why the {topic} Event Was Inevitable",
        ]
        
        headline = random.choice(templates)
        logger.info(f"✅ [FABRICATE] Generated: {headline}")
        return headline

    async def close(self):
        await self.client.aclose()


# Singleton instance
fabricate_agent = FabricateAgent()
