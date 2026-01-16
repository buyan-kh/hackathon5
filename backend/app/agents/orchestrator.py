"""
Agent Orchestrator - Coordinates agents to generate Tomorrow's Paper.

Uses:
- ✅ Multiple Yutori agents in parallel for diverse angles (Target, Aggressor, Global, Econ)
- ✅ Real Freepik API for multiple contextual images
- 🔧 Mock for Fabricate simulation
"""

import asyncio
import random
import logging
from typing import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from app.agents.yutori import yutori_agent
from app.agents.freepik import freepik_agent
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class AgentType(Enum):
    YUTORI_NEWS = "yutori_news"          # General Breaking News
    YUTORI_SENTIMENT = "yutori_sentiment" # Social Sentiment
    YUTORI_ANALYSIS = "yutori_analysis"   # Expert Analysis
    YUTORI_TARGET = "yutori_target"       # Target Country/Entity Focus
    YUTORI_GLOBAL = "yutori_global"       # Global Reaction/China/EU
    YUTORI_ECON = "yutori_econ"           # Economic Impact
    FABRICATE = "fabricate"
    FREEPIK = "freepik"


@dataclass
class OrchestratorEvent:
    """Event emitted by the orchestrator."""
    event_type: str
    agent_type: AgentType | None
    data: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AgentOrchestrator:
    """Orchestrates agents to generate Tomorrow's Paper."""
    
    def __init__(self):
        self.job_id: UUID | None = None
        self.query: str | None = None
        self.settings = get_settings()
    
    def reset(self):
        self.job_id = None
        self.query = None
    
    async def process_query(
        self,
        query: str,
        mode: str = "paper",
        use_web_search: bool = True,
    ) -> AsyncGenerator[OrchestratorEvent, None]:
        """Process a user query through the agent pipeline."""
        self.reset()
        self.job_id = uuid4()
        self.query = query
        
        logger.info(f"🚀 Starting query: {query[:50]}...")
        
        yield OrchestratorEvent(
            event_type="start",
            agent_type=None,
            data={"job_id": str(self.job_id), "query": query, "mode": mode}
        )
        
        all_news = []
        simulation_result = {}
        images = []
        
        try:
            # Phase 1: Run 6 Yutori agents & Image Generation in PARALLEL
            # We want to start images immediately as they take time
            image_task = None
            if mode == "paper":
                image_task = asyncio.create_task(self._generate_contextual_images(query))
                logger.info("🎨 Contextual images started in background")
            
            if use_web_search:
                yutori_results = await self._run_multiple_yutori(query)
                for result in yutori_results:
                    all_news.extend(result.get("news_items", []))
            
            # Phase 2: Fabricate simulation
            async for event in self._run_fabricate_mock(query, mode):
                yield event
                if event.data.get("status") == "completed":
                    simulation_result = event.data.get("result", {})
            
            # Wait for Images
            if image_task:
                images = await image_task
                logger.info(f"🎨 Images generated: {len(images)}")
            
            # Compose final paper
            paper = self._compose_paper(query, all_news, simulation_result, images)
            logger.info(f"✅ Paper: {paper['headline'][:50]}...")
            
            yield OrchestratorEvent(
                event_type="complete",
                agent_type=None,
                data={"job_id": str(self.job_id), "query": query, "paper": paper}
            )
        except Exception as e:
            logger.exception(f"❌ Error: {e}")
            yield OrchestratorEvent(event_type="error", agent_type=None, data={"message": str(e)})
    
    async def _run_multiple_yutori(self, query: str) -> list[dict]:
        """Run 6 Yutori agents in parallel with heavily varied focuses."""
        
        # Parse query for entities to make searches specific
        # Simple heuristic: Split by 'vs' or 'attacking' etc to find entities roughly
        # User example: "Trump attacking France" -> Trump (Aggressor), France (Target)
        
        agents = [
            (AgentType.YUTORI_NEWS, f"Latest breaking news: {query}"),
            (AgentType.YUTORI_SENTIMENT, f"Public opinion and social media reaction to: {query}"),
            (AgentType.YUTORI_ANALYSIS, f"Expert financial analysis and market implications of: {query}"),
            (AgentType.YUTORI_TARGET, f"Impact on local economy and politics in the target country regarding: {query}"),
            (AgentType.YUTORI_GLOBAL, f"Reactions from European Union, China, and global rivals regarding: {query}"),
            (AgentType.YUTORI_ECON, f"US economic outlook and trade war consequences related to: {query}"),
        ]
        
        # Create all research tasks
        tasks = []
        for agent_type, research_query in agents:
            task = asyncio.create_task(
                self._run_single_yutori(agent_type, research_query)
            )
            tasks.append(task)
        
        # Run all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, dict)]
    
    async def _run_single_yutori(self, agent_type: AgentType, query: str) -> dict:
        """Run single Yutori agent with specific focus."""
        from app.api.websocket import manager
        
        agent_name_map = {
            AgentType.YUTORI_NEWS: "News Scout",
            AgentType.YUTORI_SENTIMENT: "Sentiment",
            AgentType.YUTORI_ANALYSIS: "Analysis",
            AgentType.YUTORI_TARGET: "Local Impact",
            AgentType.YUTORI_GLOBAL: "Global React",
            AgentType.YUTORI_ECON: "US Economy",
        }
        name = agent_name_map.get(agent_type, "Agent")
        
        # Emit start
        await self._emit_to_websockets(OrchestratorEvent(
            event_type="agent_update",
            agent_type=agent_type,
            data={"status": "running", "progress": 10, "task": f"Starting {name}..."}
        ))
        
        try:
            # 50% chance to skip some specialized agents if query is very simple to save API
            # But for this user request, we run all.
            result = await yutori_agent.research(query)
            
            if result.status == "error":
                return {"news_items": self._mock_news_for_agent(agent_type)}

            # If task was successfully queued, poll for completion
            if result.task_id:
                await self._emit_to_websockets(OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=agent_type,
                    data={"status": "running", "progress": 30, "task": f"{name} researching..."}
                ))
                
                poll_interval = 2  # seconds between polls
                timeout = 60  # total timeout in seconds
                elapsed = 0
                
                while elapsed < timeout:
                    status_result = await yutori_agent.get_research_status(result.task_id)
                    
                    if status_result.status == "completed":
                        # Use sources from the final status result
                        result = status_result
                        break
                    elif status_result.status == "error":
                        logger.warning(f"Research task {result.task_id} failed: {status_result.error}")
                        return {"news_items": self._mock_news_for_agent(agent_type)}
                    elif status_result.status in ("queued", "processing"):
                        # Update progress based on elapsed time
                        progress = min(30 + int((elapsed / timeout) * 60), 90)
                        await self._emit_to_websockets(OrchestratorEvent(
                            event_type="agent_update",
                            agent_type=agent_type,
                            data={"status": "running", "progress": progress, "task": f"{name} processing..."}
                        ))
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                    else:
                        # Unknown status, wait and retry
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                else:
                    # Timeout reached
                    logger.warning(f"Research task {result.task_id} timed out after {timeout}s")
                    return {"news_items": self._mock_news_for_agent(agent_type)}

            await self._emit_to_websockets(OrchestratorEvent(
                 event_type="agent_update",
                 agent_type=agent_type,
                 data={"status": "completed", "progress": 100, "sources": len(result.sources or [])}
            ))

            news_items = []
            for source in (result.sources or [])[:4]:
                news_items.append({
                    "title": source.get("title", "Update"),
                    "source": source.get("url", "Unknown"),
                    "agent": name,
                    "content": source.get("content", "")[:150] + "..."
                })
            
            return {"news_items": news_items}
            
        except Exception:
            return {"news_items": self._mock_news_for_agent(agent_type)}

    async def _generate_contextual_images(self, query: str) -> list[str]:
        """Generate 3 distinct images for different sections."""
        # We'll generate 3 images in parallel
        # 1. Main Cover: Dramatic, overall theme
        # 2. Economy/Market: Charts, financial visual
        # 3. Global/Political: Map or political meeting style
        
        prompts = [
            (f"Editorial news photo, highly detailed, dramatic lighting: {query}", "cinematic"),
            (f"Financial data visualization, stock market charts, economic growth graph related to: {query}", "3d-model"),
            (f"Political map or diplomatic meeting, professional news photography: {query}", "photographic"),
        ]
        
        async def gen_one(p, style):
            try:
                res = await freepik_agent.generate_cover_image(p, query, "neutral")
                if res.base64_data:
                    return f"data:image/png;base64,{res.base64_data}"
                return res.image_url
            except:
                return None

        # Run 3 generations in parallel
        tasks = [gen_one(p, s) for p, s in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, str)]

    def _compose_paper(self, query: str, news: list, sim: dict, images: list) -> dict:
        tomorrow = datetime.utcnow() + timedelta(days=1)
        headline = "Market Analysis"
        
        # Try to find a good headline from news
        if news:
            # prioritize News Scout headlines
            headlines = [n["title"] for n in news if "News Scout" in n.get("agent", "")]
            if headlines:
                headline = headlines[0]
            else:
                headline = news[0]["title"]

        final_images = images if images else [None, None, None]
        # Ensure we have at least 3 slots
        while len(final_images) < 3:
            final_images.append(None)

        return {
            "paper_id": str(uuid4()),
            "date": tomorrow.strftime("%B %d, %Y"),
            "headline": headline,
            "subheadline": f"Comprehensive coverage: US, Europe, and Global Markets react to {query[:30]}...",
            "query": query,
            "cover_image_url": final_images[0],
            "secondary_image_url": final_images[1],
            "tertiary_image_url": final_images[2],
            "articles": [{"id": str(uuid4()), "title": headline, "content": f"Full coverage of {query}.", "category": "headline", "importance": 5}],
            "market_snapshot": sim.get("assets", []),
            "trending_topics": [{"topic": "#GlobalMarkets", "sentiment": -0.4, "mentions": 1250}],
            "news_context": news,
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def _emit_to_websockets(self, event: OrchestratorEvent):
        from app.api.websocket import manager
        data = {
            "event_type": event.event_type,
            "agent": event.agent_type.value if event.agent_type else None,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
        }
        for ws in manager.active_connections:
            try:
                await ws.send_json(data)
            except:
                pass


    def _mock_news_for_agent(self, agent_type: AgentType) -> list[dict]:
        return [{"title": "Agent Researching...", "source": "Internal", "agent": agent_type.value}]

    async def _run_fabricate_mock(self, query: str, mode: str) -> AsyncGenerator[OrchestratorEvent, None]:
        tasks = [
            ("Initializing simulation...", 0, 0.2),
            ("Running Monte Carlo...", 40, 0.4),
            ("Generating projections...", 80, 0.2),
        ]
        for task, progress, delay in tasks:
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.FABRICATE,
                data={"status": "running", "progress": progress, "task": task}
            )
            await asyncio.sleep(delay)
        
        yield OrchestratorEvent(
            event_type="agent_update",
            agent_type=AgentType.FABRICATE,
            data={"status": "completed", "progress": 100, "result": self._generate_simulation(query)}
        )

    def _generate_simulation(self, query: str) -> dict:
        query_lower = query.lower()
        is_crisis = any(w in query_lower for w in ["crash", "spike", "surge", "collapse", "tariff", "war"])
        direction = -1 if is_crisis else 1
        vol = 2.0
        
        configs = {
            "S&P 500": {"base": 5000, "vol": 0.02 * vol, "bias": direction * -0.015},
            "CAC 40": {"base": 7600, "vol": 0.03 * vol, "bias": direction * -0.025}, # French index
            "EUR/USD": {"base": 1.08, "vol": 0.01 * vol, "bias": -0.01},
            "Gold": {"base": 2050, "vol": 0.015 * vol, "bias": 0.02 if is_crisis else 0},
            "VIX": {"base": 15, "vol": 0.1 * vol, "bias": 0.3 if is_crisis else -0.05},
        }
        
        assets = []
        for name, cfg in configs.items():
            change = cfg["bias"] + random.gauss(0, cfg["vol"])
            projected = cfg["base"] * (1 + change)
            assets.append({
                "asset": name, "current_value": round(cfg["base"], 2),
                "projected_value": round(projected, 2), "change": round(projected - cfg["base"], 2),
                "change_percent": round(change * 100, 2),
            })
        
        assets.sort(key=lambda a: abs(a["change_percent"]), reverse=True)
        return {"scenario": query, "assets": assets}

orchestrator = AgentOrchestrator()
