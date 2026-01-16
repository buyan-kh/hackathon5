"""
Agent Orchestrator - Coordinates agents to generate Tomorrow's Paper.

Uses:
- Real Yutori API for web scouting
- Mock data for Fabricate/Freepik (replace with real APIs when ready)
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
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class AgentType(Enum):
    YUTORI = "yutori"
    FABRICATE = "fabricate"
    FREEPIK = "freepik"


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    progress: float = 0.0
    current_task: str | None = None
    result: dict = field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class OrchestratorEvent:
    """Event emitted by the orchestrator."""
    event_type: str
    agent_type: AgentType | None
    data: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AgentOrchestrator:
    """
    Orchestrates agents to generate Tomorrow's Paper.
    Uses real Yutori API, mock data for others.
    """
    
    def __init__(self):
        self.agents: dict[AgentType, AgentState] = {
            AgentType.YUTORI: AgentState(agent_type=AgentType.YUTORI),
            AgentType.FABRICATE: AgentState(agent_type=AgentType.FABRICATE),
            AgentType.FREEPIK: AgentState(agent_type=AgentType.FREEPIK),
        }
        self.job_id: UUID | None = None
        self.query: str | None = None
        self.settings = get_settings()
    
    def reset(self):
        """Reset all agent states."""
        for agent_type in self.agents:
            self.agents[agent_type] = AgentState(agent_type=agent_type)
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
        
        # Emit start event
        yield OrchestratorEvent(
            event_type="start",
            agent_type=None,
            data={"job_id": str(self.job_id), "query": query, "mode": mode}
        )
        
        # Collected results
        news_context = []
        simulation_result = {}
        cover_image = None
        
        try:
            # Phase 1: Yutori scouts the web (REAL API)
            if use_web_search:
                async for event in self._run_yutori_real(query):
                    yield event
                    if event.data.get("status") == "completed":
                        news_context = event.data.get("result", {}).get("news_items", [])
            
            # Phase 2: Tonic Fabricate runs simulation (mock for now)
            async for event in self._run_fabricate_mock(query, mode):
                yield event
                if event.data.get("status") == "completed":
                    simulation_result = event.data.get("result", {})
            
            # Phase 3: Freepik generates cover image (mock for now)
            if mode == "paper":
                async for event in self._run_freepik_mock(query):
                    yield event
                    if event.data.get("status") == "completed":
                        cover_image = event.data.get("result", {}).get("cover_image")
            
            # Compose final paper
            paper = self._compose_paper(
                query=query,
                news_context=news_context,
                simulation_result=simulation_result,
                cover_image_url=cover_image,
            )
            
            # Emit completion event
            yield OrchestratorEvent(
                event_type="complete",
                agent_type=None,
                data={
                    "job_id": str(self.job_id),
                    "query": query,
                    "paper": paper,
                }
            )
        except Exception as e:
            logger.exception(f"Error in orchestrator: {e}")
            yield OrchestratorEvent(
                event_type="error",
                agent_type=None,
                data={"message": str(e)}
            )
    
    async def _run_yutori_real(self, query: str) -> AsyncGenerator[OrchestratorEvent, None]:
        """Run Yutori web research agent with REAL API."""
        agent = self.agents[AgentType.YUTORI]
        agent.status = AgentStatus.RUNNING
        agent.started_at = datetime.utcnow()
        
        yield OrchestratorEvent(
            event_type="agent_update",
            agent_type=AgentType.YUTORI,
            data={"status": "running", "progress": 0, "task": "Initializing Yutori..."}
        )
        
        try:
            # Call real Yutori Research API
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.YUTORI,
                data={"status": "running", "progress": 10, "task": "Launching research task..."}
            )
            
            research_result = await yutori_agent.research(
                query=f"Latest news and market analysis related to: {query}"
            )
            
            logger.info(f"Yutori research launched: {research_result.task_id}, status: {research_result.status}")
            
            if research_result.status == "error":
                # Fall back to mock on error
                logger.warning(f"Yutori error: {research_result.content}, falling back to mock")
                async for event in self._run_yutori_mock_fallback(query):
                    yield event
                return
            
            # Poll for completion
            max_polls = 30
            poll_interval = 2.0
            
            for i in range(max_polls):
                progress = 20 + (i / max_polls) * 70
                yield OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.YUTORI,
                    data={
                        "status": "running",
                        "progress": progress,
                        "task": f"Researching... ({i + 1}/{max_polls})",
                        "sources": len(research_result.sources) if research_result.sources else 0,
                    }
                )
                
                # Check status
                status_result = await yutori_agent.get_research_status(research_result.task_id)
                logger.debug(f"Poll {i+1}: status = {status_result.status}")
                
                if status_result.status == "completed":
                    research_result = status_result
                    break
                elif status_result.status == "error":
                    logger.warning(f"Yutori status error, using fallback")
                    async for event in self._run_yutori_mock_fallback(query):
                        yield event
                    return
                
                await asyncio.sleep(poll_interval)
            
            # Complete with real results
            agent.status = AgentStatus.COMPLETED
            agent.completed_at = datetime.utcnow()
            
            # Parse sources from Yutori response
            news_items = []
            if research_result.sources:
                for source in research_result.sources:
                    news_items.append({
                        "title": source.get("title", "Untitled"),
                        "source": source.get("url", "Unknown"),
                        "content": source.get("content", ""),
                    })
            
            agent.result = {
                "task_id": research_result.task_id,
                "sources_found": len(news_items),
                "content": research_result.content,
                "news_items": news_items,
                "view_url": research_result.view_url,
            }
            
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.YUTORI,
                data={"status": "completed", "progress": 100, "result": agent.result}
            )
            
        except Exception as e:
            logger.exception(f"Yutori API error: {e}")
            # Fall back to mock
            async for event in self._run_yutori_mock_fallback(query):
                yield event
    
    async def _run_yutori_mock_fallback(self, query: str) -> AsyncGenerator[OrchestratorEvent, None]:
        """Quick mock fallback if Yutori API fails."""
        yield OrchestratorEvent(
            event_type="agent_update",
            agent_type=AgentType.YUTORI,
            data={"status": "running", "progress": 50, "task": "Gathering context..."}
        )
        
        await asyncio.sleep(0.5)
        
        news_items = self._generate_mock_news(query)
        
        yield OrchestratorEvent(
            event_type="agent_update",
            agent_type=AgentType.YUTORI,
            data={
                "status": "completed",
                "progress": 100,
                "result": {
                    "sources_found": len(news_items),
                    "news_items": news_items,
                }
            }
        )
    
    async def _run_fabricate_mock(self, query: str, mode: str) -> AsyncGenerator[OrchestratorEvent, None]:
        """Mock Tonic Fabricate simulation."""
        tasks = [
            ("Initializing simulation engine...", 0, 0.3),
            ("Loading market models...", 15, 0.3),
            ("Parsing scenario parameters...", 30, 0.3),
            ("Running Monte Carlo simulations...", 50, 0.6),
            ("Analyzing market correlations...", 70, 0.4),
            ("Generating price projections...", 85, 0.3),
            ("Finalizing simulation results...", 95, 0.2),
        ]
        
        steps = 0
        for task, progress, delay in tasks:
            steps += random.randint(100, 200)
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.FABRICATE,
                data={
                    "status": "running",
                    "progress": progress,
                    "task": task,
                    "steps": steps,
                }
            )
            await asyncio.sleep(delay)
        
        # Generate simulation results
        simulation_result = self._generate_simulation(query)
        
        yield OrchestratorEvent(
            event_type="agent_update",
            agent_type=AgentType.FABRICATE,
            data={
                "status": "completed",
                "progress": 100,
                "result": simulation_result,
            }
        )
    
    async def _run_freepik_mock(self, query: str) -> AsyncGenerator[OrchestratorEvent, None]:
        """Mock Freepik image generation."""
        tasks = [
            ("Preparing image generation...", 0, 0.2),
            ("Analyzing headline context...", 25, 0.3),
            ("Generating cover composition...", 50, 0.5),
            ("Applying editorial styling...", 75, 0.3),
            ("Finalizing visuals...", 95, 0.2),
        ]
        
        for task, progress, delay in tasks:
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.FREEPIK,
                data={
                    "status": "running",
                    "progress": progress,
                    "task": task,
                }
            )
            await asyncio.sleep(delay)
        
        yield OrchestratorEvent(
            event_type="agent_update",
            agent_type=AgentType.FREEPIK,
            data={
                "status": "completed",
                "progress": 100,
                "result": {
                    "cover_image": None,
                    "articles_generated": 3,
                    "infographics": 2,
                }
            }
        )
    
    def _generate_mock_news(self, query: str) -> list[dict]:
        """Generate mock news items based on query."""
        query_lower = query.lower()
        
        base_news = [
            {"title": "Fed Signals Careful Approach to Rate Changes", "source": "Reuters", "sentiment": "neutral"},
            {"title": "Global Markets Eye Economic Indicators", "source": "Bloomberg", "sentiment": "neutral"},
        ]
        
        if "oil" in query_lower:
            base_news.extend([
                {"title": "OPEC+ Discusses Production Targets", "source": "Reuters", "sentiment": "bearish"},
                {"title": "Energy Sector Faces Volatility Concerns", "source": "Bloomberg", "sentiment": "bearish"},
            ])
        
        if "tech" in query_lower or "nasdaq" in query_lower:
            base_news.extend([
                {"title": "Big Tech Reports Strong Cloud Revenue Growth", "source": "TechCrunch", "sentiment": "bullish"},
            ])
        
        return base_news[:6]
    
    def _generate_simulation(self, query: str) -> dict:
        """Generate simulation results based on query analysis."""
        query_lower = query.lower()
        
        is_crisis = any(w in query_lower for w in ["crash", "crisis", "spike", "surge", "collapse", "war"])
        is_bullish = any(w in query_lower for w in ["growth", "rise", "rally", "boom", "positive"])
        is_bearish = any(w in query_lower for w in ["drop", "fall", "decline", "recession", "negative"])
        
        direction = 1 if is_bullish else (-1 if (is_bearish or is_crisis) else 0)
        volatility = 3.0 if is_crisis else 1.5
        
        assets_config = {
            "S&P 500": {"base": 5000, "vol": 0.02 * volatility, "bias": direction * -0.01},
            "Gold": {"base": 2050, "vol": 0.015 * volatility, "bias": direction * 0.02 if is_crisis else 0},
            "VIX": {"base": 15, "vol": 0.1 * volatility, "bias": 0.3 if is_crisis else direction * -0.05},
        }
        
        if "oil" in query_lower:
            oil_change = 0.4 if "spike" in query_lower else (-0.2 if is_bearish else 0.1)
            assets_config["Oil"] = {"base": 75, "vol": 0.04 * volatility, "bias": oil_change}
        
        if "tech" in query_lower or "nasdaq" in query_lower:
            assets_config["NASDAQ"] = {"base": 16000, "vol": 0.03 * volatility, "bias": direction * 0.03}
        
        if "bitcoin" in query_lower or "crypto" in query_lower:
            assets_config["Bitcoin"] = {"base": 42000, "vol": 0.08 * volatility, "bias": direction * 0.1}
        
        assets = []
        for asset_name, config in assets_config.items():
            base = config["base"]
            change_pct = config["bias"] + random.gauss(0, config["vol"])
            projected = base * (1 + change_pct)
            
            timeline = []
            current = base
            for i in range(7):
                daily_change = change_pct / 7 + random.gauss(0, config["vol"] / 3)
                current *= (1 + daily_change)
                timeline.append({
                    "date": f"Day {i + 1}",
                    "value": round(current, 2),
                    "projected": True,
                })
            
            assets.append({
                "asset": asset_name,
                "current_value": round(base, 2),
                "projected_value": round(projected, 2),
                "change": round(projected - base, 2),
                "change_percent": round(change_pct * 100, 2),
                "timeline": timeline,
            })
        
        assets.sort(key=lambda a: abs(a["change_percent"]), reverse=True)
        
        return {
            "scenario": f"Scenario: {query[:50]}",
            "description": query,
            "time_horizon": "1w",
            "assets": assets,
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def _compose_paper(
        self,
        query: str,
        news_context: list[dict],
        simulation_result: dict,
        cover_image_url: str | None,
    ) -> dict:
        """Compose the final Tomorrow's Paper."""
        tomorrow = datetime.utcnow() + timedelta(days=1)
        
        assets = simulation_result.get("assets", [])
        
        if assets:
            most_impacted = assets[0]
            change = most_impacted["change_percent"]
            asset = most_impacted["asset"]
            
            if abs(change) < 2:
                headline = f"Markets Hold Steady: Analysis of '{query[:30]}...'"
            elif change > 0:
                headline = f"{asset} Surges {abs(change):.1f}% as Markets React"
            else:
                headline = f"{asset} Tumbles {abs(change):.1f}% Amid Market Uncertainty"
        else:
            headline = f"Market Analysis: {query[:40]}..."
        
        articles = [
            {
                "id": str(uuid4()),
                "title": headline,
                "content": f"In response to the scenario '{query}', our AI-powered simulation projects significant market movements. Based on analysis of {len(news_context)} news sources and advanced Monte Carlo simulations, we forecast the following developments...",
                "summary": simulation_result.get("description", "Market analysis based on AI simulation."),
                "category": "headline",
                "importance": 5,
            }
        ]
        
        if assets:
            market_summary = ", ".join([
                f"{a['asset']} {'+' if a['change_percent'] > 0 else ''}{a['change_percent']:.1f}%"
                for a in assets[:4]
            ])
            articles.append({
                "id": str(uuid4()),
                "title": "Market Projections at a Glance",
                "content": f"Key projected movements: {market_summary}. These projections are based on historical correlations and scenario analysis.",
                "summary": market_summary,
                "category": "market",
                "importance": 4,
            })
        
        market_snapshot = [
            {
                "asset": a["asset"],
                "value": a["projected_value"],
                "change": a["change"],
                "changePercent": a["change_percent"],
            }
            for a in assets
        ]
        
        trending_topics = [
            {"topic": "#MarketSimulation", "sentiment": 0.2, "mentions": random.randint(400, 800)},
            {"topic": f"#{query.split()[0]}", "sentiment": 0.1, "mentions": random.randint(200, 500)},
            {"topic": "#AITrading", "sentiment": 0.3, "mentions": random.randint(100, 300)},
        ]
        
        return {
            "paper_id": str(uuid4()),
            "date": tomorrow.strftime("%B %d, %Y"),
            "headline": headline,
            "subheadline": f"AI-powered market simulation based on: {query[:60]}...",
            "query": query,
            "cover_image_url": cover_image_url,
            "articles": articles,
            "market_snapshot": market_snapshot,
            "trending_topics": trending_topics,
            "news_context": news_context,
            "simulation_data": simulation_result,
            "generated_at": datetime.utcnow().isoformat(),
        }


# Global orchestrator instance
orchestrator = AgentOrchestrator()
