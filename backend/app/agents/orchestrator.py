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
            # SIMULATE MODE: Focus on market simulation only
            if mode == "simulate":
                yield OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FABRICATE,
                    data={"status": "running", "progress": 10, "task": "Analyzing query to determine relevant assets..."}
                )
                
                # Determine which assets to track based on query
                relevant_assets = self._identify_relevant_assets(query)
                logger.info(f"📊 [SIMULATE] Identified assets: {relevant_assets}")
                
                yield OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FABRICATE,
                    data={"status": "running", "progress": 30, "task": f"Fetching historical data for {len(relevant_assets)} assets..."}
                )
                
                # Fetch market data for each asset
                from app.agents.market_data import market_data_agent
                assets_data = []
                for idx, asset_query in enumerate(relevant_assets):
                    try:
                        market_context = await market_data_agent.get_market_context(asset_query)
                        if "error" not in market_context and market_context.get("current_price"):
                            assets_data.append({
                                "query": asset_query,
                                "market_context": market_context
                            })
                            yield OrchestratorEvent(
                                event_type="agent_update",
                                agent_type=AgentType.FABRICATE,
                                data={"status": "running", "progress": 30 + (idx + 1) * 20 // len(relevant_assets), "task": f"Fetched data for {market_context.get('name', asset_query)}"}
                            )
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch data for {asset_query}: {e}")
                
                if not assets_data:
                    # Fallback to default assets
                    logger.warning("⚠️ No assets found, using defaults")
                    assets_data = [
                        {"query": "SPY", "market_context": await market_data_agent.get_market_context("sp500")},
                        {"query": "BTC-USD", "market_context": await market_data_agent.get_market_context("bitcoin")},
                        {"query": "QQQ", "market_context": await market_data_agent.get_market_context("nasdaq")},
                    ]
                
                yield OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FABRICATE,
                    data={"status": "running", "progress": 60, "task": "Generating projected prices..."}
                )
                
                # Generate simulation for each asset
                simulation_assets = []
                for asset_info in assets_data:
                    market_ctx = asset_info["market_context"]
                    if "error" in market_ctx:
                        continue
                    
                    # Generate projection using Fabricate
                    sim_data = self._generate_simulation(query, market_ctx, all_news)
                    main_asset = sim_data.get("assets", [{}])[0] if sim_data.get("assets") else {}
                    
                    simulation_assets.append({
                        "asset": market_ctx.get("name", asset_info["query"]),
                        "current_value": market_ctx.get("current_price", 1000),
                        "projected_value": main_asset.get("projected_value", market_ctx.get("current_price", 1000)),
                        "change": main_asset.get("change", 0),
                        "change_percent": main_asset.get("change_percent", 0),
                        "history": market_ctx.get("history", []),
                    })
                
                simulation_result = {"scenario": query, "assets": simulation_assets}
                
                yield OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FABRICATE,
                    data={"status": "completed", "progress": 100, "task": f"Generated projections for {len(simulation_assets)} assets"}
                )
                
                # Emit complete event with simulation data
                yield OrchestratorEvent(
                    event_type="complete",
                    agent_type=None,
                    data={
                        "job_id": str(self.job_id),
                        "query": query,
                        "simulation": simulation_result,
                        "mode": "simulate"
                    }
                )
                return
            
            # PAPER MODE: Full paper generation
            # Phase 1: Run 6 Yutori agents & Image Generation in PARALLEL
            # We want to start images immediately as they take time
            image_task = None
            if mode == "paper":
                image_task = asyncio.create_task(self._generate_contextual_images(query))
                logger.info("🎨 Contextual images started in background")
                
            agent_screenshots = []
            
            if use_web_search:
                try:
                    # Add overall timeout for Yutori phase (110 seconds max)
                    # Must be > single agent timeout (90s) to allow graceful fallback
                    yutori_results = await asyncio.wait_for(
                        self._run_multiple_yutori(query),
                        timeout=110.0
                    )
                    for result in yutori_results:
                        all_news.extend(result.get("news_items", []))
                        # Collect screenshots from agents
                        if result.get("screenshot"):
                            # Prepend screenshots to images list so they take priority as secondary visuals
                            # or append to a separate list? 
                            # Let's treat them as distinct for now, but to save modifying _compose_paper signature too much layout
                            # we can just add them to the images list which is passed to _compose_paper
                            pass 

                    # Extract screenshots properly
                    agent_screenshots = [r.get("screenshot") for r in yutori_results if r.get("screenshot")]
                    
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Yutori phase timed out after 90s, proceeding with available results")
                    # Continue with whatever news we have (could be empty)
                    agent_screenshots = []
            
            # Phase 2: Fabricate simulation & Analysis
            # NOW driven by Yutori's findings
            
            # A. Simulation (scenarios)
            async for event in self._run_fabricate_mock(query, mode, news_context=all_news):
                yield event
                if event.data.get("status") == "completed":
                    simulation_result = event.data.get("result", {})

            # B. Market Analysis (Fabricate + Freepik)
            # User Request: "headlines of market analysis that fabricate creates by yutori request with freepik visuals"
            analysis_items = []
            try:
                # Generate analysis using Fabricate based on all_news context
                # We synthesize the top news into a coherent analysis request
                from app.agents.fabricate import fabricate_agent
                
                # Context string from top 3 news items
                context_str = " ".join([n.get("title", "") + " " + n.get("summary", "") for n in all_news[:3]])
                
                yield OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FABRICATE,
                    data={"status": "running", "progress": 50, "task": "Synthesizing market analysis..."}
                )
                
                # Generate 2 distinct analysis pieces
                for i in range(2):
                    analysis = await fabricate_agent.generate_market_analysis(context_str)
                    
                    # Generate a specific visual for this analysis using Freepik
                    # using the specific headline/topic
                    visual_query = f"editorial illustration of {analysis.get('topic', 'market')}, abstract concept, {analysis.get('title')}"
                    image_url = await self._generate_single_image(visual_query)
                    
                    analysis_items.append({
                        "id": str(uuid4()),
                        "title": analysis.get("title"),
                        "summary": analysis.get("summary"),
                        "source": "Market Impact Analysis", # Distinct source name
                        "image_url": image_url,
                        "published_at": datetime.utcnow().isoformat(),
                        "agent_type": "yutori_analysis_model", # Special type for frontend filtering
                        "is_analysis": True 
                    })
                    
                yield OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FABRICATE,
                    data={"status": "completed", "progress": 100, "task": "Analysis generation complete"}
                )
                
                # Add consumption to paper
                all_news.extend(analysis_items)

            except Exception as e:
                logger.error(f"❌ Analysis generation failed: {e}")

            # Wait for Contextual Images (with timeout)
            generated_images = []
            if image_task:
                try:
                    generated_images = await asyncio.wait_for(image_task, timeout=120.0)
                    logger.info(f"🎨 Images generated: {len(generated_images)}")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Image generation timed out after 120s, proceeding without images")
                    generated_images = []
            
            # Combine generated images with agent screenshots
            # We prioritize generated images for Cover, but screenshots for secondary
            images = generated_images + agent_screenshots

            # Compose final paper
            paper = self._compose_paper(query, all_news, simulation_result, images)
            logger.info(f"✅ Paper: {paper['headline'][:50]}...")
            
            yield OrchestratorEvent(
                event_type="complete",
                agent_type=None,
                data={"job_id": str(self.job_id), "query": query, "paper": paper}
            )
        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            logger.exception(f"❌ Error in orchestrator: {error_msg}")
            yield OrchestratorEvent(
                event_type="error", 
                agent_type=None, 
                data={
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "query": self.query
                }
            )
    
    async def _run_multiple_yutori(self, query: str) -> list[dict]:
        """Run 6 Yutori agents in parallel with heavily varied focuses."""
        
        # 1. DETECT SIMULATION INTENT UPFRONT
        # Use the raw query to decide if this is a simulation.
        # This prevents "cleaning" from removing the triggers.
        is_simulation_intent = self._is_simulation_query(query)
        if is_simulation_intent:
            logger.info(f"🔮 [ORCH] Simulation/Hypothetical detected for: '{query}'. Enforcing synthetic generation.")

        # Parse query for entities to make searches specific
        clean_query = query.lower()
        # Remove all "what if" variations completely for the Base Query (for topics),
        # BUT we still know it's a simulation trigger from above.
        for phrase in ["what if", "what happens if", "what will happen if", "simulate", "simulation", "scenario"]:
            clean_query = clean_query.replace(phrase, "")
        clean_query = clean_query.strip()
        
        if not clean_query: # Fallback if query was only stop words
            clean_query = query.lower()
        
        # Extract key entities (countries, topics) for better search
        common_countries = ["israel", "japan", "china", "russia", "ukraine", "usa", "us", "united states", "iran", "north korea", "south korea", "taiwan", "uk", "britain", "france", "germany", "eu", "europe"]
        entities = []
        for country in common_countries:
            if country in clean_query:
                entities.append(country)
        
        # Extract key topics
        key_topics = []
        topics_map = {
            "nuke": "nuclear",
            "nuclear": "nuclear",
            "war": "war",
            "attack": "attack",
            "economy": "economy",
            "market": "market",
            "trade": "trade",
            "sanction": "sanction",
        }
        for word, topic in topics_map.items():
            if word in clean_query:
                key_topics.append(topic)
        
        # Build focused query from entities and topics (max 4-5 words)
        query_parts = entities[:2] + key_topics[:2]  # Max 2 entities + 2 topics
        if not query_parts:
            # Fallback: use first 4 meaningful words
            words = [w for w in clean_query.split() if len(w) > 3][:4]
            query_parts = words
        
        base_query = " ".join(query_parts[:4])  # Max 4 words total
        
        # Build agent-specific queries
        agents = [
            (AgentType.YUTORI_NEWS, f"{base_query} news"),
            (AgentType.YUTORI_SENTIMENT, f"{base_query} reaction"),
            (AgentType.YUTORI_ANALYSIS, f"{base_query} analysis"),
            (AgentType.YUTORI_TARGET, f"{base_query} impact"),
            (AgentType.YUTORI_GLOBAL, f"{base_query} response"),
            (AgentType.YUTORI_ECON, f"{base_query} economy"),
        ]
        
        # Limit concurrency to avoid DDG rate limits (Max 2 parallel searches)
        # Increased delay to avoid rate limiting
        semaphore = asyncio.Semaphore(2)
        
        async def run_throttled(atype, aquery):
            async with semaphore:
                # Add random jitter to prevent burst pattern (longer delays)
                # SKIP DELAY FOR SIMULATION (User request: "why safe we waiting?")
                if not is_simulation_intent:
                    delay = random.uniform(1.0, 3.0)
                    logger.debug(f"⏳ Throttling {atype.value} with {delay:.1f}s delay")
                    await asyncio.sleep(delay)
                
                # PASS FORCE SIMULATION FLAG
                return await self._run_single_yutori(atype, aquery, force_simulation=is_simulation_intent)
        
        # Create tasks with throttling
        tasks = []
        for agent_type, research_query in agents:
            task = asyncio.create_task(run_throttled(agent_type, research_query))
            tasks.append(task)
        
        # Run all
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, dict)]
    
    async def _run_single_yutori(self, agent_type: AgentType, query: str, force_simulation: bool = False) -> dict:
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
            # Emit progress update
            await self._emit_to_websockets(OrchestratorEvent(
                event_type="agent_update",
                agent_type=agent_type,
                data={"status": "running", "progress": 30, "task": f"{name} researching..."}
            ))
            
            # Check if this is a simulation scenario
            # If so, generate synthetic "future" news instead of searching current web
            # Accept explicit flag or internal check (though internal check might be on "cleaned" query now)
            is_simulation = force_simulation or self._is_simulation_query(query)
            
            if is_simulation:
                # Simulation Mode: Generate synthetic future headlines
                await asyncio.sleep(1.0) # Simulate "thinking" time
                news_item = await self._generate_synthetic_news(agent_type, query)
                
                # Emit success
                await self._emit_to_websockets(OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=agent_type,
                    data={"status": "completed", "progress": 100, "task": f"{name} generated scenario data", "sources": 1}
                ))
                return {"news_items": [news_item]}

            # Standard Mode: Web Search (DuckDuckGo)
            # This bypasses the deep research queue for instant results
            result = await yutori_agent.fast_search(query, max_results=10)
            
            # Ensure sources is always a list (never None)
            sources = result.sources if result.sources else []
            sources_count = len(sources)
            
            logger.info(f"📊 [YUTORI] {name} search result: status={result.status}, sources={sources_count}, error={result.error}")
            
            if result.status == "error" or sources_count == 0:
                # Log detailed error information
                if result.error:
                    logger.error(f"❌ Fast search failed for {name}: {result.error}")
                else:
                    logger.warning(f"⚠️ Fast search returned 0 sources for {name} with query: {query[:100]}")
                
                # Emit error completion event
                await self._emit_to_websockets(OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=agent_type,
                    data={
                        "status": "completed", 
                        "progress": 100, 
                        "task": f"{name} search failed - {result.error or 'no results found'}", 
                        "sources": 0
                    }
                ))
                return {"news_items": self._mock_news_for_agent(agent_type, query=query)}
            
            # Emit success
            await self._emit_to_websockets(OrchestratorEvent(
                event_type="agent_update",
                agent_type=agent_type,
                data={"status": "completed", "progress": 100, "task": f"{name} found {sources_count} sources", "sources": sources_count}
            ))

            # --- LEVEL 1 & 2 AGENTIC UPGRADE ---
            # Enhance the top result with Deep Reading and/or Visual Scouting
            agentic_content = ""
            screenshot_url = None
            
            if sources:
                top_url = sources[0].get("url") or sources[0].get("link")
                if top_url and top_url.startswith("http"):
                    try:
                        # 1. Level 1: Deep Read (Content Extraction)
                        # We do this for the TOP result to get high quality context
                        logger.info(f"🕵️ [AGENTIC] Deep reading top source: {top_url}")
                        deep_content = await self.yutori_agent.deep_read(top_url)
                        if deep_content:
                            agentic_content = f"\n\n[DEEP READ CONTENT FROM {top_url}]:\n{deep_content[:2000]}..."
                            # Update the first source snippet with deep content for better LLM context
                            sources[0]["snippet"] = deep_content[:1000]

                        # 2. Level 2: Visual Scout (Screenshot)
                        # We assume the user wants to see the source
                        # Only do this if it's not a simulation (real web only)
                        if not is_simulation: 
                            logger.info(f"📸 [AGENTIC] Scouting visual: {top_url}")
                            screenshot_url = await self.yutori_agent.take_screenshot(top_url)
                            
                    except Exception as e:
                        logger.error(f"⚠️ [AGENTIC] Enhancement failed: {e}")
            
            # Format news items with relevance filtering
            news_items = []
            query_lower = query.lower()
            # Extract key terms from query for relevance checking
            query_terms = set(query_lower.split())
            # Remove stop words
            stop_words = {"what", "will", "happen", "if", "is", "a", "an", "the", "to", "of", "and", "or", "but", "in", "on", "at", "for", "with", "by"}
            query_terms = {t for t in query_terms if t not in stop_words and len(t) > 2}
            
            for item in result.sources:
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "")
                title_lower = title.lower()
                snippet_lower = snippet.lower()
                
                # Check relevance - must contain at least one query term
                relevance_score = sum(1 for term in query_terms if term in title_lower or term in snippet_lower)
                
                # Filter out irrelevant results (dating apps, unrelated topics)
                irrelevant_keywords = ["dating", "app", "happn", "tinder", "bumble", "match.com", "singles", "romance"]
                is_irrelevant = any(kw in title_lower or kw in snippet_lower for kw in irrelevant_keywords)
                
                # Only include if relevant and not irrelevant
                if relevance_score > 0 and not is_irrelevant:
                    news_items.append({
                        "id": str(uuid4()),
                        "title": title,
                        "url": item.get("url", "#"),
                        "summary": snippet,
                        "source": "Web",
                        "published_at": datetime.utcnow().isoformat(),
                        "agent_type": agent_type.value,
                        "agent": name  # Add agent name for filtering
                    })
                else:
                    logger.debug(f"⚠️ Filtered out irrelevant result: {title[:50]}")
            
            # If we filtered out everything, use mock data
            if not news_items:
                logger.warning(f"⚠️ All search results filtered out as irrelevant for {name}, using mock data")
                return {"news_items": self._mock_news_for_agent(agent_type, query=query)}
            
            return {
                "news_items": news_items,
                "screenshot": screenshot_url
            }
            
        except Exception as e:
            # Emit error completion event on exception
            logger.exception(f"Exception in {name}: {e}")
            await self._emit_to_websockets(OrchestratorEvent(
                event_type="agent_update",
                agent_type=agent_type,
                data={"status": "completed", "progress": 100, "task": f"{name} completed with errors: {str(e)[:50]}", "sources": 0}
            ))
            return {"news_items": self._mock_news_for_agent(agent_type, query=query)}

    async def _generate_contextual_images(self, query: str) -> list[str]:
        """Generate 3 distinct images for different sections."""
        from app.core.config import get_settings
        
        settings = get_settings()
        
        # Check if API key is configured
        if not settings.freepik_api_key:
            logger.warning("⚠️ Freepik API key not configured. Skipping image generation.")
            return []
        
        # We'll generate 3 images in parallel
        # 1. Main Cover: Dramatic, overall theme
        # 2. Economy/Market: Charts, financial visual
        # 3. Global/Political: Map or political meeting style
        
        prompts = [
            f"Editorial news photo, highly detailed, dramatic lighting: {query}",
            f"Financial data visualization, stock market charts, economic growth graph related to: {query}",
            f"Political map or diplomatic meeting, professional news photography: {query}",
        ]
        
        async def gen_one(prompt: str, index: int):
            try:
                logger.info(f"🎨 Generating image {index + 1}/3: {prompt[:50]}...")
                
                # Emit progress update
                await self._emit_to_websockets(OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FREEPIK,
                    data={"status": "running", "progress": (index * 33), "task": f"Generating image {index + 1}/3..."}
                ))
                
                res = await freepik_agent.generate_image(
                    prompt=prompt,
                    negative_prompt="text, watermark, logo, blurry, low quality, cartoon",
                    guidance_scale=8.0,
                    num_images=1,
                    image_size="landscape_4_3",
                )
                
                if res.status == "error":
                    logger.error(f"❌ Image generation failed: {res.error}")
                    await self._emit_to_websockets(OrchestratorEvent(
                        event_type="agent_update",
                        agent_type=AgentType.FREEPIK,
                        data={"status": "error", "progress": ((index + 1) * 33), "task": f"Image {index + 1} failed: {res.error}"}
                    ))
                    return None
                
                if res.base64_data:
                    logger.info("✅ Image generated (base64)")
                    await self._emit_to_websockets(OrchestratorEvent(
                        event_type="agent_update",
                        agent_type=AgentType.FREEPIK,
                        data={"status": "completed", "progress": ((index + 1) * 33), "task": f"Image {index + 1} completed"}
                    ))
                    return f"data:image/png;base64,{res.base64_data}"
                
                if res.image_url:
                    logger.info(f"✅ Image generated (URL): {res.image_url[:50]}...")
                    await self._emit_to_websockets(OrchestratorEvent(
                        event_type="agent_update",
                        agent_type=AgentType.FREEPIK,
                        data={"status": "completed", "progress": ((index + 1) * 33), "task": f"Image {index + 1} completed"}
                    ))
                    return res.image_url
                
                logger.warning("⚠️ Image generation returned no data")
                return None
            except Exception as e:
                logger.exception(f"❌ Exception during image generation: {e}")
                await self._emit_to_websockets(OrchestratorEvent(
                    event_type="agent_update",
                    agent_type=AgentType.FREEPIK,
                    data={"status": "error", "progress": ((index + 1) * 33), "task": f"Image {index + 1} error: {str(e)}"}
                ))
                return None

        # Emit start event
        await self._emit_to_websockets(OrchestratorEvent(
            event_type="agent_update",
            agent_type=AgentType.FREEPIK,
            data={"status": "running", "progress": 0, "task": "Starting image generation..."}
        ))
        
        # Run 3 generations in parallel
        tasks = [gen_one(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None and exceptions
        valid_results = []
        for r in results:
            if isinstance(r, str) and r:
                valid_results.append(r)
            elif isinstance(r, Exception):
                logger.error(f"❌ Image generation exception: {r}")
        
        logger.info(f"🎨 Successfully generated {len(valid_results)}/{len(prompts)} images")
        return valid_results

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

        # Convert news items to articles format expected by frontend
        articles = []
        for idx, news_item in enumerate(news[:10]):  # Limit to 10 articles
            articles.append({
                "id": news_item.get("id", str(uuid4())),
                "title": news_item.get("title", "News Update"),
                "content": news_item.get("summary", news_item.get("snippet", f"Coverage of {query}.")),
                "summary": news_item.get("summary", news_item.get("snippet", ""))[:150] + "...",
                "category": news_item.get("agent_type", "general").lower().replace("_", " "),
                "importance": 5 - (idx // 2),  # Decrease importance for later articles
                "image": news_item.get("image") or news_item.get("screenshot"),  # Support both fields
            })
        
        # Ensure we have at least one article (the headline)
        if not articles:
            articles = [{
                "id": str(uuid4()),
                "title": headline,
                "content": f"Comprehensive coverage of {query}. Market analysis and global reactions.",
                "summary": f"Full coverage of {query}...",
                "category": "headline",
                "importance": 5,
            }]
        
        return {
            "paper_id": str(uuid4()),
            "date": tomorrow.strftime("%B %d, %Y"),
            "headline": headline,
            "subheadline": f"Comprehensive coverage: US, Europe, and Global Markets react to {query[:30]}...",
            "query": query,
            "cover_image_url": final_images[0],
            "secondary_image_url": final_images[1],
            "tertiary_image_url": final_images[2],
            "articles": articles,
            "market_snapshot": [
                {
                    "asset": asset.get("asset", "Market"),
                    "value": asset.get("projected_value", asset.get("current_value", 1000)),
                    "current_value": asset.get("current_value", 1000),
                    "projected_value": asset.get("projected_value", asset.get("current_value", 1000)),
                    "change": asset.get("change", 0),
                    "changePercent": asset.get("change_percent", 0),
                    "change_percent": asset.get("change_percent", 0),
                    "history": asset.get("history", []),
                }
                for asset in sim.get("assets", [])
            ],
            "trending_topics": self._extract_trending_topics(query, news, sim),
            "news_context": news,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _extract_trending_topics(self, query: str, news: list, sim: dict) -> list[dict]:
        """Extract trending topics from query, news, and simulation data."""
        topics = []
        query_lower = query.lower()
        
        # Extract key themes from query
        theme_keywords = {
            "market": ["market", "stock", "trading", "invest", "equity", "dow", "nasdaq", "sp500"],
            "geopolitical": ["war", "conflict", "tension", "sanction", "diplomat", "treaty", "alliance", "attack"],
            "tech": ["tech", "technology", "ai", "silicon", "startup", "innovation", "digital"],
            "economy": ["economy", "gdp", "inflation", "recession", "growth", "unemployment", "fed", "central bank"],
            "energy": ["oil", "gas", "energy", "petrol", "crude", "fossil"],
            "crypto": ["bitcoin", "crypto", "blockchain", "ethereum", "digital currency"],
            "black swan": ["crisis", "crash", "collapse", "surge", "spike", "unexpected", "shock"],
        }
        
        # Find matching themes
        matched_themes = []
        for theme, keywords in theme_keywords.items():
            if any(kw in query_lower for kw in keywords):
                matched_themes.append(theme.title())
        
        # Extract from news titles
        news_keywords = set()
        for item in news[:5]:  # Check first 5 news items
            title = item.get("title", "").lower()
            # Extract capitalized words (likely entities/topics)
            words = title.split()
            for word in words:
                if len(word) > 4 and word[0].isupper():
                    news_keywords.add(word.title())
        
        # Build topics list
        # Add query-based themes
        for theme in matched_themes[:3]:  # Max 3 from query
            sentiment = -0.3 if "black swan" in theme.lower() or "crisis" in query_lower else 0.2
            topics.append({
                "topic": f"#{theme.replace(' ', '')}",
                "sentiment": sentiment,
                "mentions": random.randint(800, 2000)
            })
        
        # Add news-derived topics (if we have space)
        for keyword in list(news_keywords)[:2]:  # Max 2 from news
            if len(topics) < 5:  # Keep total under 5
                topics.append({
                    "topic": keyword,
                    "sentiment": random.uniform(-0.2, 0.3),
                    "mentions": random.randint(500, 1500)
                })
        
        # Add market-related topic if we have simulation data
        if sim.get("assets") and len(topics) < 5:
            main_asset = sim["assets"][0] if sim["assets"] else {}
            asset_name = main_asset.get("asset", "Markets")
            topics.append({
                "topic": asset_name,
                "sentiment": -0.1 if main_asset.get("change_percent", 0) < 0 else 0.2,
                "mentions": random.randint(1000, 2500)
            })
        
        # Fallback if no topics found
        if not topics:
            topics = [{"topic": "#BreakingNews", "sentiment": 0.0, "mentions": 1200}]
        
        return topics[:5]  # Return max 5 topics

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


    def _is_simulation_query(self, query: str) -> bool:
        """Check if query is asking for a hypothetical simulation."""
        q = query.lower()
        triggers = ["what if", "simulate", "suppose", "imagine", "what happens if", "scenario"]
        return any(t in q for t in triggers)

    async def _generate_synthetic_news(self, agent_type: AgentType, query: str) -> dict:
        """Generate a synthetic news item for a simulation scenario."""
        # Remove trigger words to get the core scenario
        clean_query = query.lower().replace("what if", "").replace("simulate", "").replace("what happens if", "").strip()
        
        # Call Fabricate Agent
        try:
            from app.agents.fabricate import fabricate_agent
            headline = await fabricate_agent.generate_headline(clean_query.title(), context=agent_type.value)
        except Exception as e:
            logger.error(f"❌ Fabricate headline generation failed: {e}")
            headline = f"Simulation: {clean_query.title()} Scenario Active"

        return {
            "id": str(uuid4()),
            "title": headline,
            "url": "#simulation",
            "summary": f"Detailed simulation report regarding the hypothetical scenario: {clean_query}. This is a generated projection of potential outcomes.",
            "source": "Fabricate Engine (Simulation)",
            "published_at": (datetime.utcnow() + timedelta(hours=random.randint(1, 24))).isoformat(),
            "agent_type": agent_type.value
        }

    def _mock_news_for_agent(self, agent_type: AgentType, query: str = "") -> list[dict]:
        """Generate realistic-looking mock news when search fails."""
        # Use simple heuristics to generate a title based on the query
        topic = "Market"
        if query:
            # Extract main topic from query (e.g. "bitcoin latest news..." -> "Bitcoin")
            words = query.split()
            if len(words) > 0:
                topic = words[0].title()
                
        titles = {
            AgentType.YUTORI_NEWS: f"Breaking: {topic} Market Movements Analysis",
            AgentType.YUTORI_SENTIMENT: f"Social Sentiment Shifts on {topic}",
            AgentType.YUTORI_ANALYSIS: f"Expert Forecast: {topic} Volatility Ahead",
            AgentType.YUTORI_TARGET: f"Local Impact: {topic} Policy Changes",
            AgentType.YUTORI_GLOBAL: f"Global Markets React to {topic} News",
            AgentType.YUTORI_ECON: f"Economic Outlook: {topic} Implications",
        }
        
        headline = titles.get(agent_type, f"Analysis: {topic} Update")
        
        return [{
            "id": str(uuid4()),
            "title": headline,
            "url": "#",
            "summary": f"Comprehensive analysis and live updates regarding {topic}. Market participants are monitoring the situation closely as volatility increases.",
            "source": "Global Wire (Simulated)",
            "published_at": datetime.utcnow().isoformat(),
            "agent_type": agent_type.value
        }]

    async def _run_fabricate_mock(self, query: str, mode: str, news_context: list = None) -> AsyncGenerator[OrchestratorEvent, None]:
        from app.agents.market_data import market_data_agent
        
        if news_context is None:
            news_context = []
        
        try:
            # Step 1: Analyze market context (Real Data)
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.FABRICATE,
                data={"status": "running", "progress": 10, "task": "Fetching real market data..."}
            )
            
            # Use news context to refine ticker search if available
            search_query = query
            if news_context:
                # Extract potential entities from news titles to improve search
                keywords = [n["title"] for n in news_context[:2]]
                search_query = f"{query} {' '.join(keywords)}"

            try:
                market_context = await market_data_agent.get_market_context(search_query)
                if "error" in market_context:
                    logger.warning(f"⚠️ Market data error: {market_context.get('error')}, retrying with S&P 500")
                    # Retry with explicit S&P 500 query
                    market_context = await market_data_agent.get_market_context("sp500")
                    if "error" in market_context or not market_context.get("current_price"):
                        logger.error(f"❌ Failed to fetch real market data, using mock")
                        market_context = {"name": "S&P 500", "current_price": 5000, "history": []}
            except Exception as e:
                logger.exception(f"❌ Error fetching market data: {e}")
                # Try one more time with explicit S&P 500
                try:
                    market_context = await market_data_agent.get_market_context("sp500")
                except:
                    logger.error(f"❌ Complete failure, using realistic fallback")
                    market_context = {"name": "S&P 500", "current_price": 5000, "history": []}
            
            ticker_name = market_context.get("name", "Market")
            
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.FABRICATE,
                data={"status": "running", "progress": 40, "task": f"Analyzing {ticker_name} history..."}
            )

            # Step 2: Run Simulation (Projection)
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.FABRICATE,
                data={"status": "running", "progress": 80, "task": "Running predictive models..."}
            )
            
            simulation_result = self._generate_simulation(query, market_context)
            
            yield OrchestratorEvent(
                event_type="agent_update",
                agent_type=AgentType.FABRICATE,
                data={"status": "completed", "progress": 100, "result": simulation_result}
            )
        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            logger.exception(f"❌ Error in _run_fabricate_mock: {error_msg}")
            yield OrchestratorEvent(
                event_type="error",
                agent_type=AgentType.FABRICATE,
                data={"message": f"Simulation error: {error_msg}", "error_type": type(e).__name__}
            )

    
    def _generate_simulation(self, query: str, market_ctx: dict, news_context: list = []) -> dict:
        """Generate simulation data using real historical context and news findings."""
        # Combine query + news headlines for sentiment analysis
        text_corpus = query.lower()
        if news_context:
            text_corpus += " " + " ".join([n["title"].lower() for n in news_context])
            
        # Sentiment Keywords
        negative_keywords = ["crash", "spike", "surge", "collapse", "tariff", "war", "crisis", "attack", "tension", "nuclear", "sanction", "ban"]
        positive_keywords = ["boom", "growth", "record", "peace", "deal", "agreement", "stimulus", "cut", "rally"]
        
        # Simple Sentiment Scoring
        neg_score = sum(1 for w in negative_keywords if w in text_corpus)
        pos_score = sum(1 for w in positive_keywords if w in text_corpus)
        
        # Determine Direction
        if "oil" in text_corpus or "gold" in text_corpus:
            # Commodities often go UP in crisis
            direction = 1 if neg_score > pos_score else -1
        else:
            # Equities generally go DOWN in crisis
            direction = -1 if neg_score > pos_score else 1
            
        # Determine Volatility based on "loudness" of news
        volatility_multiplier = 1.0 + (neg_score + pos_score) * 0.2
        volatility = 0.02 * volatility_multiplier
        
        main_asset_name = market_ctx.get("name", "Market Index")
        main_asset_hist = market_ctx.get("history", [])
        current_val = market_ctx.get("current_price", 1000)
        
        # Projection Logic
        projected_change = direction * (volatility * 100)
        
        # Cap extreme moves for realism unless "nuclear" involved
        if "nuclear" not in text_corpus and abs(projected_change) > 15:
            projected_change = 15 * (1 if projected_change > 0 else -1)
            
        projected_val = current_val * (1 + (projected_change / 100))

        
        # Build the main asset result
        main_asset = {
            "asset": main_asset_name,
            "current_value": round(float(current_val), 2),
            "projected_value": round(float(projected_val), 2),
            "change": round(float(projected_val) - float(current_val), 2),
            "change_percent": round(float(projected_change), 2),
            "history": main_asset_hist, # REAL HISTORY
            "is_primary": True
        }
        
        # Generate correlated assets (Mocked relative to main)
        assets = [main_asset]
        
        # Helper to add correlated asset
        def add_correlated(name, base_price, correlation):
            # If main asset drops 5%, and corr is 0.5, this drops 2.5%
            pct = projected_change * correlation
            # Add some noise
            pct += random.gauss(0, 1.0)
            val = base_price * (1 + pct/100)
            assets.append({
                "asset": name,
                "current_value": float(base_price),
                "projected_value": round(val, 2),
                "change": round(val - base_price, 2),
                "change_percent": round(pct, 2),
                "is_primary": False
            })

        # Add standard context assets
        if "Gold" not in main_asset_name:
            add_correlated("Gold", 2050, -0.6 if neg_score > pos_score else 0.1) # Inverse to crisis
        
        if "VIX" not in main_asset_name:
            # High neg_score (crisis) -> VIX spike
            add_correlated("VIX", 15, -4.0 if neg_score > pos_score else 0.5) 
            
        if "EUR/USD" not in main_asset_name:
            add_correlated("EUR/USD", 1.08, 0.3)

        return {"scenario": query, "assets": assets}

    async def _generate_single_image(self, prompt: str) -> str:
        """Helper to generate a single image for analysis items."""
        try:
            # Add some style keywords automatically
            styled_prompt = f"{prompt}, editorial, detailed, 8k, professional"
            result = await freepik_agent.generate_image(styled_prompt)
            # Return URL or Base64 string, not the object
            if result.image_url:
                return result.image_url
            elif result.base64_data:
                return f"data:image/png;base64,{result.base64_data}"
            return ""
        except Exception as e:
            logger.error(f"❌ Single image generation failed: {e}")
            return ""
    
    def _identify_relevant_assets(self, query: str) -> list[str]:
        """Identify which market assets/tickers are relevant to the query."""
        q_lower = query.lower()
        assets = []
        
        # Crypto mentions
        if any(word in q_lower for word in ["bitcoin", "btc", "crypto", "cryptocurrency"]):
            assets.append("BTC-USD")
        
        # Stock indices
        if any(word in q_lower for word in ["sp500", "s&p", "spy", "sp 500"]):
            assets.append("SPY")
        elif any(word in q_lower for word in ["nasdaq", "tech", "technology"]):
            assets.append("QQQ")
        elif any(word in q_lower for word in ["dow", "dow jones"]):
            assets.append("DIA")
        
        # Country-specific
        if any(word in q_lower for word in ["japan", "japanese", "nikkei"]):
            assets.append("EWJ")  # Japan ETF
        if any(word in q_lower for word in ["china", "chinese"]):
            assets.append("FXI")  # China ETF
        if any(word in q_lower for word in ["europe", "eu", "euro"]):
            assets.append("VGK")  # Europe ETF
        
        # Commodities
        if any(word in q_lower for word in ["gold", "precious metal"]):
            assets.append("GC=F")
        if any(word in q_lower for word in ["oil", "crude", "petroleum"]):
            assets.append("CL=F")
        
        # VIX (volatility/fear)
        if any(word in q_lower for word in ["volatility", "fear", "panic", "crisis", "crash"]):
            assets.append("^VIX")
        
        # Default assets if none found
        if not assets:
            # Always include major indices
            assets = ["SPY", "BTC-USD", "QQQ"]
        
        # Limit to 6 assets max for performance
        return assets[:6]

orchestrator = AgentOrchestrator()
