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
from duckduckgo_search import DDGS

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
    
    async def deep_read(self, url: str) -> str:
        """
        Level 1 Agentic: Deep Reading
        Uses Jina Reader to convert a URL into clean, LLM-friendly markdown.
        """
        logger.info(f"📖 [YUTORI] Deep reading: {url}")
        try:
            # Jina Reader is a free API that converts any URL to markdown
            reader_url = f"https://r.jina.ai/{url}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(reader_url)
                if response.status_code == 200:
                    content = response.text
                    # Limit content length to avoid overflowing context
                    return content[:5000] 
                else:
                    logger.warning(f"⚠️ [YUTORI] Jina Reader failed: {response.status_code}")
                    return ""
        except Exception as e:
            logger.error(f"❌ [YUTORI] Deep read error: {e}")
            return ""

    async def take_screenshot(self, url: str) -> str:
        """
        Level 2 Agentic: Visual Scout
        Uses Playwright to take a screenshot of the page.
        Returns a base64 encoded image string.
        """
        logger.info(f"👁️ [YUTORI] Visual scouting: {url}")
        try:
            from playwright.async_api import async_playwright
            import base64
            
            async with async_playwright() as p:
                # Use a specific channel if needed, or default
                browser = await p.chromium.launch(headless=True)
                # Create a new context with a specific viewport for "desktop" feel
                context = await browser.new_context(viewport={"width": 1280, "height": 720})
                page = await context.new_page()
                
                try:
                    await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                    # Wait a tiny bit for animations/popups to settle
                    await page.wait_for_timeout(2000)
                    
                    screenshot_bytes = await page.screenshot(type="jpeg", quality=60)
                    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                    
                    await browser.close()
                    return f"data:image/jpeg;base64,{screenshot_b64}"
                    
                except Exception as page_err:
                     logger.error(f"❌ [YUTORI] Page load error: {page_err}")
                     await browser.close()
                     return ""
                     
        except ImportError:
            logger.error("❌ Playwright not installed. Run `pip install playwright && playwright install`")
            return ""
        except Exception as e:
            logger.error(f"❌ [YUTORI] Visual scout error: {e}")
            return ""

    async def fast_search(self, query: str, max_results: int = 10) -> YutoriResearchResult:
        """Perform a fast web search using DuckDuckGo with multiple fallback strategies."""
        logger.info(f"⚡ [FAST] Searching: {query[:100]}")
        
        import asyncio
        import time
        
    async def deep_read(self, url: str) -> str:
        """
        Level 1 Agentic: Deep Reading
        Uses Jina Reader to convert a URL into clean, LLM-friendly markdown.
        """
        logger.info(f"📖 [YUTORI] Deep reading: {url}")
        try:
            # Jina Reader is a free API that converts any URL to markdown
            reader_url = f"https://r.jina.ai/{url}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(reader_url)
                if response.status_code == 200:
                    content = response.text
                    # Limit content length to avoid overflowing context
                    return content[:5000] 
                else:
                    logger.warning(f"⚠️ [YUTORI] Jina Reader failed: {response.status_code}")
                    return ""
        except Exception as e:
            logger.error(f"❌ [YUTORI] Deep read error: {e}")
            return ""

    # Clean and optimize query
    def clean_query(q: str) -> str:
            # Remove special characters that might cause issues
            q = q.replace('$', '').replace('€', '').replace('£', '').replace('%', '')
            # Remove extra whitespace
            q = ' '.join(q.split())
            # Limit length (DuckDuckGo works better with shorter queries)
            if len(q) > 100:
                words = q.split()
                # Keep first 10 words max
                q = ' '.join(words[:10])
            return q
        
        cleaned_query = clean_query(query)
        
        # Try multiple backends in order of reliability
        backends = ["api", "html", "lite"]
        
        for backend in backends:
            try:
                def run_search():
                    try:
                        # Add small delay to avoid rate limiting
                        time.sleep(0.3)
                        with DDGS() as ddgs:
                            results = list(ddgs.text(cleaned_query, max_results=max_results, backend=backend))
                            logger.debug(f"🔍 [FAST] Backend '{backend}' returned {len(results)} raw results for: {cleaned_query[:50]}")
                            return results
                    except Exception as e:
                        logger.warning(f"⚠️ [FAST] Backend '{backend}' failed: {type(e).__name__}: {str(e)[:100]}")
                        return []
                
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, run_search)
                
                # If we got results, process them
                if results:
                    sources = []
                    for r in results:
                        title = r.get('title') or r.get('text', 'No Title')
                        url = r.get('href') or r.get('url', '#')
                        snippet = r.get('body') or r.get('snippet') or r.get('text', '')
                        
                        # More lenient validation - accept if we have title OR snippet
                        if title and title != 'No Title' and len(title.strip()) > 3:
                            sources.append({
                                "title": title.strip(),
                                "url": url if url else '#',
                                "snippet": (snippet[:500] if snippet else "").strip()
                            })
                    
                    if sources:
                        logger.info(f"✅ [FAST] Found {len(sources)} valid results using backend '{backend}' for: {cleaned_query[:50]}")
                        return YutoriResearchResult(
                            task_id=f"fast-{hash(query)}",
                            status="completed",
                            sources=sources
                        )
                
                # If no results, try next backend
                logger.debug(f"⚠️ [FAST] Backend '{backend}' returned 0 valid results, trying next...")
                
            except Exception as e:
                logger.warning(f"⚠️ [FAST] Backend '{backend}' exception: {type(e).__name__}: {str(e)[:100]}")
                # Add delay before trying next backend
                await asyncio.sleep(0.5)
                continue
        
        # If all backends failed, try simplified query
        logger.warning(f"⚠️ [FAST] All backends failed for '{query[:50]}', trying simplified query...")
        try:
            # Extract key terms - take first 5-7 most important words
            words = cleaned_query.split()
            # Try different simplification strategies
            simplified_queries = [
                " ".join(words[:7]),  # First 7 words
                " ".join(words[:5]),  # First 5 words
                " ".join([w for w in words[:10] if len(w) > 3]),  # Filter short words
            ]
            
            for simple_query in simplified_queries:
                if not simple_query or len(simple_query) < 3:
                    continue
                    
                def run_simple_search():
                    try:
                        time.sleep(0.5)  # Longer delay for retry
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            with DDGS() as ddgs:
                                return list(ddgs.text(simple_query, max_results=max_results, backend="api"))
                    except Exception as e:
                        logger.debug(f"Simplified search failed: {e}")
                        return []
                
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, run_simple_search)
                
                if results:
                    sources = []
                    for r in results:
                        title = r.get('title') or r.get('text', 'No Title')
                        url = r.get('href') or r.get('url', '#')
                        snippet = r.get('body') or r.get('snippet') or r.get('text', '')
                        
                        if title and title != 'No Title' and len(title.strip()) > 3:
                            sources.append({
                                "title": title.strip(),
                                "url": url if url else '#',
                                "snippet": (snippet[:500] if snippet else "").strip()
                            })
                    
                    if sources:
                        logger.info(f"✅ [FAST] Found {len(sources)} results with simplified query: {simple_query[:50]}")
                        return YutoriResearchResult(
                            task_id=f"fast-{hash(query)}",
                            status="completed",
                            sources=sources
                        )
        except Exception as e:
            logger.error(f"❌ [FAST] Simplified search also failed: {type(e).__name__}: {str(e)[:100]}")
        
        # Final fallback: return error with empty sources
        logger.error(f"❌ [FAST] All search attempts failed for: {query[:100]}")
        return YutoriResearchResult(
            task_id="error",
            status="error",
            sources=[],  # Ensure sources is always a list
            error=f"All search backends failed for query: {query[:50]}"
        )

    async def close(self):
        await self.client.aclose()


# Singleton instance
yutori_agent = YutoriAgent()
