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
try:
    from ddgs import DDGS
except ImportError:
    # Fallback for old package name
    from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


@dataclass
class YutoriResearchResult:
    """Result from a Yutori research task."""
    task_id: str
    status: str  # "pending", "running", "completed", "error"
    sources: list = None
    view_url: str = None
    content: str = None
    error: str = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class YutoriAgent:
    """Agent for web research using Yutori API."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.yutori.com/v1" 
        self.api_key = self.settings.yutori_api_key
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else None,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self.logger = logger

    async def create_research_task(self, query: str, max_results: int = 10) -> YutoriResearchResult:
        """Create a new research task."""
        logger.info(f"🔍 [YUTORI] Creating research task: {query[:100]}")
        
        try:
            response = await self.client.post(
                "/research/tasks",
                json={
                    "query": query,
                    "max_results": max_results,
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            task_id = data.get("task_id")
            logger.info(f"✅ [YUTORI] Created task: {task_id}")
            
            return YutoriResearchResult(
                task_id=task_id,
                status="pending",
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
            return q.strip()
        
        cleaned_query = clean_query(query)
        logger.info(f"🔍 [FAST] Cleaned query: '{cleaned_query}'")
        
        # Use 'auto' backend (handles backend selection automatically, recommended)
        try:
            def run_search():
                try:
                    # Add delay to avoid rate limiting
                    time.sleep(1.5)  # Increased delay
                    logger.debug(f"🔍 [FAST] Searching with query: '{cleaned_query}'")
                    
                    # Use 'auto' backend (recommended, handles backend selection automatically)
                    ddgs = DDGS()
                    # Suppress warnings about deprecated backend parameter
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = list(ddgs.text(cleaned_query, max_results=max_results, backend="auto"))
                    
                    logger.debug(f"🔍 [FAST] Backend 'auto' returned {len(results)} raw results for: '{cleaned_query}'")
                    if results:
                        logger.debug(f"🔍 [FAST] First result keys: {list(results[0].keys()) if results else 'none'}")
                    return results
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"❌ [FAST] Search failed: {error_msg}")
                    logger.exception(f"Full exception:")
                    return []
            
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, run_search)
            
            # If we got results, process them
            if results and len(results) > 0:
                sources = []
                for r in results:
                    # Handle different result formats from DDGS
                    title = r.get('title') or r.get('text') or r.get('heading') or 'No Title'
                    url = r.get('href') or r.get('url') or r.get('link') or '#'
                    snippet = r.get('body') or r.get('snippet') or r.get('text') or r.get('description') or ''
                    
                    # More lenient validation - accept if we have title OR snippet
                    if title and title != 'No Title' and len(title.strip()) > 3:
                        sources.append({
                            "title": title.strip(),
                            "url": url if url else '#',
                            "snippet": (snippet[:500] if snippet else "").strip()
                        })
                
                if sources:
                    logger.info(f"✅ [FAST] Found {len(sources)} valid results using backend 'auto' for: {cleaned_query[:50]}")
                    return YutoriResearchResult(
                        task_id=f"fast-{hash(query)}",
                        status="completed",
                        sources=sources
                    )
                else:
                    logger.warning(f"⚠️ [FAST] Got {len(results)} raw results but 0 valid sources after processing for: '{cleaned_query}'")
                    # Log first result structure to debug
                    if results:
                        logger.debug(f"🔍 Sample result keys: {list(results[0].keys()) if results else 'none'}")
                        logger.debug(f"🔍 Sample result: {results[0] if results else 'none'}")
            else:
                logger.warning(f"⚠️ [FAST] Backend 'auto' returned 0 raw results for '{cleaned_query}'")
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ [FAST] Search exception: {error_msg}")
            logger.exception(f"Full exception traceback:")
            results = []
        
        # If all backends failed, try simplified query
        logger.warning(f"⚠️ [FAST] Primary search failed for '{query[:50]}', trying simplified query...")
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
                        time.sleep(1.5)  # Longer delay for retry
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ddgs = DDGS()
                            return list(ddgs.text(simple_query, max_results=max_results, backend="auto"))
                    except Exception as e:
                        logger.debug(f"Simplified search failed: {e}")
                        return []
                
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, run_simple_search)
                
                if results:
                    sources = []
                    for r in results:
                        title = r.get('title') or r.get('text') or r.get('heading') or 'No Title'
                        url = r.get('href') or r.get('url') or r.get('link') or '#'
                        snippet = r.get('body') or r.get('snippet') or r.get('text') or r.get('description') or ''
                        
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
