from fastapi import APIRouter
from uuid import uuid4
from datetime import datetime, timedelta

from app.models.paper import Paper, Article, MarketSnapshot, TrendingTopic

router = APIRouter(prefix="/paper", tags=["paper"])


@router.post("/generate")
async def generate_paper(query: str, mode: str = "paper"):
    """
    Generate Tomorrow's Paper based on a query.
    
    For streaming updates, use the WebSocket endpoint instead.
    """
    paper_id = uuid4()
    tomorrow = datetime.utcnow() + timedelta(days=1)
    
    # Mock paper - real implementation would use all agents
    paper = Paper(
        id=paper_id,
        date=tomorrow.strftime("%B %d, %Y"),
        headline="Markets React to Scenario Analysis",
        subheadline=f"AI-powered simulation explores: {query}",
        query=query,
        scenario=f"Scenario based on: {query}",
        articles=[
            Article(
                title="Lead Story Placeholder",
                content="This is where the main article content would go...",
                summary="A summary of the lead story.",
                category="headline",
                importance=5,
            )
        ],
        market_snapshot=[
            MarketSnapshot(asset="S&P 500", value=5000, change=-150, change_percent=-3.0),
            MarketSnapshot(asset="Gold", value=2100, change=85, change_percent=4.2),
        ],
        trending_topics=[
            TrendingTopic(topic="#MarketAnalysis", sentiment=0.3, mentions=1500),
        ],
    )
    
    return paper


@router.get("/{paper_id}")
async def get_paper(paper_id: str):
    """Get a generated paper by ID."""
    return {
        "id": paper_id,
        "status": "completed",
        "headline": "Mock Paper",
    }


@router.get("/")
async def list_papers():
    """List all generated papers."""
    return {"papers": []}
