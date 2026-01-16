from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4


class Article(BaseModel):
    """A news article in the paper."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    content: str
    summary: str
    category: str  # headline, market, analysis, feature
    importance: int = 1  # 1-5, 5 being most important
    image_url: Optional[str] = None
    sources: list[str] = []


class MarketSnapshot(BaseModel):
    """Market data snapshot for the paper."""
    asset: str
    value: float
    change: float
    change_percent: float


class TrendingTopic(BaseModel):
    """A trending topic from the simulation."""
    topic: str
    sentiment: float  # -1 to 1
    mentions: int


class Paper(BaseModel):
    """The generated Tomorrow's Paper."""
    id: UUID = Field(default_factory=uuid4)
    date: str  # The simulated date for this paper
    headline: str
    subheadline: Optional[str] = None
    cover_image_url: Optional[str] = None
    articles: list[Article] = []
    market_snapshot: list[MarketSnapshot] = []
    trending_topics: list[TrendingTopic] = []
    query: str  # Original user query
    scenario: str  # Brief scenario description
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PaperGenerationStatus(BaseModel):
    """Status update during paper generation."""
    paper_id: UUID
    stage: str  # scouting, simulating, writing, imaging, composing
    progress: float
    message: Optional[str] = None
    partial_paper: Optional[Paper] = None
