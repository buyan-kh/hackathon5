from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4


class SimulationRequest(BaseModel):
    """Request to run a simulation."""
    query: str
    scenario_type: str = "market"
    time_horizon: str = "1w"  # 1d, 1w, 1m, 3m, 1y
    include_news_context: bool = True


class TimeSeriesPoint(BaseModel):
    """A single point in a time series."""
    date: str
    value: float


class AssetSimulation(BaseModel):
    """Simulation results for a single asset."""
    asset: str
    asset_type: str  # stock, crypto, commodity, index
    current_value: float
    projected_value: float
    change: float
    change_percent: float
    confidence: float
    timeline: list[TimeSeriesPoint] = []


class SimulationResult(BaseModel):
    """Complete simulation results."""
    id: UUID = Field(default_factory=uuid4)
    query: str
    scenario: str
    status: str = "pending"  # pending, running, completed, error
    progress: float = 0.0
    assets: list[AssetSimulation] = []
    insights: list[str] = []
    news_context: list[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class SimulationUpdate(BaseModel):
    """Streaming update during simulation."""
    simulation_id: UUID
    status: str
    progress: float
    message: Optional[str] = None
    partial_results: Optional[list[AssetSimulation]] = None
