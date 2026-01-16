from fastapi import APIRouter
from uuid import uuid4

from app.models.simulation import SimulationRequest, SimulationResult

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/", response_model=SimulationResult)
async def create_simulation(request: SimulationRequest):
    """
    Create and run a new market simulation.
    
    For streaming updates, use the WebSocket endpoint instead.
    """
    simulation_id = uuid4()
    
    # Mock response - real implementation would trigger Tonic Fabricate
    return SimulationResult(
        id=simulation_id,
        query=request.query,
        scenario=f"Simulation based on: {request.query}",
        status="completed",
        progress=100.0,
        assets=[],
        insights=[
            "Market volatility expected to increase",
            "Safe-haven assets likely to benefit",
        ],
    )


@router.get("/{simulation_id}")
async def get_simulation(simulation_id: str):
    """Get simulation results by ID."""
    return {
        "id": simulation_id,
        "status": "completed",
        "assets": [],
    }
