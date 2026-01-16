"""
Tonic Fabricate Agent - Synthetic data generation and simulation.

Tonic Fabricate provides:
- Database management
- Data generation (AI-based and rule-based)
- Mock API generation

API Base: https://fabricate.tonic.ai/api/v1
"""

import httpx
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

from app.core.config import get_settings


@dataclass
class SimulationScenario:
    """A market simulation scenario."""
    name: str
    description: str
    variables: dict = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)


@dataclass 
class FabricateResult:
    """Result from Tonic Fabricate."""
    database_id: str | None = None
    status: str = "pending"
    data: list[dict] = field(default_factory=list)
    error: str | None = None


class FabricateAgent:
    """
    Agent for synthetic data generation using Tonic Fabricate API.
    
    Used for:
    - Generating simulated market data
    - Creating scenario-based projections
    - Fabricating realistic financial datasets
    
    API Reference: https://docs.tonic.ai/fabricate
    """
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.tonic_api_key
        self.base_url = settings.tonic_base_url
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,  # Data generation can take time
        )
    
    async def create_database(
        self, 
        name: str, 
        description: str | None = None,
    ) -> FabricateResult:
        """
        Create a new Fabricate database for simulation data.
        
        Args:
            name: Database name
            description: Optional description
            
        Returns:
            FabricateResult with database details
        """
        payload = {
            "name": name,
            "description": description or f"Simulation database: {name}",
        }
        
        try:
            response = await self.client.post("/databases", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return FabricateResult(
                database_id=data.get("id"),
                status="created",
            )
        except httpx.HTTPError as e:
            return FabricateResult(
                status="error",
                error=f"Database creation error: {str(e)}",
            )
    
    async def generate_data(
        self,
        database_id: str,
        entity: str,
        count: int = 100,
        constraints: dict | None = None,
    ) -> FabricateResult:
        """
        Generate synthetic data for an entity in the database.
        
        Args:
            database_id: Target database ID
            entity: Entity/table name to generate data for
            count: Number of records to generate
            constraints: Optional constraints for data generation
            
        Returns:
            FabricateResult with generated data
        """
        payload = {
            "entity": entity,
            "count": count,
        }
        if constraints:
            payload["constraints"] = constraints
        
        try:
            response = await self.client.post(
                f"/databases/{database_id}/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return FabricateResult(
                database_id=database_id,
                status="completed",
                data=data.get("records", []),
            )
        except httpx.HTTPError as e:
            return FabricateResult(
                database_id=database_id,
                status="error",
                error=f"Data generation error: {str(e)}",
            )
    
    async def generate_market_simulation(
        self,
        scenario: SimulationScenario,
        assets: list[str],
        time_horizon: str = "1w",
    ) -> dict:
        """
        Generate a market simulation based on a scenario.
        
        This is a higher-level method that uses Fabricate to generate
        realistic market data projections.
        
        Args:
            scenario: The simulation scenario
            assets: List of assets to simulate (e.g., ["S&P 500", "Gold", "VIX"])
            time_horizon: Simulation period (1d, 1w, 1m, 3m, 1y)
            
        Returns:
            Simulation results with projected values
        """
        # Map time horizon to data points
        time_points = {
            "1d": 24,    # Hourly
            "1w": 7,     # Daily
            "1m": 30,    # Daily
            "3m": 90,    # Daily
            "1y": 52,    # Weekly
        }
        
        num_points = time_points.get(time_horizon, 7)
        
        # For now, generate simulated data
        # In production, this would use Fabricate's AI data generation
        results = []
        
        for asset in assets:
            # Generate time series data
            base_value = self._get_base_value(asset)
            volatility = self._get_volatility(asset, scenario)
            
            timeline = []
            current_value = base_value
            
            for i in range(num_points):
                # Apply scenario-based adjustment
                change = self._calculate_change(asset, scenario, volatility, i)
                current_value *= (1 + change)
                
                timeline.append({
                    "date": f"Day {i + 1}",
                    "value": round(current_value, 2),
                    "projected": True,
                })
            
            projected_value = timeline[-1]["value"] if timeline else base_value
            change_pct = ((projected_value - base_value) / base_value) * 100
            
            results.append({
                "asset": asset,
                "current_value": base_value,
                "projected_value": projected_value,
                "change": round(projected_value - base_value, 2),
                "change_percent": round(change_pct, 2),
                "timeline": timeline,
            })
        
        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "time_horizon": time_horizon,
            "assets": results,
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def _get_base_value(self, asset: str) -> float:
        """Get baseline value for an asset."""
        defaults = {
            "S&P 500": 5000,
            "NASDAQ": 16000,
            "Gold": 2050,
            "Oil": 75,
            "VIX": 15,
            "Bitcoin": 42000,
            "10Y Treasury": 4.2,
        }
        return defaults.get(asset, 100)
    
    def _get_volatility(self, asset: str, scenario: SimulationScenario) -> float:
        """Get volatility factor based on asset and scenario."""
        base_volatility = {
            "S&P 500": 0.02,
            "NASDAQ": 0.03,
            "Gold": 0.015,
            "Oil": 0.04,
            "VIX": 0.1,
            "Bitcoin": 0.08,
            "10Y Treasury": 0.01,
        }
        
        # Increase volatility for high-impact scenarios
        scenario_multiplier = scenario.variables.get("impact_level", 1.0)
        
        return base_volatility.get(asset, 0.02) * scenario_multiplier
    
    def _calculate_change(
        self, 
        asset: str, 
        scenario: SimulationScenario, 
        volatility: float,
        time_step: int,
    ) -> float:
        """Calculate price change based on scenario."""
        import random
        
        # Base random change
        random_change = random.gauss(0, volatility)
        
        # Scenario-driven directional bias
        direction = scenario.variables.get("market_direction", 0)
        
        # Asset-specific scenario effects
        asset_effects = scenario.variables.get("asset_effects", {})
        asset_bias = asset_effects.get(asset, 0)
        
        return random_change + (direction * 0.01) + asset_bias
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton instance
fabricate_agent = FabricateAgent()
