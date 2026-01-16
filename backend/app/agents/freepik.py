"""
Freepik Agent - Image and content generation.

Freepik provides AI-powered image generation capabilities.
Used for generating cover images and visuals for Tomorrow's Paper.
"""

import httpx
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

from app.core.config import get_settings


@dataclass
class GeneratedImage:
    """Result from image generation."""
    image_url: str | None = None
    image_id: str | None = None
    prompt: str = ""
    status: str = "pending"
    error: str | None = None


class FreepikAgent:
    """
    Agent for image generation using Freepik API.
    
    Used for:
    - Generating cover images for the paper
    - Creating infographics
    - Generating article illustrations
    """
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.freepik_api_key
        # Freepik API base URL (adjust based on actual API)
        self.base_url = "https://api.freepik.com/v1"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
    
    async def generate_image(
        self,
        prompt: str,
        style: str = "photorealistic",
        aspect_ratio: str = "16:9",
        quality: str = "high",
    ) -> GeneratedImage:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            style: Image style (photorealistic, illustration, etc.)
            aspect_ratio: Aspect ratio (16:9, 1:1, 4:3, etc.)
            quality: Image quality (low, medium, high)
            
        Returns:
            GeneratedImage with URL and details
        """
        payload = {
            "prompt": prompt,
            "style": style,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
        }
        
        try:
            response = await self.client.post("/ai/image/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return GeneratedImage(
                image_url=data.get("url"),
                image_id=data.get("id"),
                prompt=prompt,
                status="completed",
            )
        except httpx.HTTPError as e:
            return GeneratedImage(
                prompt=prompt,
                status="error",
                error=f"Image generation error: {str(e)}",
            )
    
    async def generate_cover_image(
        self,
        headline: str,
        scenario: str,
        market_sentiment: str = "neutral",
    ) -> GeneratedImage:
        """
        Generate a cover image for Tomorrow's Paper.
        
        Args:
            headline: The paper's main headline
            scenario: The simulation scenario description
            market_sentiment: Overall market sentiment (bullish, bearish, neutral)
            
        Returns:
            GeneratedImage suitable for paper cover
        """
        # Craft a prompt based on the headline and scenario
        sentiment_colors = {
            "bullish": "warm golden light, optimistic atmosphere",
            "bearish": "dramatic shadows, tense atmosphere, red accents",
            "neutral": "balanced lighting, professional atmosphere, blue tones",
        }
        
        color_mood = sentiment_colors.get(market_sentiment, sentiment_colors["neutral"])
        
        prompt = f"""
        Professional newspaper cover image for financial news.
        Theme: {headline}
        Context: {scenario}
        Style: Modern editorial photography, {color_mood}.
        High quality, sharp focus, dramatic composition.
        No text overlays.
        """.strip()
        
        return await self.generate_image(
            prompt=prompt,
            style="photorealistic",
            aspect_ratio="16:9",
            quality="high",
        )
    
    async def generate_infographic(
        self,
        title: str,
        data_points: list[dict],
        chart_type: str = "bar",
    ) -> GeneratedImage:
        """
        Generate an infographic visualization.
        
        Args:
            title: Infographic title
            data_points: Data to visualize
            chart_type: Type of chart (bar, line, pie, etc.)
            
        Returns:
            GeneratedImage of the infographic
        """
        data_summary = ", ".join([
            f"{d.get('label', 'Item')}: {d.get('value', 0)}" 
            for d in data_points[:5]
        ])
        
        prompt = f"""
        Clean, modern infographic design.
        Title: {title}
        Data: {data_summary}
        Style: Minimalist, professional, {chart_type} chart visualization.
        Color scheme: Blue gradient accent colors.
        White background, clear typography.
        """
        
        return await self.generate_image(
            prompt=prompt,
            style="illustration",
            aspect_ratio="4:3",
            quality="high",
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton instance
freepik_agent = FreepikAgent()
