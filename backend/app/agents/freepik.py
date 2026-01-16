"""
Freepik Agent - AI Image generation.

Freepik API:
- Endpoint: https://api.freepik.com/v1/ai/text-to-image
- Auth: x-freepik-api-key header
- Docs: https://www.freepik.com/api
"""

import httpx
import logging
from dataclasses import dataclass
from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GeneratedImage:
    """Result from image generation."""
    image_url: str | None = None
    image_id: str | None = None
    prompt: str = ""
    status: str = "pending"
    error: str | None = None
    base64_data: str | None = None


class FreepikAgent:
    """Agent for image generation using Freepik API."""
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.freepik_api_key
        self.base_url = "https://api.freepik.com/v1"
        
        logger.info(f"🔑 Freepik API Key configured: {'Yes' if self.api_key else 'No'}")
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-freepik-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=120.0,
        )
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        image_size: str = "square_1_1",
        styling: dict | None = None,
    ) -> GeneratedImage:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            negative_prompt: Elements to exclude from the image
            guidance_scale: How closely to follow the prompt (1-20)
            num_images: Number of images to generate (1-4)
            image_size: Size preset (square_1_1, landscape_4_3, portrait_3_4, etc.)
            styling: Optional styling parameters
            
        Returns:
            GeneratedImage with URL or base64 data
        """
        logger.info(f"🎨 [FREEPIK] Generating image: {prompt[:50]}...")
        
        payload = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "image": {
                "size": image_size,
            },
        }
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        
        if styling:
            payload["styling"] = styling
        
        try:
            response = await self.client.post("/ai/text-to-image", json=payload)
            
            logger.info(f"🎨 [FREEPIK] Response status: {response.status_code}")
            logger.debug(f"🎨 [FREEPIK] Response: {response.text[:500]}")
            
            response.raise_for_status()
            data = response.json()
            
            # Extract image data from response
            images = data.get("data", [])
            if images:
                image = images[0]
                image_url = image.get("url") or image.get("base64")
                
                logger.info(f"✅ [FREEPIK] Image generated successfully")
                
                return GeneratedImage(
                    image_url=image_url if not image.get("base64") else None,
                    base64_data=image.get("base64"),
                    prompt=prompt,
                    status="completed",
                )
            else:
                logger.warning(f"⚠️ [FREEPIK] No images in response")
                return GeneratedImage(
                    prompt=prompt,
                    status="error",
                    error="No images returned",
                )
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"❌ [FREEPIK] API Error: {error_msg}")
            return GeneratedImage(
                prompt=prompt,
                status="error",
                error=error_msg,
            )
        except httpx.HTTPError as e:
            error_msg = str(e)
            logger.error(f"❌ [FREEPIK] Connection error: {error_msg}")
            return GeneratedImage(
                prompt=prompt,
                status="error",
                error=error_msg,
            )
    
    async def generate_cover_image(
        self,
        headline: str,
        scenario: str,
        market_sentiment: str = "neutral",
    ) -> GeneratedImage:
        """Generate a cover image for Tomorrow's Paper."""
        
        sentiment_moods = {
            "bullish": "optimistic golden sunrise, warm colors, upward trending lines",
            "bearish": "dramatic stormy sky, cool blue tones, intense atmosphere",
            "neutral": "professional business setting, balanced lighting, modern office",
        }
        
        mood = sentiment_moods.get(market_sentiment, sentiment_moods["neutral"])
        
        prompt = f"""Professional financial news cover image. 
Theme: {headline}
Context: {scenario}
Mood: {mood}
Style: Editorial photography, high quality, cinematic lighting, no text overlays."""

        return await self.generate_image(
            prompt=prompt,
            negative_prompt="text, watermark, logo, blurry, low quality, cartoon",
            guidance_scale=8.0,
            num_images=1,
            image_size="landscape_4_3",
        )
    
    async def test_connection(self) -> dict:
        """Test API connection with a simple request."""
        logger.info("🧪 [FREEPIK] Testing API connection...")
        
        # Use a simple prompt to test
        result = await self.generate_image(
            prompt="A simple blue gradient background",
            num_images=1,
        )
        
        if result.status == "completed":
            return {"status": "success", "message": "API connected"}
        else:
            return {"status": "error", "error": result.error}
    
    async def close(self):
        await self.client.aclose()


# Singleton instance
freepik_agent = FreepikAgent()
