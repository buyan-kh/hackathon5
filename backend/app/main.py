"""
Tomorrow's Paper - FastAPI Backend

AI-powered market simulation and news generation platform.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import chat, simulation, paper
from app.api.websocket import websocket_endpoint

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Set specific loggers
logging.getLogger("app.agents").setLevel(logging.DEBUG)
logging.getLogger("app.api").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered market simulation and news generation API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(paper.router, prefix="/api")

# WebSocket endpoint
app.add_api_websocket_route("/ws", websocket_endpoint)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/test-apis")
async def test_apis():
    """Test all API connections and show detailed results."""
    from app.agents.yutori import yutori_agent
    from app.agents.fabricate import fabricate_agent
    from app.agents.freepik import freepik_agent
    
    results = {
        "yutori": {"status": "testing..."},
        "fabricate": {"status": "testing..."},
        "freepik": {"status": "testing..."},
        "config": {
            "yutori_api_key": "***" + settings.yutori_api_key[-4:] if settings.yutori_api_key else "NOT SET",
            "tonic_api_key": "***" + settings.tonic_api_key[-4:] if settings.tonic_api_key else "NOT SET",
            "freepik_api_key": "***" + settings.freepik_api_key[-4:] if settings.freepik_api_key else "NOT SET",
            "gemini_api_key": "***" + settings.gemini_api_key[-4:] if settings.gemini_api_key else "NOT SET",
        }
    }
    
    # Test Yutori
    logger.info("Testing Yutori API...")
    try:
        yutori_result = await yutori_agent.research("test connection")
        results["yutori"] = {
            "status": yutori_result.status,
            "task_id": yutori_result.task_id,
            "error": yutori_result.error,
        }
    except Exception as e:
        results["yutori"] = {"status": "error", "error": str(e)}
    
    # Test Fabricate
    logger.info("Testing Tonic Fabricate API...")
    try:
        fabricate_result = await fabricate_agent.test_connection()
        results["fabricate"] = fabricate_result
    except Exception as e:
        results["fabricate"] = {"status": "error", "error": str(e)}
    
    # Test Freepik (skip actual generation to save API credits)
    logger.info("Checking Freepik API config...")
    results["freepik"] = {
        "status": "configured" if settings.freepik_api_key else "not configured",
        "note": "Image generation available. Use /test-freepik to test.",
    }
    
    return results


@app.get("/test-freepik")
async def test_freepik():
    """Test Freepik image generation (uses API credits)."""
    from app.agents.freepik import freepik_agent
    
    logger.info("Testing Freepik image generation...")
    
    result = await freepik_agent.generate_image(
        prompt="A professional minimalist blue gradient background for a financial news article",
        num_images=1,
    )
    
    return {
        "status": result.status,
        "image_url": result.image_url,
        "has_base64": result.base64_data is not None,
        "error": result.error,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
