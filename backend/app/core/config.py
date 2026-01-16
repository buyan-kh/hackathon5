from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Tomorrow's Paper API"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # API Keys (from .env)
    yutori_api_key: str = ""
    tonic_api_key: str = ""
    freepik_api_key: str = ""
    gemini_api_key: str = ""  # For LLM if needed
    
    # API Base URLs
    yutori_base_url: str = "https://api.yutori.com/v1"
    tonic_base_url: str = "https://fabricate.tonic.ai/api/v1"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
