"""
GuppShupp Frontend Configuration
================================

Configuration settings for the Streamlit frontend.

Author: GuppShupp Team
"""

import os
from dataclasses import dataclass


@dataclass
class FrontendConfig:
    """Frontend configuration settings."""
    
    # Backend API
    api_base_url: str = os.getenv("API_BASE_URL", "https://534da949ffaf.ngrok-free.app")
    api_version: str = "v1"
    
    # Timeouts
    heartbeat_timeout_seconds: int = 30  # Max time without heartbeat
    request_timeout_seconds: int = 120   # Max total request time
    
    # Audio
    max_audio_size_mb: int = 100
    supported_formats: tuple = ("wav", "mp3", "webm", "ogg")
    
    # UI
    page_title: str = "GuppShupp - Voice Companion"
    page_icon: str = "🎙️"
    theme_primary_color: str = "#6366F1"  # Indigo
    theme_background_color: str = "#0F0F23"  # Dark
    
    @property
    def api_url(self) -> str:
        """Get full API URL."""
        return f"{self.api_base_url}/api/{self.api_version}"


# Global config instance
config = FrontendConfig()
