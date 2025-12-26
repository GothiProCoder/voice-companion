"""
GuppShupp Configuration Management
Loads and validates all environment variables
Provides centralized config access for the entire application
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    url: str = Field(alias="DATABASE_URL")
    host: str = Field(alias="DB_HOST")
    port: int = Field(alias="DB_PORT")
    name: str = Field(alias="DB_NAME")
    user: str = Field(alias="DB_USER")
    password: str = Field(alias="DB_PASSWORD")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class APIConfig(BaseSettings):
    """API Keys configuration"""
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class WhisperConfig(BaseSettings):
    """Whisper ASR configuration"""
    model_size: str = Field(default="large-v3", alias="WHISPER_MODEL_SIZE")
    device: str = Field(default="auto", alias="WHISPER_DEVICE")
    compute_type: str = Field(default="auto", alias="WHISPER_COMPUTE_TYPE")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

class IndicBERTConfig(BaseSettings):
    """IndicBERT embeddings configuration"""

    model_name: str = Field(
        default="l3cube-pune/indic-sentence-similarity-sbert",
        alias="INDICBERT_MODEL_NAME"
    )

    device_raw: str = Field(
        default="auto",
        alias="INDICBERT_DEVICE"
    )

    @computed_field
    @property
    def device(self) -> str:
        """Resolved PyTorch device."""
        if self.device_raw == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_raw

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class ParlerTTSConfig(BaseSettings):
    """Parler TTS configuration"""
    model_name: str = Field(
        default="ai4bharat/indic-parler-tts",
        alias="PARLER_TTS_MODEL"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class GeminiConfig(BaseSettings):
    """Gemini LLM configuration"""
    model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    max_output_tokens: int = Field(default=2048, alias="GEMINI_MAX_OUTPUT_TOKENS")
    api_key: str = Field(alias="GEMINI_API_KEY")
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class ApplicationConfig(BaseSettings):
    """Application-level configuration"""
    app_name: str = Field(default="GuppShupp", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    frontend_url: str = Field(default="http://localhost:8501", alias="FRONTEND_URL")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class AudioConfig(BaseSettings):
    """Audio processing configuration"""
    sample_rate: int = Field(default=16000, alias="AUDIO_SAMPLE_RATE")
    audio_format: str = Field(default="wav", alias="AUDIO_FORMAT")
    # ⚡ NEW: Control Opus encoding for TTS output
    # False = WAV mode (fast, ~15-20s) | True = Opus mode (slow, ~60-70s but smaller files)
    enable_opus_encoding: bool = Field(default=False, alias="ENABLE_OPUS_ENCODING")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class MemoryConfig(BaseSettings):
    """Memory system configuration"""
    max_memory_context: int = Field(default=5, alias="MAX_MEMORY_CONTEXT")
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class SafetyConfig(BaseSettings):
    """Safety and moderation configuration"""
    enable_content_moderation: bool = Field(default=True, alias="ENABLE_CONTENT_MODERATION")
    gemini_safety_threshold: str = Field(
        default="BLOCK_MEDIUM_AND_ABOVE",
        alias="GEMINI_SAFETY_THRESHOLD"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class Config:
    """
    Centralized configuration access
    Usage: from backend.config import config
    """
    
    def __init__(self):
        # Load all configuration sections
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.whisper = WhisperConfig()
        self.indicbert = IndicBERTConfig()
        self.parler_tts = ParlerTTSConfig()
        self.gemini = GeminiConfig()
        self.application = ApplicationConfig()
        self.audio = AudioConfig()
        self.memory = MemoryConfig()
        self.safety = SafetyConfig()
        
        # Validate critical configurations
        self._validate()
    
    def _validate(self):
        """Validate that critical configurations are set"""
        errors = []
        
        # Check Gemini API key
        if not self.api.gemini_api_key or self.api.gemini_api_key == "your_gemini_api_key_here":
            errors.append("GEMINI_API_KEY not set in .env file")
        
        # Check database password
        if not self.database.password or self.database.password == "your_password":
            errors.append("DB_PASSWORD not set in .env file")
        
        # Check Whisper device and compute type
        if self.whisper.device == "auto":
            self.whisper.device = self._detect_device()
        
        if self.whisper.compute_type == "auto":
            self.whisper.compute_type = self._detect_compute_type()
        
        if errors:
            error_msg = "\n".join([f"  ❌ {error}" for error in errors])
            raise ValueError(
                f"\n\n⚠️  Configuration Error:\n{error_msg}\n\n"
                "Please check your .env file and update the required values.\n"
            )
    
    def _detect_device(self) -> str:
        """Auto-detect if CUDA is available"""
        try:
            import torch
            if torch.cuda.is_available():
                print("✅ CUDA detected - Using GPU for inference")
                return "cuda"
            else:
                print("ℹ️  CUDA not available - Using CPU for inference")
                return "cpu"
        except ImportError:
            print("ℹ️  PyTorch not installed - Defaulting to CPU")
            return "cpu"
    
    def _detect_compute_type(self) -> str:
        """Auto-detect compute type based on device"""
        if self.whisper.device == "cuda":
            return "float16"  # GPU uses float16
        else:
            return "int8"  # CPU uses int8 for faster inference
    
    def get_database_url(self) -> str:
        """Get formatted database URL"""
        return self.database.url
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.application.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.application.environment.lower() == "development"
    
    def print_config(self):
        """Print current configuration (masks sensitive data)"""
        print("\n" + "=" * 70)
        print("GUPPSHUPP CONFIGURATION")
        print("=" * 70)
        print(f"\n📦 Application:")
        print(f"   Name: {self.application.app_name}")
        print(f"   Version: {self.application.app_version}")
        print(f"   Environment: {self.application.environment}")
        print(f"   Backend: {self.application.backend_host}:{self.application.backend_port}")
        
        print(f"\n🗄️  Database:")
        print(f"   Host: {self.database.host}:{self.database.port}")
        print(f"   Database: {self.database.name}")
        print(f"   User: {self.database.user}")
        
        print(f"\n🤖 AI Models:")
        print(f"   Whisper: {self.whisper.model_size} (device: {self.whisper.device}, compute: {self.whisper.compute_type})")
        print(f"   Gemini: {self.gemini.model} (temp: {self.gemini.temperature})")
        print(f"   IndicBERT: {self.indicbert.model_name}")
        print(f"   Parler TTS: {self.parler_tts.model_name}")
        
        print(f"\n🎙️  Audio:")
        print(f"   Sample Rate: {self.audio.sample_rate} Hz")
        print(f"   Format: {self.audio.audio_format}")
        print(f"   Opus Encoding: {'Enabled (smaller files)' if self.audio.enable_opus_encoding else 'Disabled (WAV mode, faster)'}")
        
        print(f"\n🧠 Memory:")
        print(f"   Context Window: {self.memory.max_memory_context} conversations")
        print(f"   Embedding Dimension: {self.memory.embedding_dimension}")
        
        print(f"\n🛡️  Safety:")
        print(f"   Content Moderation: {'Enabled' if self.safety.enable_content_moderation else 'Disabled'}")
        print(f"   Gemini Safety: {self.safety.gemini_safety_threshold}")
        
        print(f"\n🔑 API Keys:")
        print(f"   Gemini API Key: {'✅ Set' if self.api.gemini_api_key else '❌ Missing'}")
        
        print("=" * 70 + "\n")


# Create global config instance
config = Config()


# Convenience function for imports
def get_config() -> Config:
    """Get the global configuration instance"""
    return config


if __name__ == "__main__":
    # Test configuration loading
    try:
        config.print_config()
        print("✅ Configuration loaded successfully!")
    except ValueError as e:
        print(e)
        exit(1)
