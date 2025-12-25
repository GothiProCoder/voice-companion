import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
"""
GuppShupp FastAPI Application
=============================

Production-grade FastAPI backend for the GuppShupp voice AI companion.
Handles audio processing, LLM orchestration, and TTS generation.

Features:
    - Lifespan management for model initialization
    - CORS middleware for Streamlit frontend
    - Request ID middleware for tracing
    - Structured exception handlers
    - Health and readiness endpoints
    - Authentication with session tokens
    - SSE-based chat with heartbeat

Usage:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Author: GuppShupp Team
Version: 1.0.0
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.api.router import api_router
from backend.config import config
from backend.schemas.common import ErrorResponse
from backend.utils.audio import TempAudioFile, cleanup_old_temp_files

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# APPLICATION METADATA
# =============================================================================

APP_TITLE = "GuppShupp Voice AI API"
APP_DESCRIPTION = """
<div align="center">
    <h2>🎙️ GuppShupp - Your Emotional Voice Companion</h2>
    <p>Production-grade voice AI backend for emotional wellness support</p>
</div>

## Features

- **Voice Processing**: Whisper ASR for Indian languages
- **Emotional Intelligence**: Emotion detection and empathetic responses
- **Memory System**: Long-term and episodic memory with IndicBERT embeddings
- **Natural TTS**: Indic Parler TTS for natural voice responses
- **Safety First**: Built-in crisis detection and safety guardrails

## Authentication

All authenticated endpoints require `X-Session-Token` header.
Obtain token via `/api/v1/auth/login` or `/api/v1/auth/signup`.

## Chat Endpoint (SSE)

The `/api/v1/conversations/chat` endpoint uses Server-Sent Events:
- `heartbeat`: Sent every 10s to indicate processing
- `progress`: Phase updates during processing
- `complete`: Final response with audio
- `error`: Error with retry guidance
"""
APP_VERSION = "1.0.0"


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup initialization and graceful shutdown.
    
    Startup:
        - Verify database connection (engine already created at module level)
        - Initialize WorkflowServices (loads ML models)
        - Clean up old temp files
        
    Shutdown:
        - Cleanup active temp files
        - Shutdown WorkflowServices
    """
    logger.info("=" * 60)
    logger.info("🚀 GuppShupp API Starting...")
    logger.info("=" * 60)
    
    startup_time = time.time()
    
    try:
        # Database is already initialized at module level!
        # Just verify connection works
        logger.info("💾 Verifying database connection...")
        try:
            from backend.database.database import verify_connection, register_vector_type
            
            if not verify_connection():
                raise RuntimeError("Database connection failed")
            
            # Register pgvector
            register_vector_type()
            logger.info("✅ Database ready")
            
        except Exception as e:
            logger.error(f"❌ Database verification failed: {e}")
            raise
        
        # Initialize workflow services
        logger.info("📦 Initializing WorkflowServices...")
        
        try:
            from backend.services.langgraph_workflow import _workflow_services
            from backend.database.database import SessionLocal
            
            # SessionLocal is now always available (created at module level)
            db = SessionLocal()
            try:
                await _workflow_services.initialize_async(db)
                logger.info("✅ WorkflowServices initialized")
            finally:
                db.close()
            
        except Exception as e:
            logger.warning(f"⚠️ WorkflowServices initialization deferred: {e}")
            # Continue - services can be lazy-loaded on first request
        
        # Clean up old temp audio files
        try:
            deleted = cleanup_old_temp_files(max_age_seconds=3600)
            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} old temp audio files")
        except Exception as e:
            logger.warning(f"⚠️ Temp cleanup failed: {e}")
        
        startup_duration = time.time() - startup_time
        logger.info(f"✅ Startup complete in {startup_duration:.2f}s")
        logger.info("=" * 60)

        
        # Application runs here
        yield
        
    finally:
        # Shutdown
        logger.info("=" * 60)
        logger.info("🛑 GuppShupp API Shutting down...")
        
        # Cleanup temp files
        try:
            TempAudioFile.cleanup_all_active()
        except Exception as e:
            logger.warning(f"⚠️ Temp cleanup on shutdown failed: {e}")
        
        # Shutdown workflow services
        try:
            from backend.services.langgraph_workflow import _workflow_services
            await _workflow_services.shutdown_async()
            logger.info("✅ WorkflowServices shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️ WorkflowServices shutdown failed: {e}")
        
        logger.info("👋 Goodbye!")
        logger.info("=" * 60)


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and readiness endpoints"
        },
        {
            "name": "Authentication",
            "description": "User signup, login, and session management"
        },
        {
            "name": "Conversations",
            "description": "Chat processing with SSE, history, and sessions"
        },
    ]
)


# =============================================================================
# MIDDLEWARE
# =============================================================================


# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",      # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",      # React dev
        "http://127.0.0.1:3000",
        "*",                          # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Add unique request ID to each request for tracing.
    """
    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Store in request state
    request.state.request_id = request_id
    
    # Process request
    start_time = time.time()
    
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled exception: {e}", exc_info=True)
        raise
    
    # Add request ID to response
    response.headers["X-Request-ID"] = request_id
    
    # Log request completion
    duration_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} ({duration_ms}ms)"
    )
    
    return response


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors with detailed feedback."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Extract error details
    errors = exc.errors()
    error_details = []
    for error in errors:
        error_details.append({
            "field": ".".join(str(x) for x in error.get("loc", [])),
            "message": error.get("msg", "Validation error"),
            "type": error.get("type", "unknown"),
        })
    
    logger.warning(f"[{request_id}] Validation error: {error_details}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed. Check the 'details' field.",
            request_id=request_id,
            retryable=False,
            details={"errors": error_details}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unhandled exceptions gracefully."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred. Please try again.",
            request_id=request_id,
            retryable=True,
            details=None
        ).model_dump()
    )


# =============================================================================
# ROUTES
# =============================================================================


# Include versioned API router
app.include_router(api_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# DEVELOPMENT SERVER
# =============================================================================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

