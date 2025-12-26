import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
"""
GuppShupp Conversational AI Workflow - LangGraph-based Agentic System
========================================================================

Production-Grade Multi-Stage Voice Processing Pipeline with LangGraph Orchestration
- Phase 1: Parallel audio analysis (Whisper ASR + Emotion Detection)
- Phase 2: Parallel context preparation (Memory Retrieval + Session Context)
- Phase 3: Sequential LLM generation (Gemini 2.5 Flash with rich context)
- Phase 4: Sequential TTS synthesis (Parler TTS with emotional tone)
- Phase 5: Sequential database persistence (Store conversation + memories)

Architecture:
- State-driven with TypedDict for clean state management
- Middleware-based dynamic prompts and guardrails
- Streaming support for token-level output and partial results
- Graceful error handling with fallback responses
- LangSmith integration for observability and debugging

Author: GuppShupp Team
Last Updated: 2025-12-23
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Annotated, Any, TypedDict, Literal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from enum import Enum

import numpy as np
import torch
from sqlalchemy.orm import Session

# LangGraph imports - production-grade orchestration
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # ✅ Correct async import

# LangChain runtime and observability
from langchain.agents import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.runtime import Runtime

from backend.utils.serialization import sanitize_for_state
from backend.utils.audio import save_audio_file

# GuppShupp service imports
from backend.services.whisper_asr import (
    transcribe_audio,
    initialize_whisper_service,
    shutdown_whisper_service,
    TranscriptionResult,
)
from backend.services.emotion_detection import (
    extract_prosody_features,
    initialize_prosody_service,
    shutdown_prosody_service,
    ProsodyResult,
)
from backend.services.indicbert_memory import (
    IndicBERTMemoryService,
    Memory,
    MemoryUpdate,
)
from backend.services.gemini_llm import (
    GeminiLLMService,
    GeminiLLMResponse,
    SafetyFlags,
)
from backend.services.parler_tts_module import (
    ParlerTTSService,
    TTSConfig,
    TTSRequest,
    TTSResponse,
)
from backend.database.models import User, Conversation, Memory as MemoryModel
from backend.config import config

# Configure logging with detailed formatting for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE DEFINITIONS - TypedDict for clean state management and LangGraph
# ============================================================================


class AudioAnalysisState(TypedDict):
    """Parallel audio analysis output - both ASR and emotion detection."""

    transcription: TranscriptionResult
    prosody_features: ProsodyResult
    processing_time_ms: int


class ContextPrepState(TypedDict):
    """Parallel context preparation output - memory + session context."""

    long_term_memories: List[Dict[str, Any]]
    episodic_memories: List[Dict[str, Any]]
    short_term_context: List[Dict[str, Any]]
    session_metadata: Dict[str, Any]
    processing_time_ms: int


class WorkflowInput(TypedDict):
    """User input to the workflow."""

    audio_path: str  # Path to MP3/WAV audio file
    user_id: str  # User UUID for memory retrieval and personalization
    session_id: str  # Session UUID for conversation tracking
    conversation_id: Optional[str]  # Current conversation UUID
    session_context: Dict[str, Any]  # Contains: current_tts_speaker, session_language, voice_preferences
    request_id: Optional[str]  # For tracing and debugging


class WorkflowState(TypedDict):
    """Complete workflow state - combines all phases."""

    # Input
    audio_path: str
    user_id: str
    session_id: str
    conversation_id: Optional[str]
    session_context: Dict[str, Any]  # Contains: current_tts_speaker, session_language, voice_preferences
    request_id: str

    # Phase 1: Audio Analysis (parallel)
    transcription: Optional[TranscriptionResult]
    prosody_features: Optional[ProsodyResult]
    audio_analysis_error: Optional[str]
    audio_analysis_time_ms: int

    # Phase 2: Context Preparation (parallel)
    long_term_memories: List[Dict[str, Any]]
    episodic_memories: List[Dict[str, Any]]
    short_term_context: List[Dict[str, Any]]
    session_metadata: Dict[str, Any]
    context_prep_error: Optional[str]
    context_prep_time_ms: int

    # Phase 3: LLM Generation (sequential)
    llm_response: Optional[GeminiLLMResponse]
    llm_error: Optional[str]
    llm_time_ms: int

    # Safety pre-check before dedicated guardrails
    safety_flags: Optional[SafetyFlags]
    safety_passed: bool
    safety_action: Literal["continue", "escalate", "block"]

    # Phase 4: TTS Generation (sequential)
    tts_response: Optional[TTSResponse]
    tts_error: Optional[str]
    tts_time_ms: int
    
    # TTS Speaker Tracking (for session persistence)
    current_tts_speaker: Optional[str]  # Track the current speaker for session consistency
    voice_preferences: Dict[str, Any]  # User's voice preferences (gender, etc.)

    # Phase 5: Database Persistence (sequential)
    conversation_stored: bool
    memories_stored: bool
    db_error: Optional[str]
    db_time_ms: int

    # Streaming messages for LangSmith observability
    messages: Annotated[List[BaseMessage], add_messages]

    # Metadata
    total_time_ms: int
    workflow_status: Literal["pending", "processing", "completed", "failed"]
    created_at: str
    completed_at: Optional[str]


# ============================================================================
# SERVICE INITIALIZATION - Lazy-loaded singletons with proper cleanup
# ============================================================================


class WorkflowServices:
    """Centralized service management for the workflow."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.llm_service = None
        self.memory_service = None
        self.tts_service = None
        self.db_session = None

        self._initialized = True
        logger.info("WorkflowServices singleton created")

    async def initialize_async(self, db_session: Session) -> None:
        """Initialize services asynchronously (called at startup)."""
        logger.info("Initializing workflow services...")

        try:
            # Initialize Whisper ASR service
            await initialize_whisper_service()
            logger.info("✓ Whisper ASR service initialized")

            # Initialize Prosody extraction service
            await initialize_prosody_service()
            logger.info("✓ Prosody extraction service initialized")

            # Initialize LLM service
            self.llm_service = GeminiLLMService(
                api_key=config.gemini.api_key,
                model=config.gemini.model,
            )
            logger.info("✓ Gemini LLM service initialized")

            # Initialize Memory service with IndicBERT embeddings
            self.memory_service = IndicBERTMemoryService(
                model_name=config.indicbert.model_name,
                device=config.indicbert.device,
                cache_size=1000,
                batch_size=32,
            )
            logger.info("✓ IndicBERT Memory service initialized")

            # Initialize TTS service
            self.tts_service = ParlerTTSService(
                config=TTSConfig(
                    model_name="ai4bharat/indic-parler-tts",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    sampling_rate=44100,
                    cache_enabled=True,
                )
            )
            logger.info("✓ Parler TTS service initialized")
            
            # ⚡ PHASE 1: Warmup TTS model to pre-compile torch graphs
            # This eliminates the first-request latency spike (30s → 3s)
            logger.info("🔥 Warming up TTS model (one-time, may take 20-30s)...")
            self.tts_service.warmup()

            self.db_session = db_session
            logger.info("✓ Database session configured")
            logger.info("All workflow services initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize workflow services: {e}", exc_info=True)
            raise

    async def shutdown_async(self) -> None:
        """Graceful shutdown of all services."""
        logger.info("Shutting down workflow services...")

        try:
            await shutdown_whisper_service()
            await shutdown_prosody_service()

            if self.llm_service:
                await self.llm_service.aclose()
            if self.tts_service:
                self.tts_service.close()
            if self.memory_service:
                self.memory_service.close()

            logger.info("All services shut down gracefully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# Global services singleton
_workflow_services = WorkflowServices()


# ============================================================================
# PHASE 1: PARALLEL AUDIO ANALYSIS
# ============================================================================


async def phase_1_audio_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Phase 1: Parallel Audio Analysis
    - Whisper ASR for transcription (~2-3s)
    - Emotion detection for prosody features (~1-2s)
    Runs concurrently for ~2-3s total instead of ~4-5s sequential.
    """
    logger.info(f"[PHASE 1] Starting parallel audio analysis for {state['request_id']}")
    start_time = time.time()

    try:
        # Start both tasks in parallel
        audio_path = str(state["audio_path"])  # Force string
        
        # ✅ ADD FILE EXISTENCE CHECK
        from pathlib import Path
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}. "
                f"File may have been deleted prematurely."
            )
        
        logger.info(f"[DEBUG] Audio file verified: {audio_path} (exists: {audio_file.exists()})")
        
        transcription_task = transcribe_audio(audio_path, state["request_id"])
        prosody_task = extract_prosody_features(audio_path, state["request_id"])

        # Wait for both to complete
        transcription, prosody_features = await asyncio.gather(
            transcription_task, prosody_task, return_exceptions=True
        )

        # Handle exceptions from parallel tasks
        transcription_error = None
        prosody_error = None

        if isinstance(transcription, Exception):
            logger.error(f"Transcription error: {transcription}", exc_info=True)
            transcription = None
            transcription_error = str(transcription)

        if isinstance(prosody_features, Exception):
            logger.error(f"Prosody extraction error: {prosody_features}", exc_info=True)
            prosody_features = None
            prosody_error = str(prosody_features)

        # Log results
        if transcription:
            logger.info(
                f"[PHASE 1] ✓ Transcription: {transcription.text[:100]}... "
                f"({transcription.word_count} words, {transcription.processing_time_ms}ms)"
            )
        if prosody_features:
            logger.info(
                f"[PHASE 1] ✓ Prosody extracted: "
                f"pitch={prosody_features.prosody_pitch.get('f0_mean', 0):.1f}Hz, "
                f"energy={prosody_features.energy_loudness.get('rms_mean_db', 0):.1f}dB "
                f"({prosody_features.processing_time_ms}ms)"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Return state updates
        # ✅ Convert ProsodyResult to dict for msgpack serialization (checkpointer requirement)
        return {
            **state,  # ← CRITICAL: Keep all previous state
            "transcription": sanitize_for_state(transcription),
            "prosody_features": sanitize_for_state(prosody_features),
            "audio_analysis_error": transcription_error or prosody_error,
            "audio_analysis_time_ms": elapsed_ms,
            "messages": state.get("messages", []) + [AIMessage(content=f"Phase 1: Audio analysis completed in {elapsed_ms}ms")],
        }


    except Exception as e:
        logger.error(f"[PHASE 1] Critical error: {e}", exc_info=True)
        return {
            "audio_analysis_error": str(e),
            "audio_analysis_time_ms": int((time.time() - start_time) * 1000),
            "messages": [AIMessage(content=f"[Phase 1] Error: {str(e)}")],
        }


# ============================================================================
# PHASE 2: PARALLEL CONTEXT PREPARATION
# ============================================================================


async def phase_2_context_preparation(state: WorkflowState) -> Dict[str, Any]:
    """
    Phase 2: Parallel Context Preparation
    - Memory retrieval with IndicBERT semantic search (~200-500ms)
    - Session context aggregation (~50ms)
    Runs concurrently for ~200-500ms instead of ~250-550ms sequential.
    """
    logger.info(f"[PHASE 2] Starting parallel context preparation for {state['request_id']}")
    start_time = time.time()

    try:
        # Ensure we have transcription from Phase 1
        if not state["transcription"]:
            logger.warning("[PHASE 2] No transcription available, skipping memory retrieval")
            long_term_memories = []
            episodic_memories = []
        else:
            # ⚠️ transcription is now a dict after sanitize_for_state()
            query_text = state["transcription"].get("text", "")

            memory_task = _workflow_services.memory_service.retrieve_memories_async(
                db=_workflow_services.db_session,
                user_id=state["user_id"],
                query_text=query_text,
                top_k=5,
                memory_types=["long_term", "episodic"],
                min_importance=0.3,
                apply_decay=True,
                apply_recency_boost=True,
            )

            # Aggregate short-term context (previous N turns in current session)
            session_context_task = _get_session_context(
                state["user_id"],
                state["session_id"],
                limit=5,
            )

            # Get or create conversation
            conversation_task = _get_or_create_conversation(
                state["user_id"],
                state["session_id"],
                state["conversation_id"],
            )

            # ⚠️ CRITICAL FIX: Execute database operations SEQUENTIALLY
            # SQLAlchemy sessions are NOT thread-safe for concurrent use
            # The previous asyncio.gather caused "isce" error:
            # "This session is provisioning a new connection; concurrent operations are not permitted"
            
            # Step 1: Get memories (uses db_session)
            try:
                memories_list = await memory_task
            except Exception as e:
                logger.warning(f"[PHASE 2] Memory retrieval failed: {e}")
                memories_list = []
            
            # Step 2: Get session context (uses db_session)
            try:
                short_term_ctx = await session_context_task
            except Exception as e:
                logger.warning(f"[PHASE 2] Session context failed: {e}")
                short_term_ctx = []
            
            # Step 3: Get or create conversation (uses db_session)
            try:
                conversation = await conversation_task
            except Exception as e:
                logger.warning(f"[PHASE 2] Conversation task failed: {e}")
                conversation = None


            # Split memories into long-term and episodic
            long_term_memories = [
                m for m in memories_list if m.get("memory_type") == "long_term"
            ]
            episodic_memories = [
                m for m in memories_list if m.get("memory_type") == "episodic"
            ]

            logger.info(
                f"[PHASE 2] ✓ Retrieved {len(long_term_memories)} long-term + "
                f"{len(episodic_memories)} episodic memories"
            )
            logger.info(f"[PHASE 2] ✓ Session context: {len(short_term_ctx)} turns")

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            **state,  
            "long_term_memories": long_term_memories,
            "episodic_memories": episodic_memories,
            "short_term_context": short_term_ctx if "short_term_ctx" in locals() else [],
            "session_metadata": {
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "conversation_id": state["conversation_id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "context_prep_error": None,
            "context_prep_time_ms": elapsed_ms,
            "messages": state.get("messages", []) + [AIMessage(content=f"Phase 2: Context preparation completed in {elapsed_ms}ms")],
        }

    except Exception as e:
        logger.error(f"[PHASE 2] Critical error: {e}", exc_info=True)
        return {
            "context_prep_error": str(e),
            "context_prep_time_ms": int((time.time() - start_time) * 1000),
            "messages": [AIMessage(content=f"[Phase 2] Error: {str(e)}")],
        }


# ============================================================================
# PHASE 3: SEQUENTIAL LLM GENERATION WITH SAFETY GUARDRAILS
# ============================================================================


async def phase_3_llm_generation_with_guardrails(
    state: WorkflowState,
) -> Dict[str, Any]:
    """
    Phase 3: Sequential LLM Generation with Gemini Safety Guardrails
    - Gemini handles ALL safety internally (no pre-check needed)
    - Dynamic prompt with rich context aggregation
    - Structured JSON output with memory extraction
    - Fallback mechanism for robustness
    """
    logger.info(f"[PHASE 3] Starting LLM generation for {state['request_id']}")
    start_time = time.time()

    try:
        # Build rich context for Gemini
        acoustic_features = sanitize_for_state(state.get("prosody_features")) or {}
        
        # Build session context with current TTS speaker for consistency
        session_context = state.get("session_context", {})
        session_context["current_tts_speaker"] = state.get("current_tts_speaker")
        session_context["voice_preferences"] = state.get("voice_preferences", {})
        
        # ⚠️ transcription is now a dict after sanitize_for_state()
        llm_response = await _workflow_services.llm_service.analyze_and_respond_async(
            transcript=state["transcription"].get("text", "") if state["transcription"] else "",
            language=state["transcription"].get("language", "en") if state["transcription"] else "en",
            acoustic_features=acoustic_features,
            short_term_context=state["short_term_context"],
            long_term_memories=state["long_term_memories"],
            episodic_memories=state["episodic_memories"],
            session_context=session_context,  # Contains TTS speaker info
            safety_context={},  # Gemini handles this internally
            temperature=config.gemini.temperature,
            max_output_tokens=config.gemini.max_output_tokens,
       )

        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[PHASE 3] ✓ LLM response generated ({elapsed_ms}ms): "
            f"emotion={llm_response.detected_emotion}, "
            f"intent={llm_response.detected_intent}, "
            f"speaker={llm_response.tts_speaker}, "
            f"voice_change={llm_response.voice_change_requested}, "
            f"memories={len(llm_response.memory_updates)}, "
            f"safety_risk={llm_response.safety_flags.crisis_risk}"
        )

        # Determine safety action based on Gemini's assessment
        safety_action = "escalate" if llm_response.safety_flags.crisis_risk == "high" else "continue"
        safety_passed = llm_response.safety_flags.crisis_risk != "high"
        
        # Update voice preferences if user specified gender
        voice_preferences = state.get("voice_preferences", {})
        if llm_response.preferred_speaker_gender != "any":
            voice_preferences["gender"] = llm_response.preferred_speaker_gender

        return {
            **state,
            "llm_response": llm_response,
            "safety_flags": llm_response.safety_flags,  # From Gemini
            "safety_passed": safety_passed,
            "safety_action": safety_action,
            "llm_error": None,
            "llm_time_ms": elapsed_ms,
            # Update TTS speaker for session persistence
            "current_tts_speaker": llm_response.tts_speaker,
            "voice_preferences": voice_preferences,
            "messages": state.get("messages", []) + [
                AIMessage(
                    content=f"[Phase 3] LLM generation completed in {elapsed_ms}ms\\n"
                    f"Response: {llm_response.response_text}"
                )
            ],
        }

    except Exception as e:
        logger.error(f"[PHASE 3] LLM generation error: {e}", exc_info=True)
        return {
            "llm_response": _get_fallback_llm_response(str(e)),
            "llm_error": str(e),
            "llm_time_ms": int((time.time() - start_time) * 1000),
            "messages": [AIMessage(content=f"[Phase 3] Error: {str(e)}")],
        }


# ============================================================================
# PHASE 4: SEQUENTIAL TTS GENERATION
# ============================================================================


async def phase_4_tts_generation(state: WorkflowState) -> Dict[str, Any]:
    """
    Phase 4: Sequential TTS Generation
    - Use Gemini's TTS style prompt and speaker recommendation
    - Generate high-quality Indic Parler TTS audio
    - Return base64-encoded WAV for frontend playback
    """
    logger.info(f"[PHASE 4] Starting TTS generation for {state['request_id']}")
    start_time = time.time()

    try:
        if not state["llm_response"]:
            logger.error("[PHASE 4] No LLM response available for TTS")
            return {
                "tts_error": "No LLM response available",
                "tts_time_ms": int((time.time() - start_time) * 1000),
                "messages": [AIMessage(content="[Phase 4] Error: No LLM response")],
            }

        # Extract TTS parameters from LLM response
        spoken_text = state["llm_response"].response_text
        speaker = state["llm_response"].tts_speaker or "Rohit"
        tts_style = state["llm_response"].tts_style_prompt or "speaks at a moderate pace with a calm tone in a close-sounding environment with clear audio quality."
        
        # Concatenate speaker name with description (LLM generates "speaks..." format)
        # Result: "Rohit speaks at a moderate pace with..."
        full_description = f"{speaker} {tts_style}"

        logger.info(
            f"[PHASE 4] TTS input: speaker={speaker}, "
            f"description={full_description[:60]}..., text_length={len(spoken_text)}"
        )

        # Create TTS request with full description
        tts_request = TTSRequest(
            spoken_text=spoken_text,
            speaker=speaker,
            description=full_description,  # Now includes speaker name + style
        )

        # Generate TTS audio asynchronously
        loop = asyncio.get_event_loop()
        tts_response = await loop.run_in_executor(
            None,
            _workflow_services.tts_service.generate,
            tts_request,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[PHASE 4] ✓ TTS generated ({elapsed_ms}ms): "
            f"{tts_response.duration_seconds:.2f}s, SR={tts_response.sampling_rate}Hz"
        )

        return {
            **state,
            # ⚡ CRITICAL: Include audio bytes for Phase 5 file saving
            # TTSResponse contains numpy arrays which can't be serialized by msgpack,
            # so we extract only the serializable fields here
            "tts_response": {
                "audio_base64_wav": tts_response.audio_base64_wav,  # For frontend playback
                "audio_opus_bytes": tts_response.audio_opus_bytes,  # ⚡ For Opus mode file saving
                "audio_wav_bytes": tts_response.audio_wav_bytes,    # ⚡ For WAV mode file saving
                "duration_seconds": tts_response.duration_seconds,
                "sampling_rate": tts_response.sampling_rate,
                "generation_time_ms": tts_response.generation_time_ms,
            },
            "tts_error": None,
            "tts_time_ms": elapsed_ms,
            "messages": state.get("messages", []) + [
                AIMessage(
                    content=f"[Phase 4] TTS generation completed in {elapsed_ms}ms\n"
                    f"Duration: {tts_response.duration_seconds:.2f}s"
                )
            ],
        }


    except Exception as e:
        logger.error(f"[PHASE 4] TTS generation error: {e}", exc_info=True)
        return {
            "tts_error": str(e),
            "tts_time_ms": int((time.time() - start_time) * 1000),
            "messages": [AIMessage(content=f"[Phase 4] Error: {str(e)}")],
        }


# ============================================================================
# PHASE 5: SEQUENTIAL DATABASE PERSISTENCE
# ============================================================================


async def phase_5_database_persistence(state: WorkflowState) -> Dict[str, Any]:
    """
    Phase 5: Sequential Database Persistence
    - Store conversation with full context and embeddings
    - Store proposed memories with IndicBERT embeddings
    - Update user interaction history
    """
    logger.info(f"[PHASE 5] Starting database persistence for {state['request_id']}")
    start_time = time.time()

    try:
        conversation_stored = False
        memories_stored = False

        if not state["llm_response"]:
            logger.warning("[PHASE 5] No LLM response to persist")
            return {
                "conversation_stored": False,
                "memories_stored": False,
                "db_error": "No LLM response available",
                "db_time_ms": int((time.time() - start_time) * 1000),
            }

        # Store conversation
        loop = asyncio.get_event_loop()
        stored_conversation_id = await loop.run_in_executor(
            None,
            _store_conversation,
            state,
        )

        conversation_stored = stored_conversation_id is not None

        if conversation_stored:
            logger.info(f"[PHASE 5] ✓ Conversation stored: {stored_conversation_id}")
            # 2. CRITICAL FIX: Update state with the ACTUAL ID so memories link correctly
            # state["conversation_id"] = stored_conversation_id

        # Only try to store memories if we have a valid conversation ID to link to
            if state["llm_response"] and state["llm_response"].memory_updates:
                memories_stored = await loop.run_in_executor(
                    None, 
                    _store_memories_with_id,  # New function
                    state,
                    stored_conversation_id  # Pass ID explicitly
                )
                if memories_stored:
                    logger.info(f"[PHASE 5] ✓ Stored memories linked to {stored_conversation_id}")
        else:
            logger.error("[PHASE 5] Cannot store memories: Conversation save failed (No ID)")

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            **state,
            "conversation_stored": conversation_stored,
            "memories_stored": memories_stored if 'memories_stored' in locals() else False,
            "conversation_id": stored_conversation_id,  # ← For memory linking
            "db_error": None,
            "db_time_ms": elapsed_ms,
            "messages": state.get("messages", []) + [
                AIMessage(
                    content=f"[Phase 5] Database persistence completed in {elapsed_ms}ms"
                )
            ],
        }

    except Exception as e:
        logger.error(f"[PHASE 5] Database persistence error: {e}", exc_info=True)
        return {
            "conversation_stored": False,
            "memories_stored": False,
            "db_error": str(e),
            "db_time_ms": int((time.time() - start_time) * 1000),
            "messages": [AIMessage(content=f"[Phase 5] Error: {str(e)}")],
        }


# ============================================================================
# WORKFLOW ORCHESTRATION - LangGraph State Machine
# ============================================================================


def create_workflow_graph(db_session: Session) -> StateGraph:
    """
    Create the complete LangGraph workflow with proper state management,
    error handling, and streaming support.
    """
    # Create state graph
    workflow = StateGraph(WorkflowState)

    # Add phase nodes
    workflow.add_node("phase_1_audio_analysis", phase_1_audio_analysis)
    workflow.add_node("phase_2_context_preparation", phase_2_context_preparation)
    workflow.add_node(
        "phase_3_llm_generation", phase_3_llm_generation_with_guardrails
    )
    workflow.add_node("phase_4_tts_generation", phase_4_tts_generation)
    workflow.add_node("phase_5_database_persistence", phase_5_database_persistence)

    # Define edges - sequential execution with error handling
    workflow.add_edge(START, "phase_1_audio_analysis")
    workflow.add_edge("phase_1_audio_analysis", "phase_2_context_preparation")
    workflow.add_edge("phase_2_context_preparation", "phase_3_llm_generation")
    workflow.add_edge("phase_3_llm_generation", "phase_4_tts_generation")
    workflow.add_edge("phase_4_tts_generation", "phase_5_database_persistence")
    workflow.add_edge("phase_5_database_persistence", END)

    # Compile WITHOUT checkpointer - checkpointer is added at runtime
    # Per LangGraph docs: AsyncPostgresSaver requires context manager pattern
    # The checkpointer is passed during execution in execute_workflow()
    graph = workflow.compile()
    # Note: checkpointer will be set via graph.with_config() at runtime

    logger.info(f"Workflow graph compiled with {len(workflow.nodes)} nodes")
    return graph


# ============================================================================
# HELPER FUNCTIONS - Supporting utilities for workflow phases
# ============================================================================

def _store_memories_with_id(state: WorkflowState, conversation_id: str) -> bool:
    """
    Store memories extracted by LLM with explicit conversation ID.
    """
    try:
        if not state["llm_response"] or not state["llm_response"].memory_updates:
            return False
        
        # Use the passed conversation_id directly - no state lookup
        stored_memories = _workflow_services.memory_service.store_memories_batch(
            db=_workflow_services.db_session,
            user_id=state["user_id"],
            memory_updates=state["llm_response"].memory_updates,
            conversation_id=conversation_id,  # Direct argument
        )
        
        logger.info(f"✅ Stored {len(stored_memories)} memories linked to {conversation_id}")
        return len(stored_memories) > 0
        
    except Exception as e:
        logger.error(f"Failed to store memories: {e}", exc_info=True)
        return False


async def _get_session_context(
    user_id: str,
    session_id: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve last N conversation turns in current session."""
    try:
        # Query database for recent conversations in session
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            None,
            lambda: _workflow_services.db_session.query(Conversation)
            .filter(
                Conversation.user_id == user_id,
                Conversation.session_id == session_id,
            )
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .all(),
        )

        # Format as list of dicts
        return [
            {
                "user_input": c.user_input_text,      # ✅ CORRECT FIELD NAME
                "ai_response": c.ai_response_text,    # ✅ CORRECT FIELD NAME
                "emotion": c.detected_emotion,
                "timestamp": c.created_at.isoformat(),
            }
            for c in reversed(context)  # Reverse to chronological order
        ]

    except Exception as e:
        logger.warning(f"Failed to retrieve session context: {e}")
        return []


async def _get_or_create_conversation(
    user_id: str,
    session_id: str,
    conversation_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Get or create conversation for tracking."""
    try:
        loop = asyncio.get_event_loop()

        if conversation_id:
            # Fetch existing conversation
            conv = await loop.run_in_executor(
                None,
                lambda: _workflow_services.db_session.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first(),
            )
            return conv
        else:
            # Create new conversation placeholder (will be updated later in Phase 5)
            conv = Conversation(
                user_id=user_id,
                session_id=session_id,
                user_input_text="",           # ✅ CORRECT FIELD NAME
                ai_response_text="",          # ✅ CORRECT FIELD NAME
                detected_language="en",
                response_language="en",
                detected_emotion="neutral",
                emotion_confidence=0.0,
                sentiment="neutral",
                detected_intent="unknown",
                intent_confidence=0.0,
                created_at=datetime.now(timezone.utc),
            )
            _workflow_services.db_session.add(conv)
            _workflow_services.db_session.commit()
            return conv

    except Exception as e:
        logger.warning(f"Failed to get/create conversation: {e}")
        return None


def _save_tts_audio_file(
    tts_response: Optional[dict],
    user_id: str,
    session_id: str,
) -> Optional[str]:
    """
    Save TTS audio to file and return relative path for database storage.
    
    ⚡ Supports both Opus (compressed) and WAV (fast) based on ENABLE_OPUS_ENCODING config.
    
    Args:
        tts_response: Sanitized TTSResponse dict from workflow state
        user_id: User UUID string for directory structure
        session_id: Session UUID for unique filename
        
    Returns:
        Relative path to saved file (e.g., "audio_storage/{user}/{uuid}.wav")
        None if no audio available
    """
    if not tts_response:
        logger.warning("[AUDIO SAVE] No TTS response provided")
        return None
    
    # Import config to check which format to use
    from backend.config import config
    from backend.utils.audio import save_audio_file
    
    # Determine which bytes to use based on config
    if config.audio.enable_opus_encoding:
        audio_bytes = tts_response.get("audio_opus_bytes")
        format_ext = "opus"
        logger.debug("[AUDIO SAVE] Using Opus format (config enabled)")
    else:
        audio_bytes = tts_response.get("audio_wav_bytes")
        format_ext = "wav"
        logger.debug("[AUDIO SAVE] Using WAV format (Opus disabled)")
    
    # Validate we have audio data
    if not audio_bytes:
        logger.warning(
            f"[AUDIO SAVE] No audio bytes found for format '{format_ext}'. "
            f"Available keys: {list(tts_response.keys())}"
        )
        return None
    
    # Generate unique filename using UUID
    import uuid
    conversation_id = str(uuid.uuid4())
    
    try:
        audio_path = save_audio_file(
            audio_bytes=audio_bytes,
            user_id=user_id,
            conversation_id=conversation_id,
            format_ext=format_ext  # "opus" or "wav"
        )
        logger.info(
            f"✅ [AUDIO SAVE] Saved TTS audio: {audio_path} "
            f"({len(audio_bytes) // 1024}KB, format={format_ext})"
        )
        return audio_path
    except Exception as e:
        logger.error(f"[AUDIO SAVE] Failed to save TTS audio file: {e}", exc_info=True)
        return None





def _get_fallback_llm_response(error_message: str) -> GeminiLLMResponse:
    """Generate fallback response when LLM fails."""
    return GeminiLLMResponse(
        response_text="I'm having trouble processing your request right now. Please try again in a moment.",
        response_language="en",
        detected_emotion="neutral",
        emotion_confidence=0.0,
        sentiment="neutral",
        detected_intent="unknown",
        intent_confidence=0.0,
        tts_style_prompt="Calm, gentle, supportive tone",
        tts_speaker="Thoma",
        memory_updates=[],
        safety_flags=SafetyFlags(
            crisis_risk="low",
            self_harm_mentioned=False,
            abuse_mentioned=False,
            medical_concern=False,
            flagged_keywords=[],
        ),
        generation_time_ms=0,
    )


def _store_conversation(state: WorkflowState) -> Optional[str]: # Updated return type
    """
    Store conversation to database with ALL required fields.
    Returns the string UUID of the conversation if successful.
    """
    try:
        if not state["llm_response"] or not state["transcription"]:
            logger.warning("Missing LLM response or transcription - skipping storage")
            return None # Return None instead of False

        user_text = state["transcription"].get("text", "")
        logger.info(f"Generating IndicBERT embedding for: {user_text[:50]}...")
        embedding = _workflow_services.memory_service.embed_text(user_text, use_cache=False)
        embedding_list = embedding.tolist()

        conversation = Conversation(
            user_id=state["user_id"],
            session_id=state["session_id"],
            user_input_text=user_text,
            ai_response_text=state["llm_response"].response_text,
            detected_language=state["transcription"].get("language", "en") if state["transcription"] else "en",
            response_language=state["llm_response"].response_language,
            is_code_mixed=state["transcription"].get("is_code_mixed", False) if state["transcription"] else False,
            code_mix_languages=state["transcription"].get("detected_languages") if state["transcription"] else None,
            detected_emotion=state["llm_response"].detected_emotion,
            emotion_confidence=state["llm_response"].emotion_confidence,
            sentiment=state["llm_response"].sentiment,
            detected_intent=state["llm_response"].detected_intent,
            intent_confidence=state["llm_response"].intent_confidence,
            prosody_features=state.get("prosody_features"),
            audio_duration_seconds=(
                state.get("prosody_features", {})
                    .get("meta_info", {})
                    .get("duration_sec", 0.0)
            ) if state["prosody_features"] else 0,
            audio_file_path=str(state["audio_path"]) if state["audio_path"] else None,
            
            # ⚡ NEW: Save Opus audio to file, store path in database
            response_audio_path=_save_tts_audio_file(
                tts_response=state.get("tts_response"),
                user_id=str(state["user_id"]),
                session_id=str(state["session_id"]),
            ),
            response_audio_duration_seconds=(
                state.get("tts_response", {}).get("duration_seconds", 0.0)
                if state.get("tts_response") else None
            ),
            
            tts_prompt=state["llm_response"].tts_style_prompt,
            response_generation_time_ms=state.get("llm_time_ms", 0),
            safety_check_passed=state["llm_response"].safety_flags.crisis_risk != "high",
            safety_flags = sanitize_for_state(state["llm_response"].safety_flags),
            created_at=datetime.now(timezone.utc),
            embedding=embedding_list,
        )

        _workflow_services.db_session.add(conversation)
        _workflow_services.db_session.commit()
        
        # Refresh to get the ID generated by the DB
        _workflow_services.db_session.refresh(conversation)

        logger.info(
            f"✅ Stored conversation {conversation.id} with embedding "
            f"(emotion={conversation.detected_emotion}, safety={conversation.safety_check_passed})"
        )
        return str(conversation.id) # RETURN THE UUID STRING

    except Exception as e:
        logger.error(f"❌ Failed to store conversation: {e}", exc_info=True)
        _workflow_services.db_session.rollback()
        return None # Return None instead of False
    

def _store_memories(state: WorkflowState) -> bool:
    """Store memories extracted by LLM with strict UUID validation."""
    try:
        # 1. Check if LLM actually proposed any memories
        if not state["llm_response"] or not state["llm_response"].memory_updates:
            return False

        # 2. CRITICAL FIX: Validate conversation_id type
        # In previous logs, this was 'True' (bool), which caused the crash.
        conv_id = state.get("conversation_id")
        
        if not conv_id or isinstance(conv_id, bool):
            logger.error(
                f"❌ Memory storage aborted: conversation_id is invalid type ({type(conv_id)}). "
                "Ensure _store_conversation returns a UUID string and not a Boolean."
            )
            return False

        # 3. Convert LLM memory updates to database models with IndicBERT embeddings
        stored_memories = _workflow_services.memory_service.store_memories_batch(
            db=_workflow_services.db_session,
            user_id=state["user_id"],
            memory_updates=state["llm_response"].memory_updates,
            conversation_id=conv_id, # Must be a String/UUID
        )

        logger.info(f"✅ Stored {len(stored_memories)} memories linked to conversation {conv_id}")
        return len(stored_memories) > 0

    except Exception as e:
        logger.error(f"❌ Failed to store memories: {e}", exc_info=True)
        # Rollback the session to maintain data integrity
        _workflow_services.db_session.rollback()
        return False
    
# ============================================================================
# PUBLIC API - Workflow execution entry point
# ============================================================================


async def execute_workflow(
    workflow_input: WorkflowInput,
    db_session: Session = None,
) -> Dict[str, Any]:
    """
    Execute the complete conversation workflow.
    
    Uses AsyncPostgresSaver with proper context manager pattern
    as specified in LangGraph documentation.
    
    Args:
        workflow_input: User audio + metadata
        db_session: Optional database session (will create one if not provided)
    
    Returns:
        Complete workflow output with all phases and results
    """
    # Create our own session if not provided
    from backend.database.database import SessionLocal
    
    own_session = False
    if db_session is None:
        db_session = SessionLocal()
        own_session = True
    
    try:
        # Initialize services on first call
        if not _workflow_services.llm_service:
            await _workflow_services.initialize_async(db_session)

        # Build initial state
        initial_state: WorkflowState = {
            "audio_path": str(workflow_input["audio_path"]),
            "user_id": workflow_input["user_id"],
            "session_id": workflow_input["session_id"],
            "conversation_id": workflow_input.get("conversation_id"),
            "session_context": workflow_input.get(
                "session_context",
                {
                    "current_tts_speaker": None,  # Will be set by LLM
                    "session_language": "auto",
                    "voice_preferences": {},
                },
            ),
            "request_id": workflow_input.get("request_id") or f"req_{int(time.time() * 1000)}",
            # Phase placeholders
            "transcription": None,
            "prosody_features": None,
            "audio_analysis_error": None,
            "audio_analysis_time_ms": 0,
            "long_term_memories": [],
            "episodic_memories": [],
            "short_term_context": [],
            "session_metadata": {},
            "context_prep_error": None,
            "context_prep_time_ms": 0,
            "llm_response": None,
            "llm_error": None,
            "llm_time_ms": 0,
            "safety_flags": None,
            "safety_passed": False,
            "safety_action": "continue",
            "tts_response": None,
            "tts_error": None,
            "tts_time_ms": 0,
            # TTS Speaker Tracking (for session persistence)
            "current_tts_speaker": workflow_input.get("session_context", {}).get("current_tts_speaker"),
            "voice_preferences": workflow_input.get("session_context", {}).get("voice_preferences", {}),
            "conversation_stored": False,
            "memories_stored": False,
            "db_error": None,
            "db_time_ms": 0,
            "messages": [],
            "total_time_ms": 0,
            "workflow_status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }

        # Execute workflow with proper checkpointer based on environment
        workflow_start = time.time()

        logger.info(
            f"Executing workflow {initial_state['request_id']} for user {workflow_input['user_id']}"
        )


        # Per LangGraph docs: Use context manager for PostgresSaver
        if config.application.environment == "production":
            # Production: AsyncPostgresSaver with context manager
            # Pattern from knowledge base lines 27501-27538
            async with AsyncPostgresSaver.from_conn_string(config.database.url) as checkpointer:
                # Optional: Auto-create checkpoint tables (uncomment if needed)
                # await checkpointer.setup()
                
                # Compile graph with checkpointer inside context manager
                graph = create_workflow_graph(db_session)
                compiled_graph = graph  # Already compiled without checkpointer
                
                # Re-compile with checkpointer (proper pattern)
                workflow = StateGraph(WorkflowState)
                workflow.add_node("phase_1_audio_analysis", phase_1_audio_analysis)
                workflow.add_node("phase_2_context_preparation", phase_2_context_preparation)
                workflow.add_node("phase_3_llm_generation", phase_3_llm_generation_with_guardrails)
                workflow.add_node("phase_4_tts_generation", phase_4_tts_generation)
                workflow.add_node("phase_5_database_persistence", phase_5_database_persistence)
                workflow.add_edge(START, "phase_1_audio_analysis")
                workflow.add_edge("phase_1_audio_analysis", "phase_2_context_preparation")
                workflow.add_edge("phase_2_context_preparation", "phase_3_llm_generation")
                workflow.add_edge("phase_3_llm_generation", "phase_4_tts_generation")
                workflow.add_edge("phase_4_tts_generation", "phase_5_database_persistence")
                workflow.add_edge("phase_5_database_persistence", END)
                
                compiled_graph = workflow.compile(checkpointer=checkpointer)
                
                # Config with thread_id for state persistence
                run_config = {
                    "configurable": {
                        "thread_id": initial_state["session_id"],  # Session = thread
                    }
                }
                
                # Stream execution
                output = None
                async for chunk in compiled_graph.astream(initial_state, run_config):
                    node_name = list(chunk.keys())[0]
                    node_state = chunk[node_name]
                    logger.debug(f"Completed node: {node_name}")
                    output = node_state
        else:
            # Development: InMemorySaver (simpler, no context manager needed)
            checkpointer = InMemorySaver()
            graph = create_workflow_graph(db_session)
            
            # For development, compile inline with checkpointer
            workflow = StateGraph(WorkflowState)
            workflow.add_node("phase_1_audio_analysis", phase_1_audio_analysis)
            workflow.add_node("phase_2_context_preparation", phase_2_context_preparation)
            workflow.add_node("phase_3_llm_generation", phase_3_llm_generation_with_guardrails)
            workflow.add_node("phase_4_tts_generation", phase_4_tts_generation)
            workflow.add_node("phase_5_database_persistence", phase_5_database_persistence)
            workflow.add_edge(START, "phase_1_audio_analysis")
            workflow.add_edge("phase_1_audio_analysis", "phase_2_context_preparation")
            workflow.add_edge("phase_2_context_preparation", "phase_3_llm_generation")
            workflow.add_edge("phase_3_llm_generation", "phase_4_tts_generation")
            workflow.add_edge("phase_4_tts_generation", "phase_5_database_persistence")
            workflow.add_edge("phase_5_database_persistence", END)
            
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            
            run_config = {
                "configurable": {
                    "thread_id": initial_state["session_id"],
                }
            }
            
            output = None
            async for chunk in compiled_graph.astream(initial_state, run_config):
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                logger.debug(f"Completed node: {node_name}")
                output = node_state

        # Mark workflow as complete
        total_ms = int((time.time() - workflow_start) * 1000)
        output["total_time_ms"] = total_ms
        output["workflow_status"] = "completed"
        output["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Workflow {initial_state['request_id']} completed in {total_ms}ms\n"
            f"  Phase 1: {output.get('audio_analysis_time_ms', 0)}ms\n"
            f"  Phase 2: {output.get('context_prep_time_ms', 0)}ms\n"
            f"  Phase 3: {output.get('llm_time_ms', 0)}ms\n"
            f"  Phase 4: {output.get('tts_time_ms', 0)}ms\n"
            f"  Phase 5: {output.get('db_time_ms', 0)}ms"
        )

        return output

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        total_ms = int((time.time() - workflow_start) * 1000)
        initial_state["total_time_ms"] = total_ms
        initial_state["workflow_status"] = "failed"
        initial_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        return initial_state
    
    finally:
        # Close session if we created it ourselves
        if own_session and db_session:
            try:
                db_session.close()
            except Exception as e:
                logger.warning(f"Error closing workflow session: {e}")

# ============================================================================
# CLI Testing
# ============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        """Test workflow with sample audio."""
        if len(sys.argv) < 2:
            print("Usage: python workflow.py <audio_file>")
            sys.exit(1)

        audio_file = sys.argv[1]
        
        # Initialize database first (creates engine and SessionLocal)
        from backend.database.database import init_database
        init_database()
        
        # Import SessionLocal AFTER init_database() so it's not None
        from backend.database.database import SessionLocal
        db_session = SessionLocal()
        
        from uuid import uuid4
        
        user_id = uuid4()
        session_id = uuid4()

        try:
            result = await execute_workflow(
                workflow_input={
                    "audio_path": audio_file,
                    "user_id": "0c52f2b6-a083-4eef-b956-390c2bc0f542",
                    "session_id": "3233a928-47ea-4e03-af4e-71328659e8f9",

                    "character_profile": {
                        "name": "Aarav",
                        "background": "Empathetic AI companion for Indian youth",
                        "traits": ["empathetic", "culturally aware"],
                        "speech_style": "conversational",
                    },
                },
                db_session=db_session,
            )

            print("\n" + "=" * 80)
            print("WORKFLOW EXECUTION COMPLETE")
            print("=" * 80)

            if result.get("transcription"):
                print(f"\n[Transcription] {result['transcription'].text}")

            if result.get("llm_response"):
                print(f"\n[LLM Response] {result['llm_response'].response_text}")
                print(f"[Emotion] {result['llm_response'].detected_emotion}")
                print(f"[Intent] {result['llm_response'].detected_intent}")

            if result.get("tts_response"):
                print(f"\n[TTS Audio] {result['tts_response'].duration_seconds:.2f}s")
                print(
                    f"[Audio Output] {result['tts_response'].audio_base64_wav[:50]}..."
                )

            print(f"\n[Total Time] {result.get('total_time_ms', 0)}ms")
            print(f"[Status] {result.get('workflow_status')}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            await _workflow_services.shutdown_async()
            db_session.close()

    # Fix for Windows: psycopg async needs SelectorEventLoop, not ProactorEventLoop
    import sys
    if sys.platform == 'win32':
        import asyncio
        import selectors
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
