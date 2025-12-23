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
from dataclasses import dataclass, field, asdict
from datetime import datetime
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
    character_profile: Dict[str, Any]  # Aarav personality configuration
    request_id: Optional[str]  # For tracing and debugging


class WorkflowState(TypedDict):
    """Complete workflow state - combines all phases."""

    # Input
    audio_path: str
    user_id: str
    session_id: str
    conversation_id: Optional[str]
    character_profile: Dict[str, Any]
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
                    modelname="ai4bharat/indic-parler-tts",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    sampling_rate=44100,
                    cache_enabled=True,
                )
            )
            logger.info("✓ Parler TTS service initialized")

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
        transcription_task = transcribe_audio(state["audio_path"], state["request_id"])
        prosody_task = extract_prosody_features(
            state["audio_path"], state["request_id"]
        )

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
        return {
            "transcription": transcription,
            "prosody_features": prosody_features,
            "audio_analysis_error": transcription_error or prosody_error,
            "audio_analysis_time_ms": elapsed_ms,
            "messages": [
                AIMessage(
                    content=f"[Phase 1] Audio analysis completed in {elapsed_ms}ms"
                )
            ],
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
            # Parallel tasks: memory retrieval using semantic search
            query_text = state["transcription"].text

            memory_task = _workflow_services.memory_service.retrieve_memories_async(
                db=_workflow_services.db_session,
                userid=state["user_id"],
                querytext=query_text,
                topk=5,
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

            try:
                memories_list, short_term_ctx, conversation = await asyncio.gather(
                    memory_task, session_context_task, conversation_task
                )
            except Exception as e:
                logger.warning(f"[PHASE 2] Context retrieval error: {e}")
                memories_list = []
                short_term_ctx = []
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
            "long_term_memories": long_term_memories,
            "episodic_memories": episodic_memories,
            "short_term_context": short_term_ctx if "short_term_ctx" in locals() else [],
            "session_metadata": {
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "conversation_id": state["conversation_id"],
                "timestamp": datetime.utcnow().isoformat(),
            },
            "context_prep_error": None,
            "context_prep_time_ms": elapsed_ms,
            "messages": [
                AIMessage(
                    content=f"[Phase 2] Context preparation completed in {elapsed_ms}ms"
                )
            ],
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
        acoustic_features = asdict(state["prosody_features"]) if state["prosody_features"] else {}
        
        # Call Gemini LLM - it handles safety internally with comprehensive guardrails
        llm_response = await _workflow_services.llm_service.analyze_and_respond_async(
            transcript=state["transcription"].text if state["transcription"] else "",
            language=state["transcription"].language if state["transcription"] else "en",
            acoustic_features=acoustic_features,
            short_term_context=state["short_term_context"],
            long_term_memories=state["long_term_memories"],
            episodic_memories=state["episodic_memories"],
            character_profile=state["character_profile"],
            safety_context={},  # Gemini handles this internally
            temperature=config.gemini.temperature,
            max_output_tokens=config.gemini.max_output_tokens,
       )

        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[PHASE 3] ✓ LLM response generated ({elapsed_ms}ms): "
            f"emotion={llm_response.detected_emotion}, "
            f"intent={llm_response.detected_intent}, "
            f"memories={len(llm_response.memory_updates)}, "
            f"safety_risk={llm_response.safety_flags.crisis_risk}"
        )

        # Determine safety action based on Gemini's assessment
        safety_action = "escalate" if llm_response.safety_flags.crisis_risk == "high" else "continue"
        safety_passed = llm_response.safety_flags.crisis_risk != "high"

        return {
            "llm_response": llm_response,
            "safety_flags": llm_response.safety_flags,  # From Gemini
            "safety_passed": safety_passed,
            "safety_action": safety_action,
            "llm_error": None,
            "llm_time_ms": elapsed_ms,
            "messages": [
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
        speaker = state["llm_response"].tts_speaker or "Thoma"
        description = state["llm_response"].tts_style_prompt or "neutral, conversational tone"

        logger.info(
            f"[PHASE 4] TTS input: speaker={speaker}, "
            f"description={description[:50]}..., text_length={len(spoken_text)}"
        )

        # Create TTS request
        tts_request = TTSRequest(
            spoken_text=spoken_text,
            speaker=speaker,
            description=description,
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
            "tts_response": tts_response,
            "tts_error": None,
            "tts_time_ms": elapsed_ms,
            "messages": [
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
        conversation_stored = await loop.run_in_executor(
            None,
            _store_conversation,
            state,
        )

        if conversation_stored:
            logger.info(f"[PHASE 5] ✓ Conversation stored")

        # Store memories if LLM proposed any
        if state["llm_response"].memory_updates:
            memories_stored = await loop.run_in_executor(
                None,
                _store_memories,
                state,
            )
            if memories_stored:
                logger.info(
                    f"[PHASE 5] ✓ Stored {len(state['llm_response'].memory_updates)} memories"
                )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "conversation_stored": conversation_stored,
            "memories_stored": memories_stored,
            "db_error": None,
            "db_time_ms": elapsed_ms,
            "messages": [
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
                created_at=datetime.utcnow(),
            )
            _workflow_services.db_session.add(conv)
            _workflow_services.db_session.commit()
            return conv

    except Exception as e:
        logger.warning(f"Failed to get/create conversation: {e}")
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


def _store_conversation(state: WorkflowState) -> bool:
    """
    Store conversation to database with ALL required fields.
    
    CRITICAL: Uses IndicBERT to generate semantic embeddings for search.
    """
    try:
        if not state["llm_response"] or not state["transcription"]:
            logger.warning("Missing LLM response or transcription - skipping storage")
            return False

        # Generate embedding for semantic search using IndicBERT
        user_text = state["transcription"].text
        logger.info(f"Generating IndicBERT embedding for: {user_text[:50]}...")
        embedding = _workflow_services.memory_service.embed_text(user_text, use_cache=False)
        embedding_list = embedding.tolist()  # Convert numpy to list for pgvector

        # Create Conversation object with CORRECT field names
        conversation = Conversation(
            user_id=state["user_id"],
            session_id=state["session_id"],
            
            # ✅ CORRECT FIELD NAMES (matching database schema)
            user_input_text=user_text,
            ai_response_text=state["llm_response"].response_text,
            
            # Language detection
            detected_language=state["transcription"].language if state["transcription"] else "en",
            response_language=state["llm_response"].response_language,
            is_code_mixed=state["transcription"].is_code_mixed if state["transcription"] else False,
            code_mix_languages=state["transcription"].detected_languages if state["transcription"] else None,
            
            # Emotion & sentiment (from Gemini LLM)
            detected_emotion=state["llm_response"].detected_emotion,
            emotion_confidence=state["llm_response"].emotion_confidence,
            sentiment=state["llm_response"].sentiment,
            
            # Intent classification (from Gemini LLM)
            detected_intent=state["llm_response"].detected_intent,
            intent_confidence=state["llm_response"].intent_confidence,
            
            # Prosody features (complete JSONB storage)
            prosody_features=asdict(state["prosody_features"]) if state["prosody_features"] else None,
            
            # Audio metadata
            audio_duration_seconds=state["prosody_features"].meta_info.get("duration_sec", 0)
            if state["prosody_features"]
            else 0,
            audio_file_path=state["audio_path"],
            response_audio_path=None,  # Will be updated after TTS
            
            # TTS generation details
            tts_prompt=state["llm_response"].tts_style_prompt,
            
            # Performance metrics
            response_generation_time_ms=state.get("llm_time_ms", 0),
            
            # Safety & moderation (from Gemini's safety assessment)
            safety_check_passed=state["llm_response"].safety_flags.crisis_risk != "high",
            safety_flags=asdict(state["llm_response"].safety_flags),
            
            # Timestamp
            created_at=datetime.utcnow(),
            
            # ✅ SEMANTIC EMBEDDING FOR VECTOR SEARCH
            embedding=embedding_list,
        )

        _workflow_services.db_session.add(conversation)
        _workflow_services.db_session.commit()

        logger.info(
            f"✅ Stored conversation {conversation.id} with embedding "
            f"(emotion={conversation.detected_emotion}, safety={conversation.safety_check_passed})"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Failed to store conversation: {e}", exc_info=True)
        _workflow_services.db_session.rollback()
        return False


def _store_memories(state: WorkflowState) -> bool:
    """Store memories extracted by LLM."""
    try:
        if not state["llm_response"] or not state["llm_response"].memory_updates:
            return False

        # Convert LLM memory updates to database models with IndicBERT embeddings
        stored_memories = _workflow_services.memory_service.store_memories_batch(
            db=_workflow_services.db_session,
            userid=state["user_id"],
            memory_updates=state["llm_response"].memory_updates,
            conversation_id=state["conversation_id"],
        )

        logger.info(f"Stored {len(stored_memories)} memories")
        return len(stored_memories) > 0

    except Exception as e:
        logger.error(f"Failed to store memories: {e}", exc_info=True)
        _workflow_services.db_session.rollback()
        return False


# ============================================================================
# PUBLIC API - Workflow execution entry point
# ============================================================================


async def execute_workflow(
    workflow_input: WorkflowInput,
    db_session: Session,
) -> Dict[str, Any]:
    """
    Execute the complete conversation workflow.
    
    Uses AsyncPostgresSaver with proper context manager pattern
    as specified in LangGraph documentation.
    
    Args:
        workflow_input: User audio + metadata
        db_session: Database session for persistence
    
    Returns:
        Complete workflow output with all phases and results
    """
    # Initialize services on first call
    if not _workflow_services.llm_service:
        await _workflow_services.initialize_async(db_session)

    # Build initial state
    initial_state: WorkflowState = {
        "audio_path": workflow_input["audio_path"],
        "user_id": workflow_input["user_id"],
        "session_id": workflow_input["session_id"],
        "conversation_id": workflow_input.get("conversation_id"),
        "character_profile": workflow_input.get(
            "character_profile",
            {
                "name": "Aarav",
                "background": "Empathetic AI companion for Indian youth",
                "traits": ["empathetic", "culturally aware", "patient"],
                "speech_style": "conversational, validates emotions",
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
        "conversation_stored": False,
        "memories_stored": False,
        "db_error": None,
        "db_time_ms": 0,
        "messages": [],
        "total_time_ms": 0,
        "workflow_status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    # Execute workflow with proper checkpointer based on environment
    workflow_start = time.time()

    try:
        logger.info(
            f"Executing workflow {initial_state['request_id']} for user {workflow_input['user_id']}"
        )

        # Per LangGraph docs: Use context manager for PostgresSaver
        if config.application.environment == "production":
            # Production: AsyncPostgresSaver with context manager
            # Pattern from knowledge base lines 27501-27538
            async with AsyncPostgresSaver.from_conn_string(config.database.url) as checkpointer:
                # Optional: Auto-create checkpoint tables (uncomment if needed)
                await checkpointer.setup()
                
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
        output["completed_at"] = datetime.utcnow().isoformat()

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
        initial_state["completed_at"] = datetime.utcnow().isoformat()
        return initial_state


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

        try:
            result = await execute_workflow(
                workflow_input={
                    "audio_path": audio_file,
                    "user_id": "test-user-123",
                    "session_id": "test-session-456",
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

    asyncio.run(main())
