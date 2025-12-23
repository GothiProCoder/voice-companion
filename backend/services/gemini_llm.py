"""
GuppShupp Gemini LLM Service Module
backend/services/gemini_llm.py

PRODUCTION-GRADE LLM SERVICE FOR GUPPSHUPP
- Strictly adheres to Google Gen AI Python SDK (google.genai, google.genai.types)
- Structured JSON output for database integration
- Rich context aggregation: transcript + acoustic + memory + personality
- No restrictions on LLM value generation (emotion/sentiment/intent are free-form)
- Safe by default, enforced by design

References:
- Google Gen AI SDK: https://googleapis.github.io/python-genai
- GuppShupp Spec: GUPPSHUPP_COMPREHENSIVE_SPECIFICATION.docx (authority)
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ============================================================================
# DATA MODELS FOR STRUCTURED OUTPUT
# ============================================================================

@dataclass
class MemoryUpdate:
    """Proposed memory to store from this conversation turn."""
    type: str  # "long_term" | "episodic" | "semantic"
    text: str
    category: str  # "work_study", "relationships", "health", "emotional", etc.
    importance: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyFlags:
    """Safety context detected by LLM (pre-guardrail info)."""
    crisis_risk: str  # "low" | "medium" | "high"
    self_harm_mentioned: bool
    abuse_mentioned: bool
    medical_concern: bool
    flagged_keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeminiLLMResponse:
    """
    Structured response from Gemini LLM.
    Maps directly to CONVERSATIONS table columns.
    """
    # Core response
    response_text: str
    response_language: str  # e.g., "hi-en" (Hinglish), "en", "hi"
    
    # Emotion/sentiment/intent (detected by LLM, free-form values)
    detected_emotion: str  # e.g., "sadness", "joy", "anxious", "frustrated"
    emotion_confidence: float  # 0.0 to 1.0
    sentiment: str  # e.g., "negative", "positive", "neutral", "mixed"
    detected_intent: str  # e.g., "expressing_emotion", "greeting", "seeking_help"
    intent_confidence: float  # 0.0 to 1.0
    
    # TTS integration
    tts_style_prompt: str
    tts_speaker: str  # e.g., "Rohit" for Hindi, "Thoma" for English
    
    # Memory operations (proposed by LLM)
    memory_updates: List[MemoryUpdate]
    
    # Safety pre-check (before dedicated guardrail layer)
    safety_flags: SafetyFlags
    
    # Latency tracking
    generation_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for database storage."""
        return {
            "response_text": self.response_text,
            "response_language": self.response_language,
            "detected_emotion": self.detected_emotion,
            "emotion_confidence": self.emotion_confidence,
            "sentiment": self.sentiment,
            "detected_intent": self.detected_intent,
            "intent_confidence": self.intent_confidence,
            "tts_prompt": self.tts_style_prompt,
            "tts_speaker": self.tts_speaker,
            "memory_updates": [m.to_dict() for m in self.memory_updates],
            "safety_flags": self.safety_flags.to_dict(),
            "response_generation_time_ms": self.generation_time_ms,
        }


# ============================================================================
# RESPONSE JSON SCHEMA (GROUND TRUTH)
# ============================================================================

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_text": {
            "type": "string",
            "description": "Main conversational response to user, emotionally attuned"
        },
        "response_language": {
            "type": "string",
            "description": "Language code (e.g., 'hi', 'en', 'hi-en' for code-mixing)"
        },
        "detected_emotion": {
            "type": "string",
            "description": "Primary emotion (free-form label, e.g., 'sadness', 'joy', 'anxious')"
        },
        "emotion_confidence": {
            "type": "number",
            "description": "Confidence in emotion detection (0.0 to 1.0)",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "sentiment": {
            "type": "string",
            "description": "Overall sentiment (free-form, e.g., 'positive', 'negative', 'neutral')"
        },
        "detected_intent": {
            "type": "string",
            "description": "User intent (free-form label, e.g., 'expressing_emotion', 'greeting')"
        },
        "intent_confidence": {
            "type": "number",
            "description": "Confidence in intent detection (0.0 to 1.0)",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "tts_style_prompt": {
            "type": "string",
            "description": "Emotion-aligned TTS description for Indic Parler (Aarav personality, tone)"
        },
        "tts_speaker": {
            "type": "string",
            "description": "Recommended speaker (e.g., 'Rohit' for Hindi, 'Thoma' for English)"
        },
        "memory_updates": {
            "type": "array",
            "description": "Proposed memories to extract and store",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["long_term", "episodic", "semantic"]
                    },
                    "text": {
                        "type": "string",
                        "description": "Memory fact or observation"
                    },
                    "category": {
                        "type": "string",
                        "description": "Memory category (work_study, relationships, health, emotional, etc.)"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["type", "text", "category", "importance"]
            }
        },
        "safety_flags": {
            "type": "object",
            "description": "Safety context detected (pre-guardrail)",
            "properties": {
                "crisis_risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high"]
                },
                "self_harm_mentioned": {"type": "boolean"},
                "abuse_mentioned": {"type": "boolean"},
                "medical_concern": {"type": "boolean"},
                "flagged_keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["crisis_risk", "self_harm_mentioned", "abuse_mentioned", "medical_concern", "flagged_keywords"]
        }
    },
    "required": [
        "response_text", "response_language", "detected_emotion", "emotion_confidence",
        "sentiment", "detected_intent", "intent_confidence", "tts_style_prompt",
        "tts_speaker", "memory_updates", "safety_flags"
    ]
}


# ============================================================================
# INDIC PARLER TTS SPEAKER MAPPING (FROM SPEC)
# ============================================================================

TTS_SPEAKER_MAP = {
    "hi": {"default": "Rohit", "female": "Divya"},
    "en": {"default": "Thoma", "female": "Mary"},
    "ta": {"default": "Prakash", "female": "Jaya"},
    "te": {"default": "Prakash", "female": "Lalitha"},
    "kn": {"default": "Suresh", "female": "Anu"},
    "ml": {"default": "Harish", "female": "Anjali"},
    "bn": {"default": "Arjun", "female": "Aditi"},
    "mr": {"default": "Sanjay", "female": "Sunita"},
    "gu": {"default": "Yash", "female": "Neha"},
    "pa": {"default": "Divjot", "female": "Gurpreet"},
    "or": {"default": "Manas", "female": "Debjani"},
}


# ============================================================================
# GEMINI LLM SERVICE
# ============================================================================

logger = logging.getLogger(__name__)


class GeminiLLMService:
    """
    Core LLM service for GuppShupp.
    
    Responsibilities:
    1. Aggregate rich context (transcript + acoustic + memory + personality)
    2. Generate structured response via Gemini 2.0 Flash
    3. Extract emotion, sentiment, intent (LLM-determined, unconstrained)
    4. Propose memories for storage
    5. Provide TTS styling guidance
    6. Pre-check safety (crisis, self-harm, abuse)
    
    Uses Google Gen AI SDK (official, production-ready).
    No restrictions on LLM value generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "gemini-2.5-flash",
        vertexai: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        http_options: Optional[types.HttpOptions] = None,
    ):
        """
        Initialize Gemini LLM client.
        
        Args:
            api_key: GOOGLE_API_KEY (env var if None)
            model: Model name (gemini-2.0-flash-exp or gemini-2.5-flash)
            vertexai: Use Vertex AI backend
            project: GCP project (for Vertex AI)
            location: GCP region (for Vertex AI)
            http_options: HTTP configuration (timeout, API version, etc.)
        
        SDK Reference:
            - Client init: https://googleapis.github.io/python-genai#client-initialization
            - API versions: v1, v1alpha (Gemini API); v1, v1alpha (Vertex)
        """
        self._client = genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            http_options=http_options,
        )
        self._model = model
        logger.info(f"GeminiLLMService initialized with model={model}")

    def _attempt_json_repair(self, broken_json: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair broken JSON by completing truncated strings/objects.
        """
        import re
        
        try:
            # Try to find where the JSON breaks
            # If it's an unterminated string, try to close it
            if broken_json.count('"') % 2 != 0:
                # Odd number of quotes - unterminated string
                broken_json = broken_json + '"'
            
            # Count braces
            open_braces = broken_json.count('{')
            close_braces = broken_json.count('}')
            if open_braces > close_braces:
                broken_json = broken_json + ('}' * (open_braces - close_braces))
            
            # Try parsing again
            return json.loads(broken_json)
        except:
            return None


    def _call_gemini_with_retry(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Call Gemini with automatic retry on JSON parsing failures.
        Guarantees valid response or fallback.
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini API call attempt {attempt + 1}/{max_retries}")
                
                # Make API call
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                
                # Extract response text (try multiple methods)
                full_text = None
                try:
                    full_text = resp.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    try:
                        full_text = resp.text
                    except:
                        pass
                
                if not full_text:
                    logger.warning(f"Attempt {attempt + 1}: Empty response")
                    continue
                
                logger.debug(f"Raw response length: {len(full_text)} chars")
                
                # Clean response
                cleaned_text = self._clean_json_response(full_text)
                
                # Try parsing
                try:
                    response_dict = json.loads(cleaned_text)
                    
                    # Validate structure
                    if self._validate_response_structure(response_dict):
                        logger.info(f"Valid response received on attempt {attempt + 1}")
                        return response_dict
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Incomplete structure")
                        continue
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: JSON parse failed - {e}")
                    
                    # Try to repair the JSON
                    repaired = self._attempt_json_repair(cleaned_text)
                    if repaired and self._validate_response_structure(repaired):
                        logger.info(f"JSON repaired successfully on attempt {attempt + 1}")
                        return repaired
                    
                    # Log the problematic response
                    logger.debug(f"Problematic JSON (first 1000 chars): {cleaned_text[:1000]}")
                    last_error = e
                    continue
            
            except genai_errors.APIError as e:
                logger.error(f"Attempt {attempt + 1}: API error [{e.code}]: {e.message}")
                last_error = e
                
                # Don't retry on quota/auth errors
                if e.code in [429, 401, 403]:
                    break
                
                # Wait before retry
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Unexpected error - {e}")
                last_error = e
                continue
        
        # All retries failed - use fallback
        logger.error(f"All {max_retries} attempts failed. Using fallback response.")
        if last_error:
            logger.error(f"Last error: {last_error}")
        
        return self._fallback_response()


    async def _call_gemini_with_retry_async(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Async version with retry logic."""
        import asyncio
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini async call attempt {attempt + 1}/{max_retries}")
                
                resp = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                
                # Extract response
                full_text = None
                try:
                    full_text = resp.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    try:
                        full_text = resp.text
                    except:
                        pass
                
                if not full_text:
                    logger.warning(f"Attempt {attempt + 1}: Empty response")
                    continue
                
                cleaned_text = self._clean_json_response(full_text)
                
                try:
                    response_dict = json.loads(cleaned_text)
                    
                    if self._validate_response_structure(response_dict):
                        logger.info(f"Valid response on attempt {attempt + 1}")
                        return response_dict
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Incomplete structure")
                        continue
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: JSON parse failed - {e}")
                    
                    repaired = self._attempt_json_repair(cleaned_text)
                    if repaired and self._validate_response_structure(repaired):
                        logger.info(f"JSON repaired on attempt {attempt + 1}")
                        return repaired
                    
                    logger.debug(f"Problematic JSON: {cleaned_text[:1000]}")
                    last_error = e
                    continue
            
            except genai_errors.APIError as e:
                logger.error(f"Attempt {attempt + 1}: API error - {e.message}")
                last_error = e
                
                if e.code in [429, 401, 403]:
                    break
                
                await asyncio.sleep(1 * (attempt + 1))
                continue
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Unexpected error - {e}")
                last_error = e
                continue
        
        logger.error(f"All {max_retries} attempts failed. Using fallback.")
        return self._fallback_response()
    
    def analyze_and_respond(
        self,
        transcript: str,
        language: str,
        acoustic_features: Dict[str, Any],
        short_term_context: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]],
        character_profile: Dict[str, Any],
        safety_context: Dict[str, Any],
        *,
        temperature: float = 0.7,
        max_output_tokens: int = 2000,
    ) -> GeminiLLMResponse:
        """
        Main entry point: aggregate context, call Gemini, parse structured response.
        
        GUARANTEED TO RETURN A VALID RESPONSE - never crashes.
        Uses retry logic with automatic JSON repair and fallback.
        
        Args:
            transcript: User's transcribed speech text
            language: Detected language code (e.g., "hi", "en", "hi-en")
            acoustic_features: JSON from emotion_detection.py (prosody, pitch, energy, etc.)
            short_term_context: Last N conversations from current session
            long_term_memories: Retrieved facts/preferences (IndicBERT semantic search)
            episodic_memories: Past emotional summaries with context
            character_profile: Aarav personality (name, background, traits, speech style)
            safety_context: Flags from previous turns (e.g., recent crisis mention)
            temperature: 0.0 to 1.0 (lower = more deterministic)
            max_output_tokens: Token limit for response
            
        Returns:
            GeminiLLMResponse with structured output (guaranteed valid)
        """
        import time
        start_time = time.time()

        # Build comprehensive prompt
        prompt = self._build_prompt(
            transcript=transcript,
            language=language,
            acoustic_features=acoustic_features,
            short_term_context=short_term_context,
            long_term_memories=long_term_memories,
            episodic_memories=episodic_memories,
            character_profile=character_profile,
            safety_context=safety_context,
        )

        # Build Gemini config with structured JSON output
        config = types.GenerateContentConfig(
            system_instruction=self._build_system_instruction(character_profile),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=GEMINI_RESPONSE_SCHEMA,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        )

        try:
            logger.info(f"Calling Gemini {self._model} for analysis")
            
            # Use retry logic - GUARANTEED to return valid response
            response_dict = self._call_gemini_with_retry(
                prompt=prompt,
                config=config,
                max_retries=3,
            )
            
            # Convert to typed response
            generation_time = int((time.time() - start_time) * 1000)
            structured_response = self._parse_gemini_response(response_dict, generation_time)
            
            logger.info(f"Response ready ({generation_time}ms)")
            return structured_response

        except Exception as e:
            # Ultimate fallback - should NEVER happen with retry logic
            logger.critical(f"CRITICAL: Fallback triggered after retry logic: {e}", exc_info=True)
            generation_time = int((time.time() - start_time) * 1000)
            return self._parse_gemini_response(self._fallback_response(), generation_time)

    def _validate_response_structure(self, response_dict: Dict[str, Any]) -> bool:
        """Check if response has all required keys."""
        required_keys = [
            "response_text", "response_language", "detected_emotion", 
            "emotion_confidence", "sentiment", "detected_intent",
            "intent_confidence", "tts_style_prompt", "tts_speaker",
            "memory_updates", "safety_flags"
        ]
        return all(key in response_dict for key in required_keys)


    async def analyze_and_respond_async(
        self,
        transcript: str,
        language: str,
        acoustic_features: Dict[str, Any],
        short_term_context: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]],
        character_profile: Dict[str, Any],
        safety_context: Dict[str, Any],
        *,
        temperature: float = 0.7,
        max_output_tokens: int = 2000,
    ) -> GeminiLLMResponse:
        """
        Async variant using client.aio.
        
        GUARANTEED TO RETURN A VALID RESPONSE - never crashes.
        Uses retry logic with automatic JSON repair and fallback.
        """
        import time
        start_time = time.time()

        prompt = self._build_prompt(
            transcript=transcript,
            language=language,
            acoustic_features=acoustic_features,
            short_term_context=short_term_context,
            long_term_memories=long_term_memories,
            episodic_memories=episodic_memories,
            character_profile=character_profile,
            safety_context=safety_context,
        )

        config = types.GenerateContentConfig(
            system_instruction=self._build_system_instruction(character_profile),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=GEMINI_RESPONSE_SCHEMA,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        )

        try:
            logger.info(f"Calling Gemini {self._model} (async)")
            
            # Use async retry logic - GUARANTEED to return valid response
            response_dict = await self._call_gemini_with_retry_async(
                prompt=prompt,
                config=config,
                max_retries=3,
            )
            
            generation_time = int((time.time() - start_time) * 1000)
            structured_response = self._parse_gemini_response(response_dict, generation_time)
            
            logger.info(f"Async response ready ({generation_time}ms)")
            return structured_response

        except Exception as e:
            logger.critical(f"CRITICAL: Fallback after async retry: {e}", exc_info=True)
            generation_time = int((time.time() - start_time) * 1000)
            return self._parse_gemini_response(self._fallback_response(), generation_time)


    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Clean Gemini response text to extract valid JSON.
        Handles markdown code blocks and other formatting issues.
        """
        text = text.strip()
        
        # Remove markdown code blocks (`````` or ``````)
        if text.startswith("```"):
            # Find the first newline after ```
            first_newline = text.find('\n')
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove trailing ```
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        return text

    # ========================================================================
    # PROMPT ENGINEERING & SYSTEM INSTRUCTIONS
    # ========================================================================

    def _build_system_instruction(self, character_profile: Dict[str, Any]) -> str:
        """
        Build system instruction for Aarav personality + safety constraints.
        
        SDK Reference: system_instruction in GenerateContentConfig
        https://googleapis.github.io/python-genai#configuring-generation
        """
        name = character_profile.get("name", "Aarav")
        background = character_profile.get("background", "")
        traits = character_profile.get("traits", [])
        speech_style = character_profile.get("speech_style", "")

        traits_str = ", ".join(traits) if traits else "empathetic, thoughtful, culturally aware"
        
        return f"""You are {name}, an emotionally intelligent AI voice assistant designed for Indian language speakers.

PERSONALITY & BACKGROUND:
- Name: {name}
- Background: {background}
- Traits: {traits_str}
- Speech Style: {speech_style}

CORE RESPONSIBILITIES:
1. Understand emotion from BOTH text content AND acoustic cues (pitch, energy, speech rate, pauses)
2. Recognize user intent and emotional state, responding with empathy
3. Use conversation history to provide contextual, personalized responses
4. Extract and propose valuable memories for future reference
5. Generate emotionally-aware TTS instructions for tone-appropriate speech

SAFETY GUARDRAILS (pre-check, before dedicated security layer):
- Recognize crisis signals (self-harm, abuse, severe distress)
- Avoid medical diagnosis; suggest consultation when appropriate
- Do NOT provide harmful instructions or validate destructive behaviors
- Offer supportive alternatives and resources when needed
- Flag suspicious patterns for the security layer

OUTPUT FORMAT:
Always respond with the exact JSON structure specified.
Never deviate from the schema.
Values are FREE-FORM (not restricted to predefined lists).

LANGUAGE & CULTURAL SENSITIVITY:
- Respond in the user's preferred language or code-mix (Hinglish, Tanglish, etc.)
- Understand Indian cultural context, family dynamics, work-study pressures
- Use natural, conversational tone appropriate to age and background
- Validate feelings; don't just provide information

NOW ANALYZE THE USER'S INPUT AND GENERATE YOUR STRUCTURED RESPONSE."""

    def _build_prompt(
        self,
        transcript: str,
        language: str,
        acoustic_features: Dict[str, Any],
        short_term_context: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]],
        character_profile: Dict[str, Any],
        safety_context: Dict[str, Any],
    ) -> str:
        """
        Build rich context prompt for Gemini.
        Aggregates all available context in a structured format.
        """
        
        # Format acoustic features
        acoustic_json = json.dumps(acoustic_features, indent=2, ensure_ascii=False)
        
        # Format short-term context (last N turns)
        short_term_str = ""
        if short_term_context:
            short_term_str = "\nSHORT-TERM CONTEXT (Recent conversation):\n"
            for turn in short_term_context[-5:]:  # Last 5 turns
                user_text = turn.get("user_input_text", "")
                ai_text = turn.get("ai_response_text", "")
                emotion = turn.get("detected_emotion", "unknown")
                short_term_str += f"- User: {user_text} [{emotion}]\n  AI: {ai_text}\n"
        
        # Format long-term memories
        long_term_str = ""
        if long_term_memories:
            long_term_str = "\nLONG-TERM MEMORIES (Facts, preferences, triggers):\n"
            for mem in long_term_memories:
                text = mem.get("memory_text", "")
                importance = mem.get("importance_score", 0)
                long_term_str += f"- {text} (importance: {importance})\n"
        
        # Format episodic memories
        episodic_str = ""
        if episodic_memories:
            episodic_str = "\nEPISODIC MEMORIES (Past emotional arcs, summaries):\n"
            for mem in episodic_memories:
                text = mem.get("memory_text", "")
                emotion_tone = mem.get("emotional_tone", "neutral")
                episodic_str += f"- {text} (tone: {emotion_tone})\n"
        
        # Format safety context
        safety_str = ""
        if safety_context:
            crisis = safety_context.get("crisis_risk", "low")
            recent_flags = safety_context.get("recent_flags", [])
            safety_str = f"\nSAFETY CONTEXT:\n- Crisis risk level: {crisis}\n"
            if recent_flags:
                safety_str += f"- Recent flags: {', '.join(recent_flags)}\n"
        
        return f"""USER INPUT ANALYSIS
================

TRANSCRIPT:
{transcript}

DETECTED LANGUAGE: {language}

ACOUSTIC FEATURES (librosa/OpenSMILE):
{acoustic_json}{short_term_str}{long_term_str}{episodic_str}{safety_str}

CHARACTER CONTEXT:
- Name: {character_profile.get('name', 'Aarav')}
- Personality: {character_profile.get('traits', [])}

TASK:
Analyze the user's input using ALL available context.
Generate a response that is:
1. Emotionally intelligent (validate, empathize, understand nuance)
2. Contextually aware (reference past conversations, memories)
3. Personality-aligned ({character_profile.get('name', 'Aarav')}'s traits and speech style)
4. Safe by default (pre-check for crisis, abuse, medical risks)
5. Memory-extractive (propose new facts to remember)
6. TTS-optimized (describe emotional tone for speech synthesis)

EMOTION/SENTIMENT/INTENT GUIDANCE (use these or appropriate alternatives):
- Emotions: joy, sadness, anger, fear, surprise, disgust, neutral (or: anxious, frustrated, calm, overwhelmed, etc.)
- Intents: greeting, question, complaint, request, expressing_emotion, crisis_signal (or: venting, seeking_advice, small_talk, etc.)
- Sentiments: positive, negative, neutral (or: mixed, slightly_positive, very_negative, etc.)

Respond with ONLY the JSON object."""

    # ========================================================================
    # RESPONSE PARSING & FALLBACK
    # ========================================================================

    def _parse_gemini_response(
        self,
        response_dict: Dict[str, Any],
        generation_time: int,
    ) -> GeminiLLMResponse:
        """
        Parse Gemini JSON response into typed GeminiLLMResponse.
        
        The response_dict has already been validated against GEMINI_RESPONSE_SCHEMA,
        but we perform additional type coercion and safety checks.
        """
        
        # Parse memory updates
        memory_updates = []
        for mem_dict in response_dict.get("memory_updates", []):
            try:
                memory_updates.append(MemoryUpdate(
                    type=mem_dict.get("type", "long_term"),
                    text=mem_dict.get("text", ""),
                    category=mem_dict.get("category", "other"),
                    importance=float(mem_dict.get("importance", 0.5)),
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse memory: {e}")
        
        # Parse safety flags
        safety_dict = response_dict.get("safety_flags", {})
        safety_flags = SafetyFlags(
            crisis_risk=safety_dict.get("crisis_risk", "low"),
            self_harm_mentioned=safety_dict.get("self_harm_mentioned", False),
            abuse_mentioned=safety_dict.get("abuse_mentioned", False),
            medical_concern=safety_dict.get("medical_concern", False),
            flagged_keywords=safety_dict.get("flagged_keywords", []),
        )
        
        # Construct response
        return GeminiLLMResponse(
            response_text=response_dict.get("response_text", ""),
            response_language=response_dict.get("response_language", "en"),
            detected_emotion=response_dict.get("detected_emotion", "neutral"),
            emotion_confidence=float(response_dict.get("emotion_confidence", 0.0)),
            sentiment=response_dict.get("sentiment", "neutral"),
            detected_intent=response_dict.get("detected_intent", "unknown"),
            intent_confidence=float(response_dict.get("intent_confidence", 0.0)),
            tts_style_prompt=response_dict.get("tts_style_prompt", ""),
            tts_speaker=response_dict.get("tts_speaker", "Thoma"),
            memory_updates=memory_updates,
            safety_flags=safety_flags,
            generation_time_ms=generation_time,
        )

    @staticmethod
    def _fallback_response() -> Dict[str, Any]:
        """
        Fallback response when Gemini fails (network, quota, etc.).
        Ensures the system never crashes.
        """
        return {
            "response_text": "I'm having trouble processing your request right now. Please try again in a moment.",
            "response_language": "en",
            "detected_emotion": "neutral",
            "emotion_confidence": 0.0,
            "sentiment": "neutral",
            "detected_intent": "unknown",
            "intent_confidence": 0.0,
            "tts_style_prompt": "Calm, gentle, supportive tone",
            "tts_speaker": "Thoma",
            "memory_updates": [],
            "safety_flags": {
                "crisis_risk": "low",
                "self_harm_mentioned": False,
                "abuse_mentioned": False,
                "medical_concern": False,
                "flagged_keywords": [],
            }
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_tts_speaker(self, language: str, emotion: str = None) -> str:
        """
        Get recommended TTS speaker for language + emotion.
        
        Args:
            language: Language code (e.g., "hi", "en", "ta")
            emotion: Optional emotion to influence speaker choice
            
        Returns:
            Speaker name (e.g., "Rohit", "Thoma")
        """
        lang_speakers = TTS_SPEAKER_MAP.get(language, {"default": "Thoma", "female": "Mary"})
        
        # Choose female speaker for certain emotions (sadness, vulnerability)
        if emotion and emotion.lower() in ["sadness", "vulnerability", "anxiety"]:
            return lang_speakers.get("female", lang_speakers.get("default"))
        
        return lang_speakers.get("default", "Thoma")

    def close(self):
        """Close synchronous client."""
        self._client.close()
        logger.info("GeminiLLMService client closed")

    async def aclose(self):
        """Close asynchronous client."""
        await self._client.aio.aclose()
        logger.info("GeminiLLMService async client closed")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize service
    llm_service = GeminiLLMService(api_key="GEMINI_API_KEY")
    
    # Example inputs
    transcript = "मैं परीक्षा में fail हो गया। बहुत depressed हूँ।"  # Hinglish
    language = "hi-en"
    acoustic_features = {
        "pitch": {"f0_mean": 165.2, "f0_variance": 35.1, "f0_range": [110, 220]},
        "energy": {"rms_mean": 0.035, "rms_variance": 0.018},
        "speech_rate": {"onset_rate": 2.1, "syllable_rate": 2.8},
        "pauses": {"count": 12, "total_duration": 3.2, "speech_to_silence_ratio": 0.55},
        "voice_quality": {"jitter": 0.035, "shimmer": 0.22, "breathiness": 0.42},
    }
    
    short_term_context = []
    long_term_memories = [
        {
            "memory_text": "User has high academic pressure from parents",
            "importance_score": 0.9,
        }
    ]
    episodic_memories = []
    
    character_profile = {
        "name": "Aarav",
        "background": "Empathetic AI companion for Indian youth",
        "traits": ["empathetic", "culturally aware", "patient", "thoughtful"],
        "speech_style": "conversational, validates emotions, offers practical support",
    }
    
    safety_context = {"crisis_risk": "low", "recent_flags": []}
    
    # Call LLM service
    try:
        response = llm_service.analyze_and_respond(
            transcript=transcript,
            language=language,
            acoustic_features=acoustic_features,
            short_term_context=short_term_context,
            long_term_memories=long_term_memories,
            episodic_memories=episodic_memories,
            character_profile=character_profile,
            safety_context=safety_context,
            temperature=0.7,
            max_output_tokens=2000,
        )
        
        print("\n" + "="*80)
        print("GEMINI LLM RESPONSE")
        print("="*80)
        print(f"Text: {response.response_text}")
        print(f"Language: {response.response_language}")
        print(f"Emotion: {response.detected_emotion} (conf: {response.emotion_confidence})")
        print(f"Sentiment: {response.sentiment}")
        print(f"Intent: {response.detected_intent} (conf: {response.intent_confidence})")
        print(f"TTS Prompt: {response.tts_style_prompt}")
        print(f"TTS Speaker: {response.tts_speaker}")
        print(f"Memory Updates: {len(response.memory_updates)}")
        print(f"Safety Flags: Crisis={response.safety_flags.crisis_risk}, "
              f"SelfHarm={response.safety_flags.self_harm_mentioned}")
        print(f"Generation Time: {response.generation_time_ms}ms")
        print("="*80)
        
        # Convert to dict for DB storage
        db_dict = response.to_dict()
        print("\nDatabase-ready dict:")
        print(json.dumps(db_dict, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        llm_service.close()
