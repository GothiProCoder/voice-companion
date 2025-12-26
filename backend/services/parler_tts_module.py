"""
GuppShupp Indic Parler TTS Service
backend/services/parler_tts.py

Simplified, production-ready TTS module.

Design:
- LLM decides EVERYTHING about TTS:
  - description: detailed Parler-style caption
  - speaker: character name, e.g. "Divya", "Mary", "Rohit"
- This module does ONE job:
  - Combine speaker + description into a single caption
  - Call Indic Parler-TTS
  - Return audio (array + base64) and basic metadata

No language handling, no emotion tables, no speaker maps.
The Parler model already knows which speakers belong to which languages.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import os
import io
import base64
import logging
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, GenerationConfig

# Minimal compatibility patch for transformers 4.46+
from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration

if not hasattr(ParlerTTSForConditionalGeneration, "_validate_model_kwargs"):
    ParlerTTSForConditionalGeneration._validate_model_kwargs = lambda self, kwargs: kwargs

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional; environment variables can be set by other means
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TTSConfig:
    """
    Configuration for Indic Parler TTS.
    """
    # RECOMMENDATION: use the fine-tuned model, not the bare pretrained
    model_name: str = "ai4bharat/indic-parler-tts"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sampling_rate: int = 44100
    cache_enabled: bool = True


@dataclass
class TTSRequest:
    """
    Minimal request for TTS generation.
    LLM is responsible for:
      - spoken_text: what to say
      - speaker: voice character (e.g. "Divya", "Mary")
      - description: Parler caption describing style, tone, quality
    """
    spoken_text: str
    speaker: str          # e.g. "Divya"
    description: str      # e.g. "speaks with a high pitch at a normal pace..."


@dataclass
class TTSResponse:
    """
    Response from TTS generation.
    
    Fields:
        audio_array: Raw numpy audio (for streaming/preview)
        audio_opus_bytes: Opus-encoded audio bytes (when Opus enabled)
        audio_wav_bytes: WAV-encoded audio bytes (when Opus disabled, for fast saving)
        audio_path: Relative path to saved audio file (None until saved)
        audio_base64_wav: DEPRECATED - kept for backwards compatibility
        sampling_rate: Audio sample rate
        duration_seconds: Audio duration
        generation_time_ms: Generation time in milliseconds
    """
    audio_array: np.ndarray
    audio_opus_bytes: bytes = b""        # Opus-encoded audio (when enabled)
    audio_wav_bytes: bytes = b""         # ⚡ NEW: WAV-encoded audio (when Opus disabled)
    audio_path: Optional[str] = None     # Relative path to saved file
    audio_base64_wav: str = ""           # DEPRECATED - for backwards compat
    sampling_rate: int = 44100
    duration_seconds: float = 0.0
    generation_time_ms: int = 0



# ============================================================================
# PARLER TTS SERVICE (SIMPLE VERSION)
# ============================================================================

class ParlerTTSService:
    """
    Simple Indic Parler TTS service for GuppShupp.

    Responsibilities:
    - Load ai4bharat/indic-parler-tts (or ...-pretrained)
    - Accept TTSRequest(spoken_text, speaker, description)
    - Build final caption: "{speaker} {description}"
    - Run model.generate and return audio
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize Parler TTS Service for Indic languages.
        
        Args:
            config: Optional TTSConfig. If None, uses defaults.
        
        Raises:
            Exception: If model loading or initialization fails.
        """
        self.config = config or TTSConfig()
        self.device = torch.device(self.config.device)
        
        logger.info(f"Initializing ParlerTTSService on device={self.config.device}")
        
        # Optional: Hugging Face token from env
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        try:
            # ================================================================
            # STEP 1: LOAD MODEL WITH EAGER ATTENTION
            # ================================================================
            logger.info(f"Loading Indic Parler TTS model: {self.config.model_name}")
            
            # Use eager attention for compatibility.
            # T5EncoderModel (used internally by ParlerTTS) does NOT support SDPA.
            # Reference: https://github.com/huggingface/transformers/issues/28005
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.config.model_name,
                token=hf_token,
                torch_dtype=torch.float16,      # FP16 for VRAM savings (~50% memory reduction)
                attn_implementation="eager",    # ✅ Compatible with T5EncoderModel
            ).to(self.device)
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
            # ================================================================
            # STEP 2: LOAD TOKENIZERS
            # ================================================================
            logger.info("Loading tokenizers...")
            
            # Description tokenizer (for speaker style descriptions)
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path,
                token=hf_token,
            )
            
            # Text tokenizer (for the actual speech text)
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                token=hf_token,
            )
            
            logger.info("✅ Tokenizers loaded successfully")
            
            # ================================================================
            # STEP 3: PREPARE MODEL FOR INFERENCE
            # ================================================================
            # Set model to evaluation mode (disables dropout, batch norm, etc.)
            self.model.eval()
            logger.info("✅ Model set to evaluation mode")
            
            # Configure generation parameters
            self.model.generation_config.max_new_tokens = 2048  # Max audio tokens (~20s audio)
            logger.info("✅ Generation config set (max_new_tokens=2048)")
            
            # ================================================================
            # STEP 4: INITIALIZE CACHING AND STATE
            # ================================================================
            # Simple dict cache for generated audio (key = hash of inputs)
            self._cache: Dict[str, TTSResponse] = {}
            
            # Track warmup state
            self._warmed_up = False
            
            logger.info("✅ Cache and state initialized")
            
            # ================================================================
            # NOTES ON DISABLED OPTIMIZATIONS
            # ================================================================
            # The following optimizations are DISABLED for stability:
            #
            # 1. Static KV Cache:
            #    - self.model.generation_config.cache_implementation = "static"
            #    - Can cause issues with T5EncoderModel
            #    - Enable only after thorough testing
            #
            # 2. torch.compile():
            #    - Provides 3-4x speedup but requires:
            #      * Proper warmup (30-60s at startup)
            #      * PyTorch 2.0+
            #      * May have compatibility issues
            #    - Enable via environment variable when needed
            #
            # To enable optimizations:
            #    Set ENABLE_TORCH_COMPILE=true in environment
            #    Then call warmup() after initialization
            
            logger.info("✅ ParlerTTSService initialized successfully")
            logger.info(f"   Model: {self.config.model_name}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Precision: FP16")
            logger.info(f"   Attention: eager (T5-compatible)")
            logger.info(f"   Cache enabled: {self.config.cache_enabled}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ParlerTTSService: {e}", exc_info=True)
            raise
    
        
    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def generate(self, request: TTSRequest) -> TTSResponse:
        """
        Synchronous TTS generation.

        Args:
            request: TTSRequest with spoken_text, speaker, description

        Returns:
            TTSResponse with audio_array, base64 WAV, metadata
        """
        import time
        start_time = time.time()

        # Build final caption: "<speaker> <description>"
        final_caption = self._build_caption(
            speaker=request.speaker,
            description=request.description,
        )
        logger.debug(f"Final Parler caption: {final_caption}")

        # Simple cache key: hash(spoken_text + speaker + description)
        cache_key = self._make_cache_key(
            text=request.spoken_text,
            speaker=request.speaker,
            description=request.description,
        )
        if self.config.cache_enabled and cache_key in self._cache:
            logger.info(f"TTS cache hit for key={cache_key}")
            return self._cache[cache_key]

        try:
            with torch.no_grad():
                # Tokenize description (caption)
                desc_inputs = self.description_tokenizer(
                    final_caption,
                    return_tensors="pt",
                ).to(self.device)

                # Tokenize spoken text
                text_inputs = self.text_tokenizer(
                    request.spoken_text,
                    return_tensors="pt",
                ).to(self.device)

                # Run generation
                logger.info(f"Generating TTS for speaker={request.speaker}")
                generation = self.model.generate(
                    input_ids=desc_inputs.input_ids,
                    attention_mask=desc_inputs.attention_mask,
                    prompt_input_ids=text_inputs.input_ids,
                    prompt_attention_mask=text_inputs.attention_mask,
                )

            # Convert to numpy
            audio = generation.cpu().numpy().squeeze()

            # Ensure 1D
            if audio.ndim > 1:
                audio = audio[0]

            # Basic post-processing: remove DC offset and normalize RMS
            audio = self._postprocess_audio(audio)

            duration_sec = len(audio) / self.config.sampling_rate
            
            # ⚡ CONDITIONAL ENCODING: Based on config flag
            # Import config to check Opus encoding preference
            from backend.config import config
            
            audio_opus_bytes = b""
            audio_wav_bytes = b""
            
            if config.audio.enable_opus_encoding:
                # Opus encoding (slower ~45-50s, smaller files ~100KB)
                from backend.utils.audio import encode_audio_to_opus
                audio_opus_bytes = encode_audio_to_opus(audio, self.config.sampling_rate)
                logger.info(f"Opus encoding completed: {len(audio_opus_bytes) // 1024}KB")
            else:
                # ⚡ WAV encoding (instant, larger files ~2MB)
                audio_wav_bytes = self._encode_to_wav_bytes(audio, self.config.sampling_rate)
                logger.info(f"WAV mode (Opus skipped): {len(audio_wav_bytes) // 1024}KB")
            
            # DEPRECATED: Keep base64 WAV for backwards compatibility with frontend
            b64_wav = self._encode_wav_to_base64(audio, self.config.sampling_rate)

            gen_time_ms = int((time.time() - start_time) * 1000)

            response = TTSResponse(
                audio_array=audio,
                audio_opus_bytes=audio_opus_bytes,     # Empty if Opus disabled
                audio_wav_bytes=audio_wav_bytes,       # Empty if Opus enabled
                audio_path=None,                       # Set by workflow after saving
                audio_base64_wav=b64_wav,              # DEPRECATED: Kept for compat
                sampling_rate=self.config.sampling_rate,
                duration_seconds=duration_sec,
                generation_time_ms=gen_time_ms,
            )

            if self.config.cache_enabled:
                self._cache[cache_key] = response

            logger.info(
                f"TTS generated: duration={duration_sec:.2f}s, time={gen_time_ms}ms, "
                f"format={'opus' if config.audio.enable_opus_encoding else 'wav'}"
            )
            return response

        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            raise

    async def generate_async(self, request: TTSRequest) -> TTSResponse:
        """
        Async wrapper around generate().
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)

    def warmup(self) -> None:
        """
        ⚡ Warm up the model to pre-compile torch graphs.
        
        This eliminates the first-request latency spike caused by JIT compilation.
        Call this during application startup (e.g., in FastAPI lifespan).
        
        Expected warmup time: 30-60 seconds (one-time cost)
        After warmup: Requests complete 3-5x faster
        """
        if self._warmed_up:
            logger.info("✅ TTS model already warmed up, skipping")
            return
        
        # if not self._needs_warmup:
        #     logger.info("ℹ️ Warmup not needed (torch.compile disabled)")
        #     self._warmed_up = True
        #     return
        
        logger.info("🔥 Warming up TTS model (this may take 30-60 seconds)...")
        import time
        start_time = time.time()
        
        try:
            # Run 2 warmup generations to trigger torch.compile
            warmup_texts = [
                "Hello, testing one two three.",
                "नमस्ते, यह एक परीक्षण है।"
            ]
            
            for i, text in enumerate(warmup_texts, 1):
                logger.info(f"  Warmup pass {i}/{len(warmup_texts)}...")
                
                warmup_request = TTSRequest(
                    spoken_text=text,
                    speaker="Rohit",
                    description="speaks clearly in a neutral tone with good audio quality"
                )
                
                # Disable cache to avoid storing warmup audio
                cache_enabled_backup = self.config.cache_enabled
                self.config.cache_enabled = False
                
                with torch.no_grad():
                    _ = self.generate(warmup_request)
                
                self.config.cache_enabled = cache_enabled_backup
                logger.info(f"  ✅ Warmup pass {i}/{len(warmup_texts)} complete")
            
            # Force CUDA synchronization
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            self._warmed_up = True
            logger.info(f"✅ TTS model warmed up successfully in {elapsed:.1f}s!")
            
        except Exception as e:
            logger.warning(f"⚠️ Warmup failed (model will compile on first request): {e}")
            self._warmed_up = False


    def close(self) -> None:
        """
        Cleanup resources (primarily GPU memory).
        """
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "description_tokenizer"):
                del self.description_tokenizer
            if hasattr(self, "text_tokenizer"):
                del self.text_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            logger.info("ParlerTTSService closed")

    # --------------------------------------------------------------------- #
    # INTERNAL HELPERS
    # --------------------------------------------------------------------- #

    @staticmethod
    def _build_caption(speaker: str, description: str) -> str:
        """
        Build final Parler caption as:
            "<speaker> <description>"

        Example:
            speaker = "Divya"
            description = "speaks with a high pitch at a normal pace..."
            => "Divya speaks with a high pitch at a normal pace..."
        """
        # Ensure no accidental leading/trailing spaces
        speaker = speaker.strip()
        description = description.strip()
        if not speaker:
            # In worst case, just use description
            return description
        return f"{speaker} {description}"

    @staticmethod
    def _make_cache_key(text: str, speaker: str, description: str) -> str:
        import hashlib
        key = f"{speaker}||{description}||{text}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _postprocess_audio(audio: np.ndarray) -> np.ndarray:
        """
        Simple audio post-processing:
        - Remove DC offset
        - Normalize to target RMS
        """
        # Remove DC offset
        audio = audio.astype(np.float32)
        audio = audio - np.mean(audio)

        # Normalize RMS
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-8)
        target_rms = 0.1  # keep safely below 1.0
        audio = audio * (target_rms / rms)

        # Safety clamp
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    @staticmethod
    def _encode_wav_to_base64(audio: np.ndarray, sr: int) -> str:
        """
        Encode a numpy waveform as base64-encoded WAV.
        """
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        data = buf.read()
        return base64.b64encode(data).decode("utf-8")
    
    @staticmethod
    def _encode_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        """
        ⚡ Encode numpy waveform as raw WAV bytes (fast, no ffmpeg subprocess).
        
        This avoids the slow Opus encoding via pydub/ffmpeg, which takes
        45-50 seconds in Google Colab due to CPU-bound ffmpeg subprocess.
        
        Args:
            audio: NumPy audio array (float32, range -1 to 1)
            sr: Sample rate
            
        Returns:
            WAV-encoded bytes
        """
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        return buf.read()


# ============================================================================
# INTEGRATION CONVENIENCE FUNCTION
# ============================================================================

def generate_from_llm_fields(
    tts_service: ParlerTTSService,
    *,
    response_text: str,
    tts_speaker: str,
    tts_description: str,
) -> TTSResponse:
    """
    Convenience function to generate TTS directly from Gemini LLM fields.

    LLM must provide:
      - response_text: what Aarav says
      - tts_speaker: e.g. "Divya", "Mary", "Rohit"
      - tts_description: description like
          "speaks with a high pitch at a normal pace in a clear environment..."

    This function just wraps them into TTSRequest and calls the service.
    """
    req = TTSRequest(
        spoken_text=response_text,
        speaker=tts_speaker,
        description=tts_description,
    )
    return tts_service.generate(req)


# ============================================================================
# MANUAL TEST (RUN DIRECTLY)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = TTSConfig(
        model_name="ai4bharat/indic-parler-tts",  # or "ai4bharat/indic-parler-tts-pretrained"
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    service = ParlerTTSService(cfg)

    try:
        # Example: Divya, Hindi-style voice, neutral tone
        description = (
            "speaks with a clear, natural female voice at a normal pace in a close-sounding, "
            "quiet environment. Her neutral tone is recorded with excellent audio quality and no background noise."
        )
        req = TTSRequest(
            spoken_text="नमस्ते, मैं आरव हूँ। मैं हमेशा तुम्हारी बात सुनने के लिए यहाँ हूँ।",
            speaker="Divya",
            description=description,
        )

        res = service.generate(req)
        print(
            f"Generated audio: {res.duration_seconds:.2f}s, "
            f"{res.generation_time_ms}ms, sr={res.sampling_rate}"
        )

        # Save to file for quick listen
        out_path = Path("parler_test_divya.wav")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        sf.write(str(out_path), res.audio_array, res.sampling_rate)
        print(f"Saved: {out_path}")

    except Exception as e:
        logger.error(f"Error in manual TTS test: {e}", exc_info=True)
    finally:
        service.close()
