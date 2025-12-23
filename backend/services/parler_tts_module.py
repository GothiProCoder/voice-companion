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
from typing import Optional, Dict

import os
import io
import base64
import logging
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

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
    model_name: str = "ai4bharat/indic-parler-tts-pretrained"
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
    """
    audio_array: np.ndarray
    audio_base64_wav: str
    sampling_rate: int
    duration_seconds: float
    generation_time_ms: int


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
        self.config = config or TTSConfig()
        self.device = torch.device(self.config.device)

        logger.info(f"Initializing ParlerTTSService on device={self.config.device}")

        # Optional: Hugging Face token from env
        hf_token = os.getenv("HUGGINGFACE_TOKEN")

        try:
            # Load model
            logger.info(f"Loading Indic Parler TTS model: {self.config.model_name}")
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.config.model_name,
                token=hf_token,
            ).to(self.device)

            # Load tokenizers
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path,
                token=hf_token,
            )
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                token=hf_token,
            )

            self.model.eval()
            self._cache: Dict[str, TTSResponse] = {}

            logger.info("ParlerTTSService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ParlerTTSService: {e}", exc_info=True)
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
            b64_wav = self._encode_wav_to_base64(audio, self.config.sampling_rate)

            gen_time_ms = int((time.time() - start_time) * 1000)

            response = TTSResponse(
                audio_array=audio,
                audio_base64_wav=b64_wav,
                sampling_rate=self.config.sampling_rate,
                duration_seconds=duration_sec,
                generation_time_ms=gen_time_ms,
            )

            if self.config.cache_enabled:
                self._cache[cache_key] = response

            logger.info(
                f"TTS generated: duration={duration_sec:.2f}s, time={gen_time_ms}ms"
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
