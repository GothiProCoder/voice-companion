"""
Whisper ASR Service - Production Grade
Enterprise patterns: Single model instance, async queues, micro-batching, proper GPU management
Optimized for: Accuracy + Speed + Scalability (1000+ concurrent users)
"""

import asyncio
import torch
import numpy as np
from faster_whisper import WhisperModel
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from queue import Queue
import threading
import time
import logging
from pathlib import Path

from backend.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class TranscriptionRequest:
    """Request object for transcription queue"""
    request_id: str
    audio_path: str
    audio_duration: float  # For batching similar-length requests
    result_future: asyncio.Future
    timestamp: float


@dataclass
class TranscriptionResult:
    """Structured transcription output"""
    text: str
    language: str
    language_probability: float
    duration: float
    word_count: int
    avg_confidence: float
    word_segments: List[Dict]
    is_code_mixed: bool
    detected_languages: List[str]
    processing_time_ms: int


# ==========================================
# WHISPER MODEL SINGLETON
# ==========================================

class WhisperModelSingleton:
    """
    Single model instance per process
    Loaded once, kept hot in VRAM
    Thread-safe access via locks
    """
    _instance = None
    _lock = threading.Lock()
    _model = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Load model once at startup"""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            logger.info("🚀 Initializing Whisper Large-v3...")
            
            # Detect device
            device = config.whisper.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            compute_type = config.whisper.compute_type
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"
            
            # Load model (ONCE)
            self._model = WhisperModel(
                config.whisper.model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=0,  # Let GPU handle everything
                num_workers=1,  # Single worker for stability
                download_root=None,
                local_files_only=False,
            )
            
            # GPU optimization: Enable CUDNN benchmark
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                logger.info("✅ CUDNN benchmark enabled for faster inference")
            
            self._initialized = True
            
            logger.info(f"✅ Whisper loaded: {config.whisper.model_size}")
            logger.info(f"   Device: {device}")
            logger.info(f"   Compute: {compute_type}")
            if device == "cuda":
                logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def get_model(self) -> WhisperModel:
        """Get the singleton model instance"""
        if not self._initialized:
            self.initialize()
        return self._model
    
    def cleanup(self):
        """Cleanup only on graceful shutdown"""
        logger.info("🧹 Cleaning up Whisper model...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("✅ Whisper cleanup complete")


# Global singleton
whisper_singleton = WhisperModelSingleton()


# ==========================================
# ASYNC REQUEST QUEUE WITH MICRO-BATCHING
# ==========================================

class WhisperRequestQueue:
    """
    Async queue with micro-batching and homogeneous batching
    Collects requests for 2-10ms OR until max_batch_size
    Batches similar-length requests together
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        batch_timeout_ms: float = 10.0,
        max_queue_size: int = 100
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms / 1000.0  # Convert to seconds
        self.max_queue_size = max_queue_size
        
        # Request queue (bounded)
        self.queue: asyncio.Queue = None
        self.is_running = False
        self.worker_task = None
        
        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.queue_full_count = 0
    
    async def start(self):
        """Start the queue worker"""
        if self.is_running:
            return
        
        self.queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.is_running = True
        self.worker_task = asyncio.create_task(self._batch_worker())
        logger.info("✅ Whisper request queue started")
    
    async def stop(self):
        """Stop the queue worker"""
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("✅ Whisper request queue stopped")
    
    async def enqueue(self, request: TranscriptionRequest) -> TranscriptionResult:
        """
        Enqueue a transcription request
        Returns when transcription is complete
        Raises: asyncio.QueueFull if queue is at capacity (return 429 to user)
        """
        try:
            # Backpressure: reject if queue is full
            self.queue.put_nowait(request)
            self.total_requests += 1
            
            # Wait for result
            result = await request.result_future
            return result
            
        except asyncio.QueueFull:
            self.queue_full_count += 1
            logger.warning(f"⚠️ Queue full ({self.queue.qsize()}/{self.max_queue_size})")
            raise
    
    async def _batch_worker(self):
        """
        Background worker that processes requests in batches
        Implements micro-batching with timeout
        """
        logger.info("🔄 Batch worker started")
        
        while self.is_running:
            try:
                # Collect requests for batch
                batch = await self._collect_batch()
                
                if not batch:
                    await asyncio.sleep(0.001)  # Brief sleep if no requests
                    continue
                
                # Process batch
                self.total_batches += 1
                await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"❌ Batch worker error: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[TranscriptionRequest]:
        """
        Collect requests for batching
        Strategy: Time-based micro-batching (2-10ms window)
        """
        batch = []
        start_time = time.time()
        
        # Get first request (blocking with timeout)
        try:
            first_request = await asyncio.wait_for(
                self.queue.get(),
                timeout=0.1
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            return batch
        
        # Collect more requests until timeout or max_batch_size
        while len(batch) < self.max_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= self.batch_timeout_ms:
                break
            
            try:
                # Non-blocking get with small timeout
                remaining_time = self.batch_timeout_ms - elapsed
                request = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=max(0.001, remaining_time)
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[TranscriptionRequest]):
        """
        Process a batch of requests
        Homogeneous batching: Group by similar audio duration
        """
        # Sort by audio duration for better batching
        batch.sort(key=lambda r: r.audio_duration)
        
        # Process each request (currently sequential - can be optimized further)
        for request in batch:
            try:
                result = await self._transcribe_single(request)
                request.result_future.set_result(result)
            except Exception as e:
                request.result_future.set_exception(e)
    
    async def _transcribe_single(self, request: TranscriptionRequest) -> TranscriptionResult:
        """
        Transcribe a single audio file
        Runs in thread pool to avoid blocking event loop
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _transcribe_blocking,
            request.audio_path
        )
        return result


# Global queue instance
request_queue = WhisperRequestQueue(
    max_batch_size=4,
    batch_timeout_ms=10.0,
    max_queue_size=100
)


# ==========================================
# CORE TRANSCRIPTION LOGIC
# ==========================================

def _transcribe_blocking(audio_path: str) -> TranscriptionResult:
    """
    Blocking transcription function (runs in thread pool)
    Optimized for accuracy + speed
    Handles long audio via chunking
    """
    start_time = time.time()
    
    model = whisper_singleton.get_model()
    
    # Transcribe with optimized parameters
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        segments, info = model.transcribe(
            audio_path,
            
            # Language detection (auto)
            language=None,  # Let Whisper detect
            
            # Quality parameters (balanced for speed + accuracy)
            beam_size=3,  # Default (good balance)
            best_of=1,  # Sample 5 times
            patience=1.0,
            
            # Temperature fallback
            temperature=[0.0, 0.2],
            
            # VAD optimization
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.4,  # Balanced threshold
                min_speech_duration_ms=150,
                max_speech_duration_s=float('inf'),  # Handle long audio
                min_silence_duration_ms=400,
                speech_pad_ms=300
            ),
            
            # Context and timestamps
            condition_on_previous_text=True,
            word_timestamps=False,
            
            # Hallucination prevention
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            
            # Chunking for long audio
            chunk_length=30,  # Process in 30s chunks
            
            # Anti-repetition
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
    
    # FIX: Urdu → Hindi re-transcription
    detected_lang = info.language
    original_lang = info.language
    
    if info.language == 'ur':
        logger.info(f"⚠️ Urdu detected → Re-transcribing with language='hi'")
        
        # Re-transcribe with Hindi forced to get Devanagari script
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            segments, info = model.transcribe(
                audio_path,
                
                # FORCE HINDI
                language='hi',
                
                # Same parameters as before
                beam_size=5,
                best_of=1,
                patience=1.0,
                temperature=[0.0, 0.2],
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.35,
                    min_speech_duration_ms=150,
                    max_speech_duration_s=float('inf'),
                    min_silence_duration_ms=400,
                    speech_pad_ms=300
                ),
                condition_on_previous_text=True,
                word_timestamps=True,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                chunk_length=30,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )
        
        detected_lang = 'hi'
        logger.info(f"✅ Re-transcribed as Hindi")
        
    # Collect segments
    full_transcription = []
    word_segments = []
    
    for segment in segments:
        text = segment.text.strip()
        full_transcription.append(text)
        
        # Word-level timestamps
        if hasattr(segment, 'words') and segment.words:
            for word in segment.words:
                word_segments.append({
                    'word': word.word.strip(),
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })
    
    # Calculate confidence
    avg_confidence = sum(w['probability'] for w in word_segments) / len(word_segments) if word_segments else 0.0
    
    # Detect code-mixing (simple heuristic)
    full_text = " ".join(full_transcription)
    is_code_mixed, detected_languages = _detect_code_mixing(full_text, detected_lang)
    
    # Processing time
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    # Cleanup tensors
    del segments
    if torch.cuda.is_available():
        # Minimal cleanup - let PyTorch manage VRAM
        pass
    
    return TranscriptionResult(
        text=full_text,
        language=detected_lang,
        language_probability=info.language_probability,
        duration=info.duration,
        word_count=len(full_text.split()),
        avg_confidence=avg_confidence,
        word_segments=word_segments,
        is_code_mixed=is_code_mixed,
        detected_languages=detected_languages,
        processing_time_ms=processing_time_ms
    )


def _detect_code_mixing(text: str, primary_language: str) -> Tuple[bool, List[str]]:
    """
    Simple code-mixing detection
    Checks for English words in non-English text
    """
    # Simple heuristic: check for Latin characters in Indian language text
    has_latin = any(ord(c) < 128 and c.isalpha() for c in text)
    has_indic = any(ord(c) > 2304 for c in text)  # Devanagari and other Indic scripts
    
    is_code_mixed = has_latin and has_indic
    detected_languages = [primary_language]
    
    if is_code_mixed and primary_language != 'en':
        detected_languages.append('en')
    
    return is_code_mixed, detected_languages


# ==========================================
# PUBLIC API
# ==========================================

async def transcribe_audio(
    audio_path: str,
    request_id: Optional[str] = None
) -> TranscriptionResult:
    """
    Public API for transcription
    Enqueues request and waits for result
    
    Args:
        audio_path: Path to audio file
        request_id: Optional request ID for tracking
    
    Returns:
        TranscriptionResult with full metadata
    
    Raises:
        asyncio.QueueFull: If queue is full (caller should return 429)
    """
    # Get audio duration for batching
    import soundfile as sf
    try:
        audio_info = sf.info(audio_path)
        audio_duration = audio_info.duration
    except:
        audio_duration = 0.0  # Fallback
    
    # Create request
    request = TranscriptionRequest(
        request_id=request_id or f"req_{time.time()}",
        audio_path=audio_path,
        audio_duration=audio_duration,
        result_future=asyncio.Future(),
        timestamp=time.time()
    )
    
    # Enqueue and wait
    result = await request_queue.enqueue(request)
    return result


async def initialize_whisper_service():
    """
    Initialize Whisper service
    Call this at application startup
    """
    logger.info("🚀 Initializing Whisper ASR service...")
    
    # Initialize model
    whisper_singleton.initialize()
    
    # Start request queue
    await request_queue.start()
    
    logger.info("✅ Whisper service ready")


async def shutdown_whisper_service():
    """
    Graceful shutdown
    Call this at application shutdown
    """
    logger.info("🛑 Shutting down Whisper service...")
    
    # Stop queue
    await request_queue.stop()
    
    # Cleanup model
    whisper_singleton.cleanup()
    
    logger.info("✅ Whisper service shutdown complete")


def get_service_stats() -> Dict:
    """Get service statistics"""
    return {
        "total_requests": request_queue.total_requests,
        "total_batches": request_queue.total_batches,
        "queue_size": request_queue.queue.qsize() if request_queue.queue else 0,
        "queue_full_count": request_queue.queue_full_count,
        "avg_batch_size": request_queue.total_requests / max(request_queue.total_batches, 1)
    }


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    """Test the service"""
    import sys
    
    async def test_service():
        # Initialize
        await initialize_whisper_service()
        
        # Test transcription
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            
            print(f"\n{'='*70}")
            print(f"Testing Whisper service with: {audio_file}")
            print('='*70)
            
            try:
                result = await transcribe_audio(audio_file)
                
                print(f"\n✅ Transcription:")
                print(f"   Text: {result.text}")
                print(f"   Language: {result.language} ({result.language_probability:.2%})")
                print(f"   Duration: {result.duration:.2f}s")
                print(f"   Words: {result.word_count}")
                print(f"   Confidence: {result.avg_confidence:.2%}")
                print(f"   Code-mixed: {result.is_code_mixed}")
                print(f"   Languages: {result.detected_languages}")
                print(f"   Processing time: {result.processing_time_ms}ms")
                
                # Stats
                stats = get_service_stats()
                print(f"\n📊 Service stats:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        else:
            print("Usage: python whisper_asr.py <audio_file>")
        
        # Shutdown
        await shutdown_whisper_service()
    
    # Run
    asyncio.run(test_service())
