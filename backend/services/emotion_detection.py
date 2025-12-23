"""
Prosody & Acoustic Feature Extraction Service - Production Grade
Extracts research-backed acoustic features for emotion analysis
Schema: Finalized after in-depth research
"""

import asyncio
import numpy as np
import librosa
import soundfile as sf
import opensmile
from typing import Dict, Optional
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

from backend.config import config

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class ProsodyRequest:
    """Request for prosody extraction"""
    request_id: str
    audio_path: str
    result_future: asyncio.Future
    timestamp: float


@dataclass
class ProsodyResult:
    """Complete acoustic features matching finalized schema"""
    meta_info: Dict
    prosody_pitch: Dict
    energy_loudness: Dict
    voice_quality: Dict
    spectral_timbre: Dict
    rhythm_tempo: Dict
    processing_time_ms: int


# ==========================================
# OPENSMILE SINGLETON
# ==========================================

class OpenSmileSingleton:
    """Single OpenSMILE instance per process"""
    _instance = None
    _lock = asyncio.Lock()
    _smile = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self):
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            logger.info("🚀 Initializing OpenSMILE...")
            
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            
            self._initialized = True
            logger.info("✅ OpenSMILE ready")
    
    def get_smile(self):
        if not self._initialized:
            raise RuntimeError("OpenSMILE not initialized")
        return self._smile


opensmile_singleton = OpenSmileSingleton()


# ==========================================
# CPU THREAD POOL
# ==========================================

import multiprocessing
NUM_WORKERS = max(2, multiprocessing.cpu_count() - 1)
prosody_executor = ThreadPoolExecutor(
    max_workers=NUM_WORKERS,
    thread_name_prefix="prosody_worker"
)

logger.info(f"✅ Prosody thread pool: {NUM_WORKERS} workers")


# ==========================================
# ASYNC REQUEST QUEUE
# ==========================================

class ProsodyRequestQueue:
    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.queue: asyncio.Queue = None
        self.is_running = False
        self.total_requests = 0
        self.queue_full_count = 0
    
    async def start(self):
        if self.is_running:
            return
        self.queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.is_running = True
        logger.info("✅ Prosody request queue started")
    
    async def stop(self):
        self.is_running = False
        logger.info("✅ Prosody request queue stopped")
    
    async def enqueue(self, request: ProsodyRequest) -> ProsodyResult:
        try:
            self.total_requests += 1
            result = await self._process_request(request)
            return result
        except Exception as e:
            self.queue_full_count += 1
            logger.error(f"❌ Prosody extraction error: {e}")
            raise
    
    async def _process_request(self, request: ProsodyRequest) -> ProsodyResult:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            prosody_executor,
            _extract_features_blocking,
            request.audio_path
        )
        return result


prosody_queue = ProsodyRequestQueue(max_queue_size=100)


# ==========================================
# CORE FEATURE EXTRACTION
# ==========================================

def _extract_features_blocking(audio_path: str) -> ProsodyResult:
    """Extract all acoustic features matching finalized schema"""
    start_time = time.time()
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract feature groups
        meta_info = _extract_meta_info(y, sr, duration)
        prosody_pitch = _extract_prosody_pitch(y, sr)
        energy_loudness = _extract_energy_loudness(y, sr)
        voice_quality = _extract_voice_quality(audio_path)
        spectral_timbre = _extract_spectral_timbre(y, sr)
        rhythm_tempo = _extract_rhythm_tempo(y, sr, duration)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return ProsodyResult(
            meta_info=meta_info,
            prosody_pitch=prosody_pitch,
            energy_loudness=energy_loudness,
            voice_quality=voice_quality,
            spectral_timbre=spectral_timbre,
            rhythm_tempo=rhythm_tempo,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}")
        return _get_empty_features()


def _extract_meta_info(y: np.ndarray, sr: int, duration: float) -> Dict:
    """Extract metadata: duration, voiced_ratio, SNR"""
    try:
        # Voice activity detection
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.percentile(rms, 30)
        voiced_frames = np.sum(rms > threshold)
        total_frames = len(rms)
        voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0
        
        # SNR estimation (signal-to-noise ratio)
        # Simple approach: compare voiced vs unvoiced energy
        voiced_energy = np.mean(rms[rms > threshold]) if np.any(rms > threshold) else 0
        noise_energy = np.mean(rms[rms <= threshold]) if np.any(rms <= threshold) else 1e-10
        snr_db = 10 * np.log10(voiced_energy / noise_energy) if noise_energy > 0 else 0.0
        
        return {
            "duration_sec": round(duration, 1),
            "voiced_ratio": round(float(voiced_ratio), 2),
            "snr_db": round(float(snr_db), 1)
        }
    except Exception as e:
        logger.error(f"Meta extraction error: {e}")
        return {"duration_sec": 0.0, "voiced_ratio": 0.0, "snr_db": 0.0}


def _extract_prosody_pitch(y: np.ndarray, sr: int) -> Dict:
    """Extract pitch features: F0 mean, std, range, slope"""
    try:
        # Extract F0 using pyin (more accurate than piptrack)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sr
        )
        
        # Filter valid F0 values
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 0:
            f0_mean = float(np.mean(f0_valid))
            f0_std = float(np.std(f0_valid))
            f0_min = float(np.min(f0_valid))
            f0_max = float(np.max(f0_valid))
            f0_range = f0_max - f0_min
            
            # F0 slope (linear regression)
            x = np.arange(len(f0_valid))
            if len(x) > 1:
                f0_slope = float(np.polyfit(x, f0_valid, 1)[0])
            else:
                f0_slope = 0.0
            
            return {
                "f0_mean_hz": round(f0_mean, 1),
                "f0_std_hz": round(f0_std, 1),
                "f0_range_hz": round(f0_range, 1),
                "f0_slope": round(f0_slope, 2)
            }
        else:
            return _get_empty_prosody_pitch()
            
    except Exception as e:
        logger.error(f"Prosody extraction error: {e}")
        return _get_empty_prosody_pitch()


def _extract_energy_loudness(y: np.ndarray, sr: int) -> Dict:
    """Extract energy features in dB: RMS mean, std, peak, silent segments"""
    try:
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Convert to dB (reference: max possible value = 1.0)
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        
        rms_mean_db = float(np.mean(rms_db))
        rms_std_db = float(np.std(rms_db))
        rms_peak_db = float(np.max(rms_db))
        
        # Silent segments (absolute silence)
        silence_threshold = np.percentile(rms, 5)  # Bottom 5%
        silent_segments = 0
        in_silence = False
        
        for i in range(len(rms)):
            if rms[i] < silence_threshold:
                if not in_silence:
                    silent_segments += 1
                    in_silence = True
            else:
                in_silence = False
        
        return {
            "rms_mean_db": round(rms_mean_db, 1),
            "rms_std_db": round(rms_std_db, 1),
            "rms_peak_db": round(rms_peak_db, 1),
            "silent_segments_count": silent_segments
        }
        
    except Exception as e:
        logger.error(f"Energy extraction error: {e}")
        return _get_empty_energy_loudness()


def _extract_voice_quality(audio_path: str) -> Dict:
    """Extract voice quality: jitter, shimmer, HNR, alpha_ratio"""
    try:
        smile = opensmile_singleton.get_smile()
        features = smile.process_file(audio_path)
        
        # Map OpenSMILE features
        jitter = 0.0
        shimmer = 0.0
        hnr = 0.0
        alpha_ratio = 0.0
        
        if 'jitterLocal_sma3nz_amean' in features.columns:
            jitter = float(features['jitterLocal_sma3nz_amean'].values[0])
        
        if 'shimmerLocaldB_sma3nz_amean' in features.columns:
            shimmer = float(features['shimmerLocaldB_sma3nz_amean'].values[0])
        
        if 'HNRdBACF_sma3nz_amean' in features.columns:
            hnr = float(features['HNRdBACF_sma3nz_amean'].values[0])
        
        if 'alphaRatio_sma3_amean' in features.columns:
            alpha_ratio = float(features['alphaRatio_sma3_amean'].values[0])
        
        return {
            "jitter_percent": round(jitter * 100, 2),  # Convert to percentage
            "shimmer_db": round(shimmer, 2),
            "hnr_db": round(hnr, 1),
            "alpha_ratio": round(alpha_ratio, 1)
        }
        
    except Exception as e:
        logger.error(f"Voice quality extraction error: {e}")
        return _get_empty_voice_quality()


def _extract_spectral_timbre(y: np.ndarray, sr: int) -> Dict:
    """Extract spectral features: formants, centroid, ZCR, MFCCs"""
    try:
        # Formants (F1, F2, F3) estimation
        formants = _estimate_formants(y, sr)
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = float(np.mean(centroid))
        
        # Zero Crossing Rate (fricative density)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))
        
        # MFCCs (audio fingerprint)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_stats = {
            "mfcc_1_mean": round(float(np.mean(mfccs[0])), 1),
            "mfcc_2_mean": round(float(np.mean(mfccs[1])), 1)
        }
        
        return {
            "formants": formants,
            "spectral_centroid_hz": round(centroid_mean, 1),
            "zcr_rate": round(zcr_mean, 2),
            "mfcc_stats": mfcc_stats
        }
        
    except Exception as e:
        logger.error(f"Spectral extraction error: {e}")
        return _get_empty_spectral_timbre()


def _estimate_formants(y: np.ndarray, sr: int) -> Dict:
    """
    Estimate formants F1, F2, F3 using LPC (Linear Predictive Coding)
    Formants = resonant frequencies of the vocal tract
    """
    try:
        # Apply pre-emphasis filter
        pre_emphasized = librosa.effects.preemphasis(y)
        
        # LPC analysis (order = sr/1000 + 2, typical for formant estimation)
        lpc_order = int(sr / 1000) + 2
        lpc_coeffs = librosa.lpc(pre_emphasized, order=lpc_order)
        
        # Find roots of LPC polynomial
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]  # Keep positive imaginary parts
        
        # Convert to frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))
        
        # Filter valid formant range (typically 200-4000 Hz)
        formant_freqs = freqs[(freqs > 200) & (freqs < 4000)]
        formant_freqs = np.sort(formant_freqs)
        
        # Extract first 3 formants
        f1 = float(formant_freqs[0]) if len(formant_freqs) > 0 else 500.0
        f2 = float(formant_freqs[1]) if len(formant_freqs) > 1 else 1500.0
        f3 = float(formant_freqs[2]) if len(formant_freqs) > 2 else 2500.0
        
        return {
            "f1_mean_hz": round(f1, 0),
            "f2_mean_hz": round(f2, 0),
            "f3_mean_hz": round(f3, 0)
        }
        
    except Exception as e:
        logger.error(f"Formant extraction error: {e}")
        return {"f1_mean_hz": 500, "f2_mean_hz": 1500, "f3_mean_hz": 2500}


def _extract_rhythm_tempo(y: np.ndarray, sr: int, duration: float) -> Dict:
    """Extract rhythm features: speech rate, articulation rate, pauses"""
    try:
        # Syllable detection using onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onsets, sr=sr)
        
        # Approximate syllables (onsets are rough proxy)
        syllable_count = len(onsets)
        speech_rate = syllable_count / duration if duration > 0 else 0.0
        
        # Split into speech/silence for articulation rate
        intervals = librosa.effects.split(y, top_db=30)
        speech_duration = sum([(end - start) / sr for start, end in intervals])
        articulation_rate = syllable_count / speech_duration if speech_duration > 0 else 0.0
        
        # Pause analysis
        pause_duration_total = duration - speech_duration
        
        # Count pauses (gaps between speech intervals)
        pause_count = len(intervals) - 1 if len(intervals) > 1 else 0
        pause_frequency_per_min = (pause_count / duration) * 60 if duration > 0 else 0.0
        
        return {
            "speech_rate_syllables_sec": round(speech_rate, 1),
            "articulation_rate": round(articulation_rate, 1),
            "pause_duration_total_sec": round(pause_duration_total, 1),
            "pause_frequency_per_min": round(pause_frequency_per_min, 1)
        }
        
    except Exception as e:
        logger.error(f"Rhythm extraction error: {e}")
        return _get_empty_rhythm_tempo()


# ==========================================
# FALLBACK EMPTY FEATURES
# ==========================================

def _get_empty_features() -> ProsodyResult:
    return ProsodyResult(
        meta_info={"duration_sec": 0.0, "voiced_ratio": 0.0, "snr_db": 0.0},
        prosody_pitch=_get_empty_prosody_pitch(),
        energy_loudness=_get_empty_energy_loudness(),
        voice_quality=_get_empty_voice_quality(),
        spectral_timbre=_get_empty_spectral_timbre(),
        rhythm_tempo=_get_empty_rhythm_tempo(),
        processing_time_ms=0
    )

def _get_empty_prosody_pitch():
    return {"f0_mean_hz": 0.0, "f0_std_hz": 0.0, "f0_range_hz": 0.0, "f0_slope": 0.0}

def _get_empty_energy_loudness():
    return {"rms_mean_db": 0.0, "rms_std_db": 0.0, "rms_peak_db": 0.0, "silent_segments_count": 0}

def _get_empty_voice_quality():
    return {"jitter_percent": 0.0, "shimmer_db": 0.0, "hnr_db": 0.0, "alpha_ratio": 0.0}

def _get_empty_spectral_timbre():
    return {
        "formants": {"f1_mean_hz": 500, "f2_mean_hz": 1500, "f3_mean_hz": 2500},
        "spectral_centroid_hz": 0.0,
        "zcr_rate": 0.0,
        "mfcc_stats": {"mfcc_1_mean": 0.0, "mfcc_2_mean": 0.0}
    }

def _get_empty_rhythm_tempo():
    return {
        "speech_rate_syllables_sec": 0.0,
        "articulation_rate": 0.0,
        "pause_duration_total_sec": 0.0,
        "pause_frequency_per_min": 0.0
    }


# ==========================================
# PUBLIC API
# ==========================================

async def extract_prosody_features(
    audio_path: str,
    request_id: Optional[str] = None
) -> ProsodyResult:
    """
    Extract prosody features from audio
    Returns finalized schema with all acoustic features
    """
    request = ProsodyRequest(
        request_id=request_id or f"prosody_{time.time()}",
        audio_path=audio_path,
        result_future=asyncio.Future(),
        timestamp=time.time()
    )
    
    result = await prosody_queue.enqueue(request)
    return result


async def initialize_prosody_service():
    """Initialize prosody extraction service"""
    logger.info("🚀 Initializing Prosody service...")
    await opensmile_singleton.initialize()
    await prosody_queue.start()
    logger.info("✅ Prosody service ready")


async def shutdown_prosody_service():
    """Graceful shutdown"""
    logger.info("🛑 Shutting down Prosody service...")
    await prosody_queue.stop()
    prosody_executor.shutdown(wait=True)
    logger.info("✅ Prosody service shutdown complete")


def get_prosody_stats() -> Dict:
    """Get service statistics"""
    return {
        "total_requests": prosody_queue.total_requests,
        "queue_full_count": prosody_queue.queue_full_count,
        "thread_pool_workers": NUM_WORKERS
    }


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    import sys
    import json
    
    async def test_prosody():
        await initialize_prosody_service()
        
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            
            print(f"\n{'='*70}")
            print(f"Testing Prosody Extraction: {audio_file}")
            print('='*70)
            
            result = await extract_prosody_features(audio_file)
            
            features = {
                "meta_info": result.meta_info,
                "prosody_pitch": result.prosody_pitch,
                "energy_loudness": result.energy_loudness,
                "voice_quality": result.voice_quality,
                "spectral_timbre": result.spectral_timbre,
                "rhythm_tempo": result.rhythm_tempo,
                "processing_time_ms": result.processing_time_ms
            }
            
            print(f"\n✅ Extracted Features:")
            print(json.dumps(features, indent=2))
            
            print(f"\n📊 Stats: {get_prosody_stats()}")
        else:
            print("Usage: python emotion_detection.py <audio_file>")
        
        await shutdown_prosody_service()
    
    asyncio.run(test_prosody())
