# FINAL PERFECTED SOLUTION - audio_processor.py - VAD-CHUNK-FIXED
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import tempfile
import os
import subprocess
import queue
import webrtcvad

logger = logging.getLogger(__name__)

class PerfectAudioProcessor:
    """FINAL PERFECTED: Audio processor with bulletproof WebM handling and frame-perfect VAD."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, conversation_manager=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.transcribe_audio_queue = queue.Queue()
        
        try:
            self.vad = webrtcvad.Vad(3)  # Mode 3: most aggressive
            self.vad_enabled = True
            # VAD requires specific frame lengths (10, 20, or 30 ms)
            self.vad_frame_duration_ms = 30 
            self.vad_frame_size_bytes = int(sample_rate * (self.vad_frame_duration_ms / 1000.0) * 2) # 16-bit PCM
            logger.info(f"✅ FINAL PERFECTED WebRTC VAD initialized with {self.vad_frame_duration_ms}ms frames.")
        except Exception as e:
            self.vad = None
            self.vad_enabled = False
            logger.warning(f"⚠️ WebRTC VAD not available, falling back to energy detection: {e}")
    
    async def process_webm_chunk_understand(self, webm_data: bytes) -> Optional[Dict[str, Any]]:
        """FINAL PERFECTED: Understanding mode with direct WebM to PCM conversion for gap detection"""
        if not self._is_valid_webm_chunk(webm_data): return None
        pcm_data = await self._webm_to_pcm_perfect(webm_data)
        if pcm_data:
            speech_detected = self._detect_speech_perfect(pcm_data)
            return {"pcm_data": pcm_data, "speech_detected": speech_detected}
        return None
            
    def _is_valid_webm_chunk(self, webm_data: bytes) -> bool:
        return bool(webm_data and len(webm_data) > 100)

    async def _webm_to_pcm_perfect(self, webm_data: bytes) -> Optional[bytes]:
        cmd = ['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate), 'pipe:1']
        proc = await asyncio.create_subprocess_exec(*cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pcm_data, stderr = await proc.communicate(input=webm_data)
        if proc.returncode != 0:
            logger.warning(f"FFmpeg failed to convert to PCM: {stderr.decode()}")
            return None
        return pcm_data

    def _detect_speech_perfect(self, pcm_data: bytes) -> bool:
        """FINAL PERFECTED: The CRITICAL FIX for VAD by correctly chunking audio into valid frame sizes."""
        if not self.vad_enabled or not self.vad:
            return self._detect_speech_by_energy(pcm_data)

        num_frames = len(pcm_data) // self.vad_frame_size_bytes
        if num_frames == 0: return False
            
        try:
            for i in range(num_frames):
                start = i * self.vad_frame_size_bytes
                end = start + self.vad_frame_size_bytes
                if self.vad.is_speech(pcm_data[start:end], self.sample_rate):
                    return True # Found speech, no need to check further
        except Exception as e:
            logger.warning(f"VAD frame error, falling back to energy check: {e}")
            return self._detect_speech_by_energy(pcm_data)
        
        return False

    def _detect_speech_by_energy(self, pcm_data: bytes) -> bool:
        if not pcm_data: return False
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        if audio_array.size == 0: return False
        rms_energy = np.sqrt(np.mean(np.square(audio_array.astype(np.float64))))
        return rms_energy > 350
